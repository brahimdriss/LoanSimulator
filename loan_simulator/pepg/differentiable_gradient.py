"""
Differentiable (reparameterized) performative gradient for PePG.

Replaces the dead `transition_component` / `reward_component` terms in
`PePGAgentV2._compute_standard_gradient` (built from `torch.FloatTensor`
constants with no `grad_fn`) with a single, fully-connected computation
graph: a mean-field "shadow" rollout of (mu_g, lambda_g) driven by the
policy's own reparameterized action.

IMPORTANT (found via direct diagnostic, not assumed): an earlier version of
this rollout fed the policy a single "representative applicant" per group
per timestep, built from X = mu_g (the group MEAN wealth). That meant every
gradient step only ever showed the policy states clustered around the
slowly-drifting mean -- it never saw the real population's spread of
individual wealth, so nothing ever taught it that X (or the correlated
theta_approval_prob) should affect its decision at all. Verified directly:
a trained policy's output moved by <0.001 across the full observed X range
(X=5, theta_approval_prob=0.08, to X=190, theta_approval_prob=0.93) while
approving ~98% either way -- i.e. it had not learned to look at who it was
deciding about, not "rationally decided volume doesn't matter." Fixed by
sampling a real batch of individuals (their actual current wealth, default
probability, loan amount -- drawn straight from env.current_X_male/female
and theta_params.individual_*) each step, running all of them through the
policy in one batched call, and aggregating wealth/reward from each
individual's own outcome instead of one point estimate. This is what
actually gives the gradient a reason to discriminate between applicants.

alpha_g, beta_g (Hawkes excitation/decay) are FIXED constants owned by the
environment's creator -- identical for whichever agent (PG or PePG) is
deployed, read directly off `env.alpha_R/beta_R/alpha_B/beta_B`, never
learned. This does NOT make the Hawkes dynamics agent-independent: the
excitation accumulator S_g(t) that alpha_g multiplies is built entirely
from the agent's own past decisions (`new_events` below), so a decision
today still reshapes lambda_g several steps into the future -- alpha,beta
only fix the SHAPE of that reaction (how strong, how fast it decays), not
whether it happens. That multi-step chain (decision -> excitation ->
future lambda -> future state -> future decision -> future reward) is
exactly what one `backward()` through the whole unrolled episode credits;
with alpha,beta fixed, there is no separate "Term 2 object" left to
construct -- correctly-computed Term 1, over the full trajectory, already
is the complete gradient.

Why this needs its own rollout instead of reusing the real simulator:
the real simulator advances mu_g/lambda_g using a HARD, np.random-sampled
approve/deny decision and a discrete Python list of past event timestamps
-- both steps are non-differentiable by construction. This module mirrors
the same closed-form dynamics (see environment.py: _f_networth_to_rate,
_compute_lambda_R/_B, _generate_timestep_applications, step) but replaces
each hard sample with its exact conditional expectation in the policy's
continuous output, and replaces the growing event-time list with the
Markovian (exponentially-decaying running sum) form of the same Hawkes
kernel -- so the whole trajectory is one torch graph and one
`backward()` gives the complete gradient w.r.t. policy_net.

What is NOT in this graph, on purpose:
  - lambda_wealth / lambda_approval (the Lagrangian dual variables).
    They stay on PePGAgentV2's existing separate dual-ascent optimizer.
    Folding them in here would push them the wrong direction: the dual
    ascends on constraint violation, the primal ascends on reward, and
    reward contains -lambda*violation -- maximizing reward w.r.t. lambda
    directly drives lambda to 0, which defeats the constraint. This is a
    saddle-point problem, not a single joint objective.
  - alpha_g, beta_g (Hawkes shape): environment constants, see above.
  - each individual's default probability and loan amount: fixed traits
    (TransitionParameterLearner), not policy parameters. They enter the
    recursion as plain floats/arrays, deliberately with no grad_fn --
    only the policy's OWN action for that individual is differentiable.
"""

from dataclasses import dataclass

from ..environment import MATTHEW_C

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta


# ---------------------------------------------------------------------------
# Reward/entropy normalisation
# ---------------------------------------------------------------------------

class RunningNormalizer:
    """
    Welford running mean/std, same scheme PolicyGradientAgent.train_episode
    uses for its reward normalisation. Needed here for the same reason: raw
    step rewards run into the thousands (see differentiable_episode_return),
    so an entropy bonus added directly on that scale would be numerically
    swamped and have no effect -- PG's entropy_coef only has teeth because
    PG normalises reward first. Mean/var/n persist across episodes (own it
    once per agent, not per call) so the running estimate actually
    stabilises over training, matching PG's convention.
    """

    WARMUP_N = 2     # skip ONLY the degenerate n=1 sample (see below)
    MIN_STD = 1e-3   # floor, so a near-constant reward can't blow up 1/std

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.n = 0

    def update_and_get_stats(self, x: float):
        """Update running stats with x (a plain float -- pass r.detach().item(),
        never the tensor itself, or this silently detaches the graph) and
        return (mean, std) AFTER the update. Apply those to the ORIGINAL
        tensor yourself ((r - mean) / std) so r's grad_fn is preserved --
        this method only ever touches plain floats."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.var = (self.var * (self.n - 1) + delta * delta2) / self.n

        # WARMUP GUARD. Deliberately minimal: only n=1 is degenerate, so
        # only n=1 is skipped. A longer warmup (200) was tried and is worse
        # -- it runs a whole episode on RAW rewards, and more importantly it
        # is unnecessary: from n=2 onward the variance is well defined.
        # At n=1 the sample variance is identically zero
        # (delta2 = x - mean = 0), so a bare sqrt(var) returns ~0 and the
        # caller's (r - mean)/std multiplies the GRADIENT by ~1e8 -- the
        # value is 0 but std is a detached constant, so 1/std scales
        # d(r)/dtheta directly. Measured: ||grad|| = 241,291 from a fresh
        # normalizer vs 12.7 unnormalized. Clipping bounds the step size but
        # not the direction, and it corrupts Adam's moment estimates on the
        # first updates. Until enough samples have accumulated for the
        # variance to mean anything, pass the reward through unnormalized.
        if self.n < self.WARMUP_N:
            return 0.0, 1.0
        return self.mean, max(self.var ** 0.5, self.MIN_STD)


# ---------------------------------------------------------------------------
# Fixed (non-learnable) environment constants + real population data
# ---------------------------------------------------------------------------

@dataclass
class GroupConstants:
    """Environment-owned, theta-independent data for one group. Deliberately
    plain floats/numpy arrays -- nothing here should carry a grad_fn.
    X_pop/default_prob_pop/loan_amount_pop/wealth_gain_pop are the REAL
    population arrays (same ones the actual simulator uses), sampled fresh
    each timestep in differentiable_group_step -- see module docstring for
    why a single point estimate isn't enough."""

    N: int
    d_bar: float  # population mean default probability (used only for h; see below)
    l_bar: float  # population mean loan amount (used only for two_sided wealth_norm)
    h: float  # market approval score at the group's mean wealth (arrival-rate driver)
    sensitivity: float  # application-propensity sensitivity (fixed, 0.5 in the code)
    alpha: float  # Hawkes excitation, fixed by the environment's creator
    beta: float  # Hawkes decay, fixed by the environment's creator
    arrival_scale: float  # N_g / ARRIVAL_REF_N -- per-capita arrival scaling
    X_pop: np.ndarray  # real current wealth, one entry per individual
    default_prob_pop: np.ndarray  # real (fixed) default probability, per individual
    loan_amount_pop: np.ndarray  # real (fixed) loan amount, per individual
    wealth_gain_pop: np.ndarray  # real (fixed) wealth gain on success, per individual


def build_group_constants(env, group: str) -> GroupConstants:
    """Read the fixed constants AND the real population arrays for `group`
    ('R' male / 'B' female) off the real environment. Called once per
    episode; nothing here is a tensor. alpha_R/beta_R/alpha_B/beta_B are
    read directly off `env` so the shadow rollout can never silently drift
    from whatever the real simulator uses -- same environment, same
    constants, for whichever agent is deployed."""
    tp = env.theta_params
    is_male = group == "R"
    N = env.N_male if is_male else env.N_female
    d_bar = (tp.default_rate_min + tp.default_rate_max) / 2.0
    net_return = tp.investment_return_rate - tp.interest_rate
    l_bar = tp.loan_amount_mean
    X_pop = np.asarray(env.current_X_male if is_male else env.current_X_female, dtype=np.float64)
    X_repr = float(X_pop.mean())
    X_norm = (X_repr - tp.X_mean) / tp.X_std
    S = 1.0 if is_male else 0.0
    z = tp.theta_S * S + tp.theta_X * X_norm + tp.b
    h = 1.0 / (1.0 + np.exp(-z))
    sensitivity = tp.application_sensitivity.get(
        "male" if is_male else "female", {}
    ).get("sensitivity", 0.5)
    alpha = float(env.alpha_R if is_male else env.alpha_B)
    beta = float(env.beta_R if is_male else env.beta_B)
    arrival_scale = float(
        env._arrival_scale_R if is_male else env._arrival_scale_B
    )
    key = "male" if is_male else "female"
    default_prob_pop = np.asarray(tp.individual_default_probs[key], dtype=np.float64)
    loan_amount_pop = np.asarray(tp.individual_loan_amounts[key], dtype=np.float64)
    wealth_gain_pop = np.asarray(tp.individual_wealth_gains[key], dtype=np.float64)
    return GroupConstants(
        N=N, d_bar=d_bar, l_bar=l_bar, h=h, sensitivity=sensitivity,
        alpha=alpha, beta=beta, arrival_scale=arrival_scale,
        X_pop=X_pop, default_prob_pop=default_prob_pop,
        loan_amount_pop=loan_amount_pop, wealth_gain_pop=wealth_gain_pop,
    )


# ---------------------------------------------------------------------------
# The differentiable rollout
# ---------------------------------------------------------------------------

K_SAMPLE = 16  # individuals sampled per group per timestep -- see module docstring


def f_base_rate(mu: torch.Tensor, mu_bar: torch.Tensor,
                scale: float = 1.0) -> torch.Tensor:
    """Torch version of environment.py's _f_networth_to_rate: RELATIVE
    Matthew term (mu_g/mu_bar, see environment.MATTHEW_C) times the
    per-capita arrival scaling (scale = N_g/ARRIVAL_REF_N). Must stay in
    sync with the real environment or the shadow rollout silently models a
    different arrival rate than the one being simulated. mu_bar carries
    grad_fn from BOTH groups, so this also couples the two groups'
    gradients -- matching the real environment's coupling."""
    rel = mu / torch.clamp(mu_bar, min=1e-8)
    return torch.clamp(2.0 * (1.0 + MATTHEW_C * (rel - 1.0)), min=0.5) * scale


def build_group_states(
    mu_R: torch.Tensor, mu_B: torch.Tensor,
    lam_R: torch.Tensor, lam_B: torch.Tensor,
    group: str, const: GroupConstants, tp,
    sample_idx: np.ndarray,
) -> torch.Tensor:
    """
    Batched 12-dim observations, one row per sampled individual (shape
    [k, 12]), built from the running mean-field state (mu_R, mu_B, lam_R,
    lam_B -- connecting the policy call to theta through every earlier
    step) PLUS each sampled individual's own real X and loan amount --
    this per-row variation is exactly what was missing before (see module
    docstring): without it every row would be identical and the gradient
    could never teach the policy that X matters.
    """
    is_male = group == "R"
    k = len(sample_idx)
    X_i = torch.as_tensor(const.X_pop[sample_idx], dtype=torch.float32)
    loan_i = torch.as_tensor(const.loan_amount_pop[sample_idx], dtype=torch.float32)
    S = torch.full((k,), 1.0 if is_male else 0.0)
    X_norm = (X_i - tp.X_mean) / tp.X_std
    theta_approval_prob = torch.sigmoid(tp.theta_S * S + tp.theta_X * X_norm + tp.b)
    rho = torch.full((k,), const.d_bar)  # mean-field stand-in for the running default rate
    mu_R_b = mu_R.expand(k)
    mu_B_b = mu_B.expand(k)
    lam_R_b = lam_R.expand(k)
    lam_B_b = lam_B.expand(k)
    theta_S_b = torch.full((k,), float(tp.theta_S))
    theta_X_b = torch.full((k,), float(tp.theta_X))
    return torch.stack([
        X_i / 100.0, S, mu_R_b / 100.0, mu_B_b / 100.0,
        lam_R_b, lam_B_b, rho, rho,
        theta_approval_prob, theta_S_b, theta_X_b, loan_i,
    ], dim=1).float()


def _group_profit_rate(const: GroupConstants, rho, interest_rate: float):
    """Mirrors reward.py's _group_profit_rates for one group:
        r_g = mean_loan_g * ((1 - rho_g) * interest - rho_g)
    Same loss coefficient as _calculate_bank_profit/reward.py's
    _group_profit_rates -- this used to say "- 2 * rho" (a separate,
    independent copy of the formula that reward.py's own fix never
    touched), which meant PePG's actual gradient kept training against the
    old, wrong-breakeven quantity even after reward.py was corrected --
    only the LOGGED metrics (via _collect_episode -> reward.py, logging
    only, never backprop'd through) reflected the fix. See reward.py's
    RewardFunction docstring for why this coefficient must match
    _calculate_bank_profit's.

    `rho` is the default rate AMONG APPROVED LOANS. The real environment
    computes it as total_defaults_g / total_loans_g, which the policy
    directly controls -- approving lower-risk applicants lowers it. An
    earlier version here substituted the population constant d_bar, which
    made r_g a CONSTANT and therefore zero-gradient. That silently made
    rawlsian_maximin/dm completely untrainable (its whole reward is
    min(r_R, r_B), so the policy received no signal at all -- measured
    X-spread exactly 0.0000) and made fairness_lagrangian/dm's fairness
    penalty term untrainable while its profit term still worked.

    Pass a differentiable, approval-weighted rho instead (see
    differentiable_group_step) so this tracks what the policy actually does.
    """
    return const.l_bar * ((1 - rho) * interest_rate - rho)


def compute_step_reward(
    reward_function_name: str,
    constraint_type: str,
    mu_R: torch.Tensor, mu_B: torch.Tensor,
    bank_profit_R: torch.Tensor, bank_profit_B: torch.Tensor,
    n_R: torch.Tensor, n_B: torch.Tensor,
    rho_R: torch.Tensor, rho_B: torch.Tensor,
    const_R: GroupConstants, const_B: GroupConstants,
    interest_rate: float, lambda_wealth: float, lambda_approval: float,
) -> torch.Tensor:
    """
    Differentiable, mean-field equivalent of summing compute_batched_rewards()
    over every applicant that arrives this timestep (both groups combined) --
    the shadow-rollout counterpart of reward.py's 16 formulas.

    bank_profit_R/B are ALREADY the full per-timestep bank-profit sum for
    each group (n_arrivals * mean_i[a_i * profit_i], computed in
    differentiable_group_step from the REAL sampled individuals' own default
    probabilities and loan amounts -- not a population-average constant).
    These are action-dependent and the paper defines them as a SUM over
    arriving applicants, so they enter unscaled.

    State-based terms (mu_R+mu_B, min(mu_R,mu_B), group profit rates, the
    Lagrangian penalty) don't reference any applicant's action at all, and
    Table 1 defines them PER TIMESTEP -- one value per step, not one per
    applicant. So they enter with NO n_total weighting. See
    reward.compute_batched_rewards' docstring for why the previous
    n_total weighting was actively harmful (it made "minimise arrivals"
    the dominant gradient once the penalty turned the value negative).
    """
    bank_profit = bank_profit_R + bank_profit_B

    if reward_function_name == "utilitarian_profit":
        if constraint_type == "dm":
            return bank_profit
        elif constraint_type == "social":
            return torch.zeros_like(bank_profit)
        elif constraint_type == "two_sided":
            alpha = lambda_wealth
            mean_loan = const_R.l_bar + const_B.l_bar
            wealth_norm = (mu_R + mu_B) / (2 * mean_loan + 1e-8)
            return (1 - alpha) * bank_profit + alpha * wealth_norm
        raise ValueError(f"Unknown constraint_type: {constraint_type!r}")

    if reward_function_name == "social_welfare":
        if constraint_type == "social":
            return mu_R + mu_B
        elif constraint_type == "dm":
            return torch.zeros_like(bank_profit)
        elif constraint_type == "two_sided":
            N = const_R.N + const_B.N
            return (bank_profit + (mu_R + mu_B)) / (1 + N)
        raise ValueError(f"Unknown constraint_type: {constraint_type!r}")

    if reward_function_name == "rawlsian_maximin":
        if constraint_type == "social":
            return torch.minimum(mu_R, mu_B)
        elif constraint_type == "dm":
            r_R = _group_profit_rate(const_R, rho_R, interest_rate)
            r_B = _group_profit_rate(const_B, rho_B, interest_rate)
            return torch.minimum(r_R, r_B)
        elif constraint_type == "two_sided":
            alpha = lambda_wealth
            # Same mean_loan*2 normalization as utilitarian_profit's
            # two_sided (and reward.py's rawlsian_maximin/two_sided, which
            # this must mirror) -- without it, raw mu overpowers bank_profit
            # by ~300x within a deploy episode, making alpha ineffective.
            mean_loan = const_R.l_bar + const_B.l_bar
            wealth_norm = torch.minimum(mu_R, mu_B) / (2 * mean_loan + 1e-8)
            return (1 - alpha) * bank_profit + alpha * wealth_norm
        raise ValueError(f"Unknown constraint_type: {constraint_type!r}")

    if reward_function_name == "fairness_lagrangian":
        if constraint_type == "social":
            return mu_R + mu_B - lambda_wealth * torch.abs(mu_R - mu_B)
        elif constraint_type == "dm":
            r_R = _group_profit_rate(const_R, rho_R, interest_rate)
            r_B = _group_profit_rate(const_B, rho_B, interest_rate)
            return bank_profit - lambda_wealth * torch.abs(r_R - r_B)
        elif constraint_type == "two_sided":
            alpha = lambda_wealth
            # Same mean_loan*2 normalization as utilitarian_profit's
            # two_sided (and reward.py's fairness_lagrangian/two_sided,
            # which this must mirror).
            mean_loan = const_R.l_bar + const_B.l_bar
            wealth_norm = torch.abs(mu_R - mu_B) / (2 * mean_loan + 1e-8)
            return (1 - alpha) * bank_profit - alpha * wealth_norm
        raise ValueError(f"Unknown constraint_type: {constraint_type!r}")

    raise ValueError(f"Unknown reward_function_name: {reward_function_name!r}")


def differentiable_group_step(
    mu: torch.Tensor, S_exc: torch.Tensor,
    policy_net: nn.Module,
    group: str, const: GroupConstants, tp, dt: float, interest_rate: float,
    other_mu: torch.Tensor, other_lam: torch.Tensor,
):
    """
    One mean-field step for a single group: samples K_SAMPLE real
    individuals fresh, runs ALL of them through the policy in one batched
    call (each gets their OWN action, based on their OWN wealth/loan
    amount), and aggregates their outcomes into the group-level state
    update and reward contribution. This per-individual variation is the
    fix described in the module docstring -- previously this function
    queried the policy once, on a single point (X = mu_g), which gave the
    gradient no way to teach the policy that individuals differ.

    `other_mu`/`other_lam` are the OTHER group's current state, needed only
    to build the observation; they don't feed this group's own transition
    dynamics (each group's arrivals/wealth depend on its own lambda, not
    the other group's -- confirmed: groups are independent).
    """
    alpha_g, beta_g = const.alpha, const.beta

    mu_bar = 0.5 * (mu + other_mu)
    lam = f_base_rate(mu, mu_bar, const.arrival_scale) + alpha_g * S_exc
    lam_R = lam if group == "R" else other_lam
    lam_B = other_lam if group == "R" else lam
    mu_R = mu if group == "R" else other_mu
    mu_B = other_mu if group == "R" else mu

    k = min(K_SAMPLE, len(const.X_pop))
    sample_idx = np.random.choice(len(const.X_pop), size=k, replace=len(const.X_pop) < K_SAMPLE)

    states = build_group_states(mu_R, mu_B, lam_R, lam_B, group, const, tp, sample_idx)

    policy_alpha, policy_beta = policy_net(states)
    dist = Beta(policy_alpha.squeeze(-1), policy_beta.squeeze(-1))
    a = dist.rsample()  # [k]
    a = a.clamp(1e-4, 1 - 1e-4)
    entropy = dist.entropy().mean()

    d_i = torch.as_tensor(const.default_prob_pop[sample_idx], dtype=torch.float32)
    l_i = torch.as_tensor(const.loan_amount_pop[sample_idx], dtype=torch.float32)
    kappa_i = torch.as_tensor(const.wealth_gain_pop[sample_idx], dtype=torch.float32)

    mean_a = a.mean()

    p_apply = torch.clamp(
        lam * dt / const.N * (1.0 + const.sensitivity * const.h), 0.0, 1.0
    )
    n_arrivals = const.N * p_apply  # expected arrivals this step -- NOT
    # action-dependent (p_apply depends on lambda, not a), so this is safe
    # to use as the "everyone who showed up" weight for state-based rewards.

    decay = float(np.exp(-beta_g * dt))
    new_events = n_arrivals * mean_a  # expected COUNT of new approvals
    S_exc_next = S_exc * decay + new_events

    # Wealth drift: default only ever removes the (1 - d_i) success factor
    # -- there is no separate default charge, matching the code (kappa_i =
    # 0.0 on default, never negative). Averaged over the k REAL sampled
    # individuals' own (a_i, d_i, kappa_i) instead of population constants
    # -- this is what lets the gradient credit approving a GOOD-risk
    # individual differently from a BAD-risk one.
    wealth_contrib = (a * (1 - d_i) * kappa_i).mean()
    mu_next = mu + dt * p_apply * wealth_contrib

    # Per-timestep bank-profit sum for this group, same per-individual
    # heterogeneity: n_arrivals * mean_i[a_i * profit_i], not
    # n_arrivals * mean(a) * population_average_profit.
    profit_i = (1 - d_i) * l_i * interest_rate - d_i * l_i
    bank_profit = n_arrivals * (a * profit_i).mean()

    # Default rate AMONG APPROVED loans, differentiable in the policy's own
    # actions: approving lower-d individuals pulls it down. Mirrors the
    # environment's total_defaults_g / total_loans_g. See _group_profit_rate.
    rho = (a * d_i).sum() / (a.sum() + 1e-8)

    return mu_next, S_exc_next, n_arrivals, entropy, bank_profit, rho


def differentiable_episode_return(
    env, policy_net: nn.Module,
    reward_function_name: str, constraint_type: str,
    lambda_wealth: float, lambda_approval: float,
    gamma: float,
    entropy_coef: float = 0.0,
    reward_normalizer: "RunningNormalizer | None" = None,
) -> torch.Tensor:
    """
    Full differentiable shadow rollout of one episode, matching env.T/env.dt.
    Returns a single scalar: the discounted return, connected to theta
    through policy_net's parameters. alpha_R/beta_R/alpha_B/beta_B are read
    off `env` as fixed constants (see module docstring) -- same values the
    real simulator uses, for whichever agent is deployed.

    reward_function_name/constraint_type/lambda_wealth/lambda_approval must
    match whatever the agent is actually configured with -- see
    compute_step_reward() for why using the wrong (or a hardcoded) formula
    here silently trains the policy on the wrong objective.

    entropy_coef/reward_normalizer: exploration pressure, matching
    PolicyGradientAgent's entropy_coef -- without this, nothing here
    discourages the policy from collapsing to a near-deterministic corner
    (see RunningNormalizer's docstring for why reward normalisation has to
    come first for entropy_coef to mean anything at this reward's scale).
    Pass an agent-owned RunningNormalizer instance so its running stats
    persist across episodes (create once in __init__, not per call).

    Call this ALONGSIDE the real (numpy) episode collection -- it does not
    replace the logged trajectory, only supplies the gradient.
    """
    const_R = build_group_constants(env, "R")
    const_B = build_group_constants(env, "B")
    tp = env.theta_params
    interest_rate = float(env.interest_rate)
    if reward_normalizer is None:
        reward_normalizer = RunningNormalizer()

    mu_R = torch.tensor(float(env.mu_R))
    mu_B = torch.tensor(float(env.mu_B))
    S_R = torch.tensor(0.0)
    S_B = torch.tensor(0.0)
    mu_bar0 = 0.5 * (mu_R + mu_B)
    lam_R = f_base_rate(mu_R, mu_bar0, const_R.arrival_scale)
    lam_B = f_base_rate(mu_B, mu_bar0, const_B.arrival_scale)

    T_steps = int(env.T / env.dt)
    discount = 1.0
    total_return = torch.tensor(0.0)

    for _ in range(T_steps):
        mu_R_next, S_R_next, n_R, ent_R, bp_R, rho_R = differentiable_group_step(
            mu_R, S_R, policy_net, "R", const_R, tp, env.dt, interest_rate, mu_B, lam_B
        )
        mu_B_next, S_B_next, n_B, ent_B, bp_B, rho_B = differentiable_group_step(
            mu_B, S_B, policy_net, "B", const_B, tp, env.dt, interest_rate, mu_R, lam_R
        )

        r = compute_step_reward(
            reward_function_name, constraint_type,
            mu_R, mu_B, bp_R, bp_B, n_R, n_B, rho_R, rho_B,
            const_R, const_B, interest_rate, lambda_wealth, lambda_approval,
        )
        mean, std = reward_normalizer.update_and_get_stats(r.detach().item())
        r_norm = (r - mean) / std

        # Entropy is a PER-STEP regulariser and must be accumulated with the
        # same discount as the per-step reward. An earlier attempt added it
        # once as a plain mean over the rollout, which made it ~200x weaker
        # (sum(gamma^t) ~ 100) -- measured entropy/reward gradient ratio fell
        # from 4.03x to 0.025x and the policy promptly collapsed: Beta
        # concentration 4.7 -> 79, approval 0.79 -> 0.94, X-spread 0.216 ->
        # 0.0005. The structure here is correct; ENTROPY_COEF below is what
        # calibrates the strength.
        step_value = r_norm + entropy_coef * (ent_R + ent_B)
        total_return = total_return + discount * step_value
        discount *= gamma

        mu_R, S_R = mu_R_next, S_R_next
        mu_B, S_B = mu_B_next, S_B_next
        mu_bar_t = 0.5 * (mu_R + mu_B)
        lam_R = f_base_rate(mu_R, mu_bar_t, const_R.arrival_scale) + const_R.alpha * S_R
        lam_B = f_base_rate(mu_B, mu_bar_t, const_B.arrival_scale) + const_B.alpha * S_B

    return total_return
