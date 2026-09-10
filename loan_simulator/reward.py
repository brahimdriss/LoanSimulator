import numpy as np


class RewardFunction:
    """
    Reward functions for RL agent.

    constraint_type maps to the three fairness columns in Table 1 of the
    paper, plus one extra ('predictive') implemented here but not present
    in Table 1 -- kept for completeness but not selected by either
    pg_adapt.py's or pepg_adapt.py's combo lists:
        'predictive'  — not in Table 1 (accuracy / approval-rate metrics)
        'social'      — Outcome Fairness        (mean-wealth metrics)
        'dm'          — DM's Fairness           (bank-profit metrics)
        'two_sided'   — α-Two sided Fairness    (blends profit + fairness, α = lambda_wealth)

    Acc_t^perf (expected accuracy per applicant):
        approval_prob * (1 - default_prob) + (1 - approval_prob) * default_prob
        = P(approve ∩ repay) + P(reject ∩ default)

    r_t^perf,g (expected per-loan profit rate for group g):
        interest_rate * (1 - rho_g) - rho_g
        where rho_g = total_defaults_g / max(total_loans_g, 1)

    _group_profit_rates scales this per-dollar rate by mean_loan_g to put it
    on a dollar footing comparable to _calculate_bank_profit -- needed for
    fairness_lagrangian's "dm" reward, which combines the two additively
    (bank_profit - lambda * |r_R - r_B|). It must use the SAME loss
    coefficient as _calculate_bank_profit (1x, i.e. -rho_g, not -2*rho_g) --
    otherwise its breakeven default rate (rho = rate/(2+rate) = 8.3% at
    rate=0.18) sits below _calculate_bank_profit's (rho = rate/(1+rate) =
    15.3%), so at realistic ~10-12% default rates this formula reads
    NEGATIVE (the bank is "unprofitable") while cumulative_profit -- the
    actual money -- is positive. Confirmed empirically: r_R/r_B were
    negative in all 10 seeds of a real fairness_lagrangian/dm run despite
    multi-million cumulative_profit.
    """

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_bank_profit(env, action, applicant):
        """r_t^perf for the current applicant."""
        if applicant is None:
            return 0.0
        approval_prob = action[0] if isinstance(action, np.ndarray) else action
        default_prob  = applicant["default_prob"]
        loan_amount   = applicant["loan_amount"]
        revenue = (1 - default_prob) * loan_amount * env.interest_rate
        loss    = default_prob * loan_amount
        return approval_prob * (revenue - loss)

    @staticmethod
    def _calculate_accuracy(action, applicant):
        """
        Acc_t^perf: expected accuracy of the approval decision.
            = approval_prob * (1 - default_prob) + (1 - approval_prob) * default_prob
        """
        if applicant is None:
            return 0.0
        approval_prob = action[0] if isinstance(action, np.ndarray) else action
        default_prob  = applicant["default_prob"]
        return approval_prob * (1 - default_prob) + (1 - approval_prob) * default_prob

    @staticmethod
    def _group_profit_rates(env):
        """
        r_t^perf,R and r_t^perf,B: expected profit for each group,
        estimated from running default statistics and mean loan amount.
            r_g = mean_loan_g * ((1 - rho_g) * interest_rate - rho_g)
        Same loss coefficient as _calculate_bank_profit -- see the class
        docstring for why this must match.
        """
        rho_R = env.total_defaults_R / max(env.total_loans_R, 1)
        rho_B = env.total_defaults_B / max(env.total_loans_B, 1)
        mean_loan_R = float(np.mean(env.theta_params.individual_loan_amounts["male"]))
        mean_loan_B = float(np.mean(env.theta_params.individual_loan_amounts["female"]))
        r_R = mean_loan_R * ((1 - rho_R) * env.interest_rate - rho_R)
        r_B = mean_loan_B * ((1 - rho_B) * env.interest_rate - rho_B)
        return r_R, r_B

    @staticmethod
    def _group_tpr(env):
        """
        True positive rate per group, for the "eo" (equality of opportunity)
        reward: among ground-truth-qualified applicants (Y=1), what fraction
        got approved. TPR_g = tp_g / (tp_g + fn_g), running totals.
        0 if this environment has no ground-truth labels (tp_g/fn_g both 0).
        """
        tp_R = getattr(env, "tp_R", 0)
        fn_R = getattr(env, "fn_R", 0)
        tp_B = getattr(env, "tp_B", 0)
        fn_B = getattr(env, "fn_B", 0)
        tpr_R = tp_R / max(tp_R + fn_R, 1)
        tpr_B = tp_B / max(tp_B + fn_B, 1)
        return tpr_R, tpr_B

    @staticmethod
    def _kappa_bar(env) -> float:
        """Population-mean wealth gain on a repaid loan, both groups. The
        Equality-of-Outcome constraint terms are measured in units of this
        -- see reward.RewardSnapshot.kappa_bar / _outcome_violation_batch."""
        g = env.theta_params.individual_wealth_gains
        return 0.5 * (float(np.mean(g["male"])) + float(np.mean(g["female"])))

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    @staticmethod
    def utilitarian_profit(
        env,
        action,
        info,
        constraint_type="dm",
        lambda_wealth=0.0,
        lambda_approval=0.0,
    ):
        """
        Utilitarian Profit.

        'predictive'  → Acc_t^perf
        'social'      → undefined (−)
        'eo'          → undefined (−), same reason as 'social'
        'dm'          → r_t^perf
        'two_sided'   → (1 − α) * r_t^perf + α * (μ_R + μ_B)
        """
        applicant = info.get("applicant")

        if constraint_type == "predictive":
            return RewardFunction._calculate_accuracy(action, applicant)

        elif constraint_type == "social":
            return 0.0  # undefined (−) in the table

        elif constraint_type == "eo":
            return 0.0  # undefined (−), same as 'social' -- no fairness term at all

        elif constraint_type == "dm":
            return RewardFunction._calculate_bank_profit(env, action, applicant)

        elif constraint_type == "two_sided":
            alpha       = lambda_wealth
            bank_profit = RewardFunction._calculate_bank_profit(env, action, applicant)
            mean_loan   = float(np.mean(env.theta_params.individual_loan_amounts["male"] +
                                        env.theta_params.individual_loan_amounts["female"]))
            wealth_norm = (env.mu_R + env.mu_B) / (mean_loan * 2 + 1e-8)
            return (1 - alpha) * bank_profit + alpha * wealth_norm

        else:
            raise ValueError(f"Unknown constraint_type: {constraint_type!r}")

    @staticmethod
    def social_welfare(
        env,
        action,
        info,
        constraint_type="social",
        lambda_wealth=2.0,
        lambda_approval=2.0,
    ):
        """
        Social Welfare.

        'predictive'  → app_t^R + app_t^B
        'social'      → r_t^perf + λ (μ_t^R + μ_t^B)        [Table 1 Lagrangian, see compute_batched_rewards]
        'eo'          → r_t^perf + λ (TPR_R + TPR_B)
        'dm'          → undefined (−)
        'two_sided'   → (r_t^perf + μ_R + μ_B) / (1 + N)
        """
        applicant = info.get("applicant")

        approval_rate_R = env.total_loans_R / max(env.total_applications_R, 1)
        approval_rate_B = env.total_loans_B / max(env.total_applications_B, 1)

        if constraint_type == "predictive":
            return approval_rate_R + approval_rate_B

        elif constraint_type == "social":
            bank_profit = RewardFunction._calculate_bank_profit(env, action, applicant)
            return bank_profit + lambda_wealth * (env.mu_R + env.mu_B) / RewardFunction._kappa_bar(env)

        elif constraint_type == "eo":
            bank_profit = RewardFunction._calculate_bank_profit(env, action, applicant)
            tpr_R, tpr_B = RewardFunction._group_tpr(env)
            return bank_profit + lambda_wealth * (tpr_R + tpr_B)

        elif constraint_type == "dm":
            return 0.0  # undefined (−) in the table

        elif constraint_type == "two_sided":
            bank_profit = RewardFunction._calculate_bank_profit(env, action, applicant)
            N = env.N_male + env.N_female
            return (bank_profit + env.mu_R + env.mu_B) / (1 + N)

        else:
            raise ValueError(f"Unknown constraint_type: {constraint_type!r}")

    @staticmethod
    def rawlsian_maximin(
        env,
        action,
        info,
        constraint_type="social",
        lambda_wealth=5.0,
        lambda_approval=5.0,
    ):
        """
        Rawlsian Max-Min.

        'predictive'  → min{app_t^R, app_t^B}
        'social'      → r_t^perf + λ min{μ_t^R, μ_t^B}      [Table 1 Lagrangian, see compute_batched_rewards]
        'eo'          → r_t^perf + λ min{TPR_R, TPR_B}
        'dm'          → min{r_t^perf,R, r_t^perf,B}
        'two_sided'   → (1 − α) * r_t^perf + α * min{μ_R, μ_B}
        """
        applicant = info.get("applicant")

        approval_rate_R = env.total_loans_R / max(env.total_applications_R, 1)
        approval_rate_B = env.total_loans_B / max(env.total_applications_B, 1)

        if constraint_type == "predictive":
            return min(approval_rate_R, approval_rate_B)

        elif constraint_type == "social":
            bank_profit = RewardFunction._calculate_bank_profit(env, action, applicant)
            return bank_profit + lambda_wealth * min(env.mu_R, env.mu_B) / RewardFunction._kappa_bar(env)

        elif constraint_type == "eo":
            bank_profit = RewardFunction._calculate_bank_profit(env, action, applicant)
            tpr_R, tpr_B = RewardFunction._group_tpr(env)
            return bank_profit + lambda_wealth * min(tpr_R, tpr_B)

        elif constraint_type == "dm":
            r_R, r_B = RewardFunction._group_profit_rates(env)
            return min(r_R, r_B)

        elif constraint_type == "two_sided":
            bank_profit = RewardFunction._calculate_bank_profit(env, action, applicant)
            alpha = lambda_wealth
            # Same mean_loan*2 normalization as utilitarian_profit's two_sided
            # -- without it, raw mu overpowers bank_profit by ~300x within a
            # deploy episode (confirmed empirically), making alpha ineffective.
            mean_loan   = float(np.mean(env.theta_params.individual_loan_amounts["male"] +
                                        env.theta_params.individual_loan_amounts["female"]))
            wealth_norm = min(env.mu_R, env.mu_B) / (mean_loan * 2 + 1e-8)
            return (1 - alpha) * bank_profit + alpha * wealth_norm

        else:
            raise ValueError(f"Unknown constraint_type: {constraint_type!r}")

    @staticmethod
    def fairness_lagrangian(
        env,
        action,
        info,
        constraint_type="social",
        lambda_wealth=10.0,
        lambda_approval=10.0,
    ):
        """
        Fairness Lagrangian.

        'predictive'  → Acc_t^perf − λ * |app_t^R − app_t^B|
        'social'      → r_t^perf − λ * |μ_R − μ_B|          [Table 1 Lagrangian, see compute_batched_rewards]
        'eo'          → r_t^perf − λ * |TPR_R − TPR_B|
        'dm'          → r_t^perf − λ * |r_t^perf,R − r_t^perf,B|
        'two_sided'   → (1 − α) * r_t^perf − α * |μ_R − μ_B|
        """
        applicant   = info.get("applicant")
        bank_profit = RewardFunction._calculate_bank_profit(env, action, applicant)

        approval_rate_R = env.total_loans_R / max(env.total_applications_R, 1)
        approval_rate_B = env.total_loans_B / max(env.total_applications_B, 1)

        if constraint_type == "predictive":
            accuracy = RewardFunction._calculate_accuracy(action, applicant)
            return accuracy - lambda_approval * abs(approval_rate_R - approval_rate_B)

        elif constraint_type == "social":
            return bank_profit - lambda_wealth * abs(env.mu_R - env.mu_B) / RewardFunction._kappa_bar(env)

        elif constraint_type == "eo":
            tpr_R, tpr_B = RewardFunction._group_tpr(env)
            return bank_profit - lambda_wealth * abs(tpr_R - tpr_B)

        elif constraint_type == "dm":
            r_R, r_B = RewardFunction._group_profit_rates(env)
            return bank_profit - lambda_wealth * abs(r_R - r_B)

        elif constraint_type == "two_sided":
            alpha = lambda_wealth
            # Same mean_loan*2 normalization as utilitarian_profit's two_sided
            # -- without it, raw mu overpowers bank_profit by ~300x within a
            # deploy episode (confirmed empirically), making alpha ineffective.
            mean_loan   = float(np.mean(env.theta_params.individual_loan_amounts["male"] +
                                        env.theta_params.individual_loan_amounts["female"]))
            wealth_norm = abs(env.mu_R - env.mu_B) / (mean_loan * 2 + 1e-8)
            return (1 - alpha) * bank_profit - alpha * wealth_norm

        else:
            raise ValueError(f"Unknown constraint_type: {constraint_type!r}")


# ---------------------------------------------------------------------------
# Batched reward computation, for step_cohort()'s vectorized rollout.
#
# Mirrors RewardFunction's 16 formulas exactly (verified line-by-line against
# the scalar versions above), but takes an explicit snapshot of the env-level
# scalars instead of reading them live off `env`. This is deliberate, not
# stylistic: env.mu_R/total_loans_R/etc. mutate as the environment advances
# to the next timestep, and step_cohort() must compute this cohort's rewards
# using the state that existed WHEN THIS COHORT ARRIVED (before their own
# approvals are folded into mu_R) -- otherwise an applicant's reward would
# depend on other applicants decided in the same batch, which no single-
# applicant call in the original code ever did. Taking an explicit snapshot
# makes that timing an explicit argument instead of an implicit ordering
# dependency on when this function happens to be called relative to env
# mutation.
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class RewardSnapshot:
    """Env-level scalars needed by the batched reward formulas, captured
    BEFORE the current cohort's approvals are applied to mu_R/mu_B/totals."""

    mu_R: float
    mu_B: float
    total_loans_R: int
    total_applications_R: int
    total_defaults_R: int
    total_loans_B: int
    total_applications_B: int
    total_defaults_B: int
    interest_rate: float
    mean_loan_R: float  # population-wide mean loan amount, group R (fixed)
    mean_loan_B: float  # population-wide mean loan amount, group B (fixed)
    N_male: int
    N_female: int
    tp_R: int = 0  # true positives (approved & ground-truth-qualified), group R
    fn_R: int = 0  # false negatives (rejected & ground-truth-qualified), group R
    tp_B: int = 0
    fn_B: int = 0
    # Population-mean wealth gain on a repaid loan, kappa_i = (tau_inv -
    # tau_interest) * l_i, per group (fixed). The Equality-of-Outcome
    # constraint terms are measured in units of this (see
    # _outcome_violation_batch), so lambda is "profit per average loan's
    # worth of borrower wealth" rather than being swamped by raw $k.
    mean_gain_R: float = 1.0
    mean_gain_B: float = 1.0

    @property
    def kappa_bar(self) -> float:
        return 0.5 * (self.mean_gain_R + self.mean_gain_B)

    @classmethod
    def from_env(cls, env) -> "RewardSnapshot":
        return cls(
            mu_R=float(env.mu_R),
            mu_B=float(env.mu_B),
            total_loans_R=env.total_loans_R,
            total_applications_R=env.total_applications_R,
            total_defaults_R=env.total_defaults_R,
            total_loans_B=env.total_loans_B,
            total_applications_B=env.total_applications_B,
            total_defaults_B=env.total_defaults_B,
            interest_rate=float(env.interest_rate),
            mean_loan_R=float(np.mean(env.theta_params.individual_loan_amounts["male"])),
            mean_loan_B=float(np.mean(env.theta_params.individual_loan_amounts["female"])),
            N_male=env.N_male,
            N_female=env.N_female,
            tp_R=getattr(env, "tp_R", 0),
            fn_R=getattr(env, "fn_R", 0),
            tp_B=getattr(env, "tp_B", 0),
            fn_B=getattr(env, "fn_B", 0),
            mean_gain_R=float(np.mean(env.theta_params.individual_wealth_gains["male"])),
            mean_gain_B=float(np.mean(env.theta_params.individual_wealth_gains["female"])),
        )


def _bank_profit_batch(a: np.ndarray, d: np.ndarray, l: np.ndarray, interest_rate: float) -> np.ndarray:
    """Vectorized _calculate_bank_profit."""
    revenue = (1 - d) * l * interest_rate
    loss = d * l
    return a * (revenue - loss)


def _accuracy_batch(a: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Vectorized _calculate_accuracy."""
    return a * (1 - d) + (1 - a) * d


def _group_profit_rates(snap: RewardSnapshot):
    """Vectorized _group_profit_rates -- scalar pair, unchanged from the
    original (doesn't depend on the current batch, only running totals).
    Same loss coefficient as _calculate_bank_profit -- see
    RewardFunction._group_profit_rates's docstring."""
    rho_R = snap.total_defaults_R / max(snap.total_loans_R, 1)
    rho_B = snap.total_defaults_B / max(snap.total_loans_B, 1)
    r_R = snap.mean_loan_R * ((1 - rho_R) * snap.interest_rate - rho_R)
    r_B = snap.mean_loan_B * ((1 - rho_B) * snap.interest_rate - rho_B)
    return r_R, r_B


def _group_tpr(snap: RewardSnapshot):
    """Vectorized RewardFunction._group_tpr -- scalar pair, unchanged from
    the original (doesn't depend on the current batch, only running
    totals)."""
    tpr_R = snap.tp_R / max(snap.tp_R + snap.fn_R, 1)
    tpr_B = snap.tp_B / max(snap.tp_B + snap.fn_B, 1)
    return tpr_R, tpr_B


# Fixed thresholds d for the Table 1 constraints (the dual ascends against
# C >= d for SW/RMM, C <= d for FL). Anchored to the COIN-FLIP lender
# (approve every applicant w.p. 0.5) in the deploy environment at the
# cluster scale (N = 12000 + 12000, T = 100, dt = 0.5), measured over four
# consecutive episodes with scratchpad/constraint_refs.py:
#
#   policy         profit  SW/soc  RMM/soc  FL/soc | SW/eo  RMM/eo  FL/eo
#   approve all       480    1730      610   0.245 |  2.00    1.00   0.00
#   coin flip         255     840      297   0.243 |  1.02    0.48   0.06
#   profit-max half   990     310       29   0.244 |  1.47    0.47   0.53
#   (SW/RMM social in kappa_bar units of wealth created per episode)
#
# SW/RMM social are stored PER POPULATION MEMBER so the same number means the
# same thing at any N: 840 / 24000 = 0.035 and 297 / 12000 = 0.025. FL/social
# is half the population's starting relative wealth gap (0.24). The eo
# thresholds sit between the profit-maximising policy (TPR gap ~0.5) and the
# coin flip (~0.06), so every one of the six binds against profit without
# being unreachable.
CONSTRAINT_TARGETS = {
    "sw_social":  0.035,   # (dW_R + dW_B) / kappa_bar / (N_R + N_B)   >= d
    "rmm_social": 0.025,   # dW_poorer   / kappa_bar / N_poorer         >= d
    "fl_social":  0.12,    # |mu_R - mu_B| / mean(mu)                   <= d
    "sw_eo":      1.5,     # TPR_R + TPR_B                              >= d
    "rmm_eo":     0.75,    # min(TPR_R, TPR_B)                          >= d
    "fl_eo":      0.10,    # |TPR_R - TPR_B|                            <= d
}


def constraint_measure(env, reward_function_name: str, constraint_type: str,
                       dW_R: float = None, dW_B: float = None):
    """
    End-of-episode value C of the Table 1 constraint for the DUAL update,
    and its sense. Returns (key, sense, C) with sense 'ge' for constraints
    the objective wants HIGH (SW, RMM) and 'le' for ones it wants LOW (FL).
    The agents then do  lambda <- clip(lambda + eta * v, eps, lambda_max)
    with v = (d - C)/|d| for 'ge' and (C - d)/|d| for 'le', d the fixed
    threshold CONSTRAINT_TARGETS[key]. lambda therefore RISES while the
    constraint is violated and DECAYS toward eps once it is satisfied, i.e.
    it settles where the constraint binds -- a multiplier, not a constant.
    (An earlier version used the first episode's own value as d; that is
    either trivially met or permanently violated, so lambda only ever
    reached the floor or the cap -- see the fixpilot lambda diagnostics.)

    Equality of Outcome (constraint_type 'social'):
        SW   C = wealth created this episode per member   (dW_R + dW_B) / kappa_bar / (N_R + N_B)
        RMM  C = wealth created per member, poorer group  dW_poorer / kappa_bar / N_poorer
        FL   C = relative wealth gap  |mu_R - mu_B| / mean(mu)
      dW_g = N_g * (mu_g_end - mu_g_start): the episode's total wealth
      created in group g (the same quantity the per-step credits sum to).
    Equality of Opportunity (constraint_type 'eo'), end-of-episode TPRs:
        SW   C = TPR_R + TPR_B      RMM  C = min(TPR_R, TPR_B)      FL  C = |TPR_R - TPR_B|
    """
    if constraint_type == "social":
        if dW_R is None or dW_B is None:
            raise ValueError("constraint_measure('social') needs this episode's dW_R, dW_B")
        kb = RewardFunction._kappa_bar(env)
        N_R, N_B = int(env.N_male), int(env.N_female)
        if reward_function_name == "social_welfare":
            return "sw_social", "ge", (dW_R + dW_B) / kb / (N_R + N_B)
        if reward_function_name == "rawlsian_maximin":
            poorer_is_R = env.mu_R < env.mu_B
            return "rmm_social", "ge", (dW_R / N_R if poorer_is_R else dW_B / N_B) / kb
        if reward_function_name == "fairness_lagrangian":
            mean_mu = max(0.5 * (env.mu_R + env.mu_B), 1e-8)
            return "fl_social", "le", abs(env.mu_R - env.mu_B) / mean_mu
    if constraint_type == "eo":
        tpr_R, tpr_B = RewardFunction._group_tpr(env)
        if reward_function_name == "social_welfare":
            return "sw_eo", "ge", tpr_R + tpr_B
        if reward_function_name == "rawlsian_maximin":
            return "rmm_eo", "ge", min(tpr_R, tpr_B)
        if reward_function_name == "fairness_lagrangian":
            return "fl_eo", "le", abs(tpr_R - tpr_B)
    raise ValueError(f"no Table 1 constraint for {reward_function_name!r}/{constraint_type!r}")


def dual_ascent_update(lam: float, sense: str, C: float, target: float,
                       eta: float, lam_max: float, eps: float = 1e-4) -> float:
    """One projected dual-ascent step shared by both agents (see
    constraint_measure / CONSTRAINT_TARGETS). The violation is normalised
    by |target| so eta means the same thing for every constraint: a
    violation of one target-width moves lambda by eta per episode,
    crossing a range of 10 in 100 episodes at eta = 0.1."""
    scale = max(abs(target), 1e-8)
    v = (target - C) / scale if sense == "ge" else (C - target) / scale
    return float(np.clip(lam + eta * v, eps, lam_max))


def _wealth_credit_batch(a: np.ndarray, d: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Expected wealth created by approving applicant i with probability a_i:
    a_i * (1 - d_i) * kappa_i. This is exactly the applicant's expected
    contribution to (N_g * mu_g) this step in the real environment, which
    adds kappa_i on approved-and-repaid and nothing otherwise
    (environment.step_cohort). Total-wealth units, not per-capita."""
    return a * (1.0 - d) * kappa


def _outcome_violation_batch(snap: RewardSnapshot, groups: np.ndarray,
                             credit: np.ndarray, mode: str) -> np.ndarray:
    """Per-applicant contribution g_i to the per-step CHANGE of the Table 1
    Equality-of-Outcome constraint VIOLATION, so that the Lagrangian reward
    is  r_i - lambda * g_i  (Eutopia Table 1, reward = r_t^pi - lambda * C):
        SW   C = mu_R + mu_B   (wanted HIGH)  -> g_i = -credit_i
        RMM  C = min(mu_R, mu_B) (wanted HIGH) -> g_i = -1[g_i is the poorer group] credit_i
        FL   C = |mu_R - mu_B| (wanted LOW)   -> g_i = +sign(mu_g - mu_other) credit_i
    credit_i = a_i (1-d_i) kappa_i is the applicant's expected addition to
    N_g * mu_g this step, so the cohort sum of g is N * Delta(violation)
    (equal N_g). Group ordering is decided on the pre-cohort snapshot; it
    never flips within a step (gap O(10) vs per-step move O(1e-3)).

    NORMALISATION: credit is divided by kappa_bar, the population-mean
    wealth gain on a repaid loan (~0.17 * mean loan), so g is measured in
    "average loans' worth of borrower wealth" (~1 per average successful
    approval) rather than $k. Profit and raw wealth are both in $k, but a
    loan creates ~10-40x more borrower wealth than bank profit in this
    economy, so in raw units lambda ~ 1 makes the constraint swamp the
    profit term (measured: 8611 vs 510 per episode under approve-all).
    With this scaling lambda reads as "$k of profit the lender gives up
    per average-loan of borrower wealth"."""
    # The SW and FL forms sum N_R*dmu_R and N_B*dmu_B directly, which equals
    # N * Delta(mu_R +/- mu_B) only when the groups are the same size (every
    # campaign so far: 12000/12000). Fail loudly rather than silently
    # mis-weight the groups if that ever changes (fix: per-capita credit,
    # credit / N_g * N_ref).
    if snap.N_male != snap.N_female:
        raise ValueError(
            f"_outcome_violation_batch assumes N_male == N_female (got "
            f"{snap.N_male} / {snap.N_female}); see comment for the per-capita fix.")
    is_R = groups == 1
    mu_own = np.where(is_R, snap.mu_R, snap.mu_B)
    mu_other = np.where(is_R, snap.mu_B, snap.mu_R)
    credit = credit / snap.kappa_bar
    if mode == "sw":
        return -credit
    if mode == "rmm":
        if snap.mu_R == snap.mu_B:
            return -0.5 * credit
        return -(mu_own < mu_other).astype(np.float64) * credit
    if mode == "fl":
        return np.sign(mu_own - mu_other) * credit
    raise ValueError(mode)


def _require_credit_inputs(groups, wealth_gains, reward_function_name):
    if groups is None or wealth_gains is None:
        raise ValueError(
            f"compute_batched_rewards({reward_function_name!r}, constraint_type='social') "
            "needs groups= and wealth_gains= (from step_cohort's info['groups'] / "
            "info['wealth_gains']): the social rewards are per-applicant wealth "
            "credits, not the level of mu -- see the docstring."
        )


def compute_batched_rewards(
    reward_function_name: str,
    snap: RewardSnapshot,
    actions: np.ndarray,
    default_probs: np.ndarray,
    loan_amounts: np.ndarray,
    constraint_type: str,
    lambda_wealth: float,
    lambda_approval: float,
    groups: np.ndarray = None,
    wealth_gains: np.ndarray = None,
) -> np.ndarray:
    """
    Batched equivalent of RewardFunction.<name>(env, action, info,
    constraint_type=..., lambda_wealth=..., lambda_approval=...), called
    once per applicant. Returns one reward per applicant in the batch
    (shape matches `actions`).

    'social' (Equality of Outcome) and 'eo' (Equality of Opportunity) follow
    Eutopia Table 1 as a Lagrangian:   reward = r_t^pi - lambda * C
    where r_t^pi is the bank's per-applicant profit (Section 2) and C is
    the row's constraint, entering as its VIOLATION (SW, RMM want their
    quantity high -> +lambda*C; FL wants the gap low -> -lambda*|gap|):
        SW   r_i + lam (mu_R + mu_B)          | r_i + lam (TPR_R + TPR_B)
        RMM  r_i + lam min(mu_R, mu_B)        | r_i + lam min(TPR_R, TPR_B)
        FL   r_i - lam |mu_R - mu_B|          | r_i - lam |TPR_R - TPR_B|

    HOW THE WEALTH TERMS ENTER (Outcome). The scalar RewardFunction.* use
    the LEVEL of mu, which as a training signal is action-independent: the
    snapshot precedes the cohort's approvals, and one step's approvals move
    a group mean over 12,000 people by ~1e-5 relative (approve-everyone vs
    reject-everyone differed by 0.3% of episode return; every run10 social
    policy sat at the uninformative prior). Here the wealth constraint
    enters as each applicant's expected contribution to its per-step
    CHANGE, in total-wealth units (see _outcome_violation_batch):
        credit_i = a_i (1 - d_i) kappa_i
    Summed over a step this is N * Delta C, and over the episode it
    telescopes to C_T - C_0: same argmax as the level (at gamma = 0.99 an
    affine rescaling plus a gamma^T terminal term), action-dependent per
    applicant, stationary when wealth is carried across episodes. The
    credit is measured in units of kappa_bar (population-mean gain on a
    repaid loan) so that lambda ~ 1 puts the constraint on the profit
    term's scale -- see _outcome_violation_batch.

    HOW THE TPR TERMS ENTER (Opportunity). As the LEVEL, split /n across
    the cohort (state-based, see below). The level is fine here: TPR is a
    ratio over the ~10^2 qualified applicants seen this episode (reset each
    episode), so one approval moves it by ~1/n_g and, persisting over the
    remaining steps, contributes O(1) to the return -- comparable to the
    per-applicant profit. Not the 1/12000 problem the wealth level has.

    The 'dm', 'predictive' and 'two_sided' branches are unchanged.

    TWO KINDS OF TERM, aggregated differently -- this distinction matters
    and was previously conflated:

      * ACTION-DEPENDENT (bank profit, accuracy): each applicant has their
        own value, driven by their own action/default_prob/loan_amount.
        The paper defines these as a SUM over arriving applicants
        (r_t = sum_i [...] A_t,i l_i, Section 2), so each applicant simply
        contributes their own term. Left per-applicant.

      * STATE-BASED (mu_R+mu_B, min(mu), approval rates, group profit
        rates, the Lagrangian penalty): these do not reference any
        individual's action at all. Table 1 defines them PER TIMESTEP --
        one value per step, e.g. r_t = mu_blue + mu_red -- NOT once per
        applicant. Since the caller sums per-applicant rewards over the
        cohort, these are divided by n so the cohort sum reproduces
        exactly one copy of the per-timestep value.

    Why this matters: without the /n, a state-based reward gets multiplied
    by the arrival count, which the formalism never specifies. Once the
    Lagrangian penalty pushes that value negative, "reward = n_arrivals x
    (negative)" makes MINIMISING ARRIVALS the dominant gradient -- the
    policy rejects almost everyone purely to shrink n, an artifact of the
    aggregation rather than anything in the objective. Verified directly:
    approval collapsed to ~7% under exactly this mechanism.
    """
    a, d, l = actions, default_probs, loan_amounts
    n = len(a)
    inv_n = 1.0 / max(n, 1)  # state-based terms -> one copy per TIMESTEP
    approval_rate_R = snap.total_loans_R / max(snap.total_applications_R, 1)
    approval_rate_B = snap.total_loans_B / max(snap.total_applications_B, 1)

    if reward_function_name == "utilitarian_profit":
        if constraint_type == "predictive":
            return _accuracy_batch(a, d)
        elif constraint_type == "social":
            return np.zeros(n)
        elif constraint_type == "eo":
            return np.zeros(n)  # undefined, same as 'social' -- no fairness term
        elif constraint_type == "dm":
            return _bank_profit_batch(a, d, l, snap.interest_rate)
        elif constraint_type == "two_sided":
            alpha = lambda_wealth
            bank_profit = _bank_profit_batch(a, d, l, snap.interest_rate)
            mean_loan = snap.mean_loan_R + snap.mean_loan_B
            wealth_norm = (snap.mu_R + snap.mu_B) / (mean_loan * 2 + 1e-8)
            return (1 - alpha) * bank_profit + alpha * wealth_norm * inv_n
        raise ValueError(f"Unknown constraint_type: {constraint_type!r}")

    if reward_function_name == "social_welfare":
        if constraint_type == "predictive":
            return np.full(n, (approval_rate_R + approval_rate_B) * inv_n)
        elif constraint_type == "social":
            _require_credit_inputs(groups, wealth_gains, reward_function_name)
            bank_profit = _bank_profit_batch(a, d, l, snap.interest_rate)
            g = _outcome_violation_batch(snap, groups, _wealth_credit_batch(a, d, wealth_gains), "sw")
            return bank_profit - lambda_wealth * g
        elif constraint_type == "eo":
            bank_profit = _bank_profit_batch(a, d, l, snap.interest_rate)
            tpr_R, tpr_B = _group_tpr(snap)
            return bank_profit + lambda_wealth * (tpr_R + tpr_B) * inv_n
        elif constraint_type == "dm":
            return np.zeros(n)
        elif constraint_type == "two_sided":
            bank_profit = _bank_profit_batch(a, d, l, snap.interest_rate)
            N = snap.N_male + snap.N_female
            return (bank_profit + (snap.mu_R + snap.mu_B) * inv_n) / (1 + N)
        raise ValueError(f"Unknown constraint_type: {constraint_type!r}")

    if reward_function_name == "rawlsian_maximin":
        if constraint_type == "predictive":
            return np.full(n, min(approval_rate_R, approval_rate_B) * inv_n)
        elif constraint_type == "social":
            _require_credit_inputs(groups, wealth_gains, reward_function_name)
            bank_profit = _bank_profit_batch(a, d, l, snap.interest_rate)
            g = _outcome_violation_batch(snap, groups, _wealth_credit_batch(a, d, wealth_gains), "rmm")
            return bank_profit - lambda_wealth * g
        elif constraint_type == "eo":
            bank_profit = _bank_profit_batch(a, d, l, snap.interest_rate)
            tpr_R, tpr_B = _group_tpr(snap)
            return bank_profit + lambda_wealth * min(tpr_R, tpr_B) * inv_n
        elif constraint_type == "dm":
            r_R, r_B = _group_profit_rates(snap)
            return np.full(n, min(r_R, r_B) * inv_n)
        elif constraint_type == "two_sided":
            bank_profit = _bank_profit_batch(a, d, l, snap.interest_rate)
            alpha = lambda_wealth
            # Same mean_loan*2 normalization as utilitarian_profit's
            # two_sided -- see RewardFunction.rawlsian_maximin's docstring.
            mean_loan = snap.mean_loan_R + snap.mean_loan_B
            wealth_norm = min(snap.mu_R, snap.mu_B) / (mean_loan * 2 + 1e-8)
            return (1 - alpha) * bank_profit + alpha * wealth_norm * inv_n
        raise ValueError(f"Unknown constraint_type: {constraint_type!r}")

    if reward_function_name == "fairness_lagrangian":
        bank_profit = _bank_profit_batch(a, d, l, snap.interest_rate)
        if constraint_type == "predictive":
            accuracy = _accuracy_batch(a, d)
            return accuracy - lambda_approval * abs(approval_rate_R - approval_rate_B) * inv_n
        elif constraint_type == "social":
            _require_credit_inputs(groups, wealth_gains, reward_function_name)
            # r_i - lam * sign(mu_g - mu_other) * credit_i: approving the
            # richer group widens the gap (penalised), approving the poorer
            # group narrows it (rewarded), on top of the applicant's own
            # profit. lam is the price of one unit of gap in profit units.
            g = _outcome_violation_batch(snap, groups, _wealth_credit_batch(a, d, wealth_gains), "fl")
            return bank_profit - lambda_wealth * g
        elif constraint_type == "eo":
            tpr_R, tpr_B = _group_tpr(snap)
            return bank_profit - lambda_wealth * abs(tpr_R - tpr_B) * inv_n
        elif constraint_type == "dm":
            r_R, r_B = _group_profit_rates(snap)
            return bank_profit - lambda_wealth * abs(r_R - r_B) * inv_n
        elif constraint_type == "two_sided":
            alpha = lambda_wealth
            # Same mean_loan*2 normalization as utilitarian_profit's
            # two_sided -- see RewardFunction.rawlsian_maximin's docstring.
            mean_loan = snap.mean_loan_R + snap.mean_loan_B
            wealth_norm = abs(snap.mu_R - snap.mu_B) / (mean_loan * 2 + 1e-8)
            return (1 - alpha) * bank_profit - alpha * wealth_norm * inv_n
        raise ValueError(f"Unknown constraint_type: {constraint_type!r}")

    raise ValueError(f"Unknown reward_function_name: {reward_function_name!r}")
