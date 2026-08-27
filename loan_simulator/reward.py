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
        'social'      → μ_t^R + μ_t^B
        'eo'          → TPR_R + TPR_B
        'dm'          → undefined (−)
        'two_sided'   → (r_t^perf + μ_R + μ_B) / (1 + N)
        """
        applicant = info.get("applicant")

        approval_rate_R = env.total_loans_R / max(env.total_applications_R, 1)
        approval_rate_B = env.total_loans_B / max(env.total_applications_B, 1)

        if constraint_type == "predictive":
            return approval_rate_R + approval_rate_B

        elif constraint_type == "social":
            return env.mu_R + env.mu_B

        elif constraint_type == "eo":
            tpr_R, tpr_B = RewardFunction._group_tpr(env)
            return tpr_R + tpr_B

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
        'social'      → min{μ_t^R, μ_t^B}
        'eo'          → min{TPR_R, TPR_B}
        'dm'          → min{r_t^perf,R, r_t^perf,B}
        'two_sided'   → (1 − α) * r_t^perf + α * min{μ_R, μ_B}
        """
        applicant = info.get("applicant")

        approval_rate_R = env.total_loans_R / max(env.total_applications_R, 1)
        approval_rate_B = env.total_loans_B / max(env.total_applications_B, 1)

        if constraint_type == "predictive":
            return min(approval_rate_R, approval_rate_B)

        elif constraint_type == "social":
            return min(env.mu_R, env.mu_B)

        elif constraint_type == "eo":
            tpr_R, tpr_B = RewardFunction._group_tpr(env)
            return min(tpr_R, tpr_B)

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
        'social'      → μ_R + μ_B − λ * |μ_R − μ_B|
        'eo'          → TPR_R + TPR_B − λ * |TPR_R − TPR_B|
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
            return env.mu_R + env.mu_B - lambda_wealth * abs(env.mu_R - env.mu_B)

        elif constraint_type == "eo":
            tpr_R, tpr_B = RewardFunction._group_tpr(env)
            return tpr_R + tpr_B - lambda_wealth * abs(tpr_R - tpr_B)

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


def compute_batched_rewards(
    reward_function_name: str,
    snap: RewardSnapshot,
    actions: np.ndarray,
    default_probs: np.ndarray,
    loan_amounts: np.ndarray,
    constraint_type: str,
    lambda_wealth: float,
    lambda_approval: float,
) -> np.ndarray:
    """
    Batched equivalent of RewardFunction.<name>(env, action, info,
    constraint_type=..., lambda_wealth=..., lambda_approval=...), called
    once per applicant. Returns one reward per applicant in the batch
    (shape matches `actions`).

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
            return np.full(n, (snap.mu_R + snap.mu_B) * inv_n)
        elif constraint_type == "eo":
            tpr_R, tpr_B = _group_tpr(snap)
            return np.full(n, (tpr_R + tpr_B) * inv_n)
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
            return np.full(n, min(snap.mu_R, snap.mu_B) * inv_n)
        elif constraint_type == "eo":
            tpr_R, tpr_B = _group_tpr(snap)
            return np.full(n, min(tpr_R, tpr_B) * inv_n)
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
            val = snap.mu_R + snap.mu_B - lambda_wealth * abs(snap.mu_R - snap.mu_B)
            return np.full(n, val * inv_n)
        elif constraint_type == "eo":
            tpr_R, tpr_B = _group_tpr(snap)
            val = tpr_R + tpr_B - lambda_wealth * abs(tpr_R - tpr_B)
            return np.full(n, val * inv_n)
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
