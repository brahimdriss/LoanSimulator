import numpy as np


class RewardFunction:
    """
    Reward functions for RL agent.

    constraint_type maps to the four fairness columns in the paper:
        'predictive'  — Predictive Fairness    (accuracy / approval-rate metrics)
        'social'      — Outcome/Social Fairness (mean-wealth metrics)
        'dm'          — DM's Fairness           (bank-profit metrics)
        'two_sided'   — α-Two sided Fairness    (blends profit + fairness, α = lambda_wealth)

    Acc_t^perf (expected accuracy per applicant):
        approval_prob * (1 - default_prob) + (1 - approval_prob) * default_prob
        = P(approve ∩ repay) + P(reject ∩ default)

    r_t^perf,g (expected per-loan profit rate for group g):
        interest_rate * (1 - rho_g) - rho_g
        where rho_g = total_defaults_g / max(total_loans_g, 1)
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
            r_g = mean_loan_g * ((1 - rho_g) * interest_rate - 2 * rho_g)
        """
        rho_R = env.total_defaults_R / max(env.total_loans_R, 1)
        rho_B = env.total_defaults_B / max(env.total_loans_B, 1)
        mean_loan_R = float(np.mean(env.theta_params.individual_loan_amounts["male"]))
        mean_loan_B = float(np.mean(env.theta_params.individual_loan_amounts["female"]))
        r_R = mean_loan_R * ((1 - rho_R) * env.interest_rate - 2 * rho_R)
        r_B = mean_loan_B * ((1 - rho_B) * env.interest_rate - 2 * rho_B)
        return r_R, r_B

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
        'dm'          → r_t^perf
        'two_sided'   → (1 − α) * r_t^perf + α * (μ_R + μ_B)
        """
        applicant = info.get("applicant")

        if constraint_type == "predictive":
            return RewardFunction._calculate_accuracy(action, applicant)

        elif constraint_type == "social":
            return 0.0  # undefined (−) in the table

        elif constraint_type == "dm":
            return RewardFunction._calculate_bank_profit(env, action, applicant)

        elif constraint_type == "two_sided":
            alpha       = lambda_wealth
            bank_profit = RewardFunction._calculate_bank_profit(env, action, applicant)
            return (1 - alpha) * bank_profit + alpha * (env.mu_R + env.mu_B)

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

        elif constraint_type == "dm":
            r_R, r_B = RewardFunction._group_profit_rates(env)
            return min(r_R, r_B)

        elif constraint_type == "two_sided":
            bank_profit = RewardFunction._calculate_bank_profit(env, action, applicant)
            alpha = lambda_wealth
            return (1 - alpha) * bank_profit + alpha * min(env.mu_R, env.mu_B)

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

        elif constraint_type == "dm":
            r_R, r_B = RewardFunction._group_profit_rates(env)
            return bank_profit - lambda_wealth * abs(r_R - r_B)

        elif constraint_type == "two_sided":
            alpha = lambda_wealth
            return (1 - alpha) * bank_profit - alpha * abs(env.mu_R - env.mu_B)

        else:
            raise ValueError(f"Unknown constraint_type: {constraint_type!r}")
