import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class TransitionParameterLearner:
    """Learn transition parameters θ that GOVERN the sequential dynamics."""

    def __init__(
        self,
        default_rate_min: float = 0.05,
        default_rate_max: float = 0.25,
        loan_amount_mean: float = 30.0,
        loan_amount_std: float = 10.0,
        investment_return_rate: float = 0.35,
        interest_rate: float = 0.18,
    ):
        self.theta_S = None
        self.theta_X = None
        self.b = None
        self.kappa_mean = {}
        self.kappa_std = {}
        self.default_rate_min = default_rate_min
        self.default_rate_max = default_rate_max
        self.sigma = None
        self.application_sensitivity = {}

        self.loan_amount_mean = loan_amount_mean
        self.loan_amount_std = loan_amount_std
        self.investment_return_rate = investment_return_rate
        self.interest_rate = interest_rate

        self.individual_default_probs = {}
        self.individual_loan_amounts = {}
        self.individual_wealth_gains = {}

    def learn_approval_model(self, data: pd.DataFrame):
        """Learn h_θ(X,S) = θ_S*S + θ_X*X + b"""
        X_norm = (data["X"] - data["X"].mean()) / data["X"].std()
        X_features = np.column_stack([data["S"].values, X_norm.values])
        y = data["approved"].values

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_features, y)

        self.theta_S = model.coef_[0, 0]
        self.theta_X = model.coef_[0, 1]
        self.b = model.intercept_[0]
        self.X_mean = data["X"].mean()
        self.X_std = data["X"].std()

        return model

    def learn_application_behavior(self, data: pd.DataFrame):
        """Learn how h_θ(X,S) affects application probability."""
        for group_name, S_val in [("male", 1), ("female", 0)]:
            group_data = data[data["S"] == S_val]
            base_rate = len(group_data) / len(data)
            self.application_sensitivity[group_name] = {
                "base_rate": base_rate,
                "sensitivity": 0.5,
            }

    def learn_wealth_gain_parameters(self, data: pd.DataFrame):
        """Learn κ parameters that govern wealth transitions."""
        for group_name, S_val in [("male", 1), ("female", 0)]:
            group_data = data[data["S"] == S_val]
            approved = group_data[group_data["approved"] == 1]

            if len(approved) > 0:
                kappa_success_mean = self.loan_amount_mean * (
                    self.investment_return_rate - self.interest_rate
                )
                kappa_success_std = self.loan_amount_std * (
                    self.investment_return_rate - self.interest_rate
                )

                self.kappa_mean[group_name] = {
                    "success": kappa_success_mean,
                    "default": 0.0,
                }
                self.kappa_std[group_name] = {
                    "success": kappa_success_std,
                    "default": 0.0,
                }

    def learn_default_model(self, data: pd.DataFrame):
        """Learn P(default | X, approved) via Bernoulli distribution."""
        pass

    def learn_variance_parameters(self, data: pd.DataFrame):
        """Learn σ² - governs wealth distribution spread."""
        self.sigma = data["X"].std()

    def initialize_individual_parameters(
        self, N_male: int, N_female: int, seed: int = None,
        X_male: np.ndarray = None, X_female: np.ndarray = None,
    ):
        """PRE-GENERATE all individual parameters at simulation start.

        Default probability is derived from each individual's creditworthiness
        RANK (via X, which is already a monotonic function of creditworthiness
        -- X = 10 + 190*sc_i, see data_loader.py -- so ranking by X is
        equivalent to ranking by creditworthiness directly): the
        lowest-creditworthiness individual gets default_rate_max, the
        highest gets default_rate_min, linear in rank in between. Rank
        (not raw score) is used specifically so the population MEAN default
        probability is exactly (default_rate_min + default_rate_max) / 2
        regardless of the population's creditworthiness distribution shape
        -- preserves the existing calibration (pretrain ~8.5%, deploy ~15%
        mean) exactly, only redistributes who carries the risk.

        Without this, default risk was drawn i.i.d. Bernoulli, independent
        of wealth/creditworthiness -- meaning no observable feature could
        ever predict an individual's true risk, so no policy could learn to
        be selective no matter how it was trained. X_male/X_female make
        that correlation real; omit them only for legacy/no-population
        callers (falls back to the previous population-uniform behaviour).

        default_prob is now a genuine per-individual PROBABILITY (not a
        pre-realized 0/1 latent outcome) -- matches how reward.py's
        expected-value formulas already treat it, and gives a continuous,
        learnable signal instead of a binary flag that only weakly shifts
        which fraction of similar-creditworthiness individuals are marked
        as guaranteed defaulters.
        """
        if seed is not None:
            np.random.seed(seed)

        for group, N, X in [
            ("male", N_male, X_male), ("female", N_female, X_female)
        ]:
            if X is not None and len(X) == N:
                ranks = np.argsort(np.argsort(X))  # 0 = lowest X (worst credit)
                percentile = ranks / max(N - 1, 1)  # 0 (worst) .. 1 (best)
                self.individual_default_probs[group] = (
                    self.default_rate_max
                    - (self.default_rate_max - self.default_rate_min) * percentile
                )
            else:
                p_default = (self.default_rate_min + self.default_rate_max) / 2.0
                self.individual_default_probs[group] = np.full(N, p_default)

            self.individual_loan_amounts[group] = np.clip(
                np.random.normal(self.loan_amount_mean, self.loan_amount_std, N),
                10.0,
                100.0,
            )
            net_return_rate = self.investment_return_rate - self.interest_rate
            self.individual_wealth_gains[group] = (
                self.individual_loan_amounts[group] * net_return_rate
            )

    def fit(self, data: pd.DataFrame):
        """Learn ALL transition parameters."""
        self.learn_approval_model(data)
        self.learn_application_behavior(data)
        self.learn_wealth_gain_parameters(data)
        self.learn_default_model(data)
        self.learn_variance_parameters(data)

    def compute_approval_score(self, X: float, S: int) -> float:
        """Compute h_θ(X,S)."""
        X_norm = (X - self.X_mean) / self.X_std
        return self.theta_S * S + self.theta_X * X_norm + self.b

    def compute_approval_probability(self, X: float, S: int) -> float:
        """Compute ν(h_θ(X,S)) = σ(h_θ(X,S))"""
        z = self.compute_approval_score(X, S)
        return 1.0 / (1.0 + np.exp(-z))

    def compute_application_probability(
        self, X: float, group: str, base_rate: float
    ) -> float:
        """Application probability DEPENDS ON learned θ."""
        S = 1 if group == "male" else 0
        sensitivity = self.application_sensitivity.get(group, {}).get(
            "sensitivity", 0.5
        )
        approval_prob = self.compute_approval_probability(X, S)
        application_prob = base_rate * (1.0 + sensitivity * approval_prob)
        return np.clip(application_prob, 0.0, 1.0)

    def get_default_probability(self, group: str, individual_id: int) -> float:
        return self.individual_default_probs[group][individual_id]

    def get_loan_amount(self, group: str, individual_id: int) -> float:
        return self.individual_loan_amounts[group][individual_id]

    def get_wealth_gain(self, group: str, individual_id: int, defaulted: bool) -> float:
        if defaulted:
            return 0.0
        return self.individual_wealth_gains[group][individual_id]
