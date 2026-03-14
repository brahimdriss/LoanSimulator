from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .transition_learner import TransitionParameterLearner


class IncomeEnvironment(gym.Env):
    """Environment where learned θ parameters GOVERN the dynamics."""

    def __init__(
        self,
        theta_params: TransitionParameterLearner,
        initial_wealth_male: np.ndarray,
        initial_wealth_female: np.ndarray,
        N_male: int = 2000,
        N_female: int = 2000,
        T: int = 100,
        dt: float = 0.1,
        interest_rate: float = 0.15,
        alpha_R: float = 0.3,
        alpha_B: float = 0.3,
        beta_R: float = 2.0,
        beta_B: float = 2.0,
        seed: int = None,
    ):
        super().__init__()

        self.theta_params = theta_params
        self.N_male = N_male
        self.N_female = N_female
        self.T = T
        self.dt = dt
        self.interest_rate = interest_rate
        self.seed = seed

        self.alpha_R = alpha_R
        self.alpha_B = alpha_B
        self.beta_R = beta_R
        self.beta_B = beta_B

        self.initial_X_male = initial_wealth_male[:N_male].copy()
        self.initial_X_female = initial_wealth_female[:N_female].copy()

        self.theta_params.initialize_individual_parameters(N_male, N_female, seed)

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0, -5.0, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [500.0, 1.0, 500.0, 500.0, 10.0, 10.0, 1.0, 1.0, 1.0, 5.0, 5.0, 200.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment."""
        super().reset(seed=seed)

        self.current_X_male = self.initial_X_male.copy()
        self.current_X_female = self.initial_X_female.copy()

        self.mu_R = np.mean(self.current_X_male)
        self.mu_B = np.mean(self.current_X_female)
        self.var_R = np.var(self.current_X_male)
        self.var_B = np.var(self.current_X_female)

        self.event_times_R = []
        self.event_times_B = []

        self.total_defaults_R = 0
        self.total_defaults_B = 0
        self.total_loans_R = 0
        self.total_loans_B = 0
        self.total_applications_R = 0
        self.total_applications_B = 0

        self.history = {
            "time": [0.0],
            "mu_R": [self.mu_R],
            "mu_B": [self.mu_B],
            "var_R": [self.var_R],
            "var_B": [self.var_B],
            "lambda_R": [self._compute_lambda_R(0.0)],
            "lambda_B": [self._compute_lambda_B(0.0)],
            "loan_arrivals_R": [0],
            "loan_arrivals_B": [0],
            "loan_approvals_R": [0],
            "loan_approvals_B": [0],
            "defaults_R": [0],
            "defaults_B": [0],
            "default_rate_R": [0.0],
            "default_rate_B": [0.0],
            "wealth_gap": [self.mu_R - self.mu_B],
            "approval_disparity": [0.0],
            "approval_rate_R": [0.0],
            "approval_rate_B": [0.0],
            "profit": [0.0],
            "covariance_R": [0.0],
            "covariance_B": [0.0],
            "expected_L_R": [0.0],
            "expected_L_B": [0.0],
        }

        self.current_time = 0.0
        self.time_steps = np.arange(0, self.T, self.dt)
        self.time_index = 0

        self.pending_applications = []
        self.current_applicant = None
        self.timestep_data = None
        self.timestep_profit = 0.0

        obs = self._get_observation()
        return obs, {}

    def _f_networth_to_rate(self, mu: float) -> float:
        mu = mu * 5.0
        return max(0.5, 2.0 + 0.01 * mu)

    def _phi_R(self, t: float) -> float:
        return self.alpha_R * np.exp(-self.beta_R * t)

    def _phi_B(self, t: float) -> float:
        return self.alpha_B * np.exp(-self.beta_B * t)

    def _compute_lambda_R(self, t: float) -> float:
        """Hawkes intensity for male group with vectorized computation."""
        base_rate = self._f_networth_to_rate(self.mu_R)
        if not self.event_times_R:
            return base_rate
        event_array = np.array(self.event_times_R)
        ages = t - event_array[event_array < t]
        if len(ages) == 0:
            return base_rate
        excitation = np.sum(self.alpha_R * np.exp(-self.beta_R * ages))
        return base_rate + excitation

    def _compute_lambda_B(self, t: float) -> float:
        """Hawkes intensity for female group with vectorized computation."""
        base_rate = self._f_networth_to_rate(self.mu_B)
        if not self.event_times_B:
            return base_rate
        event_array = np.array(self.event_times_B)
        ages = t - event_array[event_array < t]
        if len(ages) == 0:
            return base_rate
        excitation = np.sum(self.alpha_B * np.exp(-self.beta_B * ages))
        return base_rate + excitation

    def _get_observation(self):
        lambda_R = self._compute_lambda_R(self.current_time)
        lambda_B = self._compute_lambda_B(self.current_time)
        rho_R = self.total_defaults_R / max(self.total_loans_R, 1)
        rho_B = self.total_defaults_B / max(self.total_loans_B, 1)

        if self.current_applicant is None:
            mu_R_norm = self.mu_R / 100.0
            mu_B_norm = self.mu_B / 100.0
            return np.array(
                [
                    0.0,
                    0.5,
                    mu_R_norm,
                    mu_B_norm,
                    lambda_R,
                    lambda_B,
                    rho_R,
                    rho_B,
                    0.5,
                    self.theta_params.theta_S,
                    self.theta_params.theta_X,
                    0.0,
                ],
                dtype=np.float32,
            )
        else:
            theta_approval_prob = self.theta_params.compute_approval_probability(
                self.current_applicant["X"], self.current_applicant["S"]
            )
            X_norm = self.current_applicant["X"] / 100.0
            mu_R_norm = self.mu_R / 100.0
            mu_B_norm = self.mu_B / 100.0
            loan_amount = self.current_applicant["loan_amount"]

            return np.array(
                [
                    X_norm,
                    float(self.current_applicant["S"]),
                    mu_R_norm,
                    mu_B_norm,
                    lambda_R,
                    lambda_B,
                    rho_R,
                    rho_B,
                    theta_approval_prob,
                    self.theta_params.theta_S,
                    self.theta_params.theta_X,
                    loan_amount,
                ],
                dtype=np.float32,
            )

    def _generate_timestep_applications(self):
        """Vectorized application generation without individual loops."""
        t = self.current_time
        lambda_R = self._compute_lambda_R(t)
        lambda_B = self._compute_lambda_B(t)

        applications = []
        apply_select_R = np.zeros(self.N_male)
        apply_select_B = np.zeros(self.N_female)

        # ---- MALE APPLICATIONS (vectorized) ----
        base_rate_R = lambda_R * self.dt / self.N_male
        X_male = self.current_X_male
        X_norm_R = (X_male - self.theta_params.X_mean) / self.theta_params.X_std
        approval_probs_R = 1.0 / (
            1.0
            + np.exp(
                -(
                    self.theta_params.theta_S * 1
                    + self.theta_params.theta_X * X_norm_R
                    + self.theta_params.b
                )
            )
        )
        sensitivity_R = self.theta_params.application_sensitivity.get("male", {}).get(
            "sensitivity", 0.5
        )
        app_probs_R = np.clip(
            base_rate_R * (1.0 + sensitivity_R * approval_probs_R), 0.0, 1.0
        )

        random_vals_R = np.random.random(self.N_male)
        applying_indices_R = np.where(random_vals_R < app_probs_R)[0]
        if len(applying_indices_R) > 10:
            applying_indices_R = applying_indices_R[:10]

        for idx in applying_indices_R:
            apply_select_R[idx] = 1
            applications.append(
                {
                    "group": "male",
                    "S": 1,
                    "individual_id": idx,
                    "X": X_male[idx],
                    "default_prob": self.theta_params.get_default_probability(
                        "male", idx
                    ),
                    "loan_amount": self.theta_params.get_loan_amount("male", idx),
                    "wealth_gain": self.theta_params.get_wealth_gain(
                        "male", idx, defaulted=False
                    ),
                    "theta_approval_prob": approval_probs_R[idx],
                }
            )

        # ---- FEMALE APPLICATIONS (vectorized) ----
        base_rate_B = lambda_B * self.dt / self.N_female
        X_female = self.current_X_female
        X_norm_B = (X_female - self.theta_params.X_mean) / self.theta_params.X_std
        approval_probs_B = 1.0 / (
            1.0
            + np.exp(
                -(
                    self.theta_params.theta_S * 0
                    + self.theta_params.theta_X * X_norm_B
                    + self.theta_params.b
                )
            )
        )
        sensitivity_B = self.theta_params.application_sensitivity.get("female", {}).get(
            "sensitivity", 0.5
        )
        app_probs_B = np.clip(
            base_rate_B * (1.0 + sensitivity_B * approval_probs_B), 0.0, 1.0
        )

        random_vals_B = np.random.random(self.N_female)
        applying_indices_B = np.where(random_vals_B < app_probs_B)[0]
        if len(applying_indices_B) > 10:
            applying_indices_B = applying_indices_B[:10]

        for idx in applying_indices_B:
            apply_select_B[idx] = 1
            applications.append(
                {
                    "group": "female",
                    "S": 0,
                    "individual_id": idx,
                    "X": X_female[idx],
                    "default_prob": self.theta_params.get_default_probability(
                        "female", idx
                    ),
                    "loan_amount": self.theta_params.get_loan_amount("female", idx),
                    "wealth_gain": self.theta_params.get_wealth_gain(
                        "female", idx, defaulted=False
                    ),
                    "theta_approval_prob": approval_probs_B[idx],
                }
            )

        n_arrivals_R = len(applying_indices_R)
        n_arrivals_B = len(applying_indices_B)
        self.total_applications_R += n_arrivals_R
        self.total_applications_B += n_arrivals_B

        return {
            "applications": applications,
            "apply_select_R": apply_select_R,
            "apply_select_B": apply_select_B,
            "lambda_R": lambda_R,
            "lambda_B": lambda_B,
            "n_arrivals_R": n_arrivals_R,
            "n_arrivals_B": n_arrivals_B,
            "p_theta_R": app_probs_R.mean(),
            "p_theta_B": app_probs_B.mean(),
        }

    def step(self, action):
        reward = 0.0

        if isinstance(action, np.ndarray):
            approval_prob = np.clip(action[0], 0.0, 1.0)
        else:
            approval_prob = np.clip(float(action), 0.0, 1.0)

        if self.current_applicant is not None:
            applicant = self.current_applicant
            approved = np.random.random() < approval_prob

            if approved:
                if applicant["S"] == 1:
                    self.event_times_R.append(self.current_time)
                else:
                    self.event_times_B.append(self.current_time)

                defaults = np.random.random() < applicant["default_prob"]
                # Borrower gains wealth_gain on success, nothing on default
                kappa_i = 0.0 if defaults else applicant["wealth_gain"]

                if applicant["S"] == 1:
                    self.current_X_male[applicant["individual_id"]] += kappa_i
                    self.total_loans_R += 1
                    self.timestep_data["approvals_R"] += 1
                    self.timestep_data["apply_select_R"][applicant["individual_id"]] = 1

                    if defaults:
                        self.total_defaults_R += 1
                        self.timestep_data["defaults_R"] += 1
                        self.timestep_profit -= applicant["loan_amount"]
                    else:
                        self.timestep_profit += (
                            applicant["loan_amount"] * self.interest_rate
                        )
                else:
                    self.current_X_female[applicant["individual_id"]] += kappa_i
                    self.total_loans_B += 1
                    self.timestep_data["approvals_B"] += 1
                    self.timestep_data["apply_select_B"][applicant["individual_id"]] = 1

                    if defaults:
                        self.total_defaults_B += 1
                        self.timestep_data["defaults_B"] += 1
                        self.timestep_profit -= applicant["loan_amount"]
                    else:
                        self.timestep_profit += (
                            applicant["loan_amount"] * self.interest_rate
                        )

        if self.pending_applications:
            self.current_applicant = self.pending_applications.pop(0)
        else:
            self.current_applicant = None

        if self.current_applicant is None and self.time_index < len(self.time_steps):
            if self.timestep_data is not None:
                self.mu_R = np.mean(self.current_X_male)
                self.mu_B = np.mean(self.current_X_female)
                self.var_R = np.var(self.current_X_male)
                self.var_B = np.var(self.current_X_female)

                cov_R = (
                    np.cov(self.current_X_male, self.timestep_data["apply_select_R"])[
                        0, 1
                    ]
                    if self.var_R > 0
                    else 0
                )
                cov_B = (
                    np.cov(self.current_X_female, self.timestep_data["apply_select_B"])[
                        0, 1
                    ]
                    if self.var_B > 0
                    else 0
                )

                approval_rate_R = self.total_loans_R / max(self.total_applications_R, 1)
                approval_rate_B = self.total_loans_B / max(self.total_applications_B, 1)

                self.history["time"].append(self.current_time)
                self.history["mu_R"].append(self.mu_R)
                self.history["mu_B"].append(self.mu_B)
                self.history["var_R"].append(self.var_R)
                self.history["var_B"].append(self.var_B)
                self.history["lambda_R"].append(self.timestep_data["lambda_R"])
                self.history["lambda_B"].append(self.timestep_data["lambda_B"])
                self.history["loan_arrivals_R"].append(
                    self.timestep_data["n_arrivals_R"]
                )
                self.history["loan_arrivals_B"].append(
                    self.timestep_data["n_arrivals_B"]
                )
                self.history["loan_approvals_R"].append(
                    self.timestep_data["approvals_R"]
                )
                self.history["loan_approvals_B"].append(
                    self.timestep_data["approvals_B"]
                )
                self.history["defaults_R"].append(self.timestep_data["defaults_R"])
                self.history["defaults_B"].append(self.timestep_data["defaults_B"])
                self.history["default_rate_R"].append(
                    self.total_defaults_R / max(self.total_loans_R, 1)
                )
                self.history["default_rate_B"].append(
                    self.total_defaults_B / max(self.total_loans_B, 1)
                )
                self.history["wealth_gap"].append(self.mu_R - self.mu_B)
                self.history["approval_disparity"].append(
                    approval_rate_R - approval_rate_B
                )
                self.history["approval_rate_R"].append(approval_rate_R)
                self.history["approval_rate_B"].append(approval_rate_B)
                self.history["profit"].append(self.timestep_profit)
                self.history["covariance_R"].append(cov_R)
                self.history["covariance_B"].append(cov_B)
                self.history["expected_L_R"].append(approval_rate_R)
                self.history["expected_L_B"].append(approval_rate_B)

            if self.time_index < len(self.time_steps):
                self.current_time = self.time_steps[self.time_index]
                self.time_index += 1
                self.timestep_profit = 0.0

                timestep_info = self._generate_timestep_applications()
                self.pending_applications = timestep_info["applications"].copy()

                self.timestep_data = {
                    "apply_select_R": timestep_info["apply_select_R"].copy(),
                    "apply_select_B": timestep_info["apply_select_B"].copy(),
                    "lambda_R": timestep_info["lambda_R"],
                    "lambda_B": timestep_info["lambda_B"],
                    "n_arrivals_R": timestep_info["n_arrivals_R"],
                    "n_arrivals_B": timestep_info["n_arrivals_B"],
                    "approvals_R": 0,
                    "approvals_B": 0,
                    "defaults_R": 0,
                    "defaults_B": 0,
                    "p_theta_R": timestep_info["p_theta_R"],
                    "p_theta_B": timestep_info["p_theta_B"],
                }

                if self.pending_applications:
                    self.current_applicant = self.pending_applications.pop(0)

        terminated = (
            self.time_index >= len(self.time_steps) and self.current_applicant is None
        )
        truncated = False

        obs = self._get_observation()
        info = {
            "time": self.current_time,
            "mu_R": self.mu_R,
            "mu_B": self.mu_B,
            "wealth_gap": self.mu_R - self.mu_B,
            "applicant": self.current_applicant,
            "agent_approval_prob": approval_prob,
            "p_theta_R": self.timestep_data["p_theta_R"],
            "p_theta_B": self.timestep_data["p_theta_B"],
        }

        return obs, reward, terminated, truncated, info
