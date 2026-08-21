from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .reward import RewardSnapshot
from .transition_learner import TransitionParameterLearner

# Reference population against which the wealth-driven base arrival rate
# f(mu) is calibrated. The Hawkes intensity lambda_g is an ABSOLUTE arrival
# rate, and applicant i applies w.p. lambda_g*dt/N_g*(...), so the N cancels
# and the expected cohort size is independent of population -- measured flat
# at ~5.3 arrivals/timestep for N = 3k, 20k and 100k alike. That means each
# approved loan moves the group mean by kappa/N, so mu drift (and with it
# ALL performative feedback, which flows through mu) decays as 1/N: measured
# 0.132 per episode at N=3k but 0.005 at N=100k.
#
# Scaling f(mu) by N_g/ARRIVAL_REF_N makes the arrival rate per-capita, so
# cohort size grows with population and mu drift stays constant -- i.e. a
# larger N buys more individuals and less per-episode noise WITHOUT
# weakening the dynamics being studied. ARRIVAL_REF_N is set to the paper's
# published population so N = 3000 reproduces the previous behaviour exactly.
#
# The Hawkes constants alpha/beta need no recalibration: the excitation term
# sums over approval events, whose count also scales with N, so the
# excitation-to-base ratio those constants were tuned against is preserved.
ARRIVAL_REF_N = 3000

# Matthew-effect strength, on RELATIVE standing: f(mu_g) scales with
# mu_g / mu_bar (group mean vs population mean), not with absolute mu_g.
#
# The original absolute form f = 2 + 0.05*mu made inflow proportional to
# wealth itself, i.e. dmu/dt ~ mu -- exponential, unbounded growth. Measured
# over 1000 episodes it drove mu from 60 to ~43,000 with an absolute wealth
# gap of ~5,200, while the wealth RATIO stayed flat at ~1.14: the divergence
# was essentially a units artifact, not an inequality story. (It went
# unnoticed because earlier bugs suppressed lending to ~2% approval, so
# wealth never had the chance to run away.) Adding depreciation does not fix
# it -- an outflow ~delta*mu against an inflow ~c*mu is linear-vs-linear, so
# it either cancels on a knife edge or one side dominates; measured, delta
# =2e-5 still diverged to ~700k while delta=1e-4 collapsed wealth to 17.
#
# Normalising by mu_bar makes the exponent exactly MATTHEW_C:
#   c = 1  -> growth proportional to wealth, wealth ratios stay constant
#   c > 1  -> super-proportional, relative inequality genuinely diverges
#   c < 1  -> sub-proportional, inequality self-corrects
# c = 2 gives a compounding disadvantage the ADM must actively counteract
# (ratio 1.25 -> ~4 over 1000 episodes) while keeping absolute wealth in a
# sane range.
#
# NOTE: this couples the groups through mu_bar -- group R's arrival rate now
# depends on mu_F. That is intrinsic to "relative standing"; a growing
# normaliser is what prevents the exponential blow-up.
MATTHEW_C = 2.0


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
        interest_rate: float = 0.18,
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

        # Per-capita arrival scaling -- see ARRIVAL_REF_N above.
        self._arrival_scale_R = N_male / ARRIVAL_REF_N
        self._arrival_scale_B = N_female / ARRIVAL_REF_N

        # Fail fast on an undersized population. Slicing an array shorter than
        # N silently yields fewer entries while self.N_* keeps the requested
        # value, and the mismatch only surfaces ~200 lines later inside
        # _generate_timestep_applications as an opaque
        #   "operands could not be broadcast together with shapes (12000,) (6439,)"
        # after several minutes of training have already been thrown away.
        for _name, _arr, _n in (("male", initial_wealth_male, N_male),
                                ("female", initial_wealth_female, N_female)):
            if len(_arr) < _n:
                raise ValueError(
                    f"N_{_name}={_n} requested but only {len(_arr)} {_name} "
                    f"records supplied. Load more data (the loader's "
                    f"sample_size caps this: Adult is ~2/3 male, so the "
                    f"female count binds first) or lower N_{_name}."
                )

        self.initial_X_male = initial_wealth_male[:N_male].copy()
        self.initial_X_female = initial_wealth_female[:N_female].copy()

        self.theta_params.initialize_individual_parameters(
            N_male, N_female, seed,
            X_male=self.initial_X_male, X_female=self.initial_X_female,
        )

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
        """Reset environment (single-applicant interface, unchanged)."""
        self._reset_core(seed=seed)
        obs = self._get_observation()
        return obs, {}

    def reset_cohort(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Batched interface: reset, generate the first timestep's cohort, and
        return the observation MATRIX for that whole cohort (shape [n, 12])
        instead of one applicant's observation. Use with step_cohort().
        """
        self._reset_core(seed=seed)
        self._advance_to_nonempty_cohort()
        return self._get_cohort_observations(), {}

    def _reset_core(self, seed: Optional[int] = None):
        """State reset shared by reset() and reset_cohort()."""
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

    def _f_networth_to_rate(self, mu: float, scale: float = 1.0) -> float:
        """Wealth-driven base arrival intensity (the Matthew-effect term),
        on RELATIVE standing mu_g/mu_bar -- see MATTHEW_C. `scale` =
        N_g / ARRIVAL_REF_N makes this per-capita, see ARRIVAL_REF_N.
        The 0.5 floor keeps the rate positive if a group falls far behind
        (at c=2 the raw form turns negative below mu_g = 0.5*mu_bar)."""
        mu_bar = 0.5 * (self.mu_R + self.mu_B)
        rel = mu / max(mu_bar, 1e-8)
        return max(0.5, 2.0 * (1.0 + MATTHEW_C * (rel - 1.0))) * scale

    def _phi_R(self, t: float) -> float:
        return self.alpha_R * np.exp(-self.beta_R * t)

    def _phi_B(self, t: float) -> float:
        return self.alpha_B * np.exp(-self.beta_B * t)

    def _compute_lambda_R(self, t: float) -> float:
        """Hawkes intensity for male group with vectorized computation."""
        base_rate = self._f_networth_to_rate(self.mu_R, self._arrival_scale_R)
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
        base_rate = self._f_networth_to_rate(self.mu_B, self._arrival_scale_B)
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

    def _get_cohort_observations(self) -> np.ndarray:
        """
        Batched counterpart of _get_observation(): one row per applicant
        currently in self.pending_applications, shape [n, 12]. Same 12
        columns/scaling as _get_observation() -- the env-level columns
        (mu_R, mu_B, lambda_R, lambda_B, rho_R, rho_B, theta_S, theta_X)
        are identical across every row (the whole cohort shares one state);
        only X, S, theta_approval_prob, loan_amount vary per applicant.

        Returns an empty (0, 12) array if the cohort is empty -- callers
        must check for this (can happen at small N; use
        _advance_to_nonempty_cohort() to skip past it during rollout).
        """
        n = len(self.pending_applications)
        if n == 0:
            return np.zeros((0, 12), dtype=np.float32)

        lambda_R = self._compute_lambda_R(self.current_time)
        lambda_B = self._compute_lambda_B(self.current_time)
        rho_R = self.total_defaults_R / max(self.total_loans_R, 1)
        rho_B = self.total_defaults_B / max(self.total_loans_B, 1)
        mu_R_norm = self.mu_R / 100.0
        mu_B_norm = self.mu_B / 100.0

        X = np.array([a["X"] for a in self.pending_applications], dtype=np.float64)
        S = np.array([a["S"] for a in self.pending_applications], dtype=np.float64)
        loan_amount = np.array(
            [a["loan_amount"] for a in self.pending_applications], dtype=np.float64
        )
        theta_approval_prob = np.array(
            [a["theta_approval_prob"] for a in self.pending_applications],
            dtype=np.float64,
        )

        obs = np.zeros((n, 12), dtype=np.float32)
        obs[:, 0] = X / 100.0
        obs[:, 1] = S
        obs[:, 2] = mu_R_norm
        obs[:, 3] = mu_B_norm
        obs[:, 4] = lambda_R
        obs[:, 5] = lambda_B
        obs[:, 6] = rho_R
        obs[:, 7] = rho_B
        obs[:, 8] = theta_approval_prob
        obs[:, 9] = self.theta_params.theta_S
        obs[:, 10] = self.theta_params.theta_X
        obs[:, 11] = loan_amount
        return obs

    def _advance_to_nonempty_cohort(self):
        """
        Advance timesteps (generating each one's cohort) until a non-empty
        cohort is pending or the episode ends. At small N an occasional
        empty timestep is possible; at the population sizes this is meant
        to run at, this loop almost always executes exactly once.
        """
        while (
            len(self.pending_applications) == 0
            and self.time_index < len(self.time_steps)
        ):
            self._advance_one_timestep()

    def _advance_one_timestep(self):
        """Record the just-finished timestep's history (if any) and
        generate the next timestep's cohort into self.pending_applications."""
        if self.timestep_data is not None:
            self._record_timestep_history()

        if self.time_index < len(self.time_steps):
            self.current_time = self.time_steps[self.time_index]
            self.time_index += 1
            self.timestep_profit = 0.0

            timestep_info = self._generate_timestep_applications()
            self.pending_applications = timestep_info["applications"]

            self.timestep_data = {
                "apply_select_R": timestep_info["apply_select_R"],
                "apply_select_B": timestep_info["apply_select_B"],
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

    def _record_timestep_history(self):
        """Shared history-recording logic, extracted so both the
        single-applicant step() and the batched step_cohort() log
        identically."""
        self.mu_R = np.mean(self.current_X_male)
        self.mu_B = np.mean(self.current_X_female)
        self.var_R = np.var(self.current_X_male)
        self.var_B = np.var(self.current_X_female)

        cov_R = (
            np.cov(self.current_X_male, self.timestep_data["apply_select_R"])[0, 1]
            if self.var_R > 0
            else 0
        )
        cov_B = (
            np.cov(self.current_X_female, self.timestep_data["apply_select_B"])[0, 1]
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
        self.history["loan_arrivals_R"].append(self.timestep_data["n_arrivals_R"])
        self.history["loan_arrivals_B"].append(self.timestep_data["n_arrivals_B"])
        self.history["loan_approvals_R"].append(self.timestep_data["approvals_R"])
        self.history["loan_approvals_B"].append(self.timestep_data["approvals_B"])
        self.history["defaults_R"].append(self.timestep_data["defaults_R"])
        self.history["defaults_B"].append(self.timestep_data["defaults_B"])
        self.history["default_rate_R"].append(
            self.total_defaults_R / max(self.total_loans_R, 1)
        )
        self.history["default_rate_B"].append(
            self.total_defaults_B / max(self.total_loans_B, 1)
        )
        self.history["wealth_gap"].append(self.mu_R - self.mu_B)
        self.history["approval_disparity"].append(approval_rate_R - approval_rate_B)
        self.history["approval_rate_R"].append(approval_rate_R)
        self.history["approval_rate_B"].append(approval_rate_B)
        self.history["profit"].append(self.timestep_profit)
        self.history["covariance_R"].append(cov_R)
        self.history["covariance_B"].append(cov_B)
        self.history["expected_L_R"].append(approval_rate_R)
        self.history["expected_L_B"].append(approval_rate_B)

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

    def step_cohort(self, actions: np.ndarray):
        """
        Batched counterpart of step(): one action per applicant currently in
        self.pending_applications (same order), processed all at once.

        Returns (next_obs, terminated, truncated, info), where `info`
        carries a RewardSnapshot taken BEFORE this cohort's approvals are
        applied, plus this cohort's default_probs/loan_amounts/actions --
        pass these straight to reward.compute_batched_rewards() to get one
        reward per applicant. Rewards are not computed here so the
        environment stays reward-function-agnostic (matches step(), which
        also leaves reward computation to the caller).

        Economics are identical to step(), just vectorized: approve w.p.
        action, default w.p. the applicant's own fixed default_prob, wealth
        gain added only on approved+non-default, Hawkes event added on
        approval regardless of eventual default.
        """
        applications = self.pending_applications
        self.pending_applications = []  # consume the cohort now -- _advance_to_nonempty_cohort()
        # below checks this flag to decide whether to generate the next one;
        # leaving it non-empty here would make it re-see this same
        # already-processed cohort forever (time_index would never advance).
        n = len(applications)
        actions = np.clip(np.asarray(actions, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if len(actions) != n:
            raise ValueError(f"step_cohort: {len(actions)} actions for {n} applicants")

        # Snapshot BEFORE this cohort mutates anything -- every applicant in
        # the batch sees the same pre-cohort world (no within-batch ordering
        # bias, unlike a hypothetical applicant-by-applicant replay).
        snap = RewardSnapshot.from_env(self)

        if n > 0:
            default_probs = np.array([a["default_prob"] for a in applications])
            wealth_gains = np.array([a["wealth_gain"] for a in applications])
            loan_amounts = np.array([a["loan_amount"] for a in applications])
            S = np.array([a["S"] for a in applications])
            ids = np.array([a["individual_id"] for a in applications], dtype=np.int64)

            approved = np.random.random(n) < actions
            defaults = np.random.random(n) < default_probs
            kappa = np.where(defaults, 0.0, wealth_gains)

            male_mask = approved & (S == 1)
            female_mask = approved & (S == 0)

            if male_mask.any():
                np.add.at(self.current_X_male, ids[male_mask], kappa[male_mask])
                n_male = int(male_mask.sum())
                self.event_times_R.extend([self.current_time] * n_male)
                self.total_loans_R += n_male
                self.timestep_data["approvals_R"] += n_male
                self.timestep_data["apply_select_R"][ids[male_mask]] = 1
                male_def = defaults[male_mask]
                self.total_defaults_R += int(male_def.sum())
                self.timestep_data["defaults_R"] += int(male_def.sum())
                male_loans = loan_amounts[male_mask]
                self.timestep_profit += float(
                    np.where(male_def, -male_loans, male_loans * self.interest_rate).sum()
                )

            if female_mask.any():
                np.add.at(self.current_X_female, ids[female_mask], kappa[female_mask])
                n_female = int(female_mask.sum())
                self.event_times_B.extend([self.current_time] * n_female)
                self.total_loans_B += n_female
                self.timestep_data["approvals_B"] += n_female
                self.timestep_data["apply_select_B"][ids[female_mask]] = 1
                female_def = defaults[female_mask]
                self.total_defaults_B += int(female_def.sum())
                self.timestep_data["defaults_B"] += int(female_def.sum())
                female_loans = loan_amounts[female_mask]
                self.timestep_profit += float(
                    np.where(female_def, -female_loans, female_loans * self.interest_rate).sum()
                )

            default_probs_out = default_probs
            loan_amounts_out = loan_amounts
        else:
            default_probs_out = np.zeros(0)
            loan_amounts_out = np.zeros(0)

        info = {
            "time": self.current_time,
            "applicants": applications,
            "actions": actions,
            "default_probs": default_probs_out,
            "loan_amounts": loan_amounts_out,
            "reward_snapshot": snap,
            "p_theta_R": self.timestep_data["p_theta_R"] if self.timestep_data else 0.0,
            "p_theta_B": self.timestep_data["p_theta_B"] if self.timestep_data else 0.0,
        }

        # Advance to the next (non-empty) cohort -- updates mu_R/mu_B/history
        # for the timestep just completed, then generates the next arrivals.
        self._advance_to_nonempty_cohort()

        terminated = (
            self.time_index >= len(self.time_steps) and len(self.pending_applications) == 0
        )
        truncated = False
        next_obs = self._get_cohort_observations()

        return next_obs, terminated, truncated, info
