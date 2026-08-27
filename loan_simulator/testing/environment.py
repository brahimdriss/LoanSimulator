import pickle
from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..environment import ARRIVAL_REF_N, MATTHEW_C
from ..reward import RewardSnapshot
from ..transition_learner import TransitionParameterLearner


class TestingIncomeEnvironment(gym.Env):
    """
    Testing environment that does NOT reset wealth between episodes.

    State (wealth, Hawkes events, cumulative counters) persists across episodes.
    Episodes are checkpoints for tracking progress.
    Tracks ground truth approvals (from creditworthiness) vs agent decisions.

    Default outcomes use a Bernoulli distribution (via TransitionParameterLearner).
    """

    def __init__(
        self,
        theta_params: TransitionParameterLearner,
        initial_wealth_male: np.ndarray,
        initial_wealth_female: np.ndarray,
        ground_truth_male: np.ndarray,
        ground_truth_female: np.ndarray,
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

        # Per-capita arrival scaling -- see environment.ARRIVAL_REF_N.
        self._arrival_scale_R = N_male / ARRIVAL_REF_N
        self._arrival_scale_B = N_female / ARRIVAL_REF_N

        # Prune Hawkes events older than this — exp(-beta * cutoff) < 1e-6
        self._hawkes_cutoff_R = -np.log(1e-6) / beta_R
        self._hawkes_cutoff_B = -np.log(1e-6) / beta_B

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

        self.ground_truth_male = ground_truth_male[:N_male].copy()
        self.ground_truth_female = ground_truth_female[:N_female].copy()

        # Individual default probability is derived from creditworthiness rank
        # (via X) -- see TransitionParameterLearner.initialize_individual_parameters.
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

        self.total_episodes = 0
        self.episode_metrics = self._init_episode_metrics()
        self._initialize_state()

    # ------------------------------------------------------------------
    # Episode metrics init
    # ------------------------------------------------------------------

    def _init_episode_metrics(self):
        """Initialize episode-level metrics dictionary."""
        return {
            "episode": [],
            "time_start": [],
            "time_end": [],
            "timesteps_in_episode": [],
            "mu_M_start": [],
            "mu_M_end": [],
            "mu_F_start": [],
            "mu_F_end": [],
            "delta_mu_M": [],
            "delta_mu_F": [],
            "rho_episode": [],
            "R_M": [],
            "R_F": [],
            "R_total": [],
            "wealth_gap": [],
            "loans_M_episode": [],
            "loans_F_episode": [],
            "applications_M_episode": [],
            "applications_F_episode": [],
            "defaults_M_episode": [],
            "defaults_F_episode": [],
            "profit_episode": [],
            "approval_rate_M_episode": [],
            "approval_rate_F_episode": [],
            "success_prob_M_episode": [],
            "success_prob_F_episode": [],
            "total_loans_M": [],
            "total_loans_F": [],
            "total_applications_M": [],
            "total_applications_F": [],
            "total_defaults_M": [],
            "total_defaults_F": [],
            "cumulative_profit": [],
            "approval_rate_M_cumulative": [],
            "approval_rate_F_cumulative": [],
            "success_prob_M_cumulative": [],
            "success_prob_F_cumulative": [],
            "actual_approvals_M_episode": [],
            "actual_approvals_F_episode": [],
            "true_approvals_M_episode": [],
            "true_approvals_F_episode": [],
            "true_positive_M": [],
            "false_positive_M": [],
            "true_negative_M": [],
            "false_negative_M": [],
            "true_positive_F": [],
            "false_positive_F": [],
            "true_negative_F": [],
            "false_negative_F": [],
            "accuracy_M": [],
            "accuracy_F": [],
            "precision_M": [],
            "precision_F": [],
            "recall_M": [],
            "recall_F": [],
            "hawkes_events_M": [],
            "hawkes_events_F": [],
            # Paper Sec 3.3 metric #5: Reach_g(t) = |U_g(t)| / N_g, the
            # fraction of UNIQUE individuals in group g who received a loan
            # during episode t. Distinct from approval rate: it measures
            # whether credit is spread broadly or concentrated on repeat
            # recipients.
            "reach_rate_M": [],
            "reach_rate_F": [],
            "unique_recipients_M": [],
            "unique_recipients_F": [],
        }

    # ------------------------------------------------------------------
    # State initialisation (called once at construction)
    # ------------------------------------------------------------------

    def _initialize_state(self):
        self.current_X_male = self.initial_X_male.copy()
        self.current_X_female = self.initial_X_female.copy()

        self.mu_M = np.mean(self.current_X_male)
        self.mu_F = np.mean(self.current_X_female)
        self.prev_mu_M = self.mu_M
        self.prev_mu_F = self.mu_F

        self.mu_M_0 = self.mu_M
        self.mu_F_0 = self.mu_F

        self.event_times_R = deque()
        self.event_times_B = deque()

        self.total_defaults_M = 0
        self.total_defaults_F = 0
        self.total_loans_M = 0
        self.total_loans_F = 0
        self.total_applications_M = 0
        self.total_applications_F = 0
        self.cumulative_profit = 0.0

        self.tp_M = self.fp_M = self.tn_M = self.fn_M = 0
        self.tp_F = self.fp_F = self.tn_F = self.fn_F = 0

        self.current_time = 0.0
        self.time_steps = np.arange(0, self.T, self.dt)
        self.time_index = 0
        self.global_timestep = 0

        self.pending_applications = deque()
        self.current_applicant = None
        self.timestep_data = None
        self.timestep_profit = 0.0

        self._reset_episode_counters()

        self.episode_start_mu_M = self.mu_M
        self.episode_start_mu_F = self.mu_F
        self.episode_start_time = 0.0
        self.episode_timesteps = 0

    def _reset_episode_counters(self):
        self.episode_loans_M = 0
        self.episode_loans_F = 0
        self.episode_applications_M = 0
        self.episode_applications_F = 0
        self.episode_defaults_M = 0
        self.episode_defaults_F = 0
        self.episode_profit = 0.0
        self.episode_actual_approvals_M = 0
        self.episode_actual_approvals_F = 0
        self.episode_true_approvals_M = 0
        self.episode_true_approvals_F = 0
        # Per-episode unique loan recipients, for Reach Rate (paper metric
        # #5). Boolean masks rather than Python sets: O(1) vectorised
        # marking from step_cohort()'s index arrays, and cheap at large N.
        self._ep_unique_M = np.zeros(self.N_male, dtype=bool)
        self._ep_unique_F = np.zeros(self.N_female, dtype=bool)
        # Cumulative per-individual loan counts, persisting across episodes
        # (concentration / Lorenz analysis).
        if not hasattr(self, "loan_counts_M"):
            self.loan_counts_M = np.zeros(self.N_male, dtype=np.int64)
            self.loan_counts_F = np.zeros(self.N_female, dtype=np.int64)

    # ------------------------------------------------------------------
    # reset() — continues from current state (no wealth reset)
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Continue to next episode — all state is preserved."""
        super().reset(seed=seed)

        if self.total_episodes > 0:
            self._record_episode_metrics()

        self.total_episodes += 1
        if self.total_episodes % 10 == 1 or self.total_episodes == 1:
            print(
                f"\n  Starting Episode {self.total_episodes} "
                f"(ALL state preserved - continuous play)"
            )

        self.prev_mu_M = self.mu_M
        self.prev_mu_F = self.mu_F
        self.episode_start_mu_M = self.mu_M
        self.episode_start_mu_F = self.mu_F
        self.episode_start_time = self.current_time
        self.episode_timesteps = 0

        self._reset_episode_counters()

        start_time = self.current_time
        self.time_steps = np.arange(start_time, start_time + self.T, self.dt)
        self.time_index = 0

        self.pending_applications = deque()
        self.current_applicant = None
        self.timestep_data = None
        self.timestep_profit = 0.0

        return self._get_observation(), {}

    def reset_cohort(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Batched interface: same continuity semantics as reset() (wealth,
        Hawkes events, cumulative counters all preserved), but generates
        the first timestep's cohort and returns its observation MATRIX
        (shape [n, 12]) instead of a single applicant's observation.
        """
        super().reset(seed=seed)

        if self.total_episodes > 0:
            self._record_episode_metrics()

        self.total_episodes += 1
        if self.total_episodes % 10 == 1 or self.total_episodes == 1:
            print(
                f"\n  Starting Episode {self.total_episodes} "
                f"(ALL state preserved - continuous play)"
            )

        self.prev_mu_M = self.mu_M
        self.prev_mu_F = self.mu_F
        self.episode_start_mu_M = self.mu_M
        self.episode_start_mu_F = self.mu_F
        self.episode_start_time = self.current_time
        self.episode_timesteps = 0

        self._reset_episode_counters()

        start_time = self.current_time
        self.time_steps = np.arange(start_time, start_time + self.T, self.dt)
        self.time_index = 0

        self.pending_applications = deque()
        self.current_applicant = None
        self.timestep_data = None
        self.timestep_profit = 0.0

        self._advance_to_nonempty_cohort()
        return self._get_cohort_observations(), {}

    # ------------------------------------------------------------------
    # Episode metrics recording
    # ------------------------------------------------------------------

    def _record_episode_metrics(self):
        m = self.episode_metrics

        m["episode"].append(self.total_episodes)
        m["time_start"].append(self.episode_start_time)
        m["time_end"].append(self.current_time)
        m["timesteps_in_episode"].append(self.episode_timesteps)

        m["mu_M_start"].append(self.episode_start_mu_M)
        m["mu_M_end"].append(self.mu_M)
        m["mu_F_start"].append(self.episode_start_mu_F)
        m["mu_F_end"].append(self.mu_F)

        delta_mu_M = self.mu_M - self.episode_start_mu_M
        delta_mu_F = self.mu_F - self.episode_start_mu_F
        m["delta_mu_M"].append(delta_mu_M)
        m["delta_mu_F"].append(delta_mu_F)

        if abs(delta_mu_F) > 1e-8:
            rho_episode = delta_mu_M / delta_mu_F
        else:
            rho_episode = (
                0.0 if abs(delta_mu_M) < 1e-8 else np.sign(delta_mu_M) * 100.0
            )
        m["rho_episode"].append(rho_episode)

        if self.current_time > 1e-8:
            R_M = (self.mu_M - self.mu_M_0) / self.current_time
            R_F = (self.mu_F - self.mu_F_0) / self.current_time
        else:
            R_M = R_F = 0.0
        m["R_M"].append(R_M)
        m["R_F"].append(R_F)
        m["R_total"].append((R_M + R_F) / 2)

        m["wealth_gap"].append(self.mu_M - self.mu_F)

        m["loans_M_episode"].append(self.episode_loans_M)
        m["loans_F_episode"].append(self.episode_loans_F)
        m["applications_M_episode"].append(self.episode_applications_M)
        m["applications_F_episode"].append(self.episode_applications_F)
        m["defaults_M_episode"].append(self.episode_defaults_M)
        m["defaults_F_episode"].append(self.episode_defaults_F)
        m["profit_episode"].append(self.episode_profit)

        m["approval_rate_M_episode"].append(
            self.episode_loans_M / max(self.episode_applications_M, 1)
        )
        m["approval_rate_F_episode"].append(
            self.episode_loans_F / max(self.episode_applications_F, 1)
        )
        m["success_prob_M_episode"].append(
            1.0 - self.episode_defaults_M / max(self.episode_loans_M, 1)
        )
        m["success_prob_F_episode"].append(
            1.0 - self.episode_defaults_F / max(self.episode_loans_F, 1)
        )

        m["total_loans_M"].append(self.total_loans_M)
        m["total_loans_F"].append(self.total_loans_F)
        m["total_applications_M"].append(self.total_applications_M)
        m["total_applications_F"].append(self.total_applications_F)
        m["total_defaults_M"].append(self.total_defaults_M)
        m["total_defaults_F"].append(self.total_defaults_F)
        m["cumulative_profit"].append(self.cumulative_profit)

        m["approval_rate_M_cumulative"].append(
            self.total_loans_M / max(self.total_applications_M, 1)
        )
        m["approval_rate_F_cumulative"].append(
            self.total_loans_F / max(self.total_applications_F, 1)
        )
        m["success_prob_M_cumulative"].append(
            1.0 - self.total_defaults_M / max(self.total_loans_M, 1)
        )
        m["success_prob_F_cumulative"].append(
            1.0 - self.total_defaults_F / max(self.total_loans_F, 1)
        )

        m["actual_approvals_M_episode"].append(self.episode_actual_approvals_M)
        m["actual_approvals_F_episode"].append(self.episode_actual_approvals_F)
        m["true_approvals_M_episode"].append(self.episode_true_approvals_M)
        m["true_approvals_F_episode"].append(self.episode_true_approvals_F)

        m["true_positive_M"].append(self.tp_M)
        m["false_positive_M"].append(self.fp_M)
        m["true_negative_M"].append(self.tn_M)
        m["false_negative_M"].append(self.fn_M)
        m["true_positive_F"].append(self.tp_F)
        m["false_positive_F"].append(self.fp_F)
        m["true_negative_F"].append(self.tn_F)
        m["false_negative_F"].append(self.fn_F)

        total_M = self.tp_M + self.fp_M + self.tn_M + self.fn_M
        total_F = self.tp_F + self.fp_F + self.tn_F + self.fn_F
        m["accuracy_M"].append((self.tp_M + self.tn_M) / max(total_M, 1))
        m["accuracy_F"].append((self.tp_F + self.tn_F) / max(total_F, 1))
        m["precision_M"].append(self.tp_M / max(self.tp_M + self.fp_M, 1))
        m["precision_F"].append(self.tp_F / max(self.tp_F + self.fp_F, 1))
        m["recall_M"].append(self.tp_M / max(self.tp_M + self.fn_M, 1))
        m["recall_F"].append(self.tp_F / max(self.tp_F + self.fn_F, 1))

        m["hawkes_events_M"].append(len(self.event_times_R))
        m["hawkes_events_F"].append(len(self.event_times_B))

        n_unique_M = int(self._ep_unique_M.sum())
        n_unique_F = int(self._ep_unique_F.sum())
        m["unique_recipients_M"].append(n_unique_M)
        m["unique_recipients_F"].append(n_unique_F)
        m["reach_rate_M"].append(n_unique_M / max(self.N_male, 1))
        m["reach_rate_F"].append(n_unique_F / max(self.N_female, 1))

    def finalize_episode_metrics(self):
        """Call after the last episode to record final metrics."""
        if self.total_episodes > 0:
            self._record_episode_metrics()

    # ------------------------------------------------------------------
    # Hawkes / intensity helpers
    # ------------------------------------------------------------------

    def _prune_hawkes_events(self, t: float) -> None:
        """Drop events whose excitation has decayed below 1e-6 (negligible).
        Events are appended in time order so pruning from the left is O(k)
        where k = number of expired events (usually 0 or very small).
        """
        cutoff_R = t - self._hawkes_cutoff_R
        cutoff_B = t - self._hawkes_cutoff_B
        while self.event_times_R and self.event_times_R[0] <= cutoff_R:
            self.event_times_R.popleft()
        while self.event_times_B and self.event_times_B[0] <= cutoff_B:
            self.event_times_B.popleft()

    def _f_networth_to_rate(self, mu: float, scale: float = 1.0) -> float:
        """Wealth-driven base arrival intensity on RELATIVE standing
        mu_g/mu_bar; see environment.MATTHEW_C and ARRIVAL_REF_N."""
        mu_bar = 0.5 * (self.mu_M + self.mu_F)
        rel = mu / max(mu_bar, 1e-8)
        return max(0.5, 2.0 * (1.0 + MATTHEW_C * (rel - 1.0))) * scale

    def _compute_lambda_R(self, t: float) -> float:
        base_rate = self._f_networth_to_rate(self.mu_M, self._arrival_scale_R)
        if not self.event_times_R:
            return base_rate
        # After pruning, all events in deque are <= t; ages are all >= 0
        ages = t - np.fromiter(self.event_times_R, dtype=np.float64, count=len(self.event_times_R))
        excitation = self.alpha_R * np.sum(np.exp(-self.beta_R * ages))
        return base_rate + excitation

    def _compute_lambda_B(self, t: float) -> float:
        base_rate = self._f_networth_to_rate(self.mu_F, self._arrival_scale_B)
        if not self.event_times_B:
            return base_rate
        ages = t - np.fromiter(self.event_times_B, dtype=np.float64, count=len(self.event_times_B))
        excitation = self.alpha_B * np.sum(np.exp(-self.beta_B * ages))
        return base_rate + excitation

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self):
        lambda_R = self._compute_lambda_R(self.current_time)
        lambda_B = self._compute_lambda_B(self.current_time)
        rho_M = self.total_defaults_M / max(self.total_loans_M, 1)
        rho_F = self.total_defaults_F / max(self.total_loans_F, 1)

        if self.current_applicant is None:
            return np.array(
                [
                    0.0,
                    0.5,
                    self.mu_M / 100.0,
                    self.mu_F / 100.0,
                    lambda_R,
                    lambda_B,
                    rho_M,
                    rho_F,
                    0.5,
                    self.theta_params.theta_S,
                    self.theta_params.theta_X,
                    0.0,
                ],
                dtype=np.float32,
            )

        theta_approval_prob = self.theta_params.compute_approval_probability(
            self.current_applicant["X"], self.current_applicant["S"]
        )
        return np.array(
            [
                self.current_applicant["X"] / 100.0,
                float(self.current_applicant["S"]),
                self.mu_M / 100.0,
                self.mu_F / 100.0,
                lambda_R,
                lambda_B,
                rho_M,
                rho_F,
                theta_approval_prob,
                self.theta_params.theta_S,
                self.theta_params.theta_X,
                self.current_applicant["loan_amount"],
            ],
            dtype=np.float32,
        )

    def _get_cohort_observations(self) -> np.ndarray:
        """Batched counterpart of _get_observation() -- see
        IncomeEnvironment._get_cohort_observations() for the column layout,
        identical here."""
        n = len(self.pending_applications)
        if n == 0:
            return np.zeros((0, 12), dtype=np.float32)

        apps = list(self.pending_applications)
        lambda_R = self._compute_lambda_R(self.current_time)
        lambda_B = self._compute_lambda_B(self.current_time)
        rho_M = self.total_defaults_M / max(self.total_loans_M, 1)
        rho_F = self.total_defaults_F / max(self.total_loans_F, 1)

        X = np.array([a["X"] for a in apps], dtype=np.float64)
        S = np.array([a["S"] for a in apps], dtype=np.float64)
        loan_amount = np.array([a["loan_amount"] for a in apps], dtype=np.float64)
        theta_approval_prob = np.array(
            [a["theta_approval_prob"] for a in apps], dtype=np.float64
        )

        obs = np.zeros((n, 12), dtype=np.float32)
        obs[:, 0] = X / 100.0
        obs[:, 1] = S
        obs[:, 2] = self.mu_M / 100.0
        obs[:, 3] = self.mu_F / 100.0
        obs[:, 4] = lambda_R
        obs[:, 5] = lambda_B
        obs[:, 6] = rho_M
        obs[:, 7] = rho_F
        obs[:, 8] = theta_approval_prob
        obs[:, 9] = self.theta_params.theta_S
        obs[:, 10] = self.theta_params.theta_X
        obs[:, 11] = loan_amount
        return obs

    def _advance_to_nonempty_cohort(self):
        """Advance timesteps until a non-empty cohort is pending or the
        episode ends -- see IncomeEnvironment's version for rationale."""
        while (
            len(self.pending_applications) == 0
            and self.time_index < len(self.time_steps)
        ):
            self._advance_one_timestep()

    def _advance_one_timestep(self):
        """Close out the just-finished timestep's bookkeeping (if any) and
        generate the next timestep's cohort into self.pending_applications."""
        if self.timestep_data is not None:
            self.mu_M = np.mean(self.current_X_male)
            self.mu_F = np.mean(self.current_X_female)
            self.cumulative_profit += self.timestep_profit
            self.episode_timesteps += 1

        if self.time_index < len(self.time_steps):
            self.current_time = self.time_steps[self.time_index]
            self.time_index += 1
            self.global_timestep += 1
            self.timestep_profit = 0.0

            self._prune_hawkes_events(self.current_time)
            timestep_info = self._generate_timestep_applications()
            self.pending_applications = deque(timestep_info["applications"])

            self.timestep_data = {
                "apply_select_M": timestep_info["apply_select_M"],
                "apply_select_F": timestep_info["apply_select_F"],
                "lambda_R": timestep_info["lambda_R"],
                "lambda_B": timestep_info["lambda_B"],
                "n_arrivals_M": timestep_info["n_arrivals_M"],
                "n_arrivals_F": timestep_info["n_arrivals_F"],
                "p_theta_R": timestep_info["p_theta_R"],
                "p_theta_B": timestep_info["p_theta_B"],
                "approvals_M": 0,
                "approvals_F": 0,
                "defaults_M": 0,
                "defaults_F": 0,
            }

    # ------------------------------------------------------------------
    # Application generation (vectorised)
    # ------------------------------------------------------------------

    def _generate_timestep_applications(self):
        t = self.current_time
        lambda_R = self._compute_lambda_R(t)
        lambda_B = self._compute_lambda_B(t)

        applications = []
        apply_select_M = np.zeros(self.N_male)
        apply_select_F = np.zeros(self.N_female)

        # ---- MALE ----
        base_rate_M = lambda_R * self.dt / self.N_male
        X_male = self.current_X_male
        X_norm_M = (X_male - self.theta_params.X_mean) / self.theta_params.X_std
        approval_probs_M = 1.0 / (
            1.0
            + np.exp(
                -(
                    self.theta_params.theta_S * 1
                    + self.theta_params.theta_X * X_norm_M
                    + self.theta_params.b
                )
            )
        )
        sensitivity_M = self.theta_params.application_sensitivity.get("male", {}).get(
            "sensitivity", 0.5
        )
        app_probs_M = np.clip(
            base_rate_M * (1.0 + sensitivity_M * approval_probs_M), 0.0, 1.0
        )

        applying_indices_M = np.where(np.random.random(self.N_male) < app_probs_M)[0]

        for idx in applying_indices_M:
            apply_select_M[idx] = 1
            applications.append(
                {
                    "group": "male",
                    "S": 1,
                    "individual_id": idx,
                    "X": X_male[idx],
                    "default_prob": self.theta_params.individual_default_probs["male"][idx],
                    "loan_amount": self.theta_params.individual_loan_amounts["male"][idx],
                    "wealth_gain": self.theta_params.individual_wealth_gains["male"][idx],
                    "theta_approval_prob": approval_probs_M[idx],
                    "ground_truth": self.ground_truth_male[idx],
                }
            )

        # ---- FEMALE ----
        base_rate_F = lambda_B * self.dt / self.N_female
        X_female = self.current_X_female
        X_norm_F = (X_female - self.theta_params.X_mean) / self.theta_params.X_std
        approval_probs_F = 1.0 / (
            1.0
            + np.exp(
                -(
                    self.theta_params.theta_S * 0
                    + self.theta_params.theta_X * X_norm_F
                    + self.theta_params.b
                )
            )
        )
        sensitivity_F = self.theta_params.application_sensitivity.get("female", {}).get(
            "sensitivity", 0.5
        )
        app_probs_F = np.clip(
            base_rate_F * (1.0 + sensitivity_F * approval_probs_F), 0.0, 1.0
        )

        applying_indices_F = np.where(np.random.random(self.N_female) < app_probs_F)[0]

        for idx in applying_indices_F:
            apply_select_F[idx] = 1
            applications.append(
                {
                    "group": "female",
                    "S": 0,
                    "individual_id": idx,
                    "X": X_female[idx],
                    "default_prob": self.theta_params.individual_default_probs["female"][idx],
                    "loan_amount": self.theta_params.individual_loan_amounts["female"][idx],
                    "wealth_gain": self.theta_params.individual_wealth_gains["female"][idx],
                    "theta_approval_prob": approval_probs_F[idx],
                    "ground_truth": self.ground_truth_female[idx],
                }
            )

        n_M = len(applying_indices_M)
        n_F = len(applying_indices_F)
        self.total_applications_M += n_M
        self.total_applications_F += n_F
        self.episode_applications_M += n_M
        self.episode_applications_F += n_F

        return {
            "applications": applications,
            "apply_select_M": apply_select_M,
            "apply_select_F": apply_select_F,
            "lambda_R": lambda_R,
            "lambda_B": lambda_B,
            "n_arrivals_M": n_M,
            "n_arrivals_F": n_F,
            "p_theta_R": float(app_probs_M.mean()),
            "p_theta_B": float(app_probs_F.mean()),
        }

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action):
        reward = 0.0

        if isinstance(action, np.ndarray):
            approval_prob = np.clip(action[0], 0.0, 1.0)
        else:
            approval_prob = np.clip(float(action), 0.0, 1.0)

        if self.current_applicant is not None:
            applicant = self.current_applicant
            agent_approved = np.random.random() < approval_prob
            ground_truth = applicant["ground_truth"]

            # Update confusion matrix
            if applicant["S"] == 1:
                if agent_approved and ground_truth:
                    self.tp_M += 1
                elif agent_approved and not ground_truth:
                    self.fp_M += 1
                elif not agent_approved and not ground_truth:
                    self.tn_M += 1
                else:
                    self.fn_M += 1
                if ground_truth:
                    self.episode_true_approvals_M += 1
            else:
                if agent_approved and ground_truth:
                    self.tp_F += 1
                elif agent_approved and not ground_truth:
                    self.fp_F += 1
                elif not agent_approved and not ground_truth:
                    self.tn_F += 1
                else:
                    self.fn_F += 1
                if ground_truth:
                    self.episode_true_approvals_F += 1

            if agent_approved:
                if applicant["S"] == 1:
                    self.event_times_R.append(self.current_time)
                    self.episode_actual_approvals_M += 1
                else:
                    self.event_times_B.append(self.current_time)
                    self.episode_actual_approvals_F += 1

                defaults = np.random.random() < applicant["default_prob"]
                # Borrower gains wealth_gain on success, nothing on default
                kappa_i = 0.0 if defaults else applicant["wealth_gain"]

                if applicant["S"] == 1:
                    self.current_X_male[applicant["individual_id"]] += kappa_i
                    self._ep_unique_M[applicant["individual_id"]] = True  # Reach Rate
                    self.loan_counts_M[applicant["individual_id"]] += 1
                    self.total_loans_M += 1
                    self.episode_loans_M += 1
                    self.timestep_data["approvals_M"] += 1
                    if defaults:
                        self.total_defaults_M += 1
                        self.episode_defaults_M += 1
                        self.timestep_data["defaults_M"] += 1
                        self.timestep_profit -= applicant["loan_amount"]
                        self.episode_profit -= applicant["loan_amount"]
                    else:
                        profit = applicant["loan_amount"] * self.interest_rate
                        self.timestep_profit += profit
                        self.episode_profit += profit
                else:
                    self.current_X_female[applicant["individual_id"]] += kappa_i
                    self._ep_unique_F[applicant["individual_id"]] = True  # Reach Rate
                    self.loan_counts_F[applicant["individual_id"]] += 1
                    self.total_loans_F += 1
                    self.episode_loans_F += 1
                    self.timestep_data["approvals_F"] += 1
                    if defaults:
                        self.total_defaults_F += 1
                        self.episode_defaults_F += 1
                        self.timestep_data["defaults_F"] += 1
                        self.timestep_profit -= applicant["loan_amount"]
                        self.episode_profit -= applicant["loan_amount"]
                    else:
                        profit = applicant["loan_amount"] * self.interest_rate
                        self.timestep_profit += profit
                        self.episode_profit += profit

        if self.pending_applications:
            self.current_applicant = self.pending_applications.popleft()
        else:
            self.current_applicant = None

        if self.current_applicant is None and self.time_index < len(self.time_steps):
            if self.timestep_data is not None:
                self.mu_M = np.mean(self.current_X_male)
                self.mu_F = np.mean(self.current_X_female)
                self.cumulative_profit += self.timestep_profit
                self.episode_timesteps += 1

            if self.time_index < len(self.time_steps):
                self.current_time = self.time_steps[self.time_index]
                self.time_index += 1
                self.global_timestep += 1
                self.timestep_profit = 0.0

                self._prune_hawkes_events(self.current_time)
                timestep_info = self._generate_timestep_applications()
                self.pending_applications = deque(timestep_info["applications"])

                self.timestep_data = {
                    "apply_select_M": timestep_info["apply_select_M"].copy(),
                    "apply_select_F": timestep_info["apply_select_F"].copy(),
                    "lambda_R": timestep_info["lambda_R"],
                    "lambda_B": timestep_info["lambda_B"],
                    "n_arrivals_M": timestep_info["n_arrivals_M"],
                    "n_arrivals_F": timestep_info["n_arrivals_F"],
                    "p_theta_R": timestep_info["p_theta_R"],
                    "p_theta_B": timestep_info["p_theta_B"],
                    "approvals_M": 0,
                    "approvals_F": 0,
                    "defaults_M": 0,
                    "defaults_F": 0,
                    "p_theta_R": timestep_info["p_theta_R"],
                    "p_theta_B": timestep_info["p_theta_B"],
                }

                if self.pending_applications:
                    self.current_applicant = self.pending_applications.popleft()

        terminated = (
            self.time_index >= len(self.time_steps) and self.current_applicant is None
        )
        truncated = False

        obs = self._get_observation()
        info = {
            "time": self.current_time,
            "mu_M": self.mu_M,
            "mu_F": self.mu_F,
            "mu_R": self.mu_M,
            "mu_B": self.mu_F,
            "wealth_gap": self.mu_M - self.mu_F,
            "applicant": self.current_applicant,
            "agent_approval_prob": approval_prob,
            "p_theta_R": self.timestep_data["p_theta_R"] if self.timestep_data else 0.0,
            "p_theta_B": self.timestep_data["p_theta_B"] if self.timestep_data else 0.0,
            "episode": self.total_episodes,
        }
        return obs, reward, terminated, truncated, info

    def step_cohort(self, actions: np.ndarray):
        """
        Batched counterpart of step() -- see IncomeEnvironment.step_cohort()
        for the general contract (snapshot-before-mutation, reward left to
        the caller). Additionally updates the confusion matrix (tp/fp/tn/fn
        per group) and episode-level counters, vectorized the same way.
        """
        applications = list(self.pending_applications)
        self.pending_applications = deque()  # consume now; see IncomeEnvironment
        n = len(applications)
        actions = np.clip(np.asarray(actions, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if len(actions) != n:
            raise ValueError(f"step_cohort: {len(actions)} actions for {n} applicants")

        snap = RewardSnapshot.from_env(self)

        if n > 0:
            default_probs = np.array([a["default_prob"] for a in applications])
            wealth_gains = np.array([a["wealth_gain"] for a in applications])
            loan_amounts = np.array([a["loan_amount"] for a in applications])
            S = np.array([a["S"] for a in applications])
            ids = np.array([a["individual_id"] for a in applications], dtype=np.int64)
            ground_truth = np.array(
                [bool(a["ground_truth"]) for a in applications], dtype=bool
            )

            approved = np.random.random(n) < actions
            defaults = np.random.random(n) < default_probs
            kappa = np.where(defaults, 0.0, wealth_gains)

            for group_S, is_male in [(1, True), (0, False)]:
                gmask = S == group_S
                if not gmask.any():
                    continue
                g_approved = approved[gmask]
                g_gt = ground_truth[gmask]

                tp = int((g_approved & g_gt).sum())
                fp = int((g_approved & ~g_gt).sum())
                tn = int((~g_approved & ~g_gt).sum())
                fn = int((~g_approved & g_gt).sum())
                true_approvals = int(g_gt.sum())

                approved_mask = gmask & approved
                n_approved = int(approved_mask.sum())
                g_ids = ids[approved_mask]
                g_kappa = kappa[approved_mask]
                g_defaults = defaults[approved_mask]
                g_loans = loan_amounts[approved_mask]
                profit = float(
                    np.where(g_defaults, -g_loans, g_loans * self.interest_rate).sum()
                )
                n_defaults = int(g_defaults.sum())

                if is_male:
                    self.tp_M += tp; self.fp_M += fp; self.tn_M += tn; self.fn_M += fn
                    self.episode_true_approvals_M += true_approvals
                    if n_approved > 0:
                        self._ep_unique_M[g_ids] = True          # Reach Rate
                        np.add.at(self.loan_counts_M, g_ids, 1)  # concentration
                        np.add.at(self.current_X_male, g_ids, g_kappa)
                        self.event_times_R.extend([self.current_time] * n_approved)
                        self.episode_actual_approvals_M += n_approved
                        self.total_loans_M += n_approved
                        self.episode_loans_M += n_approved
                        self.timestep_data["approvals_M"] += n_approved
                        self.total_defaults_M += n_defaults
                        self.episode_defaults_M += n_defaults
                        self.timestep_data["defaults_M"] += n_defaults
                        self.timestep_profit += profit
                        self.episode_profit += profit
                else:
                    self.tp_F += tp; self.fp_F += fp; self.tn_F += tn; self.fn_F += fn
                    self.episode_true_approvals_F += true_approvals
                    if n_approved > 0:
                        self._ep_unique_F[g_ids] = True          # Reach Rate
                        np.add.at(self.loan_counts_F, g_ids, 1)  # concentration
                        np.add.at(self.current_X_female, g_ids, g_kappa)
                        self.event_times_B.extend([self.current_time] * n_approved)
                        self.episode_actual_approvals_F += n_approved
                        self.total_loans_F += n_approved
                        self.episode_loans_F += n_approved
                        self.timestep_data["approvals_F"] += n_approved
                        self.total_defaults_F += n_defaults
                        self.episode_defaults_F += n_defaults
                        self.timestep_data["defaults_F"] += n_defaults
                        self.timestep_profit += profit
                        self.episode_profit += profit

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
            "episode": self.total_episodes,
        }

        self._advance_to_nonempty_cohort()

        terminated = (
            self.time_index >= len(self.time_steps) and len(self.pending_applications) == 0
        )
        truncated = False
        next_obs = self._get_cohort_observations()

        return next_obs, terminated, truncated, info

    # ------------------------------------------------------------------
    # DataFrame helper
    # ------------------------------------------------------------------

    def get_episode_metrics_dataframe(self):
        return pd.DataFrame(self.episode_metrics)

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, checkpoint_path: str):
        checkpoint = {
            "total_episodes": self.total_episodes,
            "global_timestep": self.global_timestep,
            "current_time": self.current_time,
            "current_X_male": self.current_X_male.copy(),
            "current_X_female": self.current_X_female.copy(),
            "mu_M": self.mu_M,
            "mu_F": self.mu_F,
            "prev_mu_M": self.prev_mu_M,
            "prev_mu_F": self.prev_mu_F,
            "mu_M_0": self.mu_M_0,
            "mu_F_0": self.mu_F_0,
            "episode_start_mu_M": self.episode_start_mu_M,
            "episode_start_mu_F": self.episode_start_mu_F,
            "episode_start_time": self.episode_start_time,
            "episode_timesteps": self.episode_timesteps,
            "episode_loans_M": self.episode_loans_M,
            "episode_loans_F": self.episode_loans_F,
            "episode_applications_M": self.episode_applications_M,
            "episode_applications_F": self.episode_applications_F,
            "episode_defaults_M": self.episode_defaults_M,
            "episode_defaults_F": self.episode_defaults_F,
            "episode_profit": self.episode_profit,
            "episode_actual_approvals_M": self.episode_actual_approvals_M,
            "episode_actual_approvals_F": self.episode_actual_approvals_F,
            "episode_true_approvals_M": self.episode_true_approvals_M,
            "episode_true_approvals_F": self.episode_true_approvals_F,
            "event_times_R": self.event_times_R.copy(),
            "event_times_B": self.event_times_B.copy(),
            "total_defaults_M": self.total_defaults_M,
            "total_defaults_F": self.total_defaults_F,
            "total_loans_M": self.total_loans_M,
            "total_loans_F": self.total_loans_F,
            "total_applications_M": self.total_applications_M,
            "total_applications_F": self.total_applications_F,
            "cumulative_profit": self.cumulative_profit,
            "tp_M": self.tp_M,
            "fp_M": self.fp_M,
            "tn_M": self.tn_M,
            "fn_M": self.fn_M,
            "tp_F": self.tp_F,
            "fp_F": self.fp_F,
            "tn_F": self.tn_F,
            "fn_F": self.fn_F,
            "episode_metrics": self.episode_metrics,
        }

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str):
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        self.total_episodes = checkpoint["total_episodes"]
        self.global_timestep = checkpoint["global_timestep"]
        self.current_time = checkpoint["current_time"]
        self.current_X_male = checkpoint["current_X_male"]
        self.current_X_female = checkpoint["current_X_female"]
        self.mu_M = checkpoint["mu_M"]
        self.mu_F = checkpoint["mu_F"]
        self.prev_mu_M = checkpoint["prev_mu_M"]
        self.prev_mu_F = checkpoint["prev_mu_F"]
        self.mu_M_0 = checkpoint["mu_M_0"]
        self.mu_F_0 = checkpoint["mu_F_0"]
        self.episode_start_mu_M = checkpoint.get("episode_start_mu_M", self.mu_M)
        self.episode_start_mu_F = checkpoint.get("episode_start_mu_F", self.mu_F)
        self.episode_start_time = checkpoint.get(
            "episode_start_time", self.current_time
        )
        self.episode_timesteps = checkpoint.get("episode_timesteps", 0)
        self.episode_loans_M = checkpoint.get("episode_loans_M", 0)
        self.episode_loans_F = checkpoint.get("episode_loans_F", 0)
        self.episode_applications_M = checkpoint.get("episode_applications_M", 0)
        self.episode_applications_F = checkpoint.get("episode_applications_F", 0)
        self.episode_defaults_M = checkpoint.get("episode_defaults_M", 0)
        self.episode_defaults_F = checkpoint.get("episode_defaults_F", 0)
        self.episode_profit = checkpoint.get("episode_profit", 0.0)
        self.episode_actual_approvals_M = checkpoint.get("episode_actual_approvals_M", 0)
        self.episode_actual_approvals_F = checkpoint.get("episode_actual_approvals_F", 0)
        self.episode_true_approvals_M = checkpoint.get("episode_true_approvals_M", 0)
        self.episode_true_approvals_F = checkpoint.get("episode_true_approvals_F", 0)
        self.event_times_R = deque(checkpoint["event_times_R"])
        self.event_times_B = deque(checkpoint["event_times_B"])
        self.total_defaults_M = checkpoint["total_defaults_M"]
        self.total_defaults_F = checkpoint["total_defaults_F"]
        self.total_loans_M = checkpoint["total_loans_M"]
        self.total_loans_F = checkpoint["total_loans_F"]
        self.total_applications_M = checkpoint["total_applications_M"]
        self.total_applications_F = checkpoint["total_applications_F"]
        self.cumulative_profit = checkpoint["cumulative_profit"]
        self.tp_M = checkpoint["tp_M"]
        self.fp_M = checkpoint["fp_M"]
        self.tn_M = checkpoint["tn_M"]
        self.fn_M = checkpoint["fn_M"]
        self.tp_F = checkpoint["tp_F"]
        self.fp_F = checkpoint["fp_F"]
        self.tn_F = checkpoint["tn_F"]
        self.fn_F = checkpoint["fn_F"]
        self.episode_metrics = checkpoint.get(
            "episode_metrics", self._init_episode_metrics()
        )

        print(
            f"  Loaded checkpoint: {self.total_episodes} episodes completed, "
            f"time={self.current_time:.1f}"
        )
        return checkpoint

    # ------------------------------------------------------------------
    # Compatibility aliases for PolicyGradientAgent (uses R/B naming)
    # ------------------------------------------------------------------
    @property
    def mu_R(self):                 return self.mu_M
    @mu_R.setter
    def mu_R(self, value):          self.mu_M = value
    @property
    def mu_B(self):                 return self.mu_F
    @mu_B.setter
    def mu_B(self, value):          self.mu_F = value
    @property
    def total_loans_R(self):        return self.total_loans_M
    @property
    def total_loans_B(self):        return self.total_loans_F
    @property
    def total_defaults_R(self):     return self.total_defaults_M
    @property
    def total_defaults_B(self):     return self.total_defaults_F
    @property
    def total_applications_R(self): return self.total_applications_M
    @property
    def total_applications_B(self): return self.total_applications_F
    @property
    def tp_R(self):                 return self.tp_M
    @property
    def fn_R(self):                 return self.fn_M
    @property
    def tp_B(self):                 return self.tp_F
    @property
    def fn_B(self):                 return self.fn_F
    @property
    def history(self):
        # agent uses sum(env.history["profit"]) — episode_profit is that sum
        return {"profit": [self.episode_profit]}
