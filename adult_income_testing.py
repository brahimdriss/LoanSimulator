"""
Testing Script for Trained Policy Gradient Agent

This script loads a trained agent and evaluates it on the Income Environment
WITHOUT resetting between episodes (continuous play). No training occurs.

Metrics Tracked (ALL EPISODE-WISE):
- Inequality ratio (ρ) = (μ_M,end - μ_M,start) / (μ_F,end - μ_F,start) per episode
- Approval rates for each group per episode
- Mean wealth of each group per episode
- Total loans approved per group per episode
- Success probabilities per group per episode
- Actual loan approvals vs True loan approvals (ground truth based on creditworthiness)
"""

import glob
import json
import os
import pickle
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.distributions import Beta

warnings.filterwarnings("ignore")


def list_checkpoints(checkpoint_dir="./checkpoints"):
    """List all available checkpoints and their status."""
    if not os.path.exists(checkpoint_dir):
        print(f"No checkpoint directory found at {checkpoint_dir}")
        return []

    checkpoints = glob.glob(f"{checkpoint_dir}/checkpoint_*.pkl")

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return []

    print(f"\nAvailable checkpoints in {checkpoint_dir}:")
    print("-" * 70)

    checkpoint_info = []
    for cp_path in sorted(checkpoints):
        try:
            with open(cp_path, "rb") as f:
                cp = pickle.load(f)
            info = {
                "path": cp_path,
                "episodes": cp["total_episodes"],
                "time": cp["current_time"],
                "mu_M": cp["mu_M"],
                "mu_F": cp["mu_F"],
            }
            checkpoint_info.append(info)
            print(f"  {os.path.basename(cp_path)}")
            print(f"    Episodes: {info['episodes']}, Time: {info['time']:.1f}")
            print(f"    μ_M: ${info['mu_M']:.2f}k, μ_F: ${info['mu_F']:.2f}k")
        except Exception as e:
            print(f"  {os.path.basename(cp_path)} - ERROR: {e}")

    print("-" * 70)
    return checkpoint_info


# ============================================================================
# DATA LOADER (same as training script)
# ============================================================================


class AdultIncomeDataLoader:
    """Load and preprocess Adult Income dataset."""

    def __init__(self, filepath: str = None, sample_size: int = 50000):
        self.filepath = filepath
        self.sample_size = sample_size
        self.data = None
        self.male_data = None
        self.female_data = None
        self.creditworthiness_model = None
        self.scaler = None
        self.credit_threshold = 0.5  # Threshold for ground truth approval

    def load_data(self):
        """Load Adult Income data."""
        if self.filepath is None:
            print("Downloading Adult Income dataset...")
            self.data = self._download_adult_income_data()
        else:
            print(f"Loading Adult Income data from {self.filepath}...")
            try:
                self.data = pd.read_csv(
                    self.filepath, skipinitialspace=True, na_values="?"
                )
                self.data = self.data.dropna()

                column_mapping = {
                    "educational-num": "education_num",
                    "marital-status": "marital_status",
                    "capital-gain": "capital_gain",
                    "capital-loss": "capital_loss",
                    "hours-per-week": "hours_per_week",
                    "native-country": "native_country",
                    "gender": "sex",
                }
                self.data = self.data.rename(columns=column_mapping)

                if len(self.data) > self.sample_size:
                    self.data = self.data.sample(n=self.sample_size, random_state=42)

                print(f"✓ Loaded {len(self.data):,} records from {self.filepath}")

            except Exception as e:
                print(f"Error loading file: {e}")
                self.data = self._generate_synthetic_adult_data()

        return self.data

    def _download_adult_income_data(self):
        """Download or generate synthetic data."""
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            columns = [
                "age",
                "workclass",
                "fnlwgt",
                "education",
                "education_num",
                "marital_status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "capital_gain",
                "capital_loss",
                "hours_per_week",
                "native_country",
                "income",
            ]
            df = pd.read_csv(url, names=columns, skipinitialspace=True, na_values="?")
            df = df.dropna()
            if len(df) > self.sample_size:
                df = df.sample(n=self.sample_size, random_state=42)
            return df
        except:
            return self._generate_synthetic_adult_data()

    def _generate_synthetic_adult_data(self):
        """Generate synthetic data as fallback."""
        np.random.seed(42)
        n = self.sample_size
        sex = np.random.choice(["Male", "Female"], size=n, p=[0.67, 0.33])

        data = []
        for i in range(n):
            is_male = sex[i] == "Male"
            age = np.clip(np.random.normal(38 if is_male else 36, 13), 17, 90)
            education_num = np.clip(
                np.random.normal(10.2 if is_male else 10.0, 2.5), 1, 16
            )
            hours_per_week = np.clip(np.random.normal(42 if is_male else 37, 12), 1, 99)
            capital_gain = (
                np.random.exponential(15000 if is_male else 12000)
                if np.random.random() < 0.08
                else 0
            )
            capital_loss = (
                np.random.exponential(1500) if np.random.random() < 0.04 else 0
            )

            base_prob = 0.15 + (0.12 if is_male else 0)
            base_prob += 0.02 * max(0, (education_num - 10))
            base_prob = np.clip(base_prob, 0.05, 0.85)
            income = ">50K" if np.random.random() < base_prob else "<=50K"

            data.append(
                {
                    "age": int(age),
                    "workclass": "Private",
                    "fnlwgt": np.random.randint(10000, 1000000),
                    "education": "HS-grad",
                    "education_num": int(education_num),
                    "marital_status": "Married",
                    "occupation": "Other",
                    "relationship": "Other",
                    "race": "White",
                    "sex": sex[i],
                    "capital_gain": int(capital_gain),
                    "capital_loss": int(capital_loss),
                    "hours_per_week": int(hours_per_week),
                    "native_country": "United-States",
                    "income": income,
                }
            )

        return pd.DataFrame(data)

    def _derive_creditworthiness_optimal(self, df):
        """Derive creditworthiness using logistic regression - ALSO DETERMINES GROUND TRUTH."""
        print(
            "\n  Deriving creditworthiness (ground truth) using income prediction model..."
        )

        feature_cols = [
            "age",
            "education_num",
            "hours_per_week",
            "capital_gain",
            "capital_loss",
        ]
        X_features = df[feature_cols].copy()
        X_features["capital_gain_log"] = np.log1p(X_features["capital_gain"])
        X_features["capital_loss_log"] = np.log1p(X_features["capital_loss"])
        X_features.drop(["capital_gain", "capital_loss"], axis=1, inplace=True)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_features)

        y = (df["income"].str.strip() == ">50K").astype(int)

        self.creditworthiness_model = LogisticRegression(random_state=42, max_iter=1000)
        self.creditworthiness_model.fit(X_scaled, y)

        creditworthiness_probs = self.creditworthiness_model.predict_proba(X_scaled)[
            :, 1
        ]

        return creditworthiness_probs

    def preprocess(self):
        """Preprocess data and compute ground truth approvals."""
        print("Preprocessing Adult Income data...")

        df = self.data.copy()
        df["S"] = (df["sex"].str.strip() == "Male").astype(int)
        df["approved"] = (df["income"].str.strip() == ">50K").astype(int)
        df["creditworthiness"] = self._derive_creditworthiness_optimal(df)
        df["X"] = 10 + df["creditworthiness"] * 190

        # GROUND TRUTH: Should this person get a loan based on creditworthiness?
        # Using threshold on creditworthiness score
        df["ground_truth_approval"] = (
            df["creditworthiness"] >= self.credit_threshold
        ).astype(int)

        self.male_data = df[df["S"] == 1].reset_index(drop=True)
        self.female_data = df[df["S"] == 0].reset_index(drop=True)

        print(f"\nMale applicants: {len(self.male_data)}")
        print(f"Female applicants: {len(self.female_data)}")
        print(
            f"Male ground truth approval rate: {self.male_data['ground_truth_approval'].mean():.3f}"
        )
        print(
            f"Female ground truth approval rate: {self.female_data['ground_truth_approval'].mean():.3f}"
        )

        self.data = df
        return df


# ============================================================================
# TRANSITION PARAMETER LEARNER (same as training script)
# ============================================================================


class TransitionParameterLearner:
    """Learn transition parameters θ."""

    def __init__(
        self,
        default_rate_min: float = 0.05,
        default_rate_max: float = 0.25,
        loan_amount_mean: float = 30.0,
        loan_amount_std: float = 10.0,
        investment_return_rate: float = 0.35,
        interest_rate: float = 0.15,
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
        for group_name, S_val in [("male", 1), ("female", 0)]:
            group_data = data[data["S"] == S_val]
            base_rate = len(group_data) / len(data)
            self.application_sensitivity[group_name] = {
                "base_rate": base_rate,
                "sensitivity": 0.5,
            }

    def learn_wealth_gain_parameters(self, data: pd.DataFrame):
        for group_name, S_val in [("male", 1), ("female", 0)]:
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
            self.kappa_std[group_name] = {"success": kappa_success_std, "default": 0.0}

    def learn_default_model(self, data: pd.DataFrame):
        pass

    def learn_variance_parameters(self, data: pd.DataFrame):
        self.sigma = data["X"].std()

    def initialize_individual_parameters(
        self, N_male: int, N_female: int, seed: int = None
    ):
        if seed is not None:
            np.random.seed(seed)

        for group, N in [("male", N_male), ("female", N_female)]:
            self.individual_default_probs[group] = np.random.uniform(
                self.default_rate_min, self.default_rate_max, N
            )
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
        self.learn_approval_model(data)
        self.learn_application_behavior(data)
        self.learn_wealth_gain_parameters(data)
        self.learn_default_model(data)
        self.learn_variance_parameters(data)

    def compute_approval_score(self, X: float, S: int) -> float:
        X_norm = (X - self.X_mean) / self.X_std
        return self.theta_S * S + self.theta_X * X_norm + self.b

    def compute_approval_probability(self, X: float, S: int) -> float:
        z = self.compute_approval_score(X, S)
        return 1.0 / (1.0 + np.exp(-z))

    def compute_application_probability(
        self, X: float, group: str, base_rate: float
    ) -> float:
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


# ============================================================================
# TESTING ENVIRONMENT (NO RESET BETWEEN EPISODES - EPISODE-WISE METRICS)
# ============================================================================


class TestingIncomeEnvironment(gym.Env):
    """
    Testing environment that does NOT reset between episodes.
    Tracks ground truth approvals for comparison with agent decisions.
    ALL METRICS ARE TRACKED PER-EPISODE.
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

        # Initial wealth
        self.initial_X_male = initial_wealth_male[:N_male].copy()
        self.initial_X_female = initial_wealth_female[:N_female].copy()

        # Ground truth approvals (based on creditworthiness)
        self.ground_truth_male = ground_truth_male[:N_male].copy()
        self.ground_truth_female = ground_truth_female[:N_female].copy()

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

        # Episode tracking (persists across episodes)
        self.total_episodes = 0
        self.episode_metrics = self._init_episode_metrics()

        self._initialize_state()

    def _init_episode_metrics(self):
        """Initialize episode-level metrics dictionary."""
        return {
            "episode": [],
            "time_start": [],
            "time_end": [],
            "timesteps_in_episode": [],
            # Wealth at episode boundaries
            "mu_M_start": [],
            "mu_M_end": [],
            "mu_F_start": [],
            "mu_F_end": [],
            # Wealth changes per episode
            "delta_mu_M": [],
            "delta_mu_F": [],
            # Inequality ratio per episode: ρ = Δμ_M / Δμ_F
            "rho_episode": [],
            # Long-term social welfare at episode end: R_g(t) = (μ_g,t - μ_g,0) / t
            "R_M": [],
            "R_F": [],
            "R_total": [],
            # Wealth gap at episode end
            "wealth_gap": [],
            # Episode-specific counts (just this episode)
            "loans_M_episode": [],
            "loans_F_episode": [],
            "applications_M_episode": [],
            "applications_F_episode": [],
            "defaults_M_episode": [],
            "defaults_F_episode": [],
            "profit_episode": [],
            # Approval rates for this episode
            "approval_rate_M_episode": [],
            "approval_rate_F_episode": [],
            # Success probabilities for this episode
            "success_prob_M_episode": [],
            "success_prob_F_episode": [],
            # Cumulative totals (up to and including this episode)
            "total_loans_M": [],
            "total_loans_F": [],
            "total_applications_M": [],
            "total_applications_F": [],
            "total_defaults_M": [],
            "total_defaults_F": [],
            "cumulative_profit": [],
            # Cumulative approval rates
            "approval_rate_M_cumulative": [],
            "approval_rate_F_cumulative": [],
            # Cumulative success probabilities
            "success_prob_M_cumulative": [],
            "success_prob_F_cumulative": [],
            # Actual vs True approvals (episode)
            "actual_approvals_M_episode": [],
            "actual_approvals_F_episode": [],
            "true_approvals_M_episode": [],
            "true_approvals_F_episode": [],
            # Confusion matrix (cumulative up to this episode)
            "true_positive_M": [],
            "false_positive_M": [],
            "true_negative_M": [],
            "false_negative_M": [],
            "true_positive_F": [],
            "false_positive_F": [],
            "true_negative_F": [],
            "false_negative_F": [],
            # Accuracy metrics (cumulative)
            "accuracy_M": [],
            "accuracy_F": [],
            "precision_M": [],
            "precision_F": [],
            "recall_M": [],
            "recall_F": [],
            # Hawkes process state (total events so far)
            "hawkes_events_M": [],
            "hawkes_events_F": [],
        }

    def _initialize_state(self):
        """Initialize state at the very beginning (only called once)."""
        # Wealth state
        self.current_X_male = self.initial_X_male.copy()
        self.current_X_female = self.initial_X_female.copy()

        self.mu_M = np.mean(self.current_X_male)
        self.mu_F = np.mean(self.current_X_female)
        self.prev_mu_M = self.mu_M
        self.prev_mu_F = self.mu_F

        # Store initial wealth for long-term social welfare calculation
        self.mu_M_0 = self.mu_M  # μ_M,0
        self.mu_F_0 = self.mu_F  # μ_F,0

        # Hawkes process event times (persist across episodes)
        self.event_times_R = []
        self.event_times_B = []

        # Cumulative counters (persist across episodes)
        self.total_defaults_M = 0
        self.total_defaults_F = 0
        self.total_loans_M = 0
        self.total_loans_F = 0
        self.total_applications_M = 0
        self.total_applications_F = 0
        self.cumulative_profit = 0.0

        # Confusion matrix counters (persist across episodes)
        self.tp_M = 0  # True positive male
        self.fp_M = 0  # False positive male
        self.tn_M = 0  # True negative male
        self.fn_M = 0  # False negative male
        self.tp_F = 0
        self.fp_F = 0
        self.tn_F = 0
        self.fn_F = 0

        # Time state (continues across episodes)
        self.current_time = 0.0
        self.time_steps = np.arange(0, self.T, self.dt)
        self.time_index = 0
        self.global_timestep = 0

        # Application state
        self.pending_applications = []
        self.current_applicant = None
        self.timestep_data = None
        self.timestep_profit = 0.0

        # Episode-specific counters (reset each episode)
        self._reset_episode_counters()

        # Episode start tracking (for ρ calculation)
        self.episode_start_mu_M = self.mu_M
        self.episode_start_mu_F = self.mu_F
        self.episode_start_time = 0.0
        self.episode_timesteps = 0

    def _reset_episode_counters(self):
        """Reset counters that track within-episode stats."""
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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Continue to next episode - EVERYTHING persists (no resets).
        Episodes are just checkpoints/markers for tracking progress.
        """
        super().reset(seed=seed)

        # Record episode-end metrics for the PREVIOUS episode (if not first)
        if self.total_episodes > 0:
            self._record_episode_metrics()

        self.total_episodes += 1
        # Only print every 10 episodes or first episode
        if self.total_episodes % 10 == 1 or self.total_episodes == 1:
            print(
                f"\n  Starting Episode {self.total_episodes} (ALL state preserved - continuous play)"
            )

        # Store previous mean wealth for tracking
        self.prev_mu_M = self.mu_M
        self.prev_mu_F = self.mu_F

        # Store episode START wealth for episode-level ρ calculation
        self.episode_start_mu_M = self.mu_M
        self.episode_start_mu_F = self.mu_F
        self.episode_start_time = self.current_time
        self.episode_timesteps = 0

        # Reset episode-specific counters
        self._reset_episode_counters()

        # TIME CONTINUES - does NOT reset to 0
        # Just set up the next time window
        start_time = self.current_time  # Continue from where we left off
        self.time_steps = np.arange(start_time, start_time + self.T, self.dt)
        self.time_index = 0

        # HAWKES EVENTS PERSIST - do NOT clear
        # self.event_times_R and self.event_times_B remain as is

        # Only clear pending applications (natural - episode ended mid-timestep)
        self.pending_applications = []
        self.current_applicant = None
        self.timestep_data = None
        self.timestep_profit = 0.0

        obs = self._get_observation()
        return obs, {}

    def _record_episode_metrics(self):
        """Record all metrics at the end of an episode."""
        m = self.episode_metrics

        m["episode"].append(self.total_episodes)
        m["time_start"].append(self.episode_start_time)
        m["time_end"].append(self.current_time)
        m["timesteps_in_episode"].append(self.episode_timesteps)

        # Wealth at episode boundaries
        m["mu_M_start"].append(self.episode_start_mu_M)
        m["mu_M_end"].append(self.mu_M)
        m["mu_F_start"].append(self.episode_start_mu_F)
        m["mu_F_end"].append(self.mu_F)

        # Wealth changes this episode
        delta_mu_M = self.mu_M - self.episode_start_mu_M
        delta_mu_F = self.mu_F - self.episode_start_mu_F

        m["delta_mu_M"].append(delta_mu_M)
        m["delta_mu_F"].append(delta_mu_F)

        # Inequality ratio for this episode: ρ = Δμ_M / Δμ_F
        if abs(delta_mu_F) > 1e-8:
            rho_episode = delta_mu_M / delta_mu_F
        else:
            rho_episode = 0.0 if abs(delta_mu_M) < 1e-8 else np.sign(delta_mu_M) * 100.0

        m["rho_episode"].append(rho_episode)

        # Long-term social welfare at episode end: R_g(t) = (μ_g,t - μ_g,0) / t
        if self.current_time > 1e-8:
            R_M = (self.mu_M - self.mu_M_0) / self.current_time
            R_F = (self.mu_F - self.mu_F_0) / self.current_time
        else:
            R_M = R_F = 0.0

        m["R_M"].append(R_M)
        m["R_F"].append(R_F)
        m["R_total"].append((R_M + R_F) / 2)

        # Wealth gap
        m["wealth_gap"].append(self.mu_M - self.mu_F)

        # Episode-specific counts
        m["loans_M_episode"].append(self.episode_loans_M)
        m["loans_F_episode"].append(self.episode_loans_F)
        m["applications_M_episode"].append(self.episode_applications_M)
        m["applications_F_episode"].append(self.episode_applications_F)
        m["defaults_M_episode"].append(self.episode_defaults_M)
        m["defaults_F_episode"].append(self.episode_defaults_F)
        m["profit_episode"].append(self.episode_profit)

        # Episode approval rates
        m["approval_rate_M_episode"].append(
            self.episode_loans_M / max(self.episode_applications_M, 1)
        )
        m["approval_rate_F_episode"].append(
            self.episode_loans_F / max(self.episode_applications_F, 1)
        )

        # Episode success probabilities
        m["success_prob_M_episode"].append(
            1.0 - self.episode_defaults_M / max(self.episode_loans_M, 1)
        )
        m["success_prob_F_episode"].append(
            1.0 - self.episode_defaults_F / max(self.episode_loans_F, 1)
        )

        # Cumulative totals
        m["total_loans_M"].append(self.total_loans_M)
        m["total_loans_F"].append(self.total_loans_F)
        m["total_applications_M"].append(self.total_applications_M)
        m["total_applications_F"].append(self.total_applications_F)
        m["total_defaults_M"].append(self.total_defaults_M)
        m["total_defaults_F"].append(self.total_defaults_F)
        m["cumulative_profit"].append(self.cumulative_profit)

        # Cumulative approval rates
        m["approval_rate_M_cumulative"].append(
            self.total_loans_M / max(self.total_applications_M, 1)
        )
        m["approval_rate_F_cumulative"].append(
            self.total_loans_F / max(self.total_applications_F, 1)
        )

        # Cumulative success probabilities
        m["success_prob_M_cumulative"].append(
            1.0 - self.total_defaults_M / max(self.total_loans_M, 1)
        )
        m["success_prob_F_cumulative"].append(
            1.0 - self.total_defaults_F / max(self.total_loans_F, 1)
        )

        # Actual vs True approvals (episode)
        m["actual_approvals_M_episode"].append(self.episode_actual_approvals_M)
        m["actual_approvals_F_episode"].append(self.episode_actual_approvals_F)
        m["true_approvals_M_episode"].append(self.episode_true_approvals_M)
        m["true_approvals_F_episode"].append(self.episode_true_approvals_F)

        # Confusion matrix (cumulative)
        m["true_positive_M"].append(self.tp_M)
        m["false_positive_M"].append(self.fp_M)
        m["true_negative_M"].append(self.tn_M)
        m["false_negative_M"].append(self.fn_M)

        m["true_positive_F"].append(self.tp_F)
        m["false_positive_F"].append(self.fp_F)
        m["true_negative_F"].append(self.tn_F)
        m["false_negative_F"].append(self.fn_F)

        # Accuracy metrics (cumulative)
        total_M = self.tp_M + self.fp_M + self.tn_M + self.fn_M
        total_F = self.tp_F + self.fp_F + self.tn_F + self.fn_F

        m["accuracy_M"].append((self.tp_M + self.tn_M) / max(total_M, 1))
        m["accuracy_F"].append((self.tp_F + self.tn_F) / max(total_F, 1))
        m["precision_M"].append(self.tp_M / max(self.tp_M + self.fp_M, 1))
        m["precision_F"].append(self.tp_F / max(self.tp_F + self.fp_F, 1))
        m["recall_M"].append(self.tp_M / max(self.tp_M + self.fn_M, 1))
        m["recall_F"].append(self.tp_F / max(self.tp_F + self.fn_F, 1))

        # Hawkes process state
        m["hawkes_events_M"].append(len(self.event_times_R))
        m["hawkes_events_F"].append(len(self.event_times_B))

    def finalize_episode_metrics(self):
        """Call this after the last episode to record final metrics."""
        if self.total_episodes > 0:
            self._record_episode_metrics()

    def _f_networth_to_rate(self, mu: float) -> float:
        mu = mu * 5.0
        return max(0.5, 2.0 + 0.01 * mu)

    def _phi_R(self, t: float) -> float:
        return self.alpha_R * np.exp(-self.beta_R * t)

    def _phi_B(self, t: float) -> float:
        return self.alpha_B * np.exp(-self.beta_B * t)

    def _compute_lambda_R(self, t: float) -> float:
        """Hawkes intensity for male group with vectorized computation."""
        base_rate = self._f_networth_to_rate(self.mu_M)
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
        base_rate = self._f_networth_to_rate(self.mu_F)
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
        rho_M = self.total_defaults_M / max(self.total_loans_M, 1)
        rho_F = self.total_defaults_F / max(self.total_loans_F, 1)

        if self.current_applicant is None:
            mu_M_norm = self.mu_M / 100.0
            mu_F_norm = self.mu_F / 100.0
            return np.array(
                [
                    0.0,
                    0.5,
                    mu_M_norm,
                    mu_F_norm,
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
        else:
            theta_approval_prob = self.theta_params.compute_approval_probability(
                self.current_applicant["X"], self.current_applicant["S"]
            )
            X_norm = self.current_applicant["X"] / 100.0
            mu_M_norm = self.mu_M / 100.0
            mu_F_norm = self.mu_F / 100.0
            loan_amount = self.current_applicant["loan_amount"]

            return np.array(
                [
                    X_norm,
                    float(self.current_applicant["S"]),
                    mu_M_norm,
                    mu_F_norm,
                    lambda_R,
                    lambda_B,
                    rho_M,
                    rho_F,
                    theta_approval_prob,
                    self.theta_params.theta_S,
                    self.theta_params.theta_X,
                    loan_amount,
                ],
                dtype=np.float32,
            )

    def _generate_timestep_applications(self):
        """Vectorized application generation individual loops."""
        t = self.current_time
        lambda_R = self._compute_lambda_R(t)
        lambda_B = self._compute_lambda_B(t)

        applications = []
        apply_select_M = np.zeros(self.N_male)
        apply_select_F = np.zeros(self.N_female)

        # ---- MALE APPLICATIONS (vectorized) ----
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

        random_vals_M = np.random.random(self.N_male)
        applying_indices_M = np.where(random_vals_M < app_probs_M)[0]
        if len(applying_indices_M) > 10:
            final_num = 10 + np.int32(
                np.ceil(np.random.random(1)[0] * 0.0 * (len(applying_indices_M) - 10))
            )
            # print(final_num)
            applying_indices_M = applying_indices_M[:final_num]

        for idx in applying_indices_M:
            apply_select_M[idx] = 1
            applications.append(
                {
                    "group": "male",
                    "S": 1,
                    "individual_id": idx,
                    "X": X_male[idx],
                    "default_prob": self.theta_params.individual_default_probs["male"][
                        idx
                    ],
                    "loan_amount": self.theta_params.individual_loan_amounts["male"][
                        idx
                    ],
                    "wealth_gain": self.theta_params.individual_wealth_gains["male"][
                        idx
                    ],
                    "theta_approval_prob": approval_probs_M[idx],
                    "ground_truth": self.ground_truth_male[idx],
                }
            )

        # ---- FEMALE APPLICATIONS (vectorized) ----
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

        random_vals_F = np.random.random(self.N_female)
        applying_indices_F = np.where(random_vals_F < app_probs_F)[0]
        if len(applying_indices_F) > 10:
            final_num = 10 + np.int32(
                np.ceil(np.random.random(1)[0] * 0.0 * (len(applying_indices_F) - 10))
            )
            applying_indices_F = applying_indices_F[:final_num]

        for idx in applying_indices_F:
            apply_select_F[idx] = 1
            applications.append(
                {
                    "group": "female",
                    "S": 0,
                    "individual_id": idx,
                    "X": X_female[idx],
                    "default_prob": self.theta_params.individual_default_probs[
                        "female"
                    ][idx],
                    "loan_amount": self.theta_params.individual_loan_amounts["female"][
                        idx
                    ],
                    "wealth_gain": self.theta_params.individual_wealth_gains["female"][
                        idx
                    ],
                    "theta_approval_prob": approval_probs_F[idx],
                    "ground_truth": self.ground_truth_female[idx],
                }
            )

        n_arrivals_M = len(applying_indices_M)
        n_arrivals_F = len(applying_indices_F)
        self.total_applications_M += n_arrivals_M
        self.total_applications_F += n_arrivals_F
        self.episode_applications_M += n_arrivals_M
        self.episode_applications_F += n_arrivals_F

        return {
            "applications": applications,
            "apply_select_M": apply_select_M,
            "apply_select_F": apply_select_F,
            "lambda_R": lambda_R,
            "lambda_B": lambda_B,
            "n_arrivals_M": n_arrivals_M,
            "n_arrivals_F": n_arrivals_F,
        }

    def step(self, action):
        """Execute action and track metrics."""
        reward = 0.0

        if isinstance(action, np.ndarray):
            approval_prob = np.clip(action[0], 0.0, 1.0)
        else:
            approval_prob = np.clip(float(action), 0.0, 1.0)

        if self.current_applicant is not None:
            applicant = self.current_applicant
            agent_approved = np.random.random() < approval_prob
            ground_truth = applicant["ground_truth"]

            # Track confusion matrix and ground truth
            if applicant["S"] == 1:  # Male
                if agent_approved and ground_truth:
                    self.tp_M += 1
                elif agent_approved and not ground_truth:
                    self.fp_M += 1
                elif not agent_approved and not ground_truth:
                    self.tn_M += 1
                else:  # not approved but should have been
                    self.fn_M += 1

                if ground_truth:
                    self.episode_true_approvals_M += 1
            else:  # Female
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
                kappa_i = (
                    applicant["loan_amount"] if defaults else applicant["wealth_gain"]
                )

                if applicant["S"] == 1:
                    self.current_X_male[applicant["individual_id"]] += kappa_i
                    self.total_loans_M += 1
                    self.episode_loans_M += 1
                    self.timestep_data["approvals_M"] += 1

                    if defaults:
                        self.total_defaults_M += 1
                        self.episode_defaults_M += 1
                        self.timestep_data["defaults_M"] += 1
                        self.timestep_profit -= kappa_i
                    else:
                        profit = applicant["loan_amount"] * self.interest_rate
                        self.timestep_profit += profit
                        self.episode_profit += profit
                else:
                    self.current_X_female[applicant["individual_id"]] += kappa_i
                    self.total_loans_F += 1
                    self.episode_loans_F += 1
                    self.timestep_data["approvals_F"] += 1

                    if defaults:
                        self.total_defaults_F += 1
                        self.episode_defaults_F += 1
                        self.timestep_data["defaults_F"] += 1
                        self.timestep_profit -= kappa_i
                    else:
                        profit = applicant["loan_amount"] * self.interest_rate
                        self.timestep_profit += profit
                        self.episode_profit += profit

        # Get next applicant
        if self.pending_applications:
            self.current_applicant = self.pending_applications.pop(0)
        else:
            self.current_applicant = None

        # Advance timestep if no pending applications
        if self.current_applicant is None and self.time_index < len(self.time_steps):
            if self.timestep_data is not None:
                # Update mean wealth
                self.mu_M = np.mean(self.current_X_male)
                self.mu_F = np.mean(self.current_X_female)

                # Update cumulative profit
                self.cumulative_profit += self.timestep_profit

                # Increment episode timestep counter
                self.episode_timesteps += 1

            if self.time_index < len(self.time_steps):
                self.current_time = self.time_steps[self.time_index]
                self.time_index += 1
                self.global_timestep += 1
                self.timestep_profit = 0.0

                timestep_info = self._generate_timestep_applications()
                self.pending_applications = timestep_info["applications"].copy()

                self.timestep_data = {
                    "apply_select_M": timestep_info["apply_select_M"].copy(),
                    "apply_select_F": timestep_info["apply_select_F"].copy(),
                    "lambda_R": timestep_info["lambda_R"],
                    "lambda_B": timestep_info["lambda_B"],
                    "n_arrivals_M": timestep_info["n_arrivals_M"],
                    "n_arrivals_F": timestep_info["n_arrivals_F"],
                    "approvals_M": 0,
                    "approvals_F": 0,
                    "defaults_M": 0,
                    "defaults_F": 0,
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
            "mu_M": self.mu_M,
            "mu_F": self.mu_F,
            "wealth_gap": self.mu_M - self.mu_F,
            "episode": self.total_episodes,
        }

        return obs, reward, terminated, truncated, info

    def get_episode_metrics_dataframe(self):
        """Return episode-level metrics as a pandas DataFrame."""
        return pd.DataFrame(self.episode_metrics)

    def save_checkpoint(self, checkpoint_path):
        """Save current state to checkpoint file for resuming later."""
        checkpoint = {
            # Episode tracking
            "total_episodes": self.total_episodes,
            "global_timestep": self.global_timestep,
            "current_time": self.current_time,
            # Wealth state
            "current_X_male": self.current_X_male.copy(),
            "current_X_female": self.current_X_female.copy(),
            "mu_M": self.mu_M,
            "mu_F": self.mu_F,
            "prev_mu_M": self.prev_mu_M,
            "prev_mu_F": self.prev_mu_F,
            "mu_M_0": self.mu_M_0,
            "mu_F_0": self.mu_F_0,
            # Episode start tracking
            "episode_start_mu_M": self.episode_start_mu_M,
            "episode_start_mu_F": self.episode_start_mu_F,
            "episode_start_time": self.episode_start_time,
            "episode_timesteps": self.episode_timesteps,
            # Episode-specific counters
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
            # Hawkes process state
            "event_times_R": self.event_times_R.copy(),
            "event_times_B": self.event_times_B.copy(),
            # Cumulative counters
            "total_defaults_M": self.total_defaults_M,
            "total_defaults_F": self.total_defaults_F,
            "total_loans_M": self.total_loans_M,
            "total_loans_F": self.total_loans_F,
            "total_applications_M": self.total_applications_M,
            "total_applications_F": self.total_applications_F,
            "cumulative_profit": self.cumulative_profit,
            # Confusion matrix
            "tp_M": self.tp_M,
            "fp_M": self.fp_M,
            "tn_M": self.tn_M,
            "fn_M": self.fn_M,
            "tp_F": self.tp_F,
            "fp_F": self.fp_F,
            "tn_F": self.tn_F,
            "fn_F": self.fn_F,
            # Episode metrics history
            "episode_metrics": self.episode_metrics,
        }

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        """Load state from checkpoint file to resume."""
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        # Episode tracking
        self.total_episodes = checkpoint["total_episodes"]
        self.global_timestep = checkpoint["global_timestep"]
        self.current_time = checkpoint["current_time"]

        # Wealth state
        self.current_X_male = checkpoint["current_X_male"]
        self.current_X_female = checkpoint["current_X_female"]
        self.mu_M = checkpoint["mu_M"]
        self.mu_F = checkpoint["mu_F"]
        self.prev_mu_M = checkpoint["prev_mu_M"]
        self.prev_mu_F = checkpoint["prev_mu_F"]
        self.mu_M_0 = checkpoint["mu_M_0"]
        self.mu_F_0 = checkpoint["mu_F_0"]

        # Episode start tracking
        self.episode_start_mu_M = checkpoint.get("episode_start_mu_M", self.mu_M)
        self.episode_start_mu_F = checkpoint.get("episode_start_mu_F", self.mu_F)
        self.episode_start_time = checkpoint.get(
            "episode_start_time", self.current_time
        )
        self.episode_timesteps = checkpoint.get("episode_timesteps", 0)

        # Episode-specific counters
        self.episode_loans_M = checkpoint.get("episode_loans_M", 0)
        self.episode_loans_F = checkpoint.get("episode_loans_F", 0)
        self.episode_applications_M = checkpoint.get("episode_applications_M", 0)
        self.episode_applications_F = checkpoint.get("episode_applications_F", 0)
        self.episode_defaults_M = checkpoint.get("episode_defaults_M", 0)
        self.episode_defaults_F = checkpoint.get("episode_defaults_F", 0)
        self.episode_profit = checkpoint.get("episode_profit", 0.0)
        self.episode_actual_approvals_M = checkpoint.get(
            "episode_actual_approvals_M", 0
        )
        self.episode_actual_approvals_F = checkpoint.get(
            "episode_actual_approvals_F", 0
        )
        self.episode_true_approvals_M = checkpoint.get("episode_true_approvals_M", 0)
        self.episode_true_approvals_F = checkpoint.get("episode_true_approvals_F", 0)

        # Hawkes process state
        self.event_times_R = checkpoint["event_times_R"]
        self.event_times_B = checkpoint["event_times_B"]

        # Cumulative counters
        self.total_defaults_M = checkpoint["total_defaults_M"]
        self.total_defaults_F = checkpoint["total_defaults_F"]
        self.total_loans_M = checkpoint["total_loans_M"]
        self.total_loans_F = checkpoint["total_loans_F"]
        self.total_applications_M = checkpoint["total_applications_M"]
        self.total_applications_F = checkpoint["total_applications_F"]
        self.cumulative_profit = checkpoint["cumulative_profit"]

        # Confusion matrix
        self.tp_M = checkpoint["tp_M"]
        self.fp_M = checkpoint["fp_M"]
        self.tn_M = checkpoint["tn_M"]
        self.fn_M = checkpoint["fn_M"]
        self.tp_F = checkpoint["tp_F"]
        self.fp_F = checkpoint["fp_F"]
        self.tn_F = checkpoint["tn_F"]
        self.fn_F = checkpoint["fn_F"]

        # Episode metrics history
        self.episode_metrics = checkpoint.get(
            "episode_metrics", self._init_episode_metrics()
        )

        print(
            f"  ✓ Loaded checkpoint: {self.total_episodes} episodes completed, time={self.current_time:.1f}"
        )

        return checkpoint


# ============================================================================
# POLICY NETWORK (same architecture as training)
# ============================================================================


class PolicyNet(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        alpha = F.softplus(self.alpha_head(x)) + 1.0
        beta = F.softplus(self.beta_head(x)) + 1.0
        return alpha, beta


class LearnableLambdas(nn.Module):
    """Learnable lambda parameters."""

    def __init__(
        self, constraint_type="both", init_lambda_wealth=2.0, init_lambda_approval=2.0
    ):
        super().__init__()
        self.constraint_type = constraint_type

        if constraint_type in ["wealth", "both"]:
            self.log_lambda_wealth = nn.Parameter(
                torch.tensor(np.log(init_lambda_wealth))
            )
        else:
            self.register_buffer(
                "log_lambda_wealth", torch.tensor(np.log(init_lambda_wealth))
            )

        if constraint_type in ["approval_rate", "both"]:
            self.log_lambda_approval = nn.Parameter(
                torch.tensor(np.log(init_lambda_approval))
            )
        else:
            self.register_buffer(
                "log_lambda_approval", torch.tensor(np.log(init_lambda_approval))
            )

    @property
    def lambda_wealth(self):
        return torch.exp(self.log_lambda_wealth)

    @property
    def lambda_approval(self):
        return torch.exp(self.log_lambda_approval)


# ============================================================================
# TESTING AGENT (INFERENCE ONLY)
# ============================================================================


class TestingAgent:
    """Agent for testing/inference only (no training)."""

    def __init__(self, hidden_dim=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNet(12, hidden_dim).to(self.device)
        self.learnable_lambdas = None
        self.reward_func_name = None
        self.constraint_type = None

    def load_model(self, filepath):
        """Load trained model weights."""
        print(f"Loading model from {filepath}...")

        # PyTorch 2.6+ changed default weights_only=True, but our checkpoints contain
        # numpy arrays and other objects, so we need weights_only=False
        try:
            checkpoint = torch.load(
                filepath, map_location=self.device, weights_only=False
            )
        except TypeError:
            # Older PyTorch versions don't have weights_only parameter
            checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.policy_net.eval()  # Set to evaluation mode

        self.reward_func_name = checkpoint.get("reward_function", "unknown")
        self.constraint_type = checkpoint.get("constraint_type", "unknown")

        # Load lambdas if present
        if "learnable_lambdas_state_dict" in checkpoint:
            self.learnable_lambdas = LearnableLambdas(
                constraint_type=self.constraint_type,
                init_lambda_wealth=checkpoint.get("final_lambda_wealth", 2.0),
                init_lambda_approval=checkpoint.get("final_lambda_approval", 2.0),
            ).to(self.device)
            self.learnable_lambdas.load_state_dict(
                checkpoint["learnable_lambdas_state_dict"]
            )

        print(f"  ✓ Model loaded successfully")
        print(f"    Reward function: {self.reward_func_name}")
        print(f"    Constraint type: {self.constraint_type}")
        # print(f"    Final λ_wealth: {checkpoint.get('final_lambda_wealth', 'N/A')}")
        # print(f"    Final λ_approval: {checkpoint.get('final_lambda_approval', 'N/A')}")

        return checkpoint

    def get_action(self, obs, deterministic=False):
        """Get action from policy (no gradient computation)."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            alpha, beta = self.policy_net(obs_tensor)

            if deterministic:
                # Use mode of Beta distribution
                action = (alpha - 1) / (alpha + beta - 2)
                action = torch.clamp(action, 0.0, 1.0)
            else:
                dist = Beta(alpha, beta)
                action = dist.sample()

        return action.cpu().item()


# ============================================================================
# MAIN TESTING FUNCTION
# ============================================================================


def run_testing(
    weights_path: str,
    data_filepath: str = None,
    num_episodes: int = 10,
    seed: int = 42,
    N_male: int = 1000,
    N_female: int = 1000,
    T: int = 500,
    dt: float = 0.5,
    deterministic: bool = False,
    save_results: bool = True,
    results_dir: str = "./test_results",
    credit_threshold: float = 0.5,
    checkpoint_dir: str = "./checkpoints",
    checkpoint_every: int = 1,
    resume: bool = True,
):
    """
    Run testing with trained agent (no training, no reset between episodes).
    ALL METRICS ARE TRACKED PER-EPISODE.

    Args:
        weights_path: Path to saved model weights
        data_filepath: Path to adult.csv
        num_episodes: Number of episodes to run (continuous, no reset)
        seed: Random seed
        N_male, N_female: Number of individuals per group
        T, dt: Time horizon and step size
        deterministic: Use deterministic policy (mode instead of sampling)
        save_results: Save results to files
        results_dir: Directory for results
        credit_threshold: Threshold for ground truth approval (creditworthiness >= threshold)
        checkpoint_dir: Directory for checkpoint files
        checkpoint_every: Save checkpoint every N episodes
        resume: If True, search for existing checkpoint and resume
    """
    print("=" * 80)
    print("TESTING TRAINED AGENT (EPISODE-WISE METRICS)")
    print("=" * 80)
    print(f"  Model: {weights_path}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Continuous play: YES (no wealth reset between episodes)")
    print(f"  Credit threshold for ground truth: {credit_threshold}")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print(f"  Resume from checkpoint: {resume}")
    print("=" * 80)

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create directories
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Generate checkpoint filename based on model
    model_name = os.path.basename(weights_path).replace(".pt", "")
    checkpoint_path = f"{checkpoint_dir}/checkpoint_{model_name}_seed{seed}.pkl"

    # Load data
    print("\n[1] Loading Data...")
    loader = AdultIncomeDataLoader(filepath=data_filepath, sample_size=20000)
    loader.credit_threshold = credit_threshold
    loader.load_data()
    loader.preprocess()

    # Learn theta parameters
    # print("\n[2] Learning Transition Parameters...")
    theta_learner = TransitionParameterLearner(
        default_rate_min=0.02, default_rate_max=0.15
    )
    theta_learner.fit(loader.data)

    # Create testing environment
    # print("\n[3] Creating Testing Environment...")
    env = TestingIncomeEnvironment(
        theta_params=theta_learner,
        initial_wealth_male=loader.male_data["X"].values,
        initial_wealth_female=loader.female_data["X"].values,
        ground_truth_male=loader.male_data["ground_truth_approval"].values,
        ground_truth_female=loader.female_data["ground_truth_approval"].values,
        N_male=N_male,
        N_female=N_female,
        T=T,
        dt=dt,
        seed=seed,
    )

    # Load trained agent
    # print("\n[4] Loading Trained Agent...")
    agent = TestingAgent()
    agent.load_model(weights_path)

    # Check for existing checkpoint and resume if requested
    start_episode = 0
    if resume and os.path.exists(checkpoint_path):
        print(f"\n[4.5] Found existing checkpoint, resuming...")
        env.load_checkpoint(checkpoint_path)
        start_episode = env.total_episodes
        print(f"  Resuming from episode {start_episode + 1}")

    # Check if already completed
    if start_episode >= num_episodes:
        print(f"\n  All {num_episodes} episodes already completed!")
    else:
        # Run testing
        remaining_episodes = num_episodes - start_episode
        print(
            f"\n[5] Running {remaining_episodes} Episodes (Episodes {start_episode + 1} to {num_episodes})..."
        )

        for episode in range(start_episode, num_episodes):
            obs, _ = env.reset()
            done = False
            step_count = 0

            while not done:
                action = agent.get_action(obs, deterministic=deterministic)
                obs, _, terminated, truncated, info = env.step(np.array([action]))
                done = terminated or truncated
                step_count += 1

            # Print episode summary - only every 10 episodes or last episode
            if (episode + 1) % 10 == 0 or episode + 1 == num_episodes:
                print(f"\n  Episode {episode + 1}/{num_episodes} Complete:")
                print(f"    Timesteps this episode: {env.episode_timesteps}")
                print(f"    Total time elapsed: {env.current_time:.1f}")
                print(f"    μ_M: ${env.mu_M:.2f}k, μ_F: ${env.mu_F:.2f}k")
                print(f"    Wealth Gap: ${env.mu_M - env.mu_F:.2f}k")
                print(
                    f"    Episode Approval Rate M: {env.episode_loans_M / max(env.episode_applications_M, 1):.4f}"
                )
                print(
                    f"    Episode Approval Rate F: {env.episode_loans_F / max(env.episode_applications_F, 1):.4f}"
                )
                print(f"    Episode Profit: ${env.episode_profit:.2f}k")
                print(f"    Cumulative Profit: ${env.cumulative_profit:.2f}k")

            # Save checkpoint
            if (episode + 1) % checkpoint_every == 0:
                env.save_checkpoint(checkpoint_path)
                # print(f"    ✓ Checkpoint saved ({checkpoint_path})")

    # Finalize episode metrics (record the last episode)
    env.finalize_episode_metrics()
    episode_metrics_df = env.get_episode_metrics_dataframe()

    # Print final results
    print("\n[6] Computing Final Metrics...")

    # Compute accuracy metrics
    total_decisions_M = env.tp_M + env.fp_M + env.tn_M + env.fn_M
    total_decisions_F = env.tp_F + env.fp_F + env.tn_F + env.fn_F

    accuracy_M = (env.tp_M + env.tn_M) / max(total_decisions_M, 1)
    accuracy_F = (env.tp_F + env.tn_F) / max(total_decisions_F, 1)

    precision_M = env.tp_M / max(env.tp_M + env.fp_M, 1)
    precision_F = env.tp_F / max(env.tp_F + env.fp_F, 1)

    recall_M = env.tp_M / max(env.tp_M + env.fn_M, 1)
    recall_F = env.tp_F / max(env.tp_F + env.fn_F, 1)

    print("\n" + "=" * 80)
    print("FINAL TESTING RESULTS")
    print("=" * 80)

    print(f"\n  Model: {agent.reward_func_name} ({agent.constraint_type})")
    print(f"  Episodes: {num_episodes}")
    print(f"  Total Time: {env.current_time:.1f}")
    print(f"  Hawkes Events (M/F): {len(env.event_times_R)}/{len(env.event_times_B)}")

    print(f"\n  --- Wealth Metrics ---")
    print(f"  Initial μ_M,0: ${env.mu_M_0:.2f}k")
    print(f"  Initial μ_F,0: ${env.mu_F_0:.2f}k")
    print(f"  Final μ_M: ${env.mu_M:.2f}k")
    print(f"  Final μ_F: ${env.mu_F:.2f}k")
    print(f"  Final Wealth Gap: ${env.mu_M - env.mu_F:.2f}k")

    print(f"\n  --- Long-term Social Welfare R_g(t) = (μ_g,t - μ_g,0) / t ---")
    if env.current_time > 1e-8:
        R_M_final = (env.mu_M - env.mu_M_0) / env.current_time
        R_F_final = (env.mu_F - env.mu_F_0) / env.current_time
        R_total_final = (R_M_final + R_F_final) / 2
    else:
        R_M_final = R_F_final = R_total_final = 0.0
    print(f"  R_M (Male):   {R_M_final:.6f} ($1000s per time unit)")
    print(f"  R_F (Female): {R_F_final:.6f} ($1000s per time unit)")
    print(f"  R_total:      {R_total_final:.6f} ($1000s per time unit)")
    print(
        f"  R_M / R_F:    {R_M_final / R_F_final:.4f}"
        if abs(R_F_final) > 1e-8
        else "  R_M / R_F:    N/A (R_F ≈ 0)"
    )

    print(f"\n  --- Inequality Ratio ρ = Δμ_M / Δμ_F (per episode) ---")
    if len(episode_metrics_df) > 0:
        rho_values = episode_metrics_df["rho_episode"].values
        print(f"  Mean ρ:   {np.mean(rho_values):.4f}")
        print(f"  Std ρ:    {np.std(rho_values):.4f}")
        print(f"  Min ρ:    {np.min(rho_values):.4f}")
        print(f"  Max ρ:    {np.max(rho_values):.4f}")
        print(f"  Last 5 episodes: {[f'{r:.3f}' for r in rho_values[-5:]]}")

    print(f"\n  --- Approval Metrics ---")
    print(f"  Total Applications M: {env.total_applications_M}")
    print(f"  Total Applications F: {env.total_applications_F}")
    print(f"  Total Loans Approved M: {env.total_loans_M}")
    print(f"  Total Loans Approved F: {env.total_loans_F}")
    print(
        f"  Final Approval Rate M: {env.total_loans_M / max(env.total_applications_M, 1):.4f}"
    )
    print(
        f"  Final Approval Rate F: {env.total_loans_F / max(env.total_applications_F, 1):.4f}"
    )

    print(f"\n  --- Success Probabilities ---")
    print(f"  Total Defaults M: {env.total_defaults_M}")
    print(f"  Total Defaults F: {env.total_defaults_F}")
    print(
        f"  Final Success Prob M: {1 - env.total_defaults_M / max(env.total_loans_M, 1):.4f}"
    )
    print(
        f"  Final Success Prob F: {1 - env.total_defaults_F / max(env.total_loans_F, 1):.4f}"
    )

    print(f"\n  --- Ground Truth Comparison (Males) ---")
    print(f"  True Positives:  {env.tp_M}")
    print(f"  False Positives: {env.fp_M}")
    print(f"  True Negatives:  {env.tn_M}")
    print(f"  False Negatives: {env.fn_M}")
    print(f"  Accuracy:  {accuracy_M:.4f}")
    print(f"  Precision: {precision_M:.4f}")
    print(f"  Recall:    {recall_M:.4f}")

    print(f"\n  --- Ground Truth Comparison (Females) ---")
    print(f"  True Positives:  {env.tp_F}")
    print(f"  False Positives: {env.fp_F}")
    print(f"  True Negatives:  {env.tn_F}")
    print(f"  False Negatives: {env.fn_F}")
    print(f"  Accuracy:  {accuracy_F:.4f}")
    print(f"  Precision: {precision_F:.4f}")
    print(f"  Recall:    {recall_F:.4f}")

    print(f"\n  --- Profit ---")
    print(f"  Cumulative Profit: ${env.cumulative_profit:.2f}k")

    # Save results
    if save_results:
        print("\n[7] Saving Results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save episode metrics CSV
        csv_path = f"{results_dir}/test_episode_metrics_{agent.reward_func_name}_{agent.constraint_type}_seed{seed - 1}_{timestamp}.csv"
        episode_metrics_df.to_csv(csv_path, index=False)
        print(f"  Saved episode metrics to {csv_path}")

        # Compute final long-term social welfare
        if env.current_time > 1e-8:
            R_M_final = (env.mu_M - env.mu_M_0) / env.current_time
            R_F_final = (env.mu_F - env.mu_F_0) / env.current_time
        else:
            R_M_final = R_F_final = 0.0

        # Save summary
        summary = {
            "model_path": weights_path,
            "reward_function": agent.reward_func_name,
            "constraint_type": agent.constraint_type,
            "num_episodes": num_episodes,
            "total_time": env.current_time,
            "seed": seed,
            "credit_threshold": credit_threshold,
            "initial_mu_M": env.mu_M_0,
            "initial_mu_F": env.mu_F_0,
            "final_mu_M": env.mu_M,
            "final_mu_F": env.mu_F,
            "final_wealth_gap": env.mu_M - env.mu_F,
            "R_M_final": R_M_final,
            "R_F_final": R_F_final,
            "R_total_final": (R_M_final + R_F_final) / 2,
            "total_loans_M": env.total_loans_M,
            "total_loans_F": env.total_loans_F,
            "total_applications_M": env.total_applications_M,
            "total_applications_F": env.total_applications_F,
            "approval_rate_M": env.total_loans_M / max(env.total_applications_M, 1),
            "approval_rate_F": env.total_loans_F / max(env.total_applications_F, 1),
            "success_prob_M": 1 - env.total_defaults_M / max(env.total_loans_M, 1),
            "success_prob_F": 1 - env.total_defaults_F / max(env.total_loans_F, 1),
            "total_defaults_M": env.total_defaults_M,
            "total_defaults_F": env.total_defaults_F,
            "hawkes_events_M": len(env.event_times_R),
            "hawkes_events_F": len(env.event_times_B),
            "accuracy_M": accuracy_M,
            "accuracy_F": accuracy_F,
            "precision_M": precision_M,
            "precision_F": precision_F,
            "recall_M": recall_M,
            "recall_F": recall_F,
            "cumulative_profit": env.cumulative_profit,
            "tp_M": env.tp_M,
            "fp_M": env.fp_M,
            "tn_M": env.tn_M,
            "fn_M": env.fn_M,
            "tp_F": env.tp_F,
            "fp_F": env.fp_F,
            "tn_F": env.tn_F,
            "fn_F": env.fn_F,
        }

        # Add episode-level rho stats
        if len(episode_metrics_df) > 0:
            rho_values = episode_metrics_df["rho_episode"].values
            summary["rho_mean"] = float(np.mean(rho_values))
            summary["rho_std"] = float(np.std(rho_values))
            summary["rho_min"] = float(np.min(rho_values))
            summary["rho_max"] = float(np.max(rho_values))
            summary["rho_last"] = float(rho_values[-1])

        # summary_path = f"{results_dir}/test_summary_{agent.reward_func_name}_{agent.constraint_type}_{timestamp}.json"
        # with open(summary_path, 'w') as f:
        #     json.dump(summary, f, indent=2)
        # print(f"  Saved summary to {summary_path}")

        # Remove checkpoint after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            # print(f"  Removed checkpoint (run completed): {checkpoint_path}")

    # # Plot results
    # print("\n[8] Generating Plots...")
    # plot_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # plot_path = f"{results_dir}/test_plots_{plot_timestamp}.png" if save_results else None
    # plot_testing_results(episode_metrics_df, env, agent, save_path=plot_path)

    return episode_metrics_df, env, agent


def plot_testing_results(episode_metrics_df, env, agent, save_path=None):
    """Plot comprehensive testing results with episode number on x-axis."""

    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(5, 3, figure=fig)

    # Use episode numbers for x-axis
    episodes = episode_metrics_df["episode"].values

    # 1. Inequality Ratio (ρ) per episode
    ax = fig.add_subplot(gs[0, 0])
    rho_episode = episode_metrics_df["rho_episode"].values
    rho_clipped = np.clip(rho_episode, -10, 10)
    ax.plot(
        episodes, rho_clipped, "b-o", linewidth=2, markersize=4, label="ρ per episode"
    )
    ax.axhline(y=1.0, color="red", linestyle="--", label="Equal growth (ρ=1)")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("ρ (Inequality Ratio)")
    ax.set_title("Inequality Ratio: ρ = Δμ_M / Δμ_F (per episode)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Mean Wealth at episode end
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(
        episodes,
        episode_metrics_df["mu_M_end"],
        "b-o",
        linewidth=2,
        markersize=4,
        label="Male (μ_M)",
    )
    ax.plot(
        episodes,
        episode_metrics_df["mu_F_end"],
        "r-o",
        linewidth=2,
        markersize=4,
        label="Female (μ_F)",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Wealth ($1000s)")
    ax.set_title("Mean Wealth at Episode End")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Long-term Social Welfare R_g(t)
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(
        episodes,
        episode_metrics_df["R_M"],
        "b-o",
        linewidth=2,
        markersize=4,
        label="R_M (Male)",
    )
    ax.plot(
        episodes,
        episode_metrics_df["R_F"],
        "r-o",
        linewidth=2,
        markersize=4,
        label="R_F (Female)",
    )
    ax.plot(
        episodes,
        episode_metrics_df["R_total"],
        "g--o",
        linewidth=2,
        markersize=4,
        label="R_total",
    )
    ax.axhline(y=0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("R_g(t) = (μ_g,t - μ_g,0) / t")
    ax.set_title("Long-term Social Welfare")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Wealth Gap at episode end
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(
        episodes,
        episode_metrics_df["wealth_gap"],
        "purple",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Wealth Gap ($1000s)")
    ax.set_title("Wealth Gap: μ_M - μ_F (at episode end)")
    ax.grid(True, alpha=0.3)

    # 5. Episode Approval Rates
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(
        episodes,
        episode_metrics_df["approval_rate_M_episode"],
        "b-o",
        linewidth=2,
        markersize=4,
        label="Male (episode)",
    )
    ax.plot(
        episodes,
        episode_metrics_df["approval_rate_F_episode"],
        "r-o",
        linewidth=2,
        markersize=4,
        label="Female (episode)",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Approval Rate")
    ax.set_title("Approval Rates per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Cumulative Loans Approved
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(
        episodes,
        episode_metrics_df["total_loans_M"],
        "b-o",
        linewidth=2,
        markersize=4,
        label="Male",
    )
    ax.plot(
        episodes,
        episode_metrics_df["total_loans_F"],
        "r-o",
        linewidth=2,
        markersize=4,
        label="Female",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Loans (cumulative)")
    ax.set_title("Cumulative Loans Approved")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7. Episode Success Probabilities
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(
        episodes,
        episode_metrics_df["success_prob_M_episode"],
        "b-o",
        linewidth=2,
        markersize=4,
        label="Male",
    )
    ax.plot(
        episodes,
        episode_metrics_df["success_prob_F_episode"],
        "r-o",
        linewidth=2,
        markersize=4,
        label="Female",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Probability")
    ax.set_title("Success Probabilities per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8. Actual vs True Approvals (Male) - per episode
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(
        episodes,
        episode_metrics_df["actual_approvals_M_episode"],
        "b-o",
        linewidth=2,
        markersize=4,
        label="Actual (Agent)",
    )
    ax.plot(
        episodes,
        episode_metrics_df["true_approvals_M_episode"],
        "g--s",
        linewidth=2,
        markersize=4,
        label="True (Ground Truth)",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Approvals per Episode")
    ax.set_title("Male: Actual vs True Approvals")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 9. Actual vs True Approvals (Female) - per episode
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(
        episodes,
        episode_metrics_df["actual_approvals_F_episode"],
        "r-o",
        linewidth=2,
        markersize=4,
        label="Actual (Agent)",
    )
    ax.plot(
        episodes,
        episode_metrics_df["true_approvals_F_episode"],
        "g--s",
        linewidth=2,
        markersize=4,
        label="True (Ground Truth)",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Approvals per Episode")
    ax.set_title("Female: Actual vs True Approvals")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 10. Cumulative Profit
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(
        episodes,
        episode_metrics_df["cumulative_profit"],
        "green",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Profit ($1000s)")
    ax.set_title("Cumulative Bank Profit")
    ax.grid(True, alpha=0.3)

    # 11. Episode Profit
    ax = fig.add_subplot(gs[3, 1])
    ax.bar(
        episodes,
        episode_metrics_df["profit_episode"],
        color="green",
        alpha=0.7,
        edgecolor="darkgreen",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Profit ($1000s)")
    ax.set_title("Profit per Episode")
    ax.grid(True, alpha=0.3, axis="y")

    # 12. Accuracy Metrics over Episodes
    ax = fig.add_subplot(gs[3, 2])
    ax.plot(
        episodes,
        episode_metrics_df["accuracy_M"],
        "b-o",
        linewidth=2,
        markersize=4,
        label="Accuracy M",
    )
    ax.plot(
        episodes,
        episode_metrics_df["accuracy_F"],
        "r-o",
        linewidth=2,
        markersize=4,
        label="Accuracy F",
    )
    ax.plot(
        episodes,
        episode_metrics_df["precision_M"],
        "b--s",
        linewidth=1.5,
        markersize=3,
        alpha=0.7,
        label="Precision M",
    )
    ax.plot(
        episodes,
        episode_metrics_df["precision_F"],
        "r--s",
        linewidth=1.5,
        markersize=3,
        alpha=0.7,
        label="Precision F",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Metric Value")
    ax.set_title("Cumulative Accuracy & Precision")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 13. Confusion Matrix - Male (Final)
    ax = fig.add_subplot(gs[4, 0])
    cm_M = np.array([[env.tp_M, env.fn_M], [env.fp_M, env.tn_M]])
    im = ax.imshow(cm_M, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Approve", "Reject"])
    ax.set_yticklabels(["Should Approve", "Should Reject"])
    ax.set_xlabel("Agent Decision")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Male Confusion Matrix (Final)")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm_M[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

    # 14. Confusion Matrix - Female (Final)
    ax = fig.add_subplot(gs[4, 1])
    cm_F = np.array([[env.tp_F, env.fn_F], [env.fp_F, env.tn_F]])
    im = ax.imshow(cm_F, cmap="Reds")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Approve", "Reject"])
    ax.set_yticklabels(["Should Approve", "Should Reject"])
    ax.set_xlabel("Agent Decision")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Female Confusion Matrix (Final)")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm_F[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

    # 15. Summary Statistics
    ax = fig.add_subplot(gs[4, 2])
    ax.axis("off")

    total_M = env.tp_M + env.fp_M + env.tn_M + env.fn_M
    total_F = env.tp_F + env.fp_F + env.tn_F + env.fn_F

    accuracy_M = (env.tp_M + env.tn_M) / max(total_M, 1)
    accuracy_F = (env.tp_F + env.tn_F) / max(total_F, 1)
    precision_M = env.tp_M / max(env.tp_M + env.fp_M, 1)
    precision_F = env.tp_F / max(env.tp_F + env.fp_F, 1)
    recall_M = env.tp_M / max(env.tp_M + env.fn_M, 1)
    recall_F = env.tp_F / max(env.tp_F + env.fn_F, 1)

    # Compute final R values
    if env.current_time > 1e-8:
        R_M_final = (env.mu_M - env.mu_M_0) / env.current_time
        R_F_final = (env.mu_F - env.mu_F_0) / env.current_time
    else:
        R_M_final = R_F_final = 0.0

    rho_values = episode_metrics_df["rho_episode"].values

    text = f"Testing Summary\n"
    text += "=" * 50 + "\n\n"
    text += f"Model: {agent.reward_func_name}\n"
    text += f"Constraint: {agent.constraint_type}\n"
    text += f"Episodes: {env.total_episodes}\n"
    text += f"Total Time: {env.current_time:.1f}\n\n"

    text += f"--- Long-term Social Welfare ---\n"
    text += f"R_M: {R_M_final:.6f}\n"
    text += f"R_F: {R_F_final:.6f}\n"
    text += (
        f"R_M/R_F: {R_M_final / R_F_final:.4f}\n\n"
        if abs(R_F_final) > 1e-8
        else "R_M/R_F: N/A\n\n"
    )

    text += f"--- Inequality Ratio ---\n"
    text += f"Mean ρ: {np.mean(rho_values):.4f}\n"
    text += f"Std ρ:  {np.std(rho_values):.4f}\n\n"

    text += f"--- Accuracy (Final) ---\n"
    text += f"Male:   {accuracy_M:.4f}\n"
    text += f"Female: {accuracy_F:.4f}\n"

    ax.text(
        0.05,
        0.5,
        text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.suptitle(
        f"Testing Results: {agent.reward_func_name} ({agent.constraint_type}) - Episode-wise Metrics",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot to {save_path}")

    plt.show()
    return fig


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Trained Agent on Income Environment (Episode-wise Metrics)"
    )

    # Required (but not for --list-checkpoints)
    parser.add_argument(
        "--weights", type=str, default=None, help="Path to trained model weights (.pt)"
    )

    # Data
    parser.add_argument("--data", type=str, default=None, help="Path to adult.csv")

    # Testing parameters
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes (continuous)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--N-male", type=int, default=3000, help="Number of male individuals"
    )
    parser.add_argument(
        "--N-female", type=int, default=3000, help="Number of female individuals"
    )
    parser.add_argument("--T", type=int, default=100, help="Time horizon per episode")
    parser.add_argument("--dt", type=float, default=0.5, help="Time step")
    parser.add_argument(
        "--credit-threshold",
        type=float,
        default=0.5,
        help="Creditworthiness threshold for ground truth approval",
    )

    # Options
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic policy"
    )
    parser.add_argument("--save", action="store_false", help="Save results")
    parser.add_argument(
        "--results-dir", type=str, default="./test_results", help="Results directory"
    )

    # Checkpoint options
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for checkpoint files",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save checkpoint every N episodes",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore existing checkpoints",
    )
    parser.add_argument(
        "--clear-checkpoints",
        action="store_true",
        help="Clear existing checkpoints before starting",
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List available checkpoints and exit",
    )

    args = parser.parse_args()

    # List checkpoints and exit if requested
    if args.list_checkpoints:
        list_checkpoints(args.checkpoint_dir)
        exit(0)

    # Check that weights is provided for actual testing
    if args.weights is None:
        parser.error(
            "--weights is required for testing (use --list-checkpoints to see available checkpoints)"
        )

    # Handle checkpoint clearing
    if args.clear_checkpoints:
        import glob

        model_name = os.path.basename(args.weights).replace(".pt", "")
        pattern = f"{args.checkpoint_dir}/checkpoint_{model_name}_seed{args.seed}.pkl"
        for f in glob.glob(pattern):
            os.remove(f)
            print(f"Removed checkpoint: {f}")

    episode_metrics_df, env, agent = run_testing(
        weights_path=args.weights,
        data_filepath=args.data,
        num_episodes=args.episodes,
        seed=args.seed,
        N_male=args.N_male,
        N_female=args.N_female,
        T=args.T,
        dt=args.dt,
        deterministic=args.deterministic,
        save_results=args.save,
        results_dir=args.results_dir,
        credit_threshold=args.credit_threshold,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )
