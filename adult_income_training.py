import json
import pickle
import random
import warnings
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.distributions import Beta, Normal

warnings.filterwarnings("ignore")


class AdultIncomeDataLoader:
    """Load and preprocess Adult Income dataset (Census Bureau)."""

    def __init__(self, filepath: str = None, sample_size: int = 50000):
        self.filepath = filepath
        self.sample_size = sample_size
        self.data = None
        self.male_data = None
        self.female_data = None
        self.creditworthiness_model = None

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
                }
                self.data = self.data.rename(columns=column_mapping)

                if len(self.data) > self.sample_size:
                    self.data = self.data.sample(n=self.sample_size, random_state=42)

                print(f"✓ Loaded {len(self.data):,} records from {self.filepath}")

            except Exception as e:
                print(f"Error loading file: {e}")
                print("Trying alternative sources...")
                self.data = self._download_adult_income_data()

        return self.data

    def _download_adult_income_data(self):
        """Download Adult Income dataset from UCI, with fallback to synthetic."""
        try:
            print("  Trying UCI Repository...")
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

            print(f"  ✓ Downloaded {len(df):,} records from UCI repository")
            return df

        except Exception as e:
            print(f"  UCI failed: {e}")
            return self._generate_synthetic_adult_data()

    def _generate_synthetic_adult_data(self):
        """Generate synthetic Adult Income-like data."""
        print("  Generating realistic synthetic Adult Income data...")
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

            base_prob = 0.15
            base_prob += 0.12 if is_male else 0
            base_prob += 0.02 * max(0, (education_num - 10))
            base_prob += 0.005 * max(0, (age - 30))
            base_prob -= 0.003 * max(0, (age - 50))
            base_prob += 0.002 * max(0, (hours_per_week - 40))
            base_prob += 0.0001 * capital_gain
            base_prob = np.clip(base_prob, 0.05, 0.85)
            income = ">50K" if np.random.random() < base_prob else "<=50K"

            data.append(
                {
                    "age": int(age),
                    "workclass": np.random.choice(
                        ["Private", "Self-emp", "Gov", "Other"],
                        p=[0.7, 0.15, 0.12, 0.03],
                    ),
                    "fnlwgt": np.random.randint(10000, 1000000),
                    "education": "HS-grad",
                    "education_num": int(education_num),
                    "marital_status": np.random.choice(
                        ["Married", "Single", "Divorced"], p=[0.5, 0.35, 0.15]
                    ),
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

        df = pd.DataFrame(data)
        print(f"  ✓ Generated {len(df):,} synthetic records")
        return df

    def _derive_creditworthiness_optimal(self, df):
        """Derive creditworthiness using logistic regression on income prediction."""
        print("\n  Deriving creditworthiness using income prediction model...")

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

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)

        y = (df["income"].str.strip() == ">50K").astype(int)

        self.creditworthiness_model = LogisticRegression(random_state=42, max_iter=1000)
        self.creditworthiness_model.fit(X_scaled, y)

        creditworthiness_probs = self.creditworthiness_model.predict_proba(X_scaled)[
            :, 1
        ]
        return creditworthiness_probs

    def preprocess(self):
        """Preprocess Adult Income data."""
        print("Preprocessing Adult Income data...")

        df = self.data.copy()
        df["S"] = (df["sex"].str.strip() == "Male").astype(int)
        df["approved"] = (df["income"].str.strip() == ">50K").astype(int)
        df["creditworthiness"] = self._derive_creditworthiness_optimal(df)
        df["X"] = 10 + df["creditworthiness"] * 190

        self.male_data = df[df["S"] == 1].reset_index(drop=True)
        self.female_data = df[df["S"] == 0].reset_index(drop=True)

        print(f"\nMale applicants: {len(self.male_data)}")
        print(f"Female applicants: {len(self.female_data)}")
        print(f"Male high-income rate: {self.male_data['approved'].mean():.3f}")
        print(f"Female high-income rate: {self.female_data['approved'].mean():.3f}")

        self.data = df
        return df


class TransitionParameterLearner:
    """Learn transition parameters θ that GOVERN the sequential dynamics."""

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
        """Learn h_θ(X,S) = θ_S*S + θ_X*X + b"""
        # print("\n[Learning Transition Model h_θ(X,S)]")

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

        # print(f"  θ_S = {self.theta_S:.4f}")
        # print(f"  θ_X = {self.theta_X:.4f}")
        # print(f"  b = {self.b:.4f}")

        return model

    def learn_application_behavior(self, data: pd.DataFrame):
        """Learn how h_θ(X,S) affects application probability."""
        # print("\n[Learning Application Behavior]")

        for group_name, S_val in [("male", 1), ("female", 0)]:
            group_data = data[data["S"] == S_val]
            base_rate = len(group_data) / len(data)
            self.application_sensitivity[group_name] = {
                "base_rate": base_rate,
                "sensitivity": 0.5,
            }
            # print(f"  {group_name.capitalize()}: base_rate={base_rate:.3f}")

    def learn_wealth_gain_parameters(self, data: pd.DataFrame):
        """Learn κ parameters that govern wealth transitions."""
        # print("\n[Learning Wealth Transition Parameters κ]")

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
        """Learn P(default | X, approved)."""
        # print("\n[Learning Default Model]")
        # print(f"  Using UNIFORM distribution: [{self.default_rate_min:.2%}, {self.default_rate_max:.2%}]")

    def learn_variance_parameters(self, data: pd.DataFrame):
        """Learn σ² - governs wealth distribution spread."""
        # print("\n[Learning Variance Parameters]")
        self.sigma = data["X"].std()
        # print(f"  σ = ${self.sigma:.3f}k")

    def initialize_individual_parameters(
        self, N_male: int, N_female: int, seed: int = None
    ):
        """PRE-GENERATE all individual parameters at simulation start."""
        if seed is not None:
            np.random.seed(seed)

        # print("\n[Initializing Fixed Individual Parameters]")

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

            # print(f"  {group.capitalize()} (N={N}):")
            # print(f"    Default prob: mean={self.individual_default_probs[group].mean():.4f}")
            # print(f"    Loan amount:  mean=${self.individual_loan_amounts[group].mean():.2f}k")

    def fit(self, data: pd.DataFrame):
        """Learn ALL transition parameters."""
        # print("\n" + "="*70)
        # print("LEARNING TRANSITION DYNAMICS FROM ADULT INCOME DATA")
        # print("="*70)

        self.learn_approval_model(data)
        self.learn_application_behavior(data)
        self.learn_wealth_gain_parameters(data)
        self.learn_default_model(data)
        self.learn_variance_parameters(data)

        # print("\n" + "="*70)
        # print("TRANSITION LEARNING COMPLETE")
        # print("="*70)

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
            final_num = 10 + np.int32(
                np.ceil(np.random.random(1)[0] * 0.0 * (len(applying_indices_R) - 10))
            )
            # print(final_num)
            applying_indices_R = applying_indices_R[:final_num]
        # print(f"Male applicants - {len(applying_indices_R)}")
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
            final_num = 10 + np.int32(
                np.ceil(np.random.random(1)[0] * 0.0 * (len(applying_indices_B) - 10))
            )
            applying_indices_B = applying_indices_B[:final_num]
        # print(f"Female applicants - {len(applying_indices_B)}")

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
                # kappa_i = 0.0 if defaults else applicant["wealth_gain"]
                kappa_i = (
                    applicant["loan_amount"] if defaults else applicant["wealth_gain"]
                )

                if applicant["S"] == 1:
                    self.current_X_male[applicant["individual_id"]] += kappa_i
                    self.total_loans_R += 1
                    self.timestep_data["approvals_R"] += 1
                    self.timestep_data["apply_select_R"][applicant["individual_id"]] = 1

                    if defaults:
                        self.total_defaults_R += 1
                        self.timestep_data["defaults_R"] += 1
                        self.timestep_profit -= kappa_i
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
                        self.timestep_profit -= kappa_i
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
        }

        return obs, reward, terminated, truncated, info


class RewardFunction:
    """
    Reward functions for RL agent with constraint_type flag.

    constraint_type options:
        - 'approval_rate': Optimize based on approval rates
        - 'wealth': Optimize based on mean wealth (μ_R, μ_B)
        - 'both': Optimize both with learnable lambdas
    """

    @staticmethod
    def _calculate_bank_profit(env, action, applicant):
        """Calculate bank's profit from a lending decision."""
        if applicant is None:
            return 0.0

        approval_prob = action[0] if isinstance(action, np.ndarray) else action
        default_prob = applicant["default_prob"]
        loan_amount = applicant["loan_amount"]

        revenue = (1 - default_prob) * loan_amount * env.interest_rate
        loss = default_prob * loan_amount
        net_profit = revenue - loss
        bank_profit = approval_prob * net_profit

        return bank_profit

    @staticmethod
    def utilitarian_profit(
        env,
        action,
        info,
        constraint_type="wealth",
        lambda_wealth=0.0,
        lambda_approval=0.0,
    ):
        """Utilitarian Profit: Pure bank profit maximization."""
        applicant = info.get("applicant")
        return RewardFunction._calculate_bank_profit(env, action, applicant)

    @staticmethod
    def social_welfare(
        env,
        action,
        info,
        constraint_type="wealth",
        lambda_wealth=2.0,
        lambda_approval=2.0,
    ):
        """
        Social Welfare with constraint_type flag.

        - 'approval_rate': bank_profit + λ * approval_prob
        - 'wealth': bank_profit + λ * (μ_R + μ_B)
        - 'both': bank_profit + λ_wealth * (μ_R + μ_B) + λ_approval * approval_prob
        """
        applicant = info.get("applicant")
        bank_profit = RewardFunction._calculate_bank_profit(env, action, applicant)

        if applicant is None:
            return bank_profit

        approval_prob = action[0] if isinstance(action, np.ndarray) else action

        if constraint_type == "approval_rate":
            reward = bank_profit + lambda_approval * approval_prob

        elif constraint_type == "wealth":
            total_means = env.mu_R + env.mu_B
            reward = bank_profit + lambda_wealth * total_means

        elif constraint_type == "both":
            total_means = env.mu_R + env.mu_B
            reward = (
                bank_profit
                + lambda_wealth * total_means
                + lambda_approval * approval_prob
            )

        else:
            raise ValueError(f"Unknown constraint_type: {constraint_type}")

        return reward

    @staticmethod
    def rawlsian_maximin(
        env,
        action,
        info,
        constraint_type="wealth",
        lambda_wealth=5.0,
        lambda_approval=5.0,
    ):
        """
        Rawlsian Maximin with constraint_type flag.

        - 'approval_rate': bank_profit + λ * min(approval_rate_M, approval_rate_F)
        - 'wealth': bank_profit + λ * min(μ_R, μ_B)
        - 'both': bank_profit + λ_wealth * min(μ_R, μ_B) + λ_approval * min(rate_M, rate_F)
        """
        applicant = info.get("applicant")
        bank_profit = RewardFunction._calculate_bank_profit(env, action, applicant)

        approval_rate_M = env.total_loans_R / max(env.total_applications_R, 1)
        approval_rate_F = env.total_loans_B / max(env.total_applications_B, 1)

        if constraint_type == "approval_rate":
            approval_rate_min = min(approval_rate_M, approval_rate_F)
            reward = bank_profit + lambda_approval * approval_rate_min

        elif constraint_type == "wealth":
            min_mean = min(env.mu_R, env.mu_B)
            reward = bank_profit + lambda_wealth * min_mean

        elif constraint_type == "both":
            min_mean = min(env.mu_R, env.mu_B)
            approval_rate_min = min(approval_rate_M, approval_rate_F)
            reward = (
                bank_profit
                + lambda_wealth * min_mean
                + lambda_approval * approval_rate_min
            )

        else:
            raise ValueError(f"Unknown constraint_type: {constraint_type}")

        return reward

    @staticmethod
    def fairness_lagrangian(
        env,
        action,
        info,
        constraint_type="wealth",
        lambda_wealth=10.0,
        lambda_approval=10.0,
    ):
        """
        Fairness Lagrangian with constraint_type flag.

        - 'approval_rate': bank_profit - λ * |approval_rate_M - approval_rate_F|
        - 'wealth': bank_profit - λ * |μ_R - μ_B|
        - 'both': bank_profit - λ_wealth * |μ_R - μ_B| - λ_approval * |rate_M - rate_F|
        """
        applicant = info.get("applicant")
        bank_profit = RewardFunction._calculate_bank_profit(env, action, applicant)

        approval_rate_M = env.total_loans_R / max(env.total_applications_R, 1)
        approval_rate_F = env.total_loans_B / max(env.total_applications_B, 1)

        if constraint_type == "approval_rate":
            disparity_penalty = lambda_approval * abs(approval_rate_M - approval_rate_F)
            reward = bank_profit - disparity_penalty

        elif constraint_type == "wealth":
            mean_gap = abs(env.mu_R - env.mu_B)
            reward = bank_profit - lambda_wealth * mean_gap

        elif constraint_type == "both":
            mean_gap = abs(env.mu_R - env.mu_B)
            rate_gap = abs(approval_rate_M - approval_rate_F)
            reward = bank_profit - lambda_wealth * mean_gap - lambda_approval * rate_gap

        else:
            raise ValueError(f"Unknown constraint_type: {constraint_type}")

        return reward


class LearnableLambdas(nn.Module):
    """Learnable lambda parameters for all constraint modes."""

    def __init__(
        self, constraint_type="both", init_lambda_wealth=2.0, init_lambda_approval=2.0
    ):
        super().__init__()
        self.constraint_type = constraint_type

        # Use log-space for positivity constraint
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

    def forward(self):
        return self.lambda_wealth, self.lambda_approval


class PolicyGradientAgent:
    """Policy gradient agent with constraint_type flag and learnable lambdas."""

    def __init__(
        self,
        env,
        hidden_dim=128,
        lr=1e-3,
        reward_function="social_welfare",
        constraint_type="wealth",
        lambda_wealth=2.0,
        lambda_approval=2.0,
        lambda_lr=1e-2,
        use_amp=True,
    ):
        self.env = env
        self.reward_func_name = reward_function
        self.reward_function = getattr(RewardFunction, reward_function)
        self.constraint_type = constraint_type
        self.lambda_wealth = lambda_wealth
        self.lambda_approval = lambda_approval

        # CUDA setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {self.device}")

        # Mixed precision training
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        # if self.use_amp:
        # print(f"  Mixed precision (AMP): Enabled")

        # Policy network
        self.policy_net = self._build_network(12, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Learnable lambdas for all modes (except utilitarian_profit)
        self.learnable_lambdas = None
        self.lambda_optimizer = None
        if reward_function != "utilitarian_profit":
            self.learnable_lambdas = LearnableLambdas(
                constraint_type=constraint_type,
                init_lambda_wealth=lambda_wealth,
                init_lambda_approval=lambda_approval,
            ).to(self.device)
            self.lambda_optimizer = optim.Adam(
                self.learnable_lambdas.parameters(), lr=lambda_lr
            )
            # print(f"  Learnable lambdas enabled for '{constraint_type}' (lr={lambda_lr})")

        self.gamma = 0.99
        self.episode_rewards = []
        self.per_step_rewards = []
        self.lambda_history = {"wealth": [], "approval": []}

        # Episode-level metrics tracking (same as testing)
        self.episode_metrics = {
            "episode": [],
            "mu_M_start": [],
            "mu_M_end": [],
            "mu_F_start": [],
            "mu_F_end": [],
            "delta_mu_M": [],
            "delta_mu_F": [],
            "rho_episode": [],  # Inequality ratio per episode: Δμ_M / Δμ_F
            "R_M": [],  # Long-term social welfare for males
            "R_F": [],  # Long-term social welfare for females
            "R_total": [],  # Average of R_M and R_F
            "approval_rate_M": [],
            "approval_rate_F": [],
            "success_prob_M": [],
            "success_prob_F": [],
            "wealth_gap": [],
            "total_profit": [],
            "cumulative_time": [],
        }

        # Track initial wealth for R_g calculation
        self.initial_mu_M = None
        self.initial_mu_F = None
        self.cumulative_time = 0.0
        self.total_episodes_completed = 0

    def _build_network(self, input_dim, hidden_dim):
        class PolicyNet(nn.Module):
            def __init__(self, input_dim, hidden_dim):
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

        return PolicyNet(input_dim, hidden_dim)

    def get_action(self, obs):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                alpha, beta = self.policy_net(obs_tensor)

        dist = Beta(alpha, beta)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.cpu().item(), log_prob.cpu().item()

    def _get_current_lambdas(self):
        """Get current lambda values (learnable or fixed)."""
        if self.learnable_lambdas is not None:
            return (
                self.learnable_lambdas.lambda_wealth.item(),
                self.learnable_lambdas.lambda_approval.item(),
            )
        return self.lambda_wealth, self.lambda_approval

    def train_episode(self):
        obs, _ = self.env.reset()

        # Store episode START wealth values
        mu_M_start = self.env.mu_R
        mu_F_start = self.env.mu_B

        # Store initial wealth on first episode (for R_g calculation)
        if self.initial_mu_M is None:
            self.initial_mu_M = mu_M_start
            self.initial_mu_F = mu_F_start

        states, actions, log_probs, rewards = [], [], [], []
        done = False

        # Get current lambdas
        lambda_w, lambda_a = self._get_current_lambdas()

        while not done:
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            states.append(obs_tensor)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                alpha, beta = self.policy_net(obs_tensor.unsqueeze(0))

            dist = Beta(alpha, beta)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            actions.append(action)
            log_probs.append(log_prob)

            next_obs, _, terminated, truncated, info = self.env.step(
                action.cpu().numpy()
            )
            done = terminated or truncated

            # Compute reward with current lambdas
            reward = self.reward_function(
                self.env,
                action.cpu().item(),
                info,
                constraint_type=self.constraint_type,
                lambda_wealth=lambda_w,
                lambda_approval=lambda_a,
            )
            rewards.append(reward)
            self.per_step_rewards.append(reward)

            obs = next_obs

        # Compute returns with discount
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        loss = torch.stack(policy_loss).sum()

        # Backward pass for policy
        self.optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

        # Update learnable lambdas for all modes (except utilitarian)
        if self.learnable_lambdas is not None:
            self._update_lambdas()

        episode_reward = sum(rewards)
        self.episode_rewards.append(episode_reward)

        # Track lambda history
        lambda_w, lambda_a = self._get_current_lambdas()
        self.lambda_history["wealth"].append(lambda_w)
        self.lambda_history["approval"].append(lambda_a)

        # ===== Record Episode-Level Metrics =====
        self.total_episodes_completed += 1
        self.cumulative_time += self.env.T  # Add episode duration

        # Get episode END wealth values
        mu_M_end = self.env.mu_R
        mu_F_end = self.env.mu_B

        # Calculate wealth changes
        delta_mu_M = mu_M_end - mu_M_start
        delta_mu_F = mu_F_end - mu_F_start

        # Calculate inequality ratio: ρ = Δμ_M / Δμ_F
        if abs(delta_mu_F) > 1e-8:
            rho_episode = delta_mu_M / delta_mu_F
        else:
            rho_episode = 0.0 if abs(delta_mu_M) < 1e-8 else np.sign(delta_mu_M) * 100.0

        # Calculate long-term social welfare: R_g(t) = (μ_g,t - μ_g,0) / t
        if self.cumulative_time > 1e-8:
            R_M = (mu_M_end - self.initial_mu_M) / self.cumulative_time
            R_F = (mu_F_end - self.initial_mu_F) / self.cumulative_time
        else:
            R_M = R_F = 0.0

        # Approval rates and success probabilities
        approval_rate_M = self.env.total_loans_R / max(self.env.total_applications_R, 1)
        approval_rate_F = self.env.total_loans_B / max(self.env.total_applications_B, 1)
        success_prob_M = 1.0 - self.env.total_defaults_R / max(
            self.env.total_loans_R, 1
        )
        success_prob_F = 1.0 - self.env.total_defaults_B / max(
            self.env.total_loans_B, 1
        )

        # Total profit this episode
        total_profit = sum(self.env.history["profit"])

        # Store all episode metrics
        m = self.episode_metrics
        m["episode"].append(self.total_episodes_completed)
        m["mu_M_start"].append(mu_M_start)
        m["mu_M_end"].append(mu_M_end)
        m["mu_F_start"].append(mu_F_start)
        m["mu_F_end"].append(mu_F_end)
        m["delta_mu_M"].append(delta_mu_M)
        m["delta_mu_F"].append(delta_mu_F)
        m["rho_episode"].append(rho_episode)
        m["R_M"].append(R_M)
        m["R_F"].append(R_F)
        m["R_total"].append((R_M + R_F) / 2)
        m["approval_rate_M"].append(approval_rate_M)
        m["approval_rate_F"].append(approval_rate_F)
        m["success_prob_M"].append(success_prob_M)
        m["success_prob_F"].append(success_prob_F)
        m["wealth_gap"].append(mu_M_end - mu_F_end)
        m["total_profit"].append(total_profit)
        m["cumulative_time"].append(self.cumulative_time)

        return episode_reward

    def get_episode_metrics_dataframe(self):
        """Return episode-level metrics as a pandas DataFrame."""
        return pd.DataFrame(self.episode_metrics)

    def _update_lambdas(self):
        """
        Update learnable lambdas to maximize constraint satisfaction.
        We want to increase lambda if the constraint is violated (gap is high).
        """
        # Compute current constraint violations
        approval_rate_M = self.env.total_loans_R / max(self.env.total_applications_R, 1)
        approval_rate_F = self.env.total_loans_B / max(self.env.total_applications_B, 1)

        wealth_gap = abs(self.env.mu_R - self.env.mu_B)
        rate_gap = abs(approval_rate_M - approval_rate_F)

        self.lambda_optimizer.zero_grad()

        # Build loss based on constraint type
        if self.constraint_type == "wealth":
            # Only optimize lambda_wealth
            lambda_wealth_tensor = self.learnable_lambdas.lambda_wealth
            lambda_loss = -(lambda_wealth_tensor * wealth_gap)

        elif self.constraint_type == "approval_rate":
            # Only optimize lambda_approval
            lambda_approval_tensor = self.learnable_lambdas.lambda_approval
            lambda_loss = -(lambda_approval_tensor * rate_gap)

        elif self.constraint_type == "both":
            # Optimize both lambdas
            lambda_wealth_tensor = self.learnable_lambdas.lambda_wealth
            lambda_approval_tensor = self.learnable_lambdas.lambda_approval
            lambda_loss = -(
                lambda_wealth_tensor * wealth_gap + lambda_approval_tensor * rate_gap
            )
        else:
            return

        lambda_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.learnable_lambdas.parameters(), 1.0)
        self.lambda_optimizer.step()

    def save_model(self, filepath):
        """
        Save policy network weights and lambda parameters.

        Args:
            filepath: Path to save the model (e.g., 'model.pt')
        """
        save_dict = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "reward_function": self.reward_func_name,
            "constraint_type": self.constraint_type,
            "episode_rewards": self.episode_rewards,
            "lambda_history": self.lambda_history,
            "episode_metrics": self.episode_metrics,
            "initial_mu_M": self.initial_mu_M,
            "initial_mu_F": self.initial_mu_F,
            "cumulative_time": self.cumulative_time,
            "total_episodes_completed": self.total_episodes_completed,
        }

        # Save learnable lambdas if they exist
        if self.learnable_lambdas is not None:
            save_dict["learnable_lambdas_state_dict"] = (
                self.learnable_lambdas.state_dict()
            )
            save_dict["lambda_optimizer_state_dict"] = (
                self.lambda_optimizer.state_dict()
            )
            save_dict["final_lambda_wealth"] = (
                self.learnable_lambdas.lambda_wealth.item()
            )
            save_dict["final_lambda_approval"] = (
                self.learnable_lambdas.lambda_approval.item()
            )
        else:
            save_dict["final_lambda_wealth"] = self.lambda_wealth
            save_dict["final_lambda_approval"] = self.lambda_approval

        torch.save(save_dict, filepath)
        print(f"  Model saved to {filepath}")
        return filepath

    def load_model(self, filepath):
        """
        Load policy network weights and lambda parameters.

        Args:
            filepath: Path to the saved model
        """
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
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "episode_rewards" in checkpoint:
            self.episode_rewards = checkpoint["episode_rewards"]
        if "lambda_history" in checkpoint:
            self.lambda_history = checkpoint["lambda_history"]
        if "episode_metrics" in checkpoint:
            self.episode_metrics = checkpoint["episode_metrics"]
        if "initial_mu_M" in checkpoint:
            self.initial_mu_M = checkpoint["initial_mu_M"]
        if "initial_mu_F" in checkpoint:
            self.initial_mu_F = checkpoint["initial_mu_F"]
        if "cumulative_time" in checkpoint:
            self.cumulative_time = checkpoint["cumulative_time"]
        if "total_episodes_completed" in checkpoint:
            self.total_episodes_completed = checkpoint["total_episodes_completed"]

        # Load learnable lambdas if they exist
        if (
            self.learnable_lambdas is not None
            and "learnable_lambdas_state_dict" in checkpoint
        ):
            self.learnable_lambdas.load_state_dict(
                checkpoint["learnable_lambdas_state_dict"]
            )
            if "lambda_optimizer_state_dict" in checkpoint:
                self.lambda_optimizer.load_state_dict(
                    checkpoint["lambda_optimizer_state_dict"]
                )

        print(f"  Model loaded from {filepath}")
        print(f"    Reward function: {checkpoint.get('reward_function', 'unknown')}")
        print(f"    Constraint type: {checkpoint.get('constraint_type', 'unknown')}")
        print(f"    Final λ_wealth: {checkpoint.get('final_lambda_wealth', 'N/A')}")
        print(f"    Final λ_approval: {checkpoint.get('final_lambda_approval', 'N/A')}")
        if "episode_metrics" in checkpoint and checkpoint["episode_metrics"].get(
            "episode"
        ):
            print(
                f"    Episodes completed: {len(checkpoint['episode_metrics']['episode'])}"
            )

        return checkpoint

    def train(self, num_episodes=100):
        print(
            f"Training with {self.reward_func_name} ({self.constraint_type} constraint)..."
        )

        for episode in range(num_episodes):
            episode_reward = self.train_episode()

            if episode % 20 == 0:
                avg_reward = (
                    np.mean(self.episode_rewards[-20:])
                    if len(self.episode_rewards) >= 20
                    else episode_reward
                )
                lambda_w, lambda_a = self._get_current_lambdas()

                # Get latest episode metrics
                rho = (
                    self.episode_metrics["rho_episode"][-1]
                    if self.episode_metrics["rho_episode"]
                    else 0
                )
                R_M = (
                    self.episode_metrics["R_M"][-1]
                    if self.episode_metrics["R_M"]
                    else 0
                )
                R_F = (
                    self.episode_metrics["R_F"][-1]
                    if self.episode_metrics["R_F"]
                    else 0
                )

                # Only print every 10 episodes or last episode
                if episode % 10 == 0 or episode == num_episodes - 1:
                    if self.learnable_lambdas is not None:
                        if self.constraint_type == "wealth":
                            print(
                                f"  Episode {episode}: Reward={episode_reward:.3f}, Avg={avg_reward:.3f}, "
                                f"λ_w={lambda_w:.4f}, ρ={rho:.3f}, R_M={R_M:.4f}, R_F={R_F:.4f}"
                            )
                        elif self.constraint_type == "approval_rate":
                            print(
                                f"  Episode {episode}: Reward={episode_reward:.3f}, Avg={avg_reward:.3f}, "
                                f"λ_a={lambda_a:.4f}, ρ={rho:.3f}, R_M={R_M:.4f}, R_F={R_F:.4f}"
                            )
                        else:  # both
                            print(
                                f"  Episode {episode}: Reward={episode_reward:.3f}, Avg={avg_reward:.3f}, "
                                f"λ_w={lambda_w:.4f}, λ_a={lambda_a:.4f}, ρ={rho:.3f}"
                            )
                    else:
                        print(
                            f"  Episode {episode}: Reward={episode_reward:.3f}, Avg={avg_reward:.3f}, "
                            f"ρ={rho:.3f}, R_M={R_M:.4f}, R_F={R_F:.4f}"
                        )

        return self.episode_rewards


def save_results(results, reward_func, constraint_type, seed, save_dir="./results"):
    """Save results for a specific reward function, constraint type, and seed."""
    import os

    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/{reward_func}_{constraint_type}_seed{seed}_{timestamp}.pkl"

    save_data = {
        "reward_function": reward_func,
        "constraint_type": constraint_type,
        "seed": seed,
        "history": results["history"],
        "episode_rewards": results["episode_rewards"],
        "lambda_history": results.get("lambda_history", {}),
        "timestamp": timestamp,
    }

    with open(filename, "wb") as f:
        pickle.dump(save_data, f)

    print(f"  Saved results to {filename}")
    return filename


def save_lambda_history(
    lambda_history,
    reward_func,
    constraint_type,
    seed,
    save_dir="./lambda_trajectories",
    format="both",
):
    """
    Save lambda trajectory arrays separately for easy analysis.

    Args:
        lambda_history: dict with 'wealth' and 'approval' lists
        reward_func: Name of reward function
        constraint_type: 'wealth', 'approval_rate', or 'both'
        seed: Random seed
        save_dir: Directory to save
        format: 'csv', 'npy', or 'both'
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{save_dir}/{reward_func}_{constraint_type}_seed{seed}_{timestamp}"

    # Create DataFrame
    df = pd.DataFrame(
        {
            "episode": range(len(lambda_history.get("wealth", []))),
            "lambda_wealth": lambda_history.get("wealth", []),
            "lambda_approval": lambda_history.get("approval", []),
        }
    )

    saved_files = []

    if format in ["csv", "both"]:
        csv_file = f"{base_filename}_lambda.csv"
        df.to_csv(csv_file, index=False)
        saved_files.append(csv_file)
        print(f"  Lambda history saved to {csv_file}")

    if format in ["npy", "both"]:
        npy_file = f"{base_filename}_lambda.npy"
        np.save(
            npy_file,
            {
                "wealth": np.array(lambda_history.get("wealth", [])),
                "approval": np.array(lambda_history.get("approval", [])),
            },
        )
        saved_files.append(npy_file)
        print(f"  Lambda history saved to {npy_file}")

    return saved_files


def save_episode_metrics(
    episode_metrics,
    reward_func,
    constraint_type,
    seed,
    save_dir="./episode_metrics",
    format="both",
):
    """
    Save episode-level metrics separately for analysis.

    Args:
        episode_metrics: dict with episode-level metrics
        reward_func: Name of reward function
        constraint_type: 'wealth', 'approval_rate', or 'both'
        seed: Random seed
        save_dir: Directory to save
        format: 'csv', 'npy', or 'both'
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{save_dir}/{reward_func}_{constraint_type}_seed{seed}_{timestamp}"

    # Create DataFrame
    df = pd.DataFrame(episode_metrics)

    saved_files = []

    if format in ["csv", "both"]:
        csv_file = f"{base_filename}_episode_metrics.csv"
        df.to_csv(csv_file, index=False)
        saved_files.append(csv_file)
        print(f"  Episode metrics saved to {csv_file}")

    if format in ["npy", "both"]:
        npy_file = f"{base_filename}_episode_metrics.npy"
        np.save(npy_file, episode_metrics)
        saved_files.append(npy_file)
        print(f"  Episode metrics saved to {npy_file}")

    return saved_files


def plot_episode_metrics(
    episode_metrics, reward_func, constraint_type, seed, save_path=None, show=True
):
    """
    Plot episode-level metrics: ρ, R_M, R_F, wealth gap, etc.

    Args:
        episode_metrics: dict with episode-level metrics
        reward_func: Name of reward function
        constraint_type: 'wealth', 'approval_rate', or 'both'
        seed: Random seed
        save_path: Path to save figure
        show: Whether to display
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    episodes = episode_metrics.get("episode", [])
    if not episodes:
        print("  No episode metrics to plot")
        return None

    # 1. Inequality Ratio (ρ)
    ax = axes[0, 0]
    rho = episode_metrics.get("rho_episode", [])
    rho_clipped = np.clip(rho, -10, 10)
    ax.plot(episodes, rho_clipped, "b-", linewidth=2, label="ρ per episode")
    ax.axhline(y=1.0, color="red", linestyle="--", label="Equal growth (ρ=1)")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("ρ")
    ax.set_title("Inequality Ratio: ρ = Δμ_M / Δμ_F")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Long-term Social Welfare (R_M, R_F)
    ax = axes[0, 1]
    R_M = episode_metrics.get("R_M", [])
    R_F = episode_metrics.get("R_F", [])
    R_total = episode_metrics.get("R_total", [])
    ax.plot(episodes, R_M, "b-", linewidth=2, label="R_M (Male)")
    ax.plot(episodes, R_F, "r-", linewidth=2, label="R_F (Female)")
    ax.plot(episodes, R_total, "g--", linewidth=2, label="R_total")
    ax.axhline(y=0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("R_g = (μ_g,t - μ_g,0) / t")
    ax.set_title("Long-term Social Welfare")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Wealth Gap
    ax = axes[0, 2]
    wealth_gap = episode_metrics.get("wealth_gap", [])
    ax.plot(episodes, wealth_gap, "purple", linewidth=2)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Wealth Gap ($1000s)")
    ax.set_title("Wealth Gap: μ_M - μ_F")
    ax.grid(True, alpha=0.3)

    # 4. Mean Wealth Changes (Δμ)
    ax = axes[1, 0]
    delta_mu_M = episode_metrics.get("delta_mu_M", [])
    delta_mu_F = episode_metrics.get("delta_mu_F", [])
    ax.plot(episodes, delta_mu_M, "b-", linewidth=2, label="Δμ_M")
    ax.plot(episodes, delta_mu_F, "r-", linewidth=2, label="Δμ_F")
    ax.axhline(y=0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Δμ ($1000s)")
    ax.set_title("Wealth Change per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Approval Rates
    ax = axes[1, 1]
    approval_rate_M = episode_metrics.get("approval_rate_M", [])
    approval_rate_F = episode_metrics.get("approval_rate_F", [])
    ax.plot(episodes, approval_rate_M, "b-", linewidth=2, label="Male")
    ax.plot(episodes, approval_rate_F, "r-", linewidth=2, label="Female")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Approval Rate")
    ax.set_title("Approval Rates")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Success Probabilities
    ax = axes[1, 2]
    success_prob_M = episode_metrics.get("success_prob_M", [])
    success_prob_F = episode_metrics.get("success_prob_F", [])
    ax.plot(episodes, success_prob_M, "b-", linewidth=2, label="Male")
    ax.plot(episodes, success_prob_F, "r-", linewidth=2, label="Female")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Probability")
    ax.set_title("Success Probabilities (1 - Default Rate)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Episode-Level Metrics: {reward_func} ({constraint_type}) - Seed {seed}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Episode metrics plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_lambda_trajectory(
    lambda_history, reward_func, constraint_type, seed, save_path=None, show=True
):
    """
    Plot lambda multiplier trajectories across training episodes.

    Args:
        lambda_history: dict with 'wealth' and 'approval' lists
        reward_func: Name of reward function
        constraint_type: 'wealth', 'approval_rate', or 'both'
        seed: Random seed
        save_path: Path to save figure (None to not save)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    episodes = range(len(lambda_history.get("wealth", [])))

    # Plot lambda_wealth
    ax1 = axes[0]
    lambda_w = lambda_history.get("wealth", [])
    if lambda_w:
        ax1.plot(episodes, lambda_w, "b-", linewidth=2, label="λ_wealth")
        ax1.axhline(
            y=lambda_w[0],
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Initial: {lambda_w[0]:.4f}",
        )
        ax1.axhline(
            y=lambda_w[-1],
            color="green",
            linestyle="--",
            alpha=0.5,
            label=f"Final: {lambda_w[-1]:.4f}",
        )

        # Add smoothed line
        if len(lambda_w) > 10:
            window = min(20, len(lambda_w) // 5)
            smoothed = np.convolve(lambda_w, np.ones(window) / window, mode="valid")
            ax1.plot(
                range(window - 1, len(lambda_w)),
                smoothed,
                "r-",
                linewidth=1.5,
                alpha=0.7,
                label="Smoothed",
            )

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("λ_wealth")
    ax1.set_title(f"Lambda Wealth Trajectory\n{reward_func} ({constraint_type})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot lambda_approval
    ax2 = axes[1]
    lambda_a = lambda_history.get("approval", [])
    if lambda_a:
        ax2.plot(episodes, lambda_a, "purple", linewidth=2, label="λ_approval")
        ax2.axhline(
            y=lambda_a[0],
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Initial: {lambda_a[0]:.4f}",
        )
        ax2.axhline(
            y=lambda_a[-1],
            color="green",
            linestyle="--",
            alpha=0.5,
            label=f"Final: {lambda_a[-1]:.4f}",
        )

        # Add smoothed line
        if len(lambda_a) > 10:
            window = min(20, len(lambda_a) // 5)
            smoothed = np.convolve(lambda_a, np.ones(window) / window, mode="valid")
            ax2.plot(
                range(window - 1, len(lambda_a)),
                smoothed,
                "r-",
                linewidth=1.5,
                alpha=0.7,
                label="Smoothed",
            )

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("λ_approval")
    ax2.set_title(f"Lambda Approval Trajectory\n{reward_func} ({constraint_type})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Lagrangian Multiplier Evolution - Seed {seed}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Lambda trajectory plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_lambda_trajectories_comparison(
    results, constraint_type, seed, save_path=None, show=True
):
    """
    Plot lambda trajectories for multiple reward functions on same plot.

    Args:
        results: dict of {reward_func: {'lambda_history': {...}, ...}}
        constraint_type: 'wealth', 'approval_rate', or 'both'
        seed: Random seed
        save_path: Path to save figure
        show: Whether to display
    """
    colors = {
        "social_welfare": "purple",
        "rawlsian_maximin": "blue",
        "fairness_lagrangian": "green",
        "utilitarian_profit": "red",
    }

    labels = {
        "social_welfare": "Social Welfare",
        "rawlsian_maximin": "Rawlsian Maximin",
        "fairness_lagrangian": "Fairness Lagrangian",
        "utilitarian_profit": "Utilitarian Profit",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot lambda_wealth
    ax1 = axes[0]
    for rf, data in results.items():
        if rf == "utilitarian_profit":
            continue
        lambda_history = data.get("lambda_history", {})
        lambda_w = lambda_history.get("wealth", [])
        if lambda_w:
            episodes = range(len(lambda_w))
            ax1.plot(
                episodes,
                lambda_w,
                color=colors[rf],
                linewidth=2,
                label=f"{labels[rf]} (final: {lambda_w[-1]:.4f})",
            )

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("λ_wealth")
    ax1.set_title("Lambda Wealth Trajectories")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot lambda_approval
    ax2 = axes[1]
    for rf, data in results.items():
        if rf == "utilitarian_profit":
            continue
        lambda_history = data.get("lambda_history", {})
        lambda_a = lambda_history.get("approval", [])
        if lambda_a:
            episodes = range(len(lambda_a))
            ax2.plot(
                episodes,
                lambda_a,
                color=colors[rf],
                linewidth=2,
                label=f"{labels[rf]} (final: {lambda_a[-1]:.4f})",
            )

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("λ_approval")
    ax2.set_title("Lambda Approval Trajectories")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Lagrangian Multiplier Comparison - {constraint_type} - Seed {seed}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Lambda comparison plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_single_seed_results(results, theta_learner, seed, constraint_type):
    """Plot results for a single seed."""

    fig = plt.figure(figsize=(28, 22))
    gs = gridspec.GridSpec(5, 4, figure=fig)

    colors = {
        "social_welfare": "purple",
        "rawlsian_maximin": "blue",
        "fairness_lagrangian": "green",
        "utilitarian_profit": "red",
    }

    labels = {
        "social_welfare": "Social Welfare",
        "rawlsian_maximin": "Rawlsian Maximin",
        "fairness_lagrangian": "Fairness Lagrangian",
        "utilitarian_profit": "Utilitarian Profit",
    }

    reward_funcs = list(results.keys())

    # Learning curves
    ax = fig.add_subplot(gs[0, :2])
    for rf in reward_funcs:
        rewards = results[rf]["episode_rewards"]
        window = 10
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        else:
            smoothed = rewards
        ax.plot(smoothed, label=labels[rf], color=colors[rf], linewidth=2.5)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(f"Learning Curves (Seed {seed}, Constraint: {constraint_type})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Lambda evolution (for all modes with learnable lambdas)
    ax = fig.add_subplot(gs[0, 2:])
    has_lambda_data = False
    for rf in reward_funcs:
        if (
            rf != "utilitarian_profit"
            and "lambda_history" in results[rf]
            and results[rf]["lambda_history"]
        ):
            lambda_w = results[rf]["lambda_history"].get("wealth", [])
            lambda_a = results[rf]["lambda_history"].get("approval", [])
            if constraint_type == "wealth" and lambda_w:
                ax.plot(
                    lambda_w,
                    label=f"{labels[rf]} λ_wealth",
                    color=colors[rf],
                    linewidth=2,
                    linestyle="-",
                )
                has_lambda_data = True
            elif constraint_type == "approval_rate" and lambda_a:
                ax.plot(
                    lambda_a,
                    label=f"{labels[rf]} λ_approval",
                    color=colors[rf],
                    linewidth=2,
                    linestyle="--",
                )
                has_lambda_data = True
            elif constraint_type == "both":
                if lambda_w:
                    ax.plot(
                        lambda_w,
                        label=f"{labels[rf]} λ_wealth",
                        color=colors[rf],
                        linewidth=2,
                        linestyle="-",
                    )
                    has_lambda_data = True
                if lambda_a:
                    ax.plot(
                        lambda_a,
                        label=f"{labels[rf]} λ_approval",
                        color=colors[rf],
                        linewidth=2,
                        linestyle="--",
                    )
                    has_lambda_data = True

    if has_lambda_data:
        ax.set_xlabel("Episode")
        ax.set_ylabel("Lambda Value")
        ax.set_title(f"Learnable Lambda Evolution ({constraint_type})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        # Fallback: show wealth gap if no lambda data
        for rf in reward_funcs:
            time = results[rf]["history"]["time"]
            gap = results[rf]["history"]["wealth_gap"]
            ax.plot(time, gap, label=labels[rf], color=colors[rf], linewidth=2)

        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("Time")
        ax.set_ylabel("Wealth Gap ($1000s)")
        ax.set_title("Gender Wealth Gap Evolution: μ_R(t) - μ_B(t)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Wealth gap
    ax = fig.add_subplot(gs[1, :2])
    for rf in reward_funcs:
        time = results[rf]["history"]["time"]
        gap = results[rf]["history"]["wealth_gap"]
        ax.plot(time, gap, label=labels[rf], color=colors[rf], linewidth=2)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Wealth Gap ($1000s)")
    ax.set_title("Gender Wealth Gap Evolution: μ_R(t) - μ_B(t)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Approval rates
    for i, (group, key) in enumerate(
        [("Male (R)", "approval_rate_R"), ("Female (B)", "approval_rate_B")]
    ):
        ax = fig.add_subplot(gs[1, 2 + i])
        for rf in reward_funcs:
            time = results[rf]["history"]["time"]
            rates = results[rf]["history"][key]
            ax.plot(time, rates, label=labels[rf], color=colors[rf], linewidth=2)

        ax.set_xlabel("Time")
        ax.set_ylabel("Approval Rate E[L_g(t)]")
        ax.set_title(f"{group} Approval Rates")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Approval disparity
    ax = fig.add_subplot(gs[2, :2])
    for rf in reward_funcs:
        time = results[rf]["history"]["time"]
        disp = results[rf]["history"]["approval_disparity"]
        ax.plot(time, disp, label=labels[rf], color=colors[rf], linewidth=2)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Approval Rate Gap (Male - Female)")
    ax.set_title("Approval Rate Disparity")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Mean wealth evolution
    ax = fig.add_subplot(gs[2, 2:])
    for rf in reward_funcs:
        time = results[rf]["history"]["time"]
        mu_R = results[rf]["history"]["mu_R"]
        mu_B = results[rf]["history"]["mu_B"]
        ax.plot(
            time,
            mu_R,
            label=f"{labels[rf]} Male",
            color=colors[rf],
            linewidth=2,
            linestyle="-",
        )
        ax.plot(
            time,
            mu_B,
            label=f"{labels[rf]} Female",
            color=colors[rf],
            linewidth=2,
            linestyle="--",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Mean Wealth ($1000s)")
    ax.set_title("Mean Wealth Evolution by Group")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Default rates
    for i, (group, key) in enumerate(
        [("Male (R)", "default_rate_R"), ("Female (B)", "default_rate_B")]
    ):
        ax = fig.add_subplot(gs[3, i])
        for rf in reward_funcs:
            time = results[rf]["history"]["time"]
            rates = results[rf]["history"][key]
            ax.plot(time, rates, label=labels[rf], color=colors[rf], linewidth=2)

        ax.set_xlabel("Time")
        ax.set_ylabel("Default Rate")
        ax.set_title(f"{group} Default Rates")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Cumulative profit
    ax = fig.add_subplot(gs[3, 2:])
    for rf in reward_funcs:
        time = results[rf]["history"]["time"]
        cum_profit = np.cumsum(results[rf]["history"]["profit"])
        ax.plot(time, cum_profit, label=labels[rf], color=colors[rf], linewidth=2.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Profit ($1000s)")
    ax.set_title("System Profit")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Final comparisons
    metrics = [
        ("Final Wealth Gap", "wealth_gap", -1),
        ("Final Approval Disparity", "approval_disparity", -1),
        ("Total Profit", "profit", None),
    ]

    for i, (title, key, idx) in enumerate(metrics):
        ax = fig.add_subplot(gs[4, i])

        if idx is not None:
            values = [results[rf]["history"][key][idx] for rf in reward_funcs]
        else:
            values = [sum(results[rf]["history"][key]) for rf in reward_funcs]

        x = np.arange(len(reward_funcs))
        ax.bar(x, values, color=[colors[rf] for rf in reward_funcs])
        ax.set_xticks(x)
        ax.set_xticklabels(
            [labels[rf].replace(" ", "\n") for rf in reward_funcs], fontsize=8
        )
        ax.set_ylabel("$1000s" if "Profit" in title or "Gap" in title else "Rate")
        ax.set_title(title)
        if "Gap" in title or "Disparity" in title:
            ax.axhline(y=0, color="black", linestyle="--")
        ax.grid(True, alpha=0.3, axis="y")

    # Parameters
    ax = fig.add_subplot(gs[4, 3])
    ax.axis("off")
    text = f"Experiment Parameters:\n"
    text += "=" * 25 + "\n\n"
    text += f"Seed: {seed}\n"
    text += f"Constraint: {constraint_type}\n\n"
    text += f"θ_S = {theta_learner.theta_S:+.4f}\n"
    text += f"θ_X = {theta_learner.theta_X:+.4f}\n"
    text += f"b = {theta_learner.b:+.4f}\n\n"
    text += f"Per-Individual (FIXED):\n"
    text += f"  Default: U[{theta_learner.default_rate_min:.0%},{theta_learner.default_rate_max:.0%}]\n"
    text += f"  Loan: N(${theta_learner.loan_amount_mean:.0f}k,${theta_learner.loan_amount_std:.0f}k)\n"

    ax.text(
        0.1,
        0.5,
        text,
        fontsize=9,
        family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.suptitle(
        f"Adult Income RL Analysis - Seed {seed}\nConstraint Type: {constraint_type}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        f"./results/adult_income_rl_{constraint_type}_seed{seed}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


def run_single_reward_function(
    data_filepath=None,
    num_episodes=100,
    seed=0,
    reward_function="social_welfare",
    constraint_type="wealth",
    lambda_wealth=2.0,
    lambda_approval=2.0,
    lambda_lr=1e-2,
    save_weights=True,
    weights_dir="./weights",
    save_lambda=True,
    lambda_dir="./lambda_trajectories",
    plot_lambda=True,
    save_episode_metrics_flag=True,
    episode_metrics_dir="./episode_metrics",
    plot_episode_metrics_flag=True,
):
    """
    Run experiment with a SINGLE reward function and optionally save weights.

    Args:
        data_filepath: Path to adult.csv (None for auto-download)
        num_episodes: Number of training episodes
        seed: Random seed
        reward_function: One of 'social_welfare', 'rawlsian_maximin', 'fairness_lagrangian', 'utilitarian_profit'
        constraint_type: 'approval_rate', 'wealth', or 'both'
        lambda_wealth: Initial lambda for wealth constraint
        lambda_approval: Initial lambda for approval rate constraint
        lambda_lr: Learning rate for lambda optimization
        save_weights: Whether to save model weights
        weights_dir: Directory to save weights
        save_lambda: Whether to save lambda trajectory arrays
        lambda_dir: Directory to save lambda trajectories
        plot_lambda: Whether to plot lambda trajectories
        save_episode_metrics_flag: Whether to save episode-level metrics
        episode_metrics_dir: Directory to save episode metrics
        plot_episode_metrics_flag: Whether to plot episode-level metrics

    Returns:
        agent: Trained PolicyGradientAgent
        env: Final environment state
        theta_learner: Learned transition parameters
    """
    import os

    os.makedirs(weights_dir, exist_ok=True)

    print("=" * 80)
    print(f"SINGLE REWARD FUNCTION EXPERIMENT")
    print(f"  Reward: {reward_function.upper()}")
    print(f"  Constraint: {constraint_type.upper()}")
    print(f"  Seed: {seed}")
    print("=" * 80)

    # Set random seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"\nCUDA Available: {torch.cuda.get_device_name(0)}")
    else:
        print("\nUsing CPU")

    # Load data
    print(f"\n[STEP 1] Loading Adult Income Data...")
    loader = AdultIncomeDataLoader(filepath=data_filepath, sample_size=20000)
    loader.load_data()
    loader.preprocess()

    # Learn θ parameters
    print(f"\n[STEP 2] Learning θ Parameters...")
    theta_learner = TransitionParameterLearner(
        default_rate_min=0.02, default_rate_max=0.15
    )
    theta_learner.fit(loader.data)

    # Create environment
    print(f"\n[STEP 3] Creating Environment...")
    env = IncomeEnvironment(
        theta_params=theta_learner,
        initial_wealth_male=loader.male_data["X"].values,
        initial_wealth_female=loader.female_data["X"].values,
        N_male=3000,
        N_female=3000,
        T=100,
        dt=0.5,
        seed=seed,
    )

    # Create agent
    print(f"\n[STEP 4] Training Agent...")
    agent = PolicyGradientAgent(
        env,
        reward_function=reward_function,
        constraint_type=constraint_type,
        lambda_wealth=lambda_wealth,
        lambda_approval=lambda_approval,
        lambda_lr=lambda_lr,
    )

    # Train
    agent.train(num_episodes=num_episodes)

    # Run final test episode
    print(f"\n[STEP 5] Running Test Episode...")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = agent.get_action(obs)
        obs, _, terminated, truncated, _ = env.step(np.array([action]))
        done = terminated or truncated

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE: {reward_function} ({constraint_type})")
    print("=" * 60)
    print(f"  Initial wealth gap: ${env.history['wealth_gap'][0]:.3f}k")
    print(f"  Final wealth gap: ${env.history['wealth_gap'][-1]:.3f}k")
    print(f"  Approval disparity: {env.history['approval_disparity'][-1]:.4f}")
    print(f"  Total profit: ${sum(env.history['profit']):.3f}k")

    if reward_function != "utilitarian_profit":
        final_lambda_w, final_lambda_a = agent._get_current_lambdas()
        if constraint_type == "wealth":
            print(f"  Final λ_wealth: {final_lambda_w:.4f}")
        elif constraint_type == "approval_rate":
            print(f"  Final λ_approval: {final_lambda_a:.4f}")
        else:
            print(f"  Final λ_wealth: {final_lambda_w:.4f}")
            print(f"  Final λ_approval: {final_lambda_a:.4f}")

    # Print episode-level metrics summary
    if agent.episode_metrics["rho_episode"]:
        rho_values = np.array(agent.episode_metrics["rho_episode"])
        R_M_values = np.array(agent.episode_metrics["R_M"])
        R_F_values = np.array(agent.episode_metrics["R_F"])
        print(f"\n  --- Episode-Level Metrics (Training) ---")
        print(
            f"  ρ (inequality ratio): mean={np.mean(rho_values):.4f}, std={np.std(rho_values):.4f}"
        )
        print(f"  R_M (male welfare):   final={R_M_values[-1]:.6f}")
        print(f"  R_F (female welfare): final={R_F_values[-1]:.6f}")
        print(
            f"  R_M / R_F:            {R_M_values[-1] / R_F_values[-1]:.4f}"
            if abs(R_F_values[-1]) > 1e-8
            else "  R_M / R_F: N/A"
        )

    # Save weights
    if save_weights:
        print(f"\n[STEP 6] Saving Model Weights...")
        # Use predictable filename for batch processing (no timestamp)
        weights_filename = (
            f"{weights_dir}/{reward_function}_{constraint_type}_seed{seed}.pt"
        )
        agent.save_model(weights_filename)
        print(f"  Saved to: {weights_filename}")

    # Save lambda trajectories
    if save_lambda and reward_function != "utilitarian_profit":
        print(f"\n[STEP 7] Saving Lambda Trajectories...")
        save_lambda_history(
            agent.lambda_history,
            reward_function,
            constraint_type,
            seed,
            save_dir=lambda_dir,
            format="both",
        )

    # Plot lambda trajectories
    if plot_lambda and reward_function != "utilitarian_profit":
        print(f"\n[STEP 8] Plotting Lambda Trajectories...")
        plot_path = f"{lambda_dir}/{reward_function}_{constraint_type}_seed{seed}_lambda_plot.png"
        import os

        os.makedirs(lambda_dir, exist_ok=True)
        plot_lambda_trajectory(
            agent.lambda_history,
            reward_function,
            constraint_type,
            seed,
            save_path=plot_path,
            show=True,
        )

    # Save episode metrics
    if save_episode_metrics_flag:
        print(f"\n[STEP 9] Saving Episode Metrics...")
        os.makedirs(episode_metrics_dir, exist_ok=True)
        save_episode_metrics(
            agent.episode_metrics,
            reward_function,
            constraint_type,
            seed,
            save_dir=episode_metrics_dir,
            format="both",
        )

    # Plot episode metrics
    if plot_episode_metrics_flag:
        print(f"\n[STEP 10] Plotting Episode Metrics...")
        os.makedirs(episode_metrics_dir, exist_ok=True)
        plot_path = f"{episode_metrics_dir}/{reward_function}_{constraint_type}_seed{seed}_episode_metrics.png"
        plot_episode_metrics(
            agent.episode_metrics,
            reward_function,
            constraint_type,
            seed,
            save_path=plot_path,
            show=True,
        )

    return agent, env, theta_learner, loader


def load_trained_agent(weights_path, data_filepath=None, seed=0):
    """
    Load a trained agent from saved weights.

    Args:
        weights_path: Path to saved weights file (.pt)
        data_filepath: Path to adult.csv (needed to recreate environment)
        seed: Random seed

    Returns:
        agent: Loaded PolicyGradientAgent
        env: Environment
        theta_learner: Transition parameters
    """
    print(f"Loading trained agent from {weights_path}...")

    # Load checkpoint to get metadata
    # PyTorch 2.6+ changed default weights_only=True, but our checkpoints contain
    # numpy arrays and other objects, so we need weights_only=False
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions don't have weights_only parameter
        checkpoint = torch.load(weights_path, map_location="cpu")

    reward_function = checkpoint.get("reward_function", "social_welfare")
    constraint_type = checkpoint.get("constraint_type", "wealth")

    print(f"  Reward function: {reward_function}")
    print(f"  Constraint type: {constraint_type}")

    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Load data and create environment
    loader = AdultIncomeDataLoader(filepath=data_filepath, sample_size=20000)
    loader.load_data()
    loader.preprocess()

    theta_learner = TransitionParameterLearner(
        default_rate_min=0.02, default_rate_max=0.15
    )
    theta_learner.fit(loader.data)

    env = IncomeEnvironment(
        theta_params=theta_learner,
        initial_wealth_male=loader.male_data["X"].values,
        initial_wealth_female=loader.female_data["X"].values,
        N_male=3000,
        N_female=3000,
        T=100,
        dt=0.5,
        seed=seed,
    )

    # Create agent with same configuration
    lambda_wealth = checkpoint.get("final_lambda_wealth", 2.0)
    lambda_approval = checkpoint.get("final_lambda_approval", 2.0)

    agent = PolicyGradientAgent(
        env,
        reward_function=reward_function,
        constraint_type=constraint_type,
        lambda_wealth=lambda_wealth,
        lambda_approval=lambda_approval,
    )

    # Load weights
    agent.load_model(weights_path)

    return agent, env, theta_learner, loader


def run_experiment(
    data_filepath=None,
    num_episodes=100,
    seed=0,
    constraint_type="wealth",
    save_results_flag=False,
    save_weights=True,
    weights_dir="./weights",
    save_lambda=False,
    lambda_dir="./lambda_trajectories",
    plot_lambda=False,
):
    """
    Run Adult Income experiment with specified constraint_type.

    Args:
        constraint_type: 'approval_rate', 'wealth', or 'both'
        save_weights: Whether to save model weights for each reward function
        weights_dir: Directory to save weights
        save_lambda: Whether to save lambda trajectory arrays
        lambda_dir: Directory to save lambda trajectories
        plot_lambda: Whether to plot lambda trajectories
    """
    import os

    if save_weights:
        os.makedirs(weights_dir, exist_ok=True)
    if save_lambda:
        os.makedirs(lambda_dir, exist_ok=True)

    print("=" * 80)
    print(
        f"ADULT INCOME EXPERIMENT - SEED {seed} - CONSTRAINT: {constraint_type.upper()}"
    )
    print("=" * 80)

    # Set random seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"\nCUDA Available: {torch.cuda.get_device_name(0)}")
    else:
        print("\nUsing CPU")

    # Load data
    # print(f"\n[STEP 1] Loading Adult Income Data...")
    loader = AdultIncomeDataLoader(filepath=data_filepath, sample_size=20000)
    loader.load_data()
    loader.preprocess()

    # Learn θ parameters
    # print(f"\n[STEP 2] Learning θ Parameters...")
    theta_learner = TransitionParameterLearner(
        default_rate_min=0.02, default_rate_max=0.15
    )
    theta_learner.fit(loader.data)

    reward_functions = [
        "social_welfare",
        "rawlsian_maximin",
        "fairness_lagrangian",
        "utilitarian_profit",
    ]

    results = {}
    saved_files = []

    for reward_func in reward_functions:
        print(f"\n{'=' * 60}")
        print(f"Training: {reward_func.upper().replace('_', ' ')} ({constraint_type})")
        print("=" * 60)

        # Reset seeds
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        env = IncomeEnvironment(
            theta_params=theta_learner,
            initial_wealth_male=loader.male_data["X"].values,
            initial_wealth_female=loader.female_data["X"].values,
            N_male=3000,
            N_female=3000,
            T=100,
            dt=0.5,
            seed=seed,
        )

        # Set lambda values based on reward function
        if reward_func == "utilitarian_profit":
            lambda_wealth, lambda_approval = 0.0, 0.0
        elif reward_func == "social_welfare":
            lambda_wealth, lambda_approval = 2.0, 2.0
        elif reward_func == "rawlsian_maximin":
            lambda_wealth, lambda_approval = 5.0, 5.0
        elif reward_func == "fairness_lagrangian":
            lambda_wealth, lambda_approval = 10.0, 10.0

        agent = PolicyGradientAgent(
            env,
            reward_function=reward_func,
            constraint_type=constraint_type,
            lambda_wealth=lambda_wealth,
            lambda_approval=lambda_approval,
            lambda_lr=1e-2,
        )
        agent.train(num_episodes=num_episodes)

        # Test episode
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = agent.get_action(obs)
            obs, _, terminated, truncated, _ = env.step(np.array([action]))
            done = terminated or truncated

        results[reward_func] = {
            "agent": agent,
            "history": env.history,
            "episode_rewards": agent.episode_rewards,
            "lambda_history": agent.lambda_history,
            "episode_metrics": agent.episode_metrics,
        }

        if save_results_flag:
            saved_file = save_results(
                results[reward_func], reward_func, constraint_type, seed
            )
            saved_files.append(saved_file)

        # Save model weights
        if save_weights:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            weights_filename = f"{weights_dir}/{reward_func}_{constraint_type}_seed{seed}_{timestamp}.pt"
            agent.save_model(weights_filename)
            # Also save a 'latest' version
            # latest_filename = f"{weights_dir}/{reward_func}_{constraint_type}_latest.pt"
            # agent.save_model(latest_filename)

        print(f"\nSummary for {reward_func}:")
        print(f"  Initial wealth gap: ${env.history['wealth_gap'][0]:.3f}k")
        print(f"  Final wealth gap: ${env.history['wealth_gap'][-1]:.3f}k")
        print(f"  Approval disparity: {env.history['approval_disparity'][-1]:.4f}")
        print(f"  Total profit: ${sum(env.history['profit']):.3f}k")
        if reward_func != "utilitarian_profit":
            final_lambda_w, final_lambda_a = agent._get_current_lambdas()
            if constraint_type == "wealth":
                print(f"  Final λ_wealth: {final_lambda_w:.4f}")
            elif constraint_type == "approval_rate":
                print(f"  Final λ_approval: {final_lambda_a:.4f}")
            else:  # both
                print(f"  Final λ_wealth: {final_lambda_w:.4f}")
                print(f"  Final λ_approval: {final_lambda_a:.4f}")

        # Print episode-level metrics summary
        if agent.episode_metrics["rho_episode"]:
            rho_values = np.array(agent.episode_metrics["rho_episode"])
            R_M_values = np.array(agent.episode_metrics["R_M"])
            R_F_values = np.array(agent.episode_metrics["R_F"])
            print(f"  --- Episode Metrics ---")
            print(
                f"  ρ mean: {np.mean(rho_values):.4f}, R_M: {R_M_values[-1]:.6f}, R_F: {R_F_values[-1]:.6f}"
            )

    return results, loader, theta_learner


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Adult Income RL Experiment")

    # Basic arguments
    parser.add_argument("--data", type=str, default=None, help="Path to adult.csv")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--constraint",
        type=str,
        default="wealth",
        choices=["approval_rate", "wealth", "both"],
        help="Constraint type: approval_rate, wealth, or both",
    )

    # Save options
    parser.add_argument("--save", action="store_true", help="Save results (pickle)")
    parser.add_argument(
        "--save-weights", action="store_false", help="Save model weights"
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="./weights",
        help="Directory for saved weights",
    )

    # Lambda trajectory options
    parser.add_argument(
        "--save-lambda",
        action="store_true",
        help="Save lambda trajectory arrays (csv and npy)",
    )
    parser.add_argument(
        "--plot-lambda", action="store_true", help="Plot lambda trajectories"
    )
    parser.add_argument(
        "--lambda-dir",
        type=str,
        default="./lambda_trajectories",
        help="Directory for lambda data",
    )

    # Episode metrics options
    parser.add_argument(
        "--save-episode-metrics",
        action="store_true",
        help="Save episode-level metrics (ρ, R_M, R_F)",
    )
    parser.add_argument(
        "--plot-episode-metrics", action="store_true", help="Plot episode-level metrics"
    )
    parser.add_argument(
        "--episode-metrics-dir",
        type=str,
        default="./episode_metrics",
        help="Directory for episode metrics",
    )

    # Single reward function mode
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run single reward function instead of all",
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="social_welfare",
        choices=[
            "social_welfare",
            "rawlsian_maximin",
            "fairness_lagrangian",
            "utilitarian_profit",
        ],
        help="Reward function (used with --single)",
    )

    # Lambda parameters (for single mode)
    parser.add_argument(
        "--lambda-wealth", type=float, default=2.0, help="Initial lambda for wealth"
    )
    parser.add_argument(
        "--lambda-approval",
        type=float,
        default=2.0,
        help="Initial lambda for approval rate",
    )
    parser.add_argument(
        "--lambda-lr",
        type=float,
        default=1e-2,
        help="Learning rate for lambda optimization",
    )

    # Load model
    parser.add_argument(
        "--load", type=str, default=None, help="Path to load saved model weights"
    )

    args = parser.parse_args()

    if args.load:
        # Load a trained model
        agent, env, theta, loader = load_trained_agent(
            weights_path=args.load, data_filepath=args.data, seed=args.seed
        )
        print("\nModel loaded successfully. Ready for inference.")

        # Optionally plot lambda history from loaded model
        if args.plot_lambda and agent.lambda_history.get("wealth"):
            plot_lambda_trajectory(
                agent.lambda_history,
                agent.reward_func_name,
                agent.constraint_type,
                args.seed,
                save_path=None,
                show=True,
            )

        # Optionally plot episode metrics from loaded model
        if args.plot_episode_metrics and agent.episode_metrics.get("episode"):
            plot_episode_metrics(
                agent.episode_metrics,
                agent.reward_func_name,
                agent.constraint_type,
                args.seed,
                save_path=None,
                show=True,
            )

    elif args.single:
        # Run single reward function
        agent, env, theta, loader = run_single_reward_function(
            data_filepath=args.data,
            num_episodes=args.episodes,
            seed=args.seed,
            reward_function=args.reward,
            constraint_type=args.constraint,
            lambda_wealth=args.lambda_wealth,
            lambda_approval=args.lambda_approval,
            lambda_lr=args.lambda_lr,
            save_weights=args.save_weights,
            weights_dir=args.weights_dir,
            save_lambda=args.save_lambda,
            lambda_dir=args.lambda_dir,
            plot_lambda=args.plot_lambda,
            save_episode_metrics_flag=args.save_episode_metrics,
            episode_metrics_dir=args.episode_metrics_dir,
            plot_episode_metrics_flag=args.plot_episode_metrics,
        )
    else:
        # Run all reward functions
        results, loader, theta = run_experiment(
            data_filepath=args.data,
            num_episodes=args.episodes,
            seed=args.seed,
            constraint_type=args.constraint,
            save_results_flag=args.save,
            save_weights=args.save_weights,
            weights_dir=args.weights_dir,
            save_lambda=args.save_lambda,
            lambda_dir=args.lambda_dir,
            plot_lambda=args.plot_lambda,
        )
