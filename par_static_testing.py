"""
Performative Policy Gradient (PePG) Agent for Fair Lending - Version 2

Explicit modeling of performative effects in lending:
1. Hawkes process: approvals add events → increase future application intensity
2. Wealth dynamics: approvals (with success) → increase borrower wealth → change μ_g
3. Reward: bank profit + fairness constraints depend on evolved distributions

Implements Theorem 2:
∇θV = E[Σt γ^t (A(st,at)·∇θ log πθ(at|st) + ∇θ log P_πθ(st+1|st,at) + ∇θ r_πθ(st,at))]

Key insight: We explicitly track how policy decisions affect:
- Hawkes intensity: λ_g(t) = base(μ_g) + α Σ exp(-β(t-t_i)) for approved events
- Wealth distribution: μ_g changes with approved loans that don't default
"""

import copy
import gc
import json
import multiprocessing as mp
import os
import pickle
import random
import time
import warnings
from collections import defaultdict, deque
from datetime import datetime
from multiprocessing import Pool, cpu_count
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.distributions import Beta, Normal

warnings.filterwarnings("ignore")


def get_gpu_count():
    """Get number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def pepg_train_worker(config):
    """Train a single PePG V2 configuration with specified seed, reward function, and constraints."""
    try:
        # Set GPU for this worker
        if config["gpu_id"] >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Set seeds
        np.random.seed(config["seed"])
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            torch.backends.cudnn.deterministic = True

        # Import here to avoid CUDA initialization conflicts
        from adult_income_training import (
            AdultIncomeDataLoader,
            IncomeEnvironment,
            TransitionParameterLearner,
            save_episode_metrics,
            save_lambda_history,
        )

        # Create individual run directory
        run_name = f"pepg_v2_{config['reward_function']}_{config['constraint_type']}_seed{config['seed']}"
        run_dir = os.path.join(config["base_output_dir"], "runs", run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Create subdirectories for this run
        run_weights_dir = os.path.join(run_dir, "weights")
        run_lambda_dir = os.path.join(run_dir, "lambda_trajectories")
        run_metrics_dir = os.path.join(run_dir, "episode_metrics")
        os.makedirs(run_weights_dir, exist_ok=True)
        os.makedirs(run_lambda_dir, exist_ok=True)
        os.makedirs(run_metrics_dir, exist_ok=True)

        print(
            f"\n[TRAIN {config.get('run_id', '?')}/{config.get('total_runs', '?')}] "
            f"Seed={config['seed']}, Reward={config['reward_function']}, "
            f"Constraint={config['constraint_type']}, GPU={config['gpu_id']}"
        )

        # Load data
        loader = AdultIncomeDataLoader(
            filepath=config["data_filepath"], sample_size=20000
        )
        loader.load_data()
        loader.preprocess()

        # Learn theta parameters
        theta_learner = TransitionParameterLearner(
            default_rate_min=0.02, default_rate_max=0.15
        )
        theta_learner.fit(loader.data)

        # Create environment
        env = IncomeEnvironment(
            theta_params=theta_learner,
            initial_wealth_male=loader.male_data["X"].values,
            initial_wealth_female=loader.female_data["X"].values,
            N_male=3000,
            N_female=3000,
            T=100,
            dt=0.5,
            seed=config["seed"],
        )

        # Create PePG V2 agent
        agent = PePGAgentV2(
            env,
            reward_function=config["reward_function"],
            constraint_type=config["constraint_type"],
            lambda_wealth=config.get("lambda_wealth", 2.0),
            lambda_approval=config.get("lambda_approval", 2.0),
            lambda_lr=config.get("lambda_lr", 1e-2),
            buffer_capacity=config.get("buffer_capacity", 0),
            warmup_episodes=config.get("warmup_episodes", 0),
            alpha_R=env.alpha_R,
            alpha_B=env.alpha_B,
            beta_R=env.beta_R,
            beta_B=env.beta_B,
            hawkes_weight=config.get("hawkes_weight", 1.0),
            wealth_weight=config.get("wealth_weight", 1.0),
            transition_weight=config.get("transition_weight", 1.0),
            reward_weight=config.get("reward_weight", 1.0),
        )

        # Train
        agent.train(num_episodes=config["num_episodes"], use_performative=True)

        # Construct weights path
        weights_filename = f"pepg_{config['reward_function']}_{config['constraint_type']}_seed{config['seed']}.pt"
        weights_path = os.path.join(run_weights_dir, weights_filename)

        # Save model
        agent.save_model(weights_path)

        # Save lambda trajectories
        if config["reward_function"] != "utilitarian_profit":
            save_lambda_history(
                agent.lambda_history,
                f"pepg_v2_{config['reward_function']}",
                config["constraint_type"],
                config["seed"],
                save_dir=run_lambda_dir,
                format="both",
            )

        # Save episode metrics
        save_episode_metrics(
            agent.episode_metrics,
            f"pg_test_{config['reward_function']}",
            config["constraint_type"],
            config["seed"],
            save_dir=run_metrics_dir,
            format="both",
        )

        # Extract final metrics
        final_metrics = {
            "seed": config["seed"],
            "reward_function": config["reward_function"],
            "constraint_type": config["constraint_type"],
            "weights_path": weights_path,
            "final_episode_reward": agent.episode_rewards[-1]
            if agent.episode_rewards
            else None,
            "num_episodes": len(agent.episode_rewards),
            "final_wealth_gap": env.history["wealth_gap"][-1]
            if env.history["wealth_gap"]
            else None,
            "final_approval_disparity": env.history["approval_disparity"][-1]
            if env.history["approval_disparity"]
            else None,
            "total_profit": sum(env.history["profit"]) if env.history["profit"] else 0,
        }

        # Add episode metrics if available
        if agent.episode_metrics.get("episode"):
            final_metrics.update(
                {
                    "train_rho_mean": float(
                        np.mean(agent.episode_metrics["rho_episode"])
                    ),
                    "train_rho_std": float(
                        np.std(agent.episode_metrics["rho_episode"])
                    ),
                    "train_rho_final": float(agent.episode_metrics["rho_episode"][-1]),
                    "train_mu_M_final": float(agent.episode_metrics["mu_M_end"][-1]),
                    "train_mu_F_final": float(agent.episode_metrics["mu_F_end"][-1]),
                }
            )

        print(
            f"[TRAIN DONE {config.get('run_id', '?')}/{config.get('total_runs', '?')}] "
            f"Seed={config['seed']}, Reward={config['reward_function']}"
        )

        return {"success": True, "config": config, "metrics": final_metrics}

    except Exception as e:
        print(
            f"[TRAIN FAILED] Seed={config['seed']}, Reward={config['reward_function']}: {e}"
        )
        import traceback

        traceback.print_exc()
        return {"success": False, "config": config, "error": str(e)}


try:
    from adult_income_training import (
        AdultIncomeDataLoader,
        IncomeEnvironment,
        LearnableLambdas,
        RewardFunction,
        TransitionParameterLearner,
        plot_episode_metrics,
        plot_lambda_trajectory,
        save_episode_metrics,
        save_lambda_history,
        save_results,
    )

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    print("Warning: Could not import from adult_income_new_refactored.py")
    print("Please ensure the file is in the same directory or PYTHONPATH")


# ============================================================================
# PERFORMATIVE REPLAY BUFFER
# ============================================================================


class PerformativeReplayBuffer:
    """
    FIFO buffer for storing recent episodes in performative settings.

    Stores complete episode data including decision-level information
    needed for computing explicit performative gradients.
    """

    def __init__(self, capacity: int = 0):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.total_episodes_added = 0

    def add_episode(self, episode_data: dict):
        """Add an episode to the buffer."""
        episode_data["buffer_id"] = self.total_episodes_added
        self.buffer.append(episode_data)
        self.total_episodes_added += 1

    def get_recent(self, n: int = None) -> List[dict]:
        """Get the n most recent episodes."""
        if n is None or n >= len(self.buffer):
            return list(self.buffer)
        return list(self.buffer)[-n:]

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def get_statistics(self) -> dict:
        if not self.buffer:
            return {"num_episodes": 0, "total_transitions": 0}

        total_transitions = sum(len(ep.get("decisions", [])) for ep in self.buffer)

        return {
            "num_episodes": len(self.buffer),
            "total_transitions": total_transitions,
            "capacity": self.capacity,
        }


# ============================================================================
# DECISION TRACKER - Tracks individual lending decisions for gradient computation
# ============================================================================


class DecisionTracker:
    """
    Tracks individual lending decisions within an episode.

    For each decision, stores:
    - State observation
    - Policy output (approval probability)
    - Whether approved (sampled action)
    - Applicant info (group, wealth, loan amount, default prob)
    - Outcome (if approved: defaulted or not)
    - Timestamps for Hawkes computation
    """

    def __init__(self):
        self.decisions = []
        self.hawkes_events_R = []  # Timestamps of male approvals
        self.hawkes_events_B = []  # Timestamps of female approvals

    def add_decision(self, decision_data: dict):
        """
        Add a lending decision.

        Args:
            decision_data: {
                'time': float,
                'state': np.array,
                'approval_prob': float (policy output),
                'approved': bool,
                'group': str ('male' or 'female'),
                'applicant_wealth': float,
                'loan_amount': float,
                'default_prob': float,
                'defaulted': bool or None (if not approved),
                'wealth_gain': float (0 if defaulted or rejected),
                'log_prob': float (log π(a|s)),
            }
        """
        self.decisions.append(decision_data)

        # Track Hawkes events (approved loans)
        if decision_data["approved"]:
            if decision_data["group"] == "male":
                self.hawkes_events_R.append(decision_data["time"])
            else:
                self.hawkes_events_B.append(decision_data["time"])

    def get_decisions(self) -> List[dict]:
        return self.decisions

    def get_approved_decisions(self) -> List[dict]:
        return [d for d in self.decisions if d["approved"]]

    def get_group_decisions(self, group: str) -> List[dict]:
        return [d for d in self.decisions if d["group"] == group]

    def clear(self):
        self.decisions = []
        self.hawkes_events_R = []
        self.hawkes_events_B = []


# ============================================================================
# PEPG AGENT V2 - Explicit Hawkes and Reward Gradients
# ============================================================================


class PePGAgentV2:
    """
    Performative Policy Gradient Agent with Explicit Gradient Modeling.

    Key differences from V1:
    1. Explicitly models Hawkes transition gradients based on approval decisions
    2. Explicitly models wealth dynamics gradients
    3. Structured reward gradients based on lending profit formula

    The three gradient terms:

    Term 1 (Standard PG): A(s,a) · ∇θ log πθ(a|s)
        - Standard REINFORCE with advantage baseline

    Term 2 (Transition): ∇θ log P^πθ(s'|s,a)
        - Hawkes: approvals increase future intensity
        - Wealth: approvals (with success) increase μ_g

    Term 3 (Reward): ∇θ r^πθ(s,a)
        - Profit depends on approval rate and default rate
        - Fairness constraints depend on μ_R, μ_B evolution
    """

    def __init__(
        self,
        env,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        reward_function: str = "social_welfare",
        constraint_type: str = "wealth",
        lambda_wealth: float = 2.0,
        lambda_approval: float = 2.0,
        lambda_lr: float = 1e-2,
        use_amp: bool = True,
        buffer_capacity: int = 0,
        warmup_episodes: int = 0,
        # Hawkes parameters (should match environment)
        alpha_R: float = 0.3,
        alpha_B: float = 0.3,
        beta_R: float = 2.0,
        beta_B: float = 2.0,
        # Gradient weights
        transition_weight: float = 1.0,
        reward_weight: float = 1.0,
        hawkes_weight: float = 1.0,
        wealth_weight: float = 1.0,
    ):
        """
        Initialize PePG Agent V2.

        Args:
            env: IncomeEnvironment instance
            hidden_dim: Hidden layer dimension for policy network
            lr: Learning rate for policy
            reward_function: One of the four reward functions
            constraint_type: 'approval_rate', 'wealth', or 'both'
            lambda_wealth: Initial lambda for wealth constraint
            lambda_approval: Initial lambda for approval rate constraint
            lambda_lr: Learning rate for lambda optimization
            use_amp: Use automatic mixed precision
            buffer_capacity: Replay buffer capacity
            warmup_episodes: Episodes before using performative gradients (can be 0)
            alpha_R, alpha_B: Hawkes excitation parameters
            beta_R, beta_B: Hawkes decay parameters
            transition_weight: Weight for combined transition gradient
            reward_weight: Weight for reward gradient
            hawkes_weight: Weight for Hawkes component of transition gradient
            wealth_weight: Weight for wealth component of transition gradient
        """
        self.env = env
        self.reward_func_name = reward_function
        self.reward_function = getattr(RewardFunction, reward_function)
        self.constraint_type = constraint_type
        self.lambda_wealth = lambda_wealth
        self.lambda_approval = lambda_approval

        # Hawkes parameters
        self.alpha_R = alpha_R
        self.alpha_B = alpha_B
        self.beta_R = beta_R
        self.beta_B = beta_B

        # Gradient weights
        self.transition_weight = transition_weight
        self.reward_weight = reward_weight
        self.hawkes_weight = hawkes_weight
        self.wealth_weight = wealth_weight

        # Performative settings
        self.buffer_capacity = buffer_capacity
        self.warmup_episodes = warmup_episodes

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {self.device}")

        # Mixed precision
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        if self.use_amp:
            print(f"  Mixed precision (AMP): Enabled")

        # Policy network (Beta distribution for approval probability)
        self.policy_net = self._build_policy_network(12, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Learnable lambdas
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
            print(
                f"  Learnable lambdas enabled for '{constraint_type}' (lr={lambda_lr})"
            )

        # Replay buffer
        self.replay_buffer = PerformativeReplayBuffer(capacity=buffer_capacity)
        print(f"  Replay buffer capacity: {buffer_capacity} episodes")
        print(f"  Warmup episodes: {warmup_episodes}")

        # Decision tracker for current episode
        self.decision_tracker = DecisionTracker()

        # Training state
        self.gamma = 0.99
        self.episode_rewards = []
        self.per_step_rewards = []
        self.lambda_history = {"wealth": [], "approval": []}

        # Episode-level metrics (comprehensive tracking)
        self.episode_metrics = {
            # Episode identification
            "episode": [],
            "time_start": [],
            "time_end": [],
            "timesteps_in_episode": [],
            
            # Wealth metrics
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
            
            # Episode-level counts
            "loans_M_episode": [],
            "loans_F_episode": [],
            "applications_M_episode": [],
            "applications_F_episode": [],
            "defaults_M_episode": [],
            "defaults_F_episode": [],
            "profit_episode": [],
            
            # Episode-level rates
            "approval_rate_M_episode": [],
            "approval_rate_F_episode": [],
            "success_prob_M_episode": [],
            "success_prob_F_episode": [],
            
            # Cumulative totals (across all episodes)
            "total_loans_M": [],
            "total_loans_F": [],
            "total_applications_M": [],
            "total_applications_F": [],
            "total_defaults_M": [],
            "total_defaults_F": [],
            "cumulative_profit": [],
            
            # Cumulative rates
            "approval_rate_M_cumulative": [],
            "approval_rate_F_cumulative": [],
            "success_prob_M_cumulative": [],
            "success_prob_F_cumulative": [],
            
            # Agent decisions vs true labels (for confusion matrix)
            # "actual" = agent's decision, "true" = would the loan have succeeded (based on default_prob threshold)
            "actual_approvals_M_episode": [],
            "actual_approvals_F_episode": [],
            "true_approvals_M_episode": [],  # Based on whether default_prob < 0.5
            "true_approvals_F_episode": [],
            
            # Confusion matrix (M = Male, F = Female)
            # True Positive: approved AND didn't default (or would not have defaulted)
            # False Positive: approved AND defaulted
            # True Negative: rejected AND would have defaulted
            # False Negative: rejected AND would not have defaulted
            "true_positive_M": [],
            "false_positive_M": [],
            "true_negative_M": [],
            "false_negative_M": [],
            "true_positive_F": [],
            "false_positive_F": [],
            "true_negative_F": [],
            "false_negative_F": [],
            
            # Classification metrics
            "accuracy_M": [],
            "accuracy_F": [],
            "precision_M": [],
            "precision_F": [],
            "recall_M": [],
            "recall_F": [],
            
            # Hawkes process events
            "hawkes_events_M": [],
            "hawkes_events_F": [],
            
            # Legacy metrics (for backwards compatibility)
            "total_profit": [],
            "cumulative_time": [],
            "approval_rate_M": [],  # Same as approval_rate_M_cumulative
            "approval_rate_F": [],  # Same as approval_rate_F_cumulative
            "success_prob_M": [],   # Same as success_prob_M_cumulative
            "success_prob_F": [],   # Same as success_prob_F_cumulative
        }
        
        # Track cumulative values across episodes
        self.cumulative_loans_M = 0
        self.cumulative_loans_F = 0
        self.cumulative_applications_M = 0
        self.cumulative_applications_F = 0
        self.cumulative_defaults_M = 0
        self.cumulative_defaults_F = 0
        self.cumulative_profit = 0.0

        # Performative gradient metrics
        self.gradient_metrics = {
            "standard_grad_norm": [],
            "hawkes_grad_norm": [],
            "wealth_grad_norm": [],
            "reward_grad_norm": [],
            "total_grad_norm": [],
            "num_decisions": [],
            "num_approvals_R": [],
            "num_approvals_B": [],
        }

        # Loss tracking for visualization
        self.loss_history = {
            "policy_loss": [],
            "lambda_loss": [],
            "total_loss": [],
            "advantage_mean": [],
            "advantage_std": [],
            "log_prob_mean": [],
            "entropy": [],
            "value_baseline": [],
            "constraint_wealth": [],
            "constraint_approval": [],
            "reward_component": [],
            "transition_component": [],
        }

        # Track initial wealth for R_g calculation
        self.initial_mu_M = None
        self.initial_mu_F = None
        self.cumulative_time = 0.0
        self.total_episodes_completed = 0

        # Store environment initial state
        self._store_initial_env_state()

    def _build_policy_network(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """Build Beta distribution policy network."""

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

    def _store_initial_env_state(self):
        """Store initial environment state."""
        self.initial_env_state = {
            "X_male": self.env.initial_X_male.copy(),
            "X_female": self.env.initial_X_female.copy(),
        }

    def get_action(self, obs: np.ndarray) -> Tuple[float, float]:
        """Get action from policy."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                alpha, beta = self.policy_net(obs_tensor)

        dist = Beta(alpha, beta)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.cpu().item(), log_prob.cpu().item()

    def _get_current_lambdas(self) -> Tuple[float, float]:
        """Get current lambda values."""
        if self.learnable_lambdas is not None:
            return (
                self.learnable_lambdas.lambda_wealth.item(),
                self.learnable_lambdas.lambda_approval.item(),
            )
        return self.lambda_wealth, self.lambda_approval

    def _soft_reset_env(self):
        """
        Soft reset for performative setting.
        Keeps wealth distributions and decayed Hawkes events.
        """
        # Store current wealth
        current_X_male = self.env.current_X_male.copy()
        current_X_female = self.env.current_X_female.copy()

        # Decay and shift Hawkes events
        current_time = self.env.current_time
        decay_window = 50.0

        event_times_R = [
            t - current_time
            for t in self.env.event_times_R
            if current_time - t < decay_window
        ]
        event_times_B = [
            t - current_time
            for t in self.env.event_times_B
            if current_time - t < decay_window
        ]

        # Standard reset
        obs, info = self.env.reset()

        # Restore performative state
        self.env.current_X_male = current_X_male
        self.env.current_X_female = current_X_female
        self.env.event_times_R = event_times_R
        self.env.event_times_B = event_times_B

        # Update means
        self.env.mu_R = np.mean(current_X_male)
        self.env.mu_B = np.mean(current_X_female)
        self.env.var_R = np.var(current_X_male)
        self.env.var_B = np.var(current_X_female)

        obs = self.env._get_observation()
        return obs, info

    # ========================================================================
    # GRADIENT COMPUTATION - The Core of PePG (VECTORIZED)
    # ========================================================================

    def _compute_all_grad_log_pi_batched(
        self, states: np.ndarray, actions: np.ndarray
    ) -> torch.Tensor:
        """
        Compute ∇θ log πθ(a|s) for ALL state-action pairs efficiently.

        Uses independent forward/backward passes to avoid retain_graph memory leak.

        Args:
            states: (T, state_dim) array
            actions: (T,) array of approval probabilities

        Returns:
            grad_log_pi_matrix: (T, num_params) tensor where each row is ∇θ log πθ(a_t|s_t)
        """
        T = len(states)
        num_params = sum(p.numel() for p in self.policy_net.parameters())

        if T == 0:
            return torch.zeros((0, num_params), device=self.device)

        grad_matrix = torch.zeros((T, num_params), device=self.device)

        # Process in batches to balance speed vs memory
        batch_size = min(64, T)

        for batch_start in range(0, T, batch_size):
            batch_end = min(batch_start + batch_size, T)
            batch_states = states[batch_start:batch_end]
            batch_actions = actions[batch_start:batch_end]

            # Each sample in batch gets its own forward/backward (no retain_graph needed)
            for i, (state, action) in enumerate(zip(batch_states, batch_actions)):
                idx = batch_start + i

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_tensor = torch.FloatTensor([action]).to(self.device)

                # Fresh forward pass for each sample
                self.policy_net.zero_grad()

                alpha, beta = self.policy_net(state_tensor)
                dist = Beta(alpha.squeeze(), beta.squeeze())
                log_prob = dist.log_prob(action_tensor.clamp(1e-6, 1 - 1e-6))

                # Backward pass (no retain_graph!)
                log_prob.backward()

                # Extract and store gradients
                grads = []
                for p in self.policy_net.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.view(-1).clone())
                    else:
                        grads.append(torch.zeros(p.numel(), device=self.device))

                grad_matrix[idx] = torch.cat(grads)

        return grad_matrix

    def _compute_standard_gradient(self, decisions: List[dict]) -> torch.Tensor:
        """
        Compute Term 1: Standard policy gradient A(s,a) · ∇θ log πθ(a|s)
        
        Returns:
            gradient: The computed gradient tensor
            loss_info: Dictionary with loss components for tracking
        """
        num_params = sum(p.numel() for p in self.policy_net.parameters())
        
        if not decisions:
            return torch.zeros(num_params, device=self.device), {
                "policy_loss": 0.0,
                "advantage_mean": 0.0,
                "advantage_std": 0.0,
                "log_prob_mean": 0.0,
                "entropy": 0.0,
                "value_baseline": 0.0,
                "reward_component": 0.0,
                "transition_component": 0.0,
            }

        # Extract data
        states = np.array([d["state"] for d in decisions])
        actions = np.array([d["approval_prob"] for d in decisions])
        rewards = np.array([d.get("reward", 0.0) for d in decisions])
        rewards = rewards/(np.linalg.norm(rewards))
        transitions = np.log(np.array([d.get("p_theta") for d in decisions]))
        
        # Compute returns (G_t = Σ γ^k r_{t+k})
        T = len(rewards)
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        # Advantage = return - baseline
        baseline = np.mean(returns)
        advantages = returns - baseline

        # Normalize advantages
        if len(advantages) > 1 and np.std(advantages) > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (
                np.std(advantages) + 1e-8
            )

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)

        # Compute policy gradient (vectorized)
        self.policy_net.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            alpha, beta = self.policy_net(states_tensor)

        dist = Beta(alpha.squeeze(-1), beta.squeeze(-1))
        log_probs = dist.log_prob(actions_tensor.clamp(1e-6, 1 - 1e-6))

        # Discounted policy gradient
        discounts = torch.FloatTensor([self.gamma**t for t in range(T)]).to(self.device)
        policy_loss = -(discounts *(advantages_tensor * log_probs)).sum()

        policy_loss.backward()

        # Extract gradient
        grads = []
        for p in self.policy_net.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1).clone())
            else:
                grads.append(torch.zeros(p.numel(), device=self.device))

        return torch.cat(grads)

    def _compute_full_pepg_gradient(
        self, decisions: List[dict]
        ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the full PePG gradient combining all three terms.

        OPTIMIZED: Computes gradient matrix ONCE and reuses for all terms.

        Returns:
            total_gradient: Combined gradient tensor
            grad_info: Dictionary with individual gradient norms
        """
        num_params = sum(p.numel() for p in self.policy_net.parameters())

        if not decisions:
            return torch.zeros(num_params, device=self.device), {
                "standard_norm": 0.0,
                "total_norm": 0.0,
            }

        # Precompute gradient matrix for ALL decisions ONCE
        states = np.array([d["state"] for d in decisions])
        actions = np.array([d["approval_prob"] for d in decisions])
        # grad_matrix = self._compute_all_grad_log_pi_batched(states, actions)

        # Term 1: Standard policy gradient (uses its own efficient computation)
        standard_grad = self._compute_standard_gradient(decisions)

        # Term 2a: Hawkes transition gradient (pass precomputed gradients)

        # Free the gradient matrix memory
        # del grad_matrix
        del states
        del actions

        # Combine with weights
       

        total_grad = (
            standard_grad
            # + self.transition_weight * transition_grad
            # + self.reward_weight * reward_grad
        )

        grad_info = {
            "standard_norm": standard_grad.norm().item(),
            "total_norm": total_grad.norm().item(),
        }

        # Clean up intermediate gradients
        del standard_grad

        return total_grad, grad_info

    # ========================================================================
    # EPISODE COLLECTION
    # ========================================================================

    def _collect_episode(self, use_soft_reset: bool = True) -> dict:
        """
        Collect one episode with detailed decision tracking.
        """
        # if use_soft_reset and self.total_episodes_completed > 0:
        #     obs, _ = self._soft_reset_env()
        # else:
        #     obs, _ = self.env.reset()

        obs = self.env._get_observation()

        # Store episode start state
        mu_M_start = self.env.mu_R
        mu_F_start = self.env.mu_B

        if self.initial_mu_M is None:
            self.initial_mu_M = mu_M_start
            self.initial_mu_F = mu_F_start

        # Clear decision tracker
        self.decision_tracker.clear()

        # Collect trajectory
        states = []
        actions = []
        rewards = []
        log_probs = []
        p_theta = []

        done = False
        lambda_w, lambda_a = self._get_current_lambdas()
        step = 0

        while not done:
            states.append(obs.copy())

            # Get action from policy
            obs_tensor = torch.FloatTensor(obs).to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                alpha, beta = self.policy_net(obs_tensor.unsqueeze(0))

            dist = Beta(alpha, beta)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            action_np = action.cpu().numpy()
            approval_prob = float(
                action_np[0].item()
            )  # Use .item() to extract the scalar value
            actions.append(approval_prob)
            log_probs.append(log_prob.cpu().item())

            # Get applicant info before step (if there's a current applicant)
            applicant = self.env.current_applicant


            
            # Step environment
            next_obs, _, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            # Compute reward
            reward = self.reward_function(
                self.env,
                approval_prob,
                info,
                constraint_type=self.constraint_type,
                lambda_wealth=lambda_w,
                lambda_approval=lambda_a,
            )
            rewards.append(reward)

            # Track decision if there was an applicant
            if applicant is not None:
                # Determine if approved (stochastic based on approval_prob)
                approved = np.random.random() < approval_prob
                
                if applicant["group"]=="male":
                    p_theta.append(info['p_theta_R'])
                    app_p_theta = info['p_theta_R']
                else:
                    p_theta.append(info['p_theta_B'])
                    app_p_theta = info['p_theta_B']

                # Determine if defaulted (if approved)
                defaulted = None
                wealth_gain = 0.0
                if approved:
                    defaulted = np.random.random() < applicant["default_prob"]
                    if not defaulted:
                        wealth_gain = applicant.get("wealth_gain", 0.0)

                decision_data = {
                    "time": self.env.current_time,
                    "step": step,
                    "state": obs.copy(),
                    "approval_prob": approval_prob,
                    "approved": approved,
                    "group": applicant["group"],
                    "applicant_wealth": applicant["X"],
                    "loan_amount": applicant.get("loan_amount", 30.0),
                    "default_prob": applicant["default_prob"],
                    "defaulted": defaulted,
                    "wealth_gain": wealth_gain,
                    "log_prob": log_prob.cpu().item(),
                    "reward": reward,
                    "p_theta": app_p_theta
                }
                self.decision_tracker.add_decision(decision_data)
                

            obs = next_obs
            step += 1

        # Create episode data
        episode_data = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "log_probs": log_probs,
            "p_theta":p_theta,
            "decisions": self.decision_tracker.get_decisions(),
            "hawkes_events_R": self.decision_tracker.hawkes_events_R.copy(),
            "hawkes_events_B": self.decision_tracker.hawkes_events_B.copy(),
            "mu_M_start": mu_M_start,
            "mu_F_start": mu_F_start,
            "mu_M_end": self.env.mu_R,
            "mu_F_end": self.env.mu_B,
            "timestamp": datetime.now().isoformat(),
        }

        return episode_data

    def _collect_episode_test(self, use_soft_reset: bool = True) -> dict:
        """
        Collect one episode with detailed decision tracking.
        """
        if use_soft_reset and self.total_episodes_completed > 0:
            obs, _ = self._soft_reset_env()
        else:
            obs, _ = self.env.reset()

        # Store episode start state
        mu_M_start = self.env.mu_R
        mu_F_start = self.env.mu_B

        if self.initial_mu_M is None:
            self.initial_mu_M = mu_M_start
            self.initial_mu_F = mu_F_start

        # Clear decision tracker
        self.decision_tracker.clear()

        # Collect trajectory
        states = []
        actions = []
        rewards = []
        log_probs = []
        p_theta = []

        done = False
        lambda_w, lambda_a = self._get_current_lambdas()
        step = 0

        while not done:
            states.append(obs.copy())

            # Get action from policy
            obs_tensor = torch.FloatTensor(obs).to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                alpha, beta = self.policy_net(obs_tensor.unsqueeze(0))

            dist = Beta(alpha, beta)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            action_np = action.cpu().numpy()
            approval_prob = float(
                action_np[0].item()
            )  # Use .item() to extract the scalar value
            actions.append(approval_prob)
            log_probs.append(log_prob.cpu().item())

            # Get applicant info before step (if there's a current applicant)
            applicant = self.env.current_applicant


            
            # Step environment
            next_obs, _, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            # Compute reward
            reward = self.reward_function(
                self.env,
                approval_prob,
                info,
                constraint_type=self.constraint_type,
                lambda_wealth=lambda_w,
                lambda_approval=lambda_a,
            )
            rewards.append(reward)

            # Track decision if there was an applicant
            if applicant is not None:
                # Determine if approved (stochastic based on approval_prob)
                approved = np.random.random() < approval_prob
                
                if applicant["group"]=="male":
                    p_theta.append(info['p_theta_R'])
                    app_p_theta = info['p_theta_R']
                else:
                    p_theta.append(info['p_theta_B'])
                    app_p_theta = info['p_theta_B']

                # Determine if defaulted (if approved)
                defaulted = None
                wealth_gain = 0.0
                if approved:
                    defaulted = np.random.random() < applicant["default_prob"]
                    if not defaulted:
                        wealth_gain = applicant.get("wealth_gain", 0.0)

                decision_data = {
                    "time": self.env.current_time,
                    "step": step,
                    "state": obs.copy(),
                    "approval_prob": approval_prob,
                    "approved": approved,
                    "group": applicant["group"],
                    "applicant_wealth": applicant["X"],
                    "loan_amount": applicant.get("loan_amount", 30.0),
                    "default_prob": applicant["default_prob"],
                    "defaulted": defaulted,
                    "wealth_gain": wealth_gain,
                    "log_prob": log_prob.cpu().item(),
                    "reward": reward,
                    "p_theta": app_p_theta
                }
                self.decision_tracker.add_decision(decision_data)
                

            obs = next_obs
            step += 1

        # Create episode data
        episode_data = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "log_probs": log_probs,
            "p_theta":p_theta,
            "decisions": self.decision_tracker.get_decisions(),
            "hawkes_events_R": self.decision_tracker.hawkes_events_R.copy(),
            "hawkes_events_B": self.decision_tracker.hawkes_events_B.copy(),
            "mu_M_start": mu_M_start,
            "mu_F_start": mu_F_start,
            "mu_M_end": self.env.mu_R,
            "mu_F_end": self.env.mu_B,
            "timestamp": datetime.now().isoformat(),
        }

        return episode_data

    # ========================================================================
    # TRAINING
    # ========================================================================

    def train_episode(self, use_performative: bool = True) -> float:
        """
        Train for one episode using PePG.
        """
        
        obs, _ = self.env.reset()

        in_warmup = self.total_episodes_completed < self.warmup_episodes
        use_perf_grad = use_performative and not in_warmup
        
        # Collect episode
        episode_data = self._collect_episode(use_soft_reset=use_performative)

        # Get decisions for gradient computation
        decisions = episode_data["decisions"]

        # Compute gradient
        self.optimizer.zero_grad()

        if use_perf_grad and decisions:
            # Full PePG gradient
            total_grad, grad_info = self._compute_full_pepg_gradient(decisions)

            # Record gradient metrics
            self.gradient_metrics["standard_grad_norm"].append(
                grad_info["standard_norm"]
            )
            self.gradient_metrics["total_grad_norm"].append(grad_info["total_norm"])
        else:
            # Standard policy gradient only
            total_grad = self._compute_standard_gradient(decisions)
            self.gradient_metrics["standard_grad_norm"].append(total_grad.norm().item())
            self.gradient_metrics["total_grad_norm"].append(total_grad.norm().item())

        # Record decision counts
        num_approvals_R = len(
            [d for d in decisions if d["approved"] and d["group"] == "male"]
        )
        num_approvals_B = len(
            [d for d in decisions if d["approved"] and d["group"] == "female"]
        )
        self.gradient_metrics["num_decisions"].append(len(decisions))
        self.gradient_metrics["num_approvals_R"].append(num_approvals_R)
        self.gradient_metrics["num_approvals_B"].append(num_approvals_B)

        # Apply gradient to policy network
        if total_grad.numel() > 0:
            idx = 0
            for p in self.policy_net.parameters():
                numel = p.numel()
                if idx + numel <= total_grad.numel():
                    p.grad = total_grad[idx : idx + numel].view(p.shape)
                idx += numel

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

            # Optimizer step
            self.optimizer.step()

        # Update learnable lambdas
        if self.learnable_lambdas is not None:
            self._update_lambdas()

        # Record metrics
        episode_reward = sum(episode_data["rewards"])
        self.episode_rewards.append(episode_reward)
        # Keep only last 10000 step rewards to prevent memory bloat
        self.per_step_rewards.extend(episode_data["rewards"])
        if len(self.per_step_rewards) > 10000:
            self.per_step_rewards = self.per_step_rewards[-10000:]

        lambda_w, lambda_a = self._get_current_lambdas()
        self.lambda_history["wealth"].append(lambda_w)
        self.lambda_history["approval"].append(lambda_a)

        # Update episode metrics
        self._update_episode_metrics(episode_data)

        # Store lightweight version in replay buffer (remove heavy state arrays)
        lightweight_episode = {
            "rewards": episode_data["rewards"],
            "mu_M_start": episode_data["mu_M_start"],
            "mu_F_start": episode_data["mu_F_start"],
            "mu_M_end": episode_data["mu_M_end"],
            "mu_F_end": episode_data["mu_F_end"],
            "num_decisions": len(decisions),
            "num_approvals_R": num_approvals_R,
            "num_approvals_B": num_approvals_B,
            "timestamp": episode_data["timestamp"],
        }
        self.replay_buffer.add_episode(lightweight_episode)

        # Clear references to help garbage collection
        del episode_data
        del decisions
        del total_grad

        return episode_reward

    def test_episode(self, use_performative: bool = True) -> float:
        """
        Train for one episode using PePG.
        """
        
        in_warmup = self.total_episodes_completed < self.warmup_episodes
        use_perf_grad = use_performative and not in_warmup
        
        # Collect episode
        episode_data = self._collect_episode_test(use_soft_reset=use_performative)

        # Get decisions for gradient computation
        decisions = episode_data["decisions"]

        # Compute gradient
        self.optimizer.zero_grad()

        if use_perf_grad and decisions:
            # Full PePG gradient
            total_grad, grad_info = self._compute_full_pepg_gradient(decisions)

            # Record gradient metrics
            self.gradient_metrics["standard_grad_norm"].append(
                grad_info["standard_norm"]
            )
            self.gradient_metrics["total_grad_norm"].append(grad_info["total_norm"])
        else:
            # Standard policy gradient only
            total_grad = self._compute_standard_gradient(decisions)
            self.gradient_metrics["standard_grad_norm"].append(total_grad.norm().item())
            self.gradient_metrics["total_grad_norm"].append(total_grad.norm().item())

        # Record decision counts
        num_approvals_R = len(
            [d for d in decisions if d["approved"] and d["group"] == "male"]
        )
        num_approvals_B = len(
            [d for d in decisions if d["approved"] and d["group"] == "female"]
        )
        self.gradient_metrics["num_decisions"].append(len(decisions))
        self.gradient_metrics["num_approvals_R"].append(num_approvals_R)
        self.gradient_metrics["num_approvals_B"].append(num_approvals_B)

        # Apply gradient to policy network
        if total_grad.numel() > 0:
            idx = 0
            for p in self.policy_net.parameters():
                numel = p.numel()
                if idx + numel <= total_grad.numel():
                    p.grad = total_grad[idx : idx + numel].view(p.shape)
                idx += numel

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

            # Optimizer step
            self.optimizer.step()

        # Update learnable lambdas
        if self.learnable_lambdas is not None:
            self._update_lambdas()

        # Record metrics
        episode_reward = sum(episode_data["rewards"])
        self.episode_rewards.append(episode_reward)
        # Keep only last 10000 step rewards to prevent memory bloat
        self.per_step_rewards.extend(episode_data["rewards"])
        if len(self.per_step_rewards) > 10000:
            self.per_step_rewards = self.per_step_rewards[-10000:]

        lambda_w, lambda_a = self._get_current_lambdas()
        self.lambda_history["wealth"].append(lambda_w)
        self.lambda_history["approval"].append(lambda_a)

        # Update episode metrics
        self._update_episode_metrics(episode_data)

        # Store lightweight version in replay buffer (remove heavy state arrays)
        lightweight_episode = {
            "rewards": episode_data["rewards"],
            "mu_M_start": episode_data["mu_M_start"],
            "mu_F_start": episode_data["mu_F_start"],
            "mu_M_end": episode_data["mu_M_end"],
            "mu_F_end": episode_data["mu_F_end"],
            "num_decisions": len(decisions),
            "num_approvals_R": num_approvals_R,
            "num_approvals_B": num_approvals_B,
            "timestamp": episode_data["timestamp"],
        }
        self.replay_buffer.add_episode(lightweight_episode)

        # Clear references to help garbage collection
        del episode_data
        del decisions
        del total_grad

        return episode_reward


    def _update_lambdas(self):
        """Update learnable lambdas based on constraint violations."""
        approval_rate_M = self.env.total_loans_R / max(self.env.total_applications_R, 1)
        approval_rate_F = self.env.total_loans_B / max(self.env.total_applications_B, 1)

        wealth_gap = abs(self.env.mu_R - self.env.mu_B)
        rate_gap = abs(approval_rate_M - approval_rate_F)

        self.lambda_optimizer.zero_grad()

        if self.constraint_type == "wealth":
            lambda_wealth_tensor = self.learnable_lambdas.lambda_wealth
            lambda_loss = -(lambda_wealth_tensor * wealth_gap)
        elif self.constraint_type == "approval_rate":
            lambda_approval_tensor = self.learnable_lambdas.lambda_approval
            lambda_loss = -(lambda_approval_tensor * rate_gap)
        elif self.constraint_type == "both":
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

    def _update_episode_metrics(self, episode_data: dict):
        """Update episode-level metrics."""
        self.total_episodes_completed += 1
        self.cumulative_time += self.env.T

        mu_M_start = episode_data["mu_M_start"]
        mu_M_end = episode_data["mu_M_end"]
        mu_F_start = episode_data["mu_F_start"]
        mu_F_end = episode_data["mu_F_end"]

        delta_mu_M = mu_M_end - mu_M_start
        delta_mu_F = mu_F_end - mu_F_start

        # Inequality ratio
        if abs(delta_mu_F) > 1e-8:
            rho_episode = delta_mu_M / delta_mu_F
        else:
            rho_episode = 0.0 if abs(delta_mu_M) < 1e-8 else np.sign(delta_mu_M) * 100.0

        # Long-term social welfare
        if self.cumulative_time > 1e-8:
            R_M = (mu_M_end - self.initial_mu_M) / self.cumulative_time
            R_F = (mu_F_end - self.initial_mu_F) / self.cumulative_time
        else:
            R_M = R_F = 0.0

        # Get decisions from episode
        decisions = episode_data.get("decisions", [])
        
        # Episode-level counts from decisions
        male_decisions = [d for d in decisions if d["group"] == "male"]
        female_decisions = [d for d in decisions if d["group"] == "female"]
        
        applications_M_episode = len(male_decisions)
        applications_F_episode = len(female_decisions)
        
        loans_M_episode = len([d for d in male_decisions if d["approved"]])
        loans_F_episode = len([d for d in female_decisions if d["approved"]])
        
        defaults_M_episode = len([d for d in male_decisions if d["approved"] and d.get("defaulted", False)])
        defaults_F_episode = len([d for d in female_decisions if d["approved"] and d.get("defaulted", False)])
        
        # Episode profit
        profit_episode = sum(episode_data.get("rewards", []))
        
        # Episode-level rates
        approval_rate_M_episode = loans_M_episode / max(applications_M_episode, 1)
        approval_rate_F_episode = loans_F_episode / max(applications_F_episode, 1)
        success_prob_M_episode = 1.0 - defaults_M_episode / max(loans_M_episode, 1)
        success_prob_F_episode = 1.0 - defaults_F_episode / max(loans_F_episode, 1)
        
        # Update cumulative totals
        self.cumulative_loans_M += loans_M_episode
        self.cumulative_loans_F += loans_F_episode
        self.cumulative_applications_M += applications_M_episode
        self.cumulative_applications_F += applications_F_episode
        self.cumulative_defaults_M += defaults_M_episode
        self.cumulative_defaults_F += defaults_F_episode
        self.cumulative_profit += profit_episode
        
        # Cumulative rates
        approval_rate_M_cumulative = self.cumulative_loans_M / max(self.cumulative_applications_M, 1)
        approval_rate_F_cumulative = self.cumulative_loans_F / max(self.cumulative_applications_F, 1)
        success_prob_M_cumulative = 1.0 - self.cumulative_defaults_M / max(self.cumulative_loans_M, 1)
        success_prob_F_cumulative = 1.0 - self.cumulative_defaults_F / max(self.cumulative_loans_F, 1)
        
        # Confusion matrix computation
        # "True" label: whether the loan SHOULD have been approved (based on default_prob < 0.5)
        # For approved loans: we know actual outcome
        # For rejected loans: we use default_prob as proxy
        
        # Initialize confusion matrix counters
        tp_M, fp_M, tn_M, fn_M = 0, 0, 0, 0
        tp_F, fp_F, tn_F, fn_F = 0, 0, 0, 0
        
        actual_approvals_M = 0
        actual_approvals_F = 0
        true_approvals_M = 0  # Count of applications that "should" be approved (low default risk)
        true_approvals_F = 0
        
        for d in male_decisions:
            # True label: should this loan have been approved? (default_prob < 0.5 means low risk)
            should_approve = d["default_prob"] < 0.5
            if should_approve:
                true_approvals_M += 1
            
            if d["approved"]:
                actual_approvals_M += 1
                if not d.get("defaulted", False):
                    tp_M += 1  # Approved and didn't default
                else:
                    fp_M += 1  # Approved but defaulted
            else:
                # Rejected - use default_prob to determine "true" outcome
                if d["default_prob"] >= 0.5:
                    tn_M += 1  # Rejected high-risk (correct)
                else:
                    fn_M += 1  # Rejected low-risk (missed opportunity)
        
        for d in female_decisions:
            should_approve = d["default_prob"] < 0.5
            if should_approve:
                true_approvals_F += 1
            
            if d["approved"]:
                actual_approvals_F += 1
                if not d.get("defaulted", False):
                    tp_F += 1
                else:
                    fp_F += 1
            else:
                if d["default_prob"] >= 0.5:
                    tn_F += 1
                else:
                    fn_F += 1
        
        # Classification metrics
        total_M = tp_M + fp_M + tn_M + fn_M
        total_F = tp_F + fp_F + tn_F + fn_F
        
        accuracy_M = (tp_M + tn_M) / max(total_M, 1)
        accuracy_F = (tp_F + tn_F) / max(total_F, 1)
        
        precision_M = tp_M / max(tp_M + fp_M, 1)
        precision_F = tp_F / max(tp_F + fp_F, 1)
        
        recall_M = tp_M / max(tp_M + fn_M, 1)
        recall_F = tp_F / max(tp_F + fn_F, 1)
        
        # Hawkes events
        hawkes_events_M = len(episode_data.get("hawkes_events_R", []))
        hawkes_events_F = len(episode_data.get("hawkes_events_B", []))
        
        # Time tracking
        time_start = 0.0  # Episode always starts at t=0
        time_end = self.env.current_time
        timesteps_in_episode = len(decisions)
        
        # Record all metrics
        m = self.episode_metrics
        
        # Episode identification
        m["episode"].append(self.total_episodes_completed)
        m["time_start"].append(time_start)
        m["time_end"].append(time_end)
        m["timesteps_in_episode"].append(timesteps_in_episode)
        
        # Wealth metrics
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
        m["wealth_gap"].append(mu_M_end - mu_F_end)
        
        # Episode-level counts
        m["loans_M_episode"].append(loans_M_episode)
        m["loans_F_episode"].append(loans_F_episode)
        m["applications_M_episode"].append(applications_M_episode)
        m["applications_F_episode"].append(applications_F_episode)
        m["defaults_M_episode"].append(defaults_M_episode)
        m["defaults_F_episode"].append(defaults_F_episode)
        m["profit_episode"].append(profit_episode)
        
        # Episode-level rates
        m["approval_rate_M_episode"].append(approval_rate_M_episode)
        m["approval_rate_F_episode"].append(approval_rate_F_episode)
        m["success_prob_M_episode"].append(success_prob_M_episode)
        m["success_prob_F_episode"].append(success_prob_F_episode)
        
        # Cumulative totals
        m["total_loans_M"].append(self.cumulative_loans_M)
        m["total_loans_F"].append(self.cumulative_loans_F)
        m["total_applications_M"].append(self.cumulative_applications_M)
        m["total_applications_F"].append(self.cumulative_applications_F)
        m["total_defaults_M"].append(self.cumulative_defaults_M)
        m["total_defaults_F"].append(self.cumulative_defaults_F)
        m["cumulative_profit"].append(self.cumulative_profit)
        
        # Cumulative rates
        m["approval_rate_M_cumulative"].append(approval_rate_M_cumulative)
        m["approval_rate_F_cumulative"].append(approval_rate_F_cumulative)
        m["success_prob_M_cumulative"].append(success_prob_M_cumulative)
        m["success_prob_F_cumulative"].append(success_prob_F_cumulative)
        
        # Agent decisions vs true labels
        m["actual_approvals_M_episode"].append(actual_approvals_M)
        m["actual_approvals_F_episode"].append(actual_approvals_F)
        m["true_approvals_M_episode"].append(true_approvals_M)
        m["true_approvals_F_episode"].append(true_approvals_F)
        
        # Confusion matrix
        m["true_positive_M"].append(tp_M)
        m["false_positive_M"].append(fp_M)
        m["true_negative_M"].append(tn_M)
        m["false_negative_M"].append(fn_M)
        m["true_positive_F"].append(tp_F)
        m["false_positive_F"].append(fp_F)
        m["true_negative_F"].append(tn_F)
        m["false_negative_F"].append(fn_F)
        
        # Classification metrics
        m["accuracy_M"].append(accuracy_M)
        m["accuracy_F"].append(accuracy_F)
        m["precision_M"].append(precision_M)
        m["precision_F"].append(precision_F)
        m["recall_M"].append(recall_M)
        m["recall_F"].append(recall_F)
        
        # Hawkes events
        m["hawkes_events_M"].append(hawkes_events_M)
        m["hawkes_events_F"].append(hawkes_events_F)
        
        # Legacy metrics (for backwards compatibility)
        m["total_profit"].append(profit_episode)
        m["cumulative_time"].append(self.cumulative_time)
        m["approval_rate_M"].append(approval_rate_M_cumulative)
        m["approval_rate_F"].append(approval_rate_F_cumulative)
        m["success_prob_M"].append(success_prob_M_cumulative)
        m["success_prob_F"].append(success_prob_F_cumulative)

    def get_episode_metrics_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.episode_metrics)

    def get_gradient_metrics_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.gradient_metrics)

    def train(self, num_episodes: int = 100, use_performative: bool = True):
        """Train the agent for multiple episodes."""
        print(
            f"Training PePG V2 with {self.reward_func_name} ({self.constraint_type} constraint)..."
        )
        print(f"  Warmup episodes: {self.warmup_episodes}")
        print(
            f"  Gradient weights: hawkes={self.hawkes_weight}, wealth={self.wealth_weight}, "
            f"transition={self.transition_weight}, reward={self.reward_weight}"
        )

        for episode in range(num_episodes):
            episode_reward = self.test_episode(use_performative=use_performative)

            # Periodic memory cleanup every 10 episodes
            if episode % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if episode % 10 == 0 or episode == num_episodes - 1:
                avg_reward = (
                    np.mean(self.episode_rewards[-20:])
                    if len(self.episode_rewards) >= 20
                    else episode_reward
                )
                lambda_w, lambda_a = self._get_current_lambdas()

                rho = (
                    self.episode_metrics["rho_episode"][-1]
                    if self.episode_metrics["rho_episode"]
                    else 0
                )

                # Gradient norms
                std_norm = self.gradient_metrics["standard_grad_norm"][-1]
                # hawkes_norm = self.gradient_metrics["hawkes_grad_norm"][-1]
                # wealth_norm = self.gradient_metrics["wealth_grad_norm"][-1]
                approvals_r = self.gradient_metrics['num_approvals_R'][-1]
                approvals_b = self.gradient_metrics['num_approvals_B'][-1]

                in_warmup = episode < self.warmup_episodes
                mode_str = "WARMUP" if in_warmup else "PePG"

                print(
                    f"  [{mode_str}] Ep {episode}: R={episode_reward:.2f}, "
                    f"Avg={avg_reward:.2f}, ρ={rho:.2f}, "
                    f"λw={lambda_w:.3f}, "
                    f"Number of Approvals R = {approvals_r:.3f}, "
                    f"Number of Approvals B = {approvals_b:.3f}, "
                )

        print(f"\nTraining complete. Buffer: {len(self.replay_buffer)} episodes")


        """Test the agent for multiple episodes.""" 
    # ========================================================================
    # SAVE / LOAD
    # ========================================================================

    def save_model(self, filepath: str) -> str:
        """Save model weights and state."""
        save_dict = {
            # Policy network
            "policy_net_state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # Configuration
            "reward_function": self.reward_func_name,
            "constraint_type": self.constraint_type,
            "buffer_capacity": self.buffer_capacity,
            "warmup_episodes": self.warmup_episodes,
            "alpha_R": self.alpha_R,
            "alpha_B": self.alpha_B,
            "beta_R": self.beta_R,
            "beta_B": self.beta_B,
            "transition_weight": self.transition_weight,
            "reward_weight": self.reward_weight,
            "hawkes_weight": self.hawkes_weight,
            "wealth_weight": self.wealth_weight,
            # Training state
            "episode_rewards": self.episode_rewards,
            "lambda_history": self.lambda_history,
            "episode_metrics": self.episode_metrics,
            "gradient_metrics": self.gradient_metrics,
            "initial_mu_M": self.initial_mu_M,
            "initial_mu_F": self.initial_mu_F,
            "cumulative_time": self.cumulative_time,
            "total_episodes_completed": self.total_episodes_completed,
            # Cumulative tracking variables
            "cumulative_loans_M": self.cumulative_loans_M,
            "cumulative_loans_F": self.cumulative_loans_F,
            "cumulative_applications_M": self.cumulative_applications_M,
            "cumulative_applications_F": self.cumulative_applications_F,
            "cumulative_defaults_M": self.cumulative_defaults_M,
            "cumulative_defaults_F": self.cumulative_defaults_F,
            "cumulative_profit": self.cumulative_profit,
            # Replay buffer
            "replay_buffer_data": list(self.replay_buffer.buffer),
        }

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
        print(f"  PePG V2 model saved to {filepath}")
        return filepath

    def load_model(self, filepath: str) -> dict:
        """Load model weights and state."""
        try:
            checkpoint = torch.load(
                filepath, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "episode_rewards" in checkpoint:
            self.episode_rewards = checkpoint["episode_rewards"]
        if "lambda_history" in checkpoint:
            self.lambda_history = checkpoint["lambda_history"]
        if "episode_metrics" in checkpoint:
            self.episode_metrics = checkpoint["episode_metrics"]
        if "gradient_metrics" in checkpoint:
            self.gradient_metrics = checkpoint["gradient_metrics"]
        if "initial_mu_M" in checkpoint:
            self.initial_mu_M = checkpoint["initial_mu_M"]
        if "initial_mu_F" in checkpoint:
            self.initial_mu_F = checkpoint["initial_mu_F"]
        if "cumulative_time" in checkpoint:
            self.cumulative_time = checkpoint["cumulative_time"]
        if "total_episodes_completed" in checkpoint:
            self.total_episodes_completed = checkpoint["total_episodes_completed"]
        
        # Load cumulative tracking variables
        if "cumulative_loans_M" in checkpoint:
            self.cumulative_loans_M = checkpoint["cumulative_loans_M"]
        if "cumulative_loans_F" in checkpoint:
            self.cumulative_loans_F = checkpoint["cumulative_loans_F"]
        if "cumulative_applications_M" in checkpoint:
            self.cumulative_applications_M = checkpoint["cumulative_applications_M"]
        if "cumulative_applications_F" in checkpoint:
            self.cumulative_applications_F = checkpoint["cumulative_applications_F"]
        if "cumulative_defaults_M" in checkpoint:
            self.cumulative_defaults_M = checkpoint["cumulative_defaults_M"]
        if "cumulative_defaults_F" in checkpoint:
            self.cumulative_defaults_F = checkpoint["cumulative_defaults_F"]
        if "cumulative_profit" in checkpoint:
            self.cumulative_profit = checkpoint["cumulative_profit"]

        if "replay_buffer_data" in checkpoint:
            self.replay_buffer.buffer = deque(
                checkpoint["replay_buffer_data"], maxlen=self.buffer_capacity
            )

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

        print(f"  PePG V2 model loaded from {filepath}")
        return checkpoint


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def run_pepg_v2_experiment(
    data_filepath: str = None,
    num_episodes: int = 100,
    seed: int = 0,
    reward_function: str = "social_welfare",
    constraint_type: str = "wealth",
    lambda_wealth: float = 2.0,
    lambda_approval: float = 2.0,
    lambda_lr: float = 1e-2,
    buffer_capacity: int = 0,
    warmup_episodes: int = 0,
    hawkes_weight: float = 1.0,
    wealth_weight: float = 1.0,
    transition_weight: float = 1.0,
    reward_weight: float = 1.0,
    save_weights: bool = True,
    weights_dir: str = "./weights_pepg_v2",
    save_lambda: bool = True,
    lambda_dir: str = "./lambda_trajectories_pepg_v2",
    plot_lambda: bool = True,
    save_episode_metrics_flag: bool = True,
    episode_metrics_dir: str = "./episode_metrics_pg_test",
    plot_episode_metrics_flag: bool = True,
    plot_results: bool = True,
):
    """Run PePG V2 experiment."""
    import os

    os.makedirs(weights_dir, exist_ok=True)

    print("=" * 80)
    print(f"PERFORMATIVE POLICY GRADIENT V2 (Explicit Hawkes) EXPERIMENT")
    print(f"  Reward: {reward_function.upper()}")
    print(f"  Constraint: {constraint_type.upper()}")
    print(f"  Seed: {seed}")
    print(f"  Warmup: {warmup_episodes} episodes")
    print("=" * 80)

    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        print(f"\nCUDA Available: {torch.cuda.get_device_name(0)}")
    else:
        print("\nUsing CPU")

    # Load data
    print(f"\n[STEP 1] Loading Adult Income Data...")
    loader = AdultIncomeDataLoader(filepath=data_filepath, sample_size=20000)
    loader.load_data()
    loader.preprocess()

    # Learn theta parameters
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

    # Create PePG V2 agent
    print(f"\n[STEP 4] Creating PePG V2 Agent...")
    agent = PePGAgentV2(
        env,
        reward_function=reward_function,
        constraint_type=constraint_type,
        lambda_wealth=lambda_wealth,
        lambda_approval=lambda_approval,
        lambda_lr=lambda_lr,
        buffer_capacity=buffer_capacity,
        warmup_episodes=warmup_episodes,
        alpha_R=env.alpha_R,
        alpha_B=env.alpha_B,
        beta_R=env.beta_R,
        beta_B=env.beta_B,
        hawkes_weight=hawkes_weight,
        wealth_weight=wealth_weight,
        transition_weight=transition_weight,
        reward_weight=reward_weight,
    )

    # Train
    print(f"\n[STEP 5] Training PePG V2 Agent...")
    agent.train(num_episodes=num_episodes, use_performative=True)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Final wealth gap: ${env.history['wealth_gap'][-1]:.3f}k")
    print(f"  Approval disparity: {env.history['approval_disparity'][-1]:.4f}")
    print(f"  Total profit: ${sum(env.history['profit']):.3f}k")

    if agent.episode_metrics["rho_episode"]:
        rho_values = np.array(agent.episode_metrics["rho_episode"])
        print(f"  Mean ρ: {np.mean(rho_values):.4f}")

    # Save
    if save_weights:
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = (
            f"{weights_dir}/pepg_{reward_function}_{constraint_type}_seed{seed}.pt"
        )
        agent.save_model(weights_path)

    # Save lambda trajectories
    if save_lambda and reward_function != "utilitarian_profit":
        os.makedirs(lambda_dir, exist_ok=True)
        save_lambda_history(
            agent.lambda_history,
            f"pepg_v2_{reward_function}",
            constraint_type,
            seed,
            save_dir=lambda_dir,
            format="both",
        )

    # Plot lambda trajectories
    if plot_lambda and reward_function != "utilitarian_profit":
        os.makedirs(lambda_dir, exist_ok=True)
        lambda_plot_path = f"{lambda_dir}/pepg_v2_{reward_function}_{constraint_type}_seed{seed}_lambda.png"
        plot_lambda_trajectory(
            agent.lambda_history,
            f"PePG_v2 {reward_function}",
            constraint_type,
            seed,
            save_path=lambda_plot_path,
            show=True,
        )

    # Save episode metrics
    if save_episode_metrics_flag:
        os.makedirs(episode_metrics_dir, exist_ok=True)
        save_episode_metrics(
            agent.episode_metrics,
            f"pg_test_v2_{reward_function}",
            constraint_type,
            seed,
            save_dir=episode_metrics_dir,
            format="both",
        )

    # Plot episode metrics
    if plot_episode_metrics_flag:
        os.makedirs(episode_metrics_dir, exist_ok=True)
        episode_plot_path = f"{episode_metrics_dir}/pg_test_{reward_function}_{constraint_type}_seed{seed}_metrics.png"
        plot_episode_metrics(
            agent.episode_metrics,
            f"PePG_v2 {reward_function}",
            constraint_type,
            seed,
            save_path=episode_plot_path,
            show=True,
        )

    # Plot gradient decomposition results
    if plot_results:
        _plot_pepg_v2_results(
            agent, reward_function, constraint_type, seed, weights_dir
        )

    return agent, env, theta_learner, loader


def _plot_pepg_v2_results(agent, reward_function, constraint_type, seed, save_dir):
    """Plot PePG V2 training results including gradient decomposition."""
    import os

    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = agent.gradient_metrics
    episodes = range(len(metrics["standard_grad_norm"]))

    # 1. Gradient norms decomposition
    ax = axes[0, 0]
    ax.plot(
        episodes, metrics["standard_grad_norm"], "b-", label="Standard PG", alpha=0.7
    )
    ax.plot(episodes, metrics.get("hawkes_grad_norm", [0]*len(episodes)), "g-", label="Hawkes", alpha=0.7)
    ax.plot(episodes, metrics.get("wealth_grad_norm", [0]*len(episodes)), "r-", label="Wealth", alpha=0.7)
    ax.plot(episodes, metrics.get("reward_grad_norm", [0]*len(episodes)), "m-", label="Reward", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("PePG Gradient Decomposition")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Total gradient norm
    ax = axes[0, 1]
    ax.plot(episodes, metrics["total_grad_norm"], "purple", linewidth=2)
    if len(metrics["total_grad_norm"]) > 10:
        window = 10
        smoothed = np.convolve(
            metrics["total_grad_norm"], np.ones(window) / window, mode="valid"
        )
        ax.plot(
            range(window - 1, len(episodes)),
            smoothed,
            "orange",
            linewidth=2,
            label="Smoothed",
        )
        ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Gradient Norm")
    ax.set_title("Combined PePG Gradient")
    ax.grid(True, alpha=0.3)

    # 3. Episode rewards
    ax = axes[0, 2]
    ax.plot(agent.episode_rewards, "b-", alpha=0.5)
    if len(agent.episode_rewards) > 10:
        window = 10
        smoothed = np.convolve(
            agent.episode_rewards, np.ones(window) / window, mode="valid"
        )
        ax.plot(
            range(window - 1, len(agent.episode_rewards)), smoothed, "r-", linewidth=2
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Learning Curve")
    ax.grid(True, alpha=0.3)

    # 4. Decision counts
    ax = axes[1, 0]
    ax.plot(
        episodes, metrics["num_decisions"], "k-", label="Total Decisions", alpha=0.7
    )
    ax.plot(
        episodes, metrics["num_approvals_R"], "b-", label="Male Approvals", alpha=0.7
    )
    ax.plot(
        episodes, metrics["num_approvals_B"], "r-", label="Female Approvals", alpha=0.7
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Count")
    ax.set_title("Decisions per Episode")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Inequality ratio
    ax = axes[1, 1]
    rho = agent.episode_metrics["rho_episode"]
    rho_clipped = np.clip(rho, -10, 10)
    ax.plot(rho_clipped, "purple", linewidth=1.5)
    ax.axhline(y=1.0, color="red", linestyle="--", label="Equal growth")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("ρ = Δμ_M / Δμ_F")
    ax.set_title("Inequality Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Lambda evolution
    ax = axes[1, 2]
    if agent.lambda_history["wealth"]:
        ax.plot(agent.lambda_history["wealth"], "b-", label="λ_wealth", linewidth=2)
    if agent.lambda_history["approval"]:
        ax.plot(agent.lambda_history["approval"], "r-", label="λ_approval", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Lambda Value")
    ax.set_title("Learnable Lambda Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"PePG V2: {reward_function} ({constraint_type}) - Seed {seed}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    save_path = (
        f"{save_dir}/pepg_v2_{reward_function}_{constraint_type}_seed{seed}_results.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Results plot saved to {save_path}")
    plt.show()


def load_pepg_v2_agent(weights_path: str, data_filepath: str = None, seed: int = 0):
    """Load a trained PePG V2 agent."""
    print(f"Loading PePG V2 agent from {weights_path}...")

    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location="cpu")

    reward_function = checkpoint.get("reward_function", "social_welfare")
    constraint_type = checkpoint.get("constraint_type", "wealth")

    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Load data
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

    # Create agent with loaded config
    agent = PePGAgentV2(
        env,
        reward_function=reward_function,
        constraint_type=constraint_type,
        lambda_wealth=checkpoint.get("final_lambda_wealth", 2.0),
        lambda_approval=checkpoint.get("final_lambda_approval", 2.0),
        buffer_capacity=checkpoint.get("buffer_capacity", 0),
        warmup_episodes=checkpoint.get("warmup_episodes", 0),
        alpha_R=checkpoint.get("alpha_R", 0.3),
        alpha_B=checkpoint.get("alpha_B", 0.3),
        beta_R=checkpoint.get("beta_R", 2.0),
        beta_B=checkpoint.get("beta_B", 2.0),
        hawkes_weight=checkpoint.get("hawkes_weight", 1.0),
        wealth_weight=checkpoint.get("wealth_weight", 1.0),
    )

    agent.load_model(weights_path)

    return agent, env, theta_learner, loader


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Run PePG V2 Experiment with Multi-Processing Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Experiment configuration
    parser.add_argument(
        "--data", type=str, default=None, help="Path to adult.csv data file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of training episodes (default: 100)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of random seeds to run in parallel (default: 1 for single run)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Starting seed or single seed if --seeds=1 (default: 1)",
    )
    parser.add_argument(
        "--constraint",
        type=str,
        default="wealth",
        choices=["approval_rate", "wealth", "both"],
        help="Constraint type (default: wealth)",
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="all",
        choices=[
            "all",
            "social_welfare",
            "rawlsian_maximin",
            "fairness_lagrangian",
            "utilitarian_profit",
        ],
        help="Reward function to use (default: all for multi-seed, social_welfare for single)",
    )

    # PePG specific parameters
    parser.add_argument("--buffer-capacity", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--hawkes-weight", type=float, default=1.0)
    parser.add_argument("--wealth-weight", type=float, default=1.0)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--reward-weight", type=float, default=1.0)

    parser.add_argument("--lambda-wealth", type=float, default=2.0)
    parser.add_argument("--lambda-approval", type=float, default=2.0)
    parser.add_argument("--lambda-lr", type=float, default=1e-2)

    # Output directories
    parser.add_argument("--save-weights", action="store_false")
    parser.add_argument("--weights-dir", type=str, default="./weights_pepg_v2")
    parser.add_argument("--save-lambda", action="store_true")
    parser.add_argument("--plot-lambda", action="store_true")
    parser.add_argument(
        "--lambda-dir", type=str, default="./lambda_trajectories_pepg_v2"
    )
    parser.add_argument("--save-episode-metrics", action="store_false")
    parser.add_argument("--plot-episode-metrics", action="store_true")
    parser.add_argument(
        "--episode-metrics-dir", type=str, default="./episode_metrics_pg_test"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Plot gradient decomposition results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./pepg_v2_results",
        help="Directory to save aggregated results (default: ./pepg_v2_results)",
    )

    # Multiprocessing parameters
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: 80%% of CPU count, max 20)",
    )

    parser.add_argument("--load", type=str, default=None)

    args = parser.parse_args()

    if args.seeds == 1:
        if args.reward == "all":
            args.reward = "social_welfare"

        if args.load:
            agent, env, theta, loader = load_pepg_v2_agent(
                weights_path=args.load, data_filepath=args.data, seed=args.seed
            )
            
            print(f"\n[] Testing PePG V2 Agent...")
            agent.train(num_episodes=args.episodes, use_performative=True)

            os.makedirs(args.episode_metrics_dir, exist_ok=True)
            save_episode_metrics(
                agent.episode_metrics,
                f"pg_test_v2_{args.reward}",
                args.constraint,
                args.seed,
                save_dir=args.episode_metrics_dir,
                format="both",
            )
        else:
            agent, env, theta, loader = run_pepg_v2_experiment(
                data_filepath=args.data,
                num_episodes=args.episodes,
                seed=args.seed,
                reward_function=args.reward,
                constraint_type=args.constraint,
                lambda_wealth=args.lambda_wealth,
                lambda_approval=args.lambda_approval,
                lambda_lr=args.lambda_lr,
                buffer_capacity=args.buffer_capacity,
                warmup_episodes=args.warmup,
                hawkes_weight=args.hawkes_weight,
                wealth_weight=args.wealth_weight,
                transition_weight=args.transition_weight,
                reward_weight=args.reward_weight,
                save_weights=args.save_weights,
                weights_dir=args.weights_dir,
                save_lambda=args.save_lambda,
                lambda_dir=args.lambda_dir,
                plot_lambda=args.plot_lambda,
                save_episode_metrics_flag=args.save_episode_metrics,
                episode_metrics_dir=args.episode_metrics_dir,
                plot_episode_metrics_flag=args.plot_episode_metrics,
                plot_results=args.plot,
            )
    else:
        start_time = time.time()

        if args.workers is None:
            available_cpus = cpu_count()
            args.workers = min(int(available_cpus * 0.8), 20)

        if args.reward == "all":
            reward_functions = [
                "social_welfare",
                "rawlsian_maximin",
                "fairness_lagrangian",
                "utilitarian_profit",
            ]
        else:
            reward_functions = [args.reward]

        print(f"\n{'=' * 80}")
        print(f"PEPG V2 MULTI-SEED EXPERIMENT")
        print(f"{'=' * 80}")
        print(f"  Seeds: {args.seeds} (starting from seed {args.seed})")
        print(f"  Reward functions: {reward_functions}")
        print(f"  Constraint: {args.constraint}")
        print(f"  Episodes: {args.episodes}")
        print(f"  Workers: {args.workers}")
        print(f"  GPUs: {get_gpu_count()}")
        print(f"{'=' * 80}\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = os.path.join(
            args.results_dir, f"pepg_v2_experiment_{timestamp}"
        )
        os.makedirs(base_output_dir, exist_ok=True)

        print(f"Base output directory: {base_output_dir}")

        configs = []
        gpu_count = get_gpu_count()

        for seed in range(args.seed, args.seed + args.seeds):
            for reward_func in reward_functions:
                config = {
                    "seed": seed,
                    "reward_function": reward_func,
                    "constraint_type": args.constraint,
                    "data_filepath": args.data,
                    "num_episodes": args.episodes,
                    "base_output_dir": base_output_dir,
                    "gpu_id": (len(configs) % gpu_count) if gpu_count > 0 else -1,
                    "run_id": len(configs) + 1,
                    # PePG specific parameters
                    "lambda_wealth": args.lambda_wealth,
                    "lambda_approval": args.lambda_approval,
                    "lambda_lr": args.lambda_lr,
                    "buffer_capacity": args.buffer_capacity,
                    "warmup_episodes": args.warmup,
                    "hawkes_weight": args.hawkes_weight,
                    "wealth_weight": args.wealth_weight,
                    "transition_weight": args.transition_weight,
                    "reward_weight": args.reward_weight,
                }
                configs.append(config)

        total_runs = len(configs)
        for config in configs:
            config["total_runs"] = total_runs

        print(f"Total runs: {total_runs}")

        aggregated_dir = os.path.join(base_output_dir, "aggregated_results")
        os.makedirs(aggregated_dir, exist_ok=True)

        print("\n[Phase 1/2] Training with parallel workers...")

        with Pool(processes=args.workers) as pool:
            train_results = pool.map(pepg_train_worker, configs)

        train_results_path = os.path.join(aggregated_dir, "train_results.pkl")
        with open(train_results_path, "wb") as f:
            pickle.dump(train_results, f)

        successful_trains = sum(1 for r in train_results if r["success"])
        print(f"\nTraining complete: {successful_trains}/{len(configs)} successful")

        print("\n[Phase 2/2] Aggregating results...")

        import pandas as pd

        summary_records = []
        for result in train_results:
            if result["success"]:
                metrics = result["metrics"]
                summary_records.append(metrics)

        if summary_records:
            summary_df = pd.DataFrame(summary_records)
            summary_path = os.path.join(aggregated_dir, "training_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"Training summary saved to: {summary_path}")

            print(f"\n{'=' * 60}")
            print("AGGREGATED RESULTS BY REWARD FUNCTION")
            print("=" * 60)

            for reward_func in reward_functions:
                subset = summary_df[summary_df["reward_function"] == reward_func]
                if len(subset) > 0:
                    print(f"\n{reward_func.upper()}:")
                    if "train_rho_mean" in subset.columns:
                        rho_values = subset["train_rho_mean"].dropna()
                        if len(rho_values) > 0:
                            print(
                                f"  ρ (mean±std): {rho_values.mean():.4f} ± {rho_values.std():.4f}"
                            )
                    if "final_wealth_gap" in subset.columns:
                        gap_values = subset["final_wealth_gap"].dropna()
                        if len(gap_values) > 0:
                            print(
                                f"  Wealth gap: ${gap_values.mean():.3f}k ± ${gap_values.std():.3f}k"
                            )
                    if "total_profit" in subset.columns:
                        profit_values = subset["total_profit"].dropna()
                        if len(profit_values) > 0:
                            print(
                                f"  Total profit: ${profit_values.mean():.2f}k ± ${profit_values.std():.2f}k"
                            )

        elapsed_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT COMPLETE")
        print(
            f"  Total time: {elapsed_time:.1f} seconds ({elapsed_time / 60:.1f} minutes)"
        )
        print(f"  Results: {base_output_dir}")
        print(f"{'=' * 60}")
