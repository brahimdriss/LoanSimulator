import gc
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta

from ..agent import LearnableLambdas
from ..reward import RewardFunction
from .buffers import DecisionTracker, PerformativeReplayBuffer


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
        buffer_capacity: int = 50,
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
        entropy_coef: float = 0.0,
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
            entropy_coef: Entropy regularisation coefficient λ (PePG paper Def. 5).
                          Shifts reward to soft reward r̃ = r - λ·log π(a|s).
                          0.0 = unregularised PePG (default).
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
        self.entropy_coef = entropy_coef

        # Performative settings
        self.buffer_capacity = buffer_capacity
        self.warmup_episodes = warmup_episodes

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {self.device}")

        # Mixed precision (compatible with older PyTorch versions)
        self.use_amp = use_amp and self.device.type == "cuda"
        if self.use_amp:
            try:
                # PyTorch >= 2.0
                self.scaler = torch.amp.GradScaler("cuda")
            except (AttributeError, TypeError):
                # PyTorch < 2.0
                self.scaler = torch.cuda.amp.GradScaler()
            print(f"  Mixed precision (AMP): Enabled")
        else:
            self.scaler = None

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
            "actual_approvals_M_episode": [],
            "actual_approvals_F_episode": [],
            "true_approvals_M_episode": [],
            "true_approvals_F_episode": [],

            # Confusion matrix (M = Male, F = Female)
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
            "approval_rate_M": [],
            "approval_rate_F": [],
            "success_prob_M": [],
            "success_prob_F": [],
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
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=self.use_amp):
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

            for i, (state, action) in enumerate(zip(batch_states, batch_actions)):
                idx = batch_start + i

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_tensor = torch.FloatTensor([action]).to(self.device)

                self.policy_net.zero_grad()

                alpha, beta = self.policy_net(state_tensor)
                dist = Beta(alpha.squeeze(), beta.squeeze())
                log_prob = dist.log_prob(action_tensor.clamp(1e-6, 1 - 1e-6))

                log_prob.backward()

                grads = []
                for p in self.policy_net.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.view(-1).clone())
                    else:
                        grads.append(torch.zeros(p.numel(), device=self.device))

                grad_matrix[idx] = torch.cat(grads)

        return grad_matrix

    def _compute_standard_gradient(
        self, decisions: List[dict]
    ) -> Tuple[torch.Tensor, dict]:
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
        rewards = rewards / (np.linalg.norm(rewards) + 1e-8)
        transitions = np.log(np.array([d.get("p_theta") for d in decisions]) + 1e-8)

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

        # Store pre-normalization statistics
        advantage_mean_raw = np.mean(advantages)
        advantage_std_raw = np.std(advantages)

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

        # Compute entropy for monitoring and regularisation
        entropy_per_step = dist.entropy().squeeze(-1) if dist.entropy().dim() > 1 else dist.entropy()
        entropy = entropy_per_step.mean().item()

        # Discounted policy gradient
        discounts = torch.FloatTensor([self.gamma**t for t in range(T)]).to(self.device)
        transitions_tensor = torch.FloatTensor(transitions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)

        # Compute individual loss components
        pg_component = advantages_tensor * log_probs
        transition_component = advantages_tensor * transitions_tensor
        reward_component = rewards_tensor

        policy_loss = -(
            discounts * (pg_component + transition_component + reward_component)
        ).sum()

        # Entropy regularisation (PePG paper Def. 5, Eq. 7):
        # soft reward r̃ = r - λ·log π(a|s)  ⟹  maximise entropy ⟹ subtract from loss
        if self.entropy_coef > 0:
            entropy_bonus = (discounts * entropy_per_step).sum()
            policy_loss = policy_loss - self.entropy_coef * entropy_bonus

        policy_loss.backward()

        # Extract gradient
        grads = []
        for p in self.policy_net.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1).clone())
            else:
                grads.append(torch.zeros(p.numel(), device=self.device))

        # Compile loss info
        loss_info = {
            "policy_loss": policy_loss.item(),
            "advantage_mean": advantage_mean_raw,
            "advantage_std": advantage_std_raw,
            "log_prob_mean": log_probs.mean().item(),
            "entropy": entropy,
            "value_baseline": baseline,
            "reward_component": (discounts * reward_component).sum().item(),
            "transition_component": (discounts * transition_component).sum().item(),
        }

        return torch.cat(grads), loss_info

    def _compute_full_pepg_gradient(
        self, decisions: List[dict]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the full PePG gradient combining all three terms.

        OPTIMIZED: Computes gradient matrix ONCE and reuses for all terms.

        Returns:
            total_gradient: Combined gradient tensor
            grad_info: Dictionary with individual gradient norms and loss info
        """
        num_params = sum(p.numel() for p in self.policy_net.parameters())

        if not decisions:
            return torch.zeros(num_params, device=self.device), {
                "standard_norm": 0.0,
                "total_norm": 0.0,
                "loss_info": {
                    "policy_loss": 0.0,
                    "advantage_mean": 0.0,
                    "advantage_std": 0.0,
                    "log_prob_mean": 0.0,
                    "entropy": 0.0,
                    "value_baseline": 0.0,
                    "reward_component": 0.0,
                    "transition_component": 0.0,
                },
            }

        # Precompute gradient matrix for ALL decisions ONCE
        states = np.array([d["state"] for d in decisions])
        actions = np.array([d["approval_prob"] for d in decisions])

        # Term 1: Standard policy gradient
        standard_grad, loss_info = self._compute_standard_gradient(decisions)

        # Free memory
        del states
        del actions

        # Combine with weights
        total_grad = standard_grad

        grad_info = {
            "standard_norm": standard_grad.norm().item(),
            "total_norm": total_grad.norm().item(),
            "loss_info": loss_info,
        }

        del standard_grad

        return total_grad, grad_info

    # ========================================================================
    # EPISODE COLLECTION
    # ========================================================================

    def _collect_episode(self, use_soft_reset: bool = True) -> dict:
        """Collect one episode with detailed decision tracking."""
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
            obs_tensor = torch.from_numpy(obs).float().to(self.device)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                alpha, beta = self.policy_net(obs_tensor.unsqueeze(0))

            dist = Beta(alpha, beta)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            action_np = action.detach().cpu().numpy()
            approval_prob = float(action_np[0])
            actions.append(approval_prob)
            log_probs.append(log_prob.cpu().item())

            # Get applicant info before step
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
                approved = np.random.random() < approval_prob

                if applicant["group"] == "male":
                    p_theta.append(info["p_theta_R"])
                    app_p_theta = info["p_theta_R"]
                else:
                    p_theta.append(info["p_theta_B"])
                    app_p_theta = info["p_theta_B"]

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
                    "p_theta": app_p_theta,
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
            "p_theta": p_theta,
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
        """Train for one episode using PePG."""
        in_warmup = self.total_episodes_completed < self.warmup_episodes
        use_perf_grad = use_performative and not in_warmup

        # Collect episode
        episode_data = self._collect_episode(use_soft_reset=use_performative)

        # Get decisions for gradient computation
        decisions = episode_data["decisions"]

        # Compute gradient
        self.optimizer.zero_grad()

        loss_info = {}
        if use_perf_grad and decisions:
            # Full PePG gradient
            total_grad, grad_info = self._compute_full_pepg_gradient(decisions)
            loss_info = grad_info.get("loss_info", {})

            self.gradient_metrics["standard_grad_norm"].append(
                grad_info["standard_norm"]
            )
            self.gradient_metrics["total_grad_norm"].append(grad_info["total_norm"])
        else:
            # Standard policy gradient only
            total_grad, loss_info = self._compute_standard_gradient(decisions)
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

            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

        # Update learnable lambdas and get lambda loss
        lambda_loss = 0.0
        constraint_wealth = 0.0
        constraint_approval = 0.0
        if self.learnable_lambdas is not None:
            lambda_loss, constraint_wealth, constraint_approval = (
                self._update_lambdas_with_loss()
            )

        # Record loss history
        self.loss_history["policy_loss"].append(loss_info.get("policy_loss", 0.0))
        self.loss_history["lambda_loss"].append(lambda_loss)
        self.loss_history["total_loss"].append(
            loss_info.get("policy_loss", 0.0) + lambda_loss
        )
        self.loss_history["advantage_mean"].append(loss_info.get("advantage_mean", 0.0))
        self.loss_history["advantage_std"].append(loss_info.get("advantage_std", 0.0))
        self.loss_history["log_prob_mean"].append(loss_info.get("log_prob_mean", 0.0))
        self.loss_history["entropy"].append(loss_info.get("entropy", 0.0))
        self.loss_history["value_baseline"].append(loss_info.get("value_baseline", 0.0))
        self.loss_history["constraint_wealth"].append(constraint_wealth)
        self.loss_history["constraint_approval"].append(constraint_approval)
        self.loss_history["reward_component"].append(
            loss_info.get("reward_component", 0.0)
        )
        self.loss_history["transition_component"].append(
            loss_info.get("transition_component", 0.0)
        )

        # Record metrics
        episode_reward = sum(episode_data["rewards"])
        self.episode_rewards.append(episode_reward)
        self.per_step_rewards.extend(episode_data["rewards"])
        if len(self.per_step_rewards) > 10000:
            self.per_step_rewards = self.per_step_rewards[-10000:]

        lambda_w, lambda_a = self._get_current_lambdas()
        self.lambda_history["wealth"].append(lambda_w)
        self.lambda_history["approval"].append(lambda_a)

        # Update episode metrics
        self._update_episode_metrics(episode_data)

        # Store lightweight version in replay buffer
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
            lambda_loss = -(self.learnable_lambdas.lambda_wealth * wealth_gap)
        elif self.constraint_type == "approval_rate":
            lambda_loss = -(self.learnable_lambdas.lambda_approval * rate_gap)
        elif self.constraint_type == "both":
            lambda_loss = -(
                self.learnable_lambdas.lambda_wealth * wealth_gap
                + self.learnable_lambdas.lambda_approval * rate_gap
            )
        else:
            return

        lambda_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.learnable_lambdas.parameters(), 1.0)
        self.lambda_optimizer.step()

    def _update_lambdas_with_loss(self) -> Tuple[float, float, float]:
        """Update learnable lambdas and return loss values for tracking."""
        approval_rate_M = self.env.total_loans_R / max(self.env.total_applications_R, 1)
        approval_rate_F = self.env.total_loans_B / max(self.env.total_applications_B, 1)

        wealth_gap = abs(self.env.mu_R - self.env.mu_B)
        rate_gap = abs(approval_rate_M - approval_rate_F)

        self.lambda_optimizer.zero_grad()

        if self.constraint_type == "wealth":
            lambda_loss = -(self.learnable_lambdas.lambda_wealth * wealth_gap)
        elif self.constraint_type == "approval_rate":
            lambda_loss = -(self.learnable_lambdas.lambda_approval * rate_gap)
        elif self.constraint_type == "both":
            lambda_loss = -(
                self.learnable_lambdas.lambda_wealth * wealth_gap
                + self.learnable_lambdas.lambda_approval * rate_gap
            )
        else:
            return 0.0, wealth_gap, rate_gap

        lambda_loss_value = lambda_loss.item()
        lambda_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.learnable_lambdas.parameters(), 1.0)
        self.lambda_optimizer.step()

        return lambda_loss_value, wealth_gap, rate_gap

    def _update_episode_metrics(self, episode_data: dict):
        """Update episode-level metrics with comprehensive tracking."""
        self.total_episodes_completed += 1
        self.cumulative_time += self.env.T

        # Basic episode info
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
            rho_episode = (
                0.0 if abs(delta_mu_M) < 1e-8 else np.sign(delta_mu_M) * 100.0
            )

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

        defaults_M_episode = len(
            [d for d in male_decisions if d["approved"] and d.get("defaulted", False)]
        )
        defaults_F_episode = len(
            [d for d in female_decisions if d["approved"] and d.get("defaulted", False)]
        )

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
        approval_rate_M_cumulative = self.cumulative_loans_M / max(
            self.cumulative_applications_M, 1
        )
        approval_rate_F_cumulative = self.cumulative_loans_F / max(
            self.cumulative_applications_F, 1
        )
        success_prob_M_cumulative = 1.0 - self.cumulative_defaults_M / max(
            self.cumulative_loans_M, 1
        )
        success_prob_F_cumulative = 1.0 - self.cumulative_defaults_F / max(
            self.cumulative_loans_F, 1
        )

        # Confusion matrix computation
        tp_M, fp_M, tn_M, fn_M = 0, 0, 0, 0
        tp_F, fp_F, tn_F, fn_F = 0, 0, 0, 0

        actual_approvals_M = 0
        actual_approvals_F = 0
        true_approvals_M = 0
        true_approvals_F = 0

        for d in male_decisions:
            should_approve = d["default_prob"] < 0.5
            if should_approve:
                true_approvals_M += 1

            if d["approved"]:
                actual_approvals_M += 1
                if not d.get("defaulted", False):
                    tp_M += 1
                else:
                    fp_M += 1
            else:
                if d["default_prob"] >= 0.5:
                    tn_M += 1
                else:
                    fn_M += 1

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
        time_start = 0.0
        time_end = self.env.current_time
        timesteps_in_episode = len(decisions)

        # Record all metrics
        m = self.episode_metrics

        m["episode"].append(self.total_episodes_completed)
        m["time_start"].append(time_start)
        m["time_end"].append(time_end)
        m["timesteps_in_episode"].append(timesteps_in_episode)

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

        m["loans_M_episode"].append(loans_M_episode)
        m["loans_F_episode"].append(loans_F_episode)
        m["applications_M_episode"].append(applications_M_episode)
        m["applications_F_episode"].append(applications_F_episode)
        m["defaults_M_episode"].append(defaults_M_episode)
        m["defaults_F_episode"].append(defaults_F_episode)
        m["profit_episode"].append(profit_episode)

        m["approval_rate_M_episode"].append(approval_rate_M_episode)
        m["approval_rate_F_episode"].append(approval_rate_F_episode)
        m["success_prob_M_episode"].append(success_prob_M_episode)
        m["success_prob_F_episode"].append(success_prob_F_episode)

        m["total_loans_M"].append(self.cumulative_loans_M)
        m["total_loans_F"].append(self.cumulative_loans_F)
        m["total_applications_M"].append(self.cumulative_applications_M)
        m["total_applications_F"].append(self.cumulative_applications_F)
        m["total_defaults_M"].append(self.cumulative_defaults_M)
        m["total_defaults_F"].append(self.cumulative_defaults_F)
        m["cumulative_profit"].append(self.cumulative_profit)

        m["approval_rate_M_cumulative"].append(approval_rate_M_cumulative)
        m["approval_rate_F_cumulative"].append(approval_rate_F_cumulative)
        m["success_prob_M_cumulative"].append(success_prob_M_cumulative)
        m["success_prob_F_cumulative"].append(success_prob_F_cumulative)

        m["actual_approvals_M_episode"].append(actual_approvals_M)
        m["actual_approvals_F_episode"].append(actual_approvals_F)
        m["true_approvals_M_episode"].append(true_approvals_M)
        m["true_approvals_F_episode"].append(true_approvals_F)

        m["true_positive_M"].append(tp_M)
        m["false_positive_M"].append(fp_M)
        m["true_negative_M"].append(tn_M)
        m["false_negative_M"].append(fn_M)
        m["true_positive_F"].append(tp_F)
        m["false_positive_F"].append(fp_F)
        m["true_negative_F"].append(tn_F)
        m["false_negative_F"].append(fn_F)

        m["accuracy_M"].append(accuracy_M)
        m["accuracy_F"].append(accuracy_F)
        m["precision_M"].append(precision_M)
        m["precision_F"].append(precision_F)
        m["recall_M"].append(recall_M)
        m["recall_F"].append(recall_F)

        m["hawkes_events_M"].append(hawkes_events_M)
        m["hawkes_events_F"].append(hawkes_events_F)

        # Legacy metrics
        m["total_profit"].append(profit_episode)
        m["cumulative_time"].append(self.cumulative_time)
        m["approval_rate_M"].append(approval_rate_M_cumulative)
        m["approval_rate_F"].append(approval_rate_F_cumulative)
        m["success_prob_M"].append(success_prob_M_cumulative)
        m["success_prob_F"].append(success_prob_F_cumulative)

    def get_episode_metrics_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.episode_metrics)

    def get_gradient_metrics_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.gradient_metrics)

    def get_loss_history_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.loss_history)

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
            episode_reward = self.train_episode(use_performative=use_performative)

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

                std_norm = self.gradient_metrics["standard_grad_norm"][-1]
                approvals_r = self.gradient_metrics["num_approvals_R"][-1]
                approvals_b = self.gradient_metrics["num_approvals_B"][-1]

                policy_loss = (
                    self.loss_history["policy_loss"][-1]
                    if self.loss_history["policy_loss"]
                    else 0
                )

                in_warmup = episode < self.warmup_episodes
                mode_str = "WARMUP" if in_warmup else "PePG"

                print(
                    f"  [{mode_str}] Ep {episode}: R={episode_reward:.2f}, "
                    f"Avg={avg_reward:.2f}, ρ={rho:.2f}, "
                    f"Loss={policy_loss:.4f}, "
                    f"λw={lambda_w:.3f}, "
                    f"Approvals R={approvals_r}, B={approvals_b}"
                )

        print(f"\nTraining complete. Buffer: {len(self.replay_buffer)} episodes")

    # ========================================================================
    # SAVE / LOAD
    # ========================================================================

    def save_model(self, filepath: str) -> str:
        """Save model weights and state."""
        save_dict = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
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
            "episode_rewards": self.episode_rewards,
            "lambda_history": self.lambda_history,
            "episode_metrics": self.episode_metrics,
            "gradient_metrics": self.gradient_metrics,
            "loss_history": self.loss_history,
            "initial_mu_M": self.initial_mu_M,
            "initial_mu_F": self.initial_mu_F,
            "cumulative_time": self.cumulative_time,
            "total_episodes_completed": self.total_episodes_completed,
            "cumulative_loans_M": self.cumulative_loans_M,
            "cumulative_loans_F": self.cumulative_loans_F,
            "cumulative_applications_M": self.cumulative_applications_M,
            "cumulative_applications_F": self.cumulative_applications_F,
            "cumulative_defaults_M": self.cumulative_defaults_M,
            "cumulative_defaults_F": self.cumulative_defaults_F,
            "cumulative_profit": self.cumulative_profit,
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

        for attr in [
            "episode_rewards", "lambda_history", "episode_metrics",
            "gradient_metrics", "loss_history", "initial_mu_M", "initial_mu_F",
            "cumulative_time", "total_episodes_completed", "cumulative_loans_M",
            "cumulative_loans_F", "cumulative_applications_M",
            "cumulative_applications_F", "cumulative_defaults_M",
            "cumulative_defaults_F", "cumulative_profit",
        ]:
            if attr in checkpoint:
                setattr(self, attr, checkpoint[attr])

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
