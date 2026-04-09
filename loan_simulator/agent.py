import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta

from .reward import RewardFunction


class LearnableLambdas(nn.Module):
    """Learnable lambda parameters for all constraint modes."""

    def __init__(
        self, constraint_type="both", init_lambda_wealth=2.0, init_lambda_approval=2.0
    ):
        super().__init__()
        self.constraint_type = constraint_type

        # Use log-space for positivity (exp recovery), logit-space for two_sided (sigmoid recovery)
        if constraint_type in ["wealth", "both", "social", "dm", "two_sided"]:
            if constraint_type == "two_sided":
                # init_lambda_wealth is the desired starting alpha ∈ (0, 1); store as logit
                alpha0 = float(np.clip(init_lambda_wealth, 1e-4, 1 - 1e-4))
                init_val = float(np.log(alpha0 / (1.0 - alpha0)))
            else:
                init_val = float(np.log(init_lambda_wealth))
            self.log_lambda_wealth = nn.Parameter(torch.tensor(init_val))
        else:
            self.register_buffer(
                "log_lambda_wealth", torch.tensor(np.log(init_lambda_wealth))
            )

        if constraint_type in ["approval_rate", "both", "predictive"]:
            self.log_lambda_approval = nn.Parameter(
                torch.tensor(np.log(init_lambda_approval))
            )
        else:
            self.register_buffer(
                "log_lambda_approval", torch.tensor(np.log(init_lambda_approval))
            )

    @property
    def lambda_wealth(self):
        # two_sided uses alpha as a blend weight — must stay in (0, 1)
        if self.constraint_type == "two_sided":
            return torch.sigmoid(self.log_lambda_wealth)
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
        entropy_coef=0.01,
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

        # Policy network
        self.policy_net = self._build_network(12, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Learnable lambdas for all modes except utilitarian_profit,
        # UNLESS constraint is two_sided (learnable alpha blend).
        self.learnable_lambdas = None
        self.lambda_optimizer = None
        if reward_function != "utilitarian_profit" or constraint_type == "two_sided":
            self.learnable_lambdas = LearnableLambdas(
                constraint_type=constraint_type,
                init_lambda_wealth=lambda_wealth,
                init_lambda_approval=lambda_approval,
            ).to(self.device)
            self.lambda_optimizer = optim.Adam(
                self.learnable_lambdas.parameters(), lr=lambda_lr
            )

        self.gamma = 0.99
        self.entropy_coef = entropy_coef
        self.episode_rewards = []
        self.per_step_rewards = []
        self.lambda_history = {"wealth": [], "approval": []}

        # Online reward normalisation (Welford's algorithm, across all steps/episodes)
        self._rew_ema_mean = 0.0
        self._rew_ema_var  = 1.0
        self._rew_ema_n    = 0

        # Episode-level metrics tracking
        self.episode_metrics = {
            "episode": [],
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
            "approval_rate_M": [],
            "approval_rate_F": [],
            "success_prob_M": [],
            "success_prob_F": [],
            "wealth_gap": [],
            "total_profit": [],
            "cumulative_time": [],
        }

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

        mu_M_start = self.env.mu_R
        mu_F_start = self.env.mu_B

        if self.initial_mu_M is None:
            self.initial_mu_M = mu_M_start
            self.initial_mu_F = mu_F_start

        states, actions, log_probs, entropies, rewards = [], [], [], [], []
        done = False

        lambda_w, lambda_a = self._get_current_lambdas()

        while not done:
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            states.append(obs_tensor)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                alpha, beta = self.policy_net(obs_tensor.unsqueeze(0))

            dist = Beta(alpha, beta)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy  = dist.entropy()

            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)

            next_obs, _, terminated, truncated, info = self.env.step(
                action.detach().cpu().numpy()
            )
            done = terminated or truncated

            reward = self.reward_function(
                self.env,
                action.cpu().item(),
                info,
                constraint_type=self.constraint_type,
                lambda_wealth=lambda_w,
                lambda_approval=lambda_a,
            )
            # Online reward normalisation (Welford) — removes scale differences
            # across constraint types and amplifies within-episode variation for
            # state-based rewards (e.g. mu_R + mu_B) that are otherwise near-constant.
            self._rew_ema_n += 1
            delta = reward - self._rew_ema_mean
            self._rew_ema_mean += delta / self._rew_ema_n
            delta2 = reward - self._rew_ema_mean
            self._rew_ema_var = (
                (self._rew_ema_var * (self._rew_ema_n - 1) + delta * delta2)
                / self._rew_ema_n
            )
            reward_norm = (reward - self._rew_ema_mean) / (
                np.sqrt(max(self._rew_ema_var, 1e-8))
            )
            rewards.append(reward_norm)
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
            returns_std = returns.std()
            if returns_std > 1e-6:
                returns = (returns - returns.mean()) / (returns_std + 1e-8)
            else:
                returns = returns - returns.mean()  # centre only; avoid amplifying noise

        # Compute policy loss + entropy bonus
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        entropy_bonus = torch.stack(entropies).mean()
        loss = torch.stack(policy_loss).sum() - self.entropy_coef * entropy_bonus

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

        # Update learnable lambdas
        if self.learnable_lambdas is not None:
            self._update_lambdas()

        episode_reward = sum(rewards)
        self.episode_rewards.append(episode_reward)

        # Track lambda history
        lambda_w, lambda_a = self._get_current_lambdas()
        self.lambda_history["wealth"].append(lambda_w)
        self.lambda_history["approval"].append(lambda_a)

        # Record episode-level metrics
        self.total_episodes_completed += 1
        self.cumulative_time += self.env.T

        mu_M_end = self.env.mu_R
        mu_F_end = self.env.mu_B

        delta_mu_M = mu_M_end - mu_M_start
        delta_mu_F = mu_F_end - mu_F_start

        if abs(delta_mu_F) > 1e-8:
            rho_episode = delta_mu_M / delta_mu_F
        else:
            rho_episode = 0.0 if abs(delta_mu_M) < 1e-8 else np.sign(delta_mu_M) * 100.0

        if self.cumulative_time > 1e-8:
            R_M = (mu_M_end - self.initial_mu_M) / self.cumulative_time
            R_F = (mu_F_end - self.initial_mu_F) / self.cumulative_time
        else:
            R_M = R_F = 0.0

        approval_rate_M = self.env.total_loans_R / max(self.env.total_applications_R, 1)
        approval_rate_F = self.env.total_loans_B / max(self.env.total_applications_B, 1)
        success_prob_M = 1.0 - self.env.total_defaults_R / max(
            self.env.total_loans_R, 1
        )
        success_prob_F = 1.0 - self.env.total_defaults_B / max(
            self.env.total_loans_B, 1
        )

        total_profit = sum(self.env.history["profit"])

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
        """Update learnable lambdas to maximize constraint satisfaction."""
        approval_rate_M = self.env.total_loans_R / max(self.env.total_applications_R, 1)
        approval_rate_F = self.env.total_loans_B / max(self.env.total_applications_B, 1)

        wealth_gap = abs(self.env.mu_R - self.env.mu_B)
        rate_gap = abs(approval_rate_M - approval_rate_F)

        self.lambda_optimizer.zero_grad()

        if self.constraint_type in ["wealth", "social", "two_sided"]:
            lambda_wealth_tensor = self.learnable_lambdas.lambda_wealth
            lambda_loss = -(lambda_wealth_tensor * wealth_gap)

        elif self.constraint_type in ["approval_rate", "predictive"]:
            lambda_approval_tensor = self.learnable_lambdas.lambda_approval
            lambda_loss = -(lambda_approval_tensor * rate_gap)

        elif self.constraint_type == "both":
            lambda_wealth_tensor = self.learnable_lambdas.lambda_wealth
            lambda_approval_tensor = self.learnable_lambdas.lambda_approval
            lambda_loss = -(
                lambda_wealth_tensor * wealth_gap + lambda_approval_tensor * rate_gap
            )

        elif self.constraint_type == "dm":
            rho_R = self.env.total_defaults_R / max(self.env.total_loans_R, 1)
            rho_B = self.env.total_defaults_B / max(self.env.total_loans_B, 1)
            r_R = self.env.interest_rate * (1 - rho_R) - rho_R
            r_B = self.env.interest_rate * (1 - rho_B) - rho_B
            profit_rate_gap = abs(r_R - r_B)
            lambda_wealth_tensor = self.learnable_lambdas.lambda_wealth
            lambda_loss = -(lambda_wealth_tensor * profit_rate_gap)

        else:
            return

        lambda_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.learnable_lambdas.parameters(), 1.0)
        self.lambda_optimizer.step()

    def save_model(self, filepath):
        """Save policy network weights and lambda parameters."""
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
        """Load policy network weights and lambda parameters."""
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
        if "initial_mu_M" in checkpoint:
            self.initial_mu_M = checkpoint["initial_mu_M"]
        if "initial_mu_F" in checkpoint:
            self.initial_mu_F = checkpoint["initial_mu_F"]
        if "cumulative_time" in checkpoint:
            self.cumulative_time = checkpoint["cumulative_time"]
        if "total_episodes_completed" in checkpoint:
            self.total_episodes_completed = checkpoint["total_episodes_completed"]

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
                        else:
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
