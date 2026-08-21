import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from tqdm import tqdm

from .reward import RewardFunction, compute_batched_rewards


class LearnableLambdas(nn.Module):
    """Learnable lambda parameters for all constraint modes."""

    def __init__(
        self, constraint_type="both", init_lambda_wealth=2.0, init_lambda_approval=2.0
    ):
        super().__init__()
        self.constraint_type = constraint_type

        # Floor before log(): callers legitimately pass 0.0 for lambdas that
        # the chosen constraint_type doesn't use (e.g. utilitarian_profit
        # passes lambda_approval=0.0), and log(0) = -inf poisons the tensor.
        _FLOOR = 1e-4

        # Use log-space for positivity (exp recovery), logit-space for two_sided (sigmoid recovery)
        if constraint_type in ["wealth", "both", "social", "dm", "two_sided"]:
            if constraint_type == "two_sided":
                # init_lambda_wealth is the desired starting alpha ∈ (0, 1); store as logit
                alpha0 = float(np.clip(init_lambda_wealth, _FLOOR, 1 - _FLOOR))
                init_val = float(np.log(alpha0 / (1.0 - alpha0)))
            else:
                init_val = float(np.log(max(init_lambda_wealth, _FLOOR)))
            self.log_lambda_wealth = nn.Parameter(torch.tensor(init_val))
        else:
            self.register_buffer(
                "log_lambda_wealth",
                torch.tensor(np.log(max(init_lambda_wealth, _FLOOR))),
            )

        if constraint_type in ["approval_rate", "both", "predictive"]:
            self.log_lambda_approval = nn.Parameter(
                torch.tensor(np.log(max(init_lambda_approval, _FLOOR)))
            )
        else:
            self.register_buffer(
                "log_lambda_approval",
                torch.tensor(np.log(max(init_lambda_approval, _FLOOR))),
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
        lambda_lr=1e-3,
        alpha_lr=None,
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
        self.lambda_lr = lambda_lr
        # alpha (two_sided blend weight) needs its own rate -- see __init__ docs
        # and _dual_ascent_step. Bounded in (0,1) + normalised signal, so a
        # shared lambda_lr saturates it while barely moving the unbounded lambdas.
        self.alpha_lr = alpha_lr if alpha_lr is not None else lambda_lr / 4.0
        # Status-quo violation references for dual ascent; see _baseline().
        self._violation_baseline = {}
        if reward_function != "utilitarian_profit" or constraint_type == "two_sided":
            self.learnable_lambdas = LearnableLambdas(
                constraint_type=constraint_type,
                init_lambda_wealth=lambda_wealth,
                init_lambda_approval=lambda_approval,
            ).to(self.device)
            # NOT used for the actual lambda update anymore (kept only so
            # save_model/load_model's optimizer-state checkpointing doesn't
            # need a schema change) -- see _update_lambdas(). Differentiating
            # a (lambda * violation) loss w.r.t. log_lambda gives a gradient
            # proportional to lambda itself (chain rule through
            # lambda=exp(log_lambda)), so ANY gradient-based optimizer
            # applied there -- Adam, SGD, doesn't matter -- turns the
            # intended additive dual-ascent step lambda += lr*violation into
            # a multiplicative, compounding-in-lambda one, causing unbounded
            # (Adam: exponential; raw SGD: worse, near finite-time-blowup)
            # growth regardless of how small the actual violation is.
            # Verified directly on a 500-episode run. _update_lambdas() now
            # applies the textbook additive update directly to lambda,
            # bypassing this optimizer and the log-space autograd path
            # entirely.
            self.lambda_optimizer = optim.SGD(
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
        """
        Batched: one policy_net forward pass per timestep-cohort (all
        applicants arriving that step decided in one call), not one per
        applicant -- see PePGAgentV2._collect_episode for the matching
        design on the PePG side, and environment.py's step_cohort() for
        the environment-side interface both agents now share. Everything
        downstream of the flat per-decision states/actions/log_probs/
        entropies/rewards lists is unchanged: they're populated in the
        same chronological order, just filled in per-cohort instead of
        per-applicant.
        """
        obs, _ = self.env.reset_cohort()

        mu_M_start = self.env.mu_R
        mu_F_start = self.env.mu_B

        if self.initial_mu_M is None:
            self.initial_mu_M = mu_M_start
            self.initial_mu_F = mu_F_start

        states, actions, log_probs, entropies, rewards = [], [], [], [], []
        raw_rewards = []  # unnormalized -- for episode_reward/logging only, see below
        cohort_sizes = []  # applicants per timestep, for per-timestep discounting
        done = False

        lambda_w, lambda_a = self._get_current_lambdas()

        while not done:
            n = obs.shape[0]
            if n == 0:
                next_obs, terminated, truncated, info = self.env.step_cohort(np.zeros(0))
                done = terminated or truncated
                obs = next_obs
                continue

            obs_tensor = torch.from_numpy(obs).float().to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                alpha, beta = self.policy_net(obs_tensor)

            dist = Beta(alpha.squeeze(-1), beta.squeeze(-1))
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            next_obs, terminated, truncated, info = self.env.step_cohort(
                action.detach().cpu().numpy()
            )
            done = terminated or truncated

            reward_arr = compute_batched_rewards(
                self.reward_func_name,
                info["reward_snapshot"],
                info["actions"],
                info["default_probs"],
                info["loan_amounts"],
                constraint_type=self.constraint_type,
                lambda_wealth=lambda_w,
                lambda_approval=lambda_a,
            )

            for i in range(n):
                states.append(obs_tensor[i])
                actions.append(action[i])
                log_probs.append(log_prob[i])
                entropies.append(entropy[i])

                reward = float(reward_arr[i])
                # Online reward normalisation (Welford) — removes scale differences
                # across constraint types and amplifies within-episode variation for
                # state-based rewards (e.g. mu_R + mu_B) that are otherwise near-constant.
                # Updated per-decision (not per-cohort) to match the original statistics.
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
                raw_rewards.append(reward)
                self.per_step_rewards.append(reward)

            # Cohort boundary: every applicant in this cohort arrived at the
            # SAME env timestamp, so they must not be discounted against each
            # other -- see the returns computation below.
            cohort_sizes.append(n)

            obs = next_obs

        # Compute returns, discounting ONCE PER TIMESTEP (cohort) rather than
        # once per applicant.
        #
        # Every applicant in a cohort shares the same self.current_time
        # (_generate_timestep_applications stamps them identically), so
        # discounting the 15th against the 1st would discount across zero
        # elapsed time. Worse, per-applicant discounting makes the effective
        # horizon depend on things it must not:
        #   - arrival volume (stochastic, driven by lambda), so the bank's
        #     time preference would vary with how busy the day was; and
        #   - POPULATION SIZE, since arrivals now scale with N. At gamma=0.99
        #     a ~100-decision horizon is ~20 timesteps at N=3000 but only ~5
        #     at N=12000 -- scaling the population would silently make the
        #     agent 4x more myopic.
        # Per-timestep discounting has neither problem, and matches the
        # paper's "T*dt decision points per episode" (= 200, per timestep).
        # Decision granularity is unchanged: the bank still evaluates every
        # applicant individually.
        returns = []
        R = 0
        idx = len(rewards)
        for n_c in reversed(cohort_sizes):
            R *= self.gamma                      # one discount step per timestep
            cohort_rewards = rewards[idx - n_c: idx]
            R = R + sum(cohort_rewards)          # cohort's rewards are simultaneous
            # every applicant in this cohort sees the same return-to-go
            returns[0:0] = [R] * n_c
            idx -= n_c

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

        # Raw (unnormalized) sum -- comparable across reward types AND across
        # agents (PePGAgent's episode_reward is likewise raw; see
        # pepg/agent.py's _collect_episode). The training signal itself still
        # uses the normalized `rewards`/`returns` computed above -- only what
        # gets recorded/displayed/plotted changes here.
        episode_reward = sum(raw_rewards)
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

    def _baseline(self, key: str, value: float) -> float:
        """Status-quo violation reference, captured on first use --
        see PePGAgentV2._baseline for the full rationale (identical
        semantics, kept in sync so PG and PePG are comparable)."""
        if key not in self._violation_baseline:
            self._violation_baseline[key] = float(value)
        return self._violation_baseline[key]

    def _update_lambdas(self):
        """
        Update learnable lambdas via dual ascent against the status-quo
        baseline: lambda += lr * (violation - baseline), applied directly
        and additively to lambda itself, clamped positive (or to (0,1) for
        the two_sided alpha blend, whose signal is additionally normalised
        since it is a bounded weight) -- not through log-space autograd.
        See __init__'s comment on self.lambda_optimizer for why the
        autograd path is broken regardless of which optimizer applies it,
        and PePGAgentV2._baseline for why the baseline is needed.
        """
        approval_rate_M = self.env.total_loans_R / max(self.env.total_applications_R, 1)
        approval_rate_F = self.env.total_loans_B / max(self.env.total_applications_B, 1)

        wealth_gap = abs(self.env.mu_R - self.env.mu_B)
        rate_gap = abs(approval_rate_M - approval_rate_F)
        ll = self.learnable_lambdas
        lr = self.lambda_lr

        eps = 1e-4
        with torch.no_grad():
            if self.constraint_type == "two_sided":
                base = self._baseline("wealth", wealth_gap)
                signal = (wealth_gap - base) / max(abs(base), eps)  # dimensionless
                # alpha uses alpha_lr, NOT lambda_lr: it is bounded in (0,1)
                # and takes a normalised signal, so the shared rate that suits
                # the unbounded lambdas exhausts alpha's whole range.
                alpha = ll.lambda_wealth.item()
                alpha_new = float(np.clip(alpha + self.alpha_lr * signal, eps, 1 - eps))
                ll.log_lambda_wealth.copy_(
                    torch.log(torch.tensor(alpha_new / (1 - alpha_new)))
                )

            elif self.constraint_type in ("wealth", "social"):
                base = self._baseline("wealth", wealth_gap)
                lw = ll.lambda_wealth.item()
                lw_new = max(lw + lr * (wealth_gap - base), eps)
                ll.log_lambda_wealth.copy_(torch.log(torch.tensor(lw_new)))

            elif self.constraint_type in ("approval_rate", "predictive"):
                base = self._baseline("rate", rate_gap)
                la = ll.lambda_approval.item()
                la_new = max(la + lr * (rate_gap - base), eps)
                ll.log_lambda_approval.copy_(torch.log(torch.tensor(la_new)))

            elif self.constraint_type == "both":
                bw = self._baseline("wealth", wealth_gap)
                br = self._baseline("rate", rate_gap)
                lw = ll.lambda_wealth.item()
                la = ll.lambda_approval.item()
                lw_new = max(lw + lr * (wealth_gap - bw), eps)
                la_new = max(la + lr * (rate_gap - br), eps)
                ll.log_lambda_wealth.copy_(torch.log(torch.tensor(lw_new)))
                ll.log_lambda_approval.copy_(torch.log(torch.tensor(la_new)))

            elif self.constraint_type == "dm":
                rho_R = self.env.total_defaults_R / max(self.env.total_loans_R, 1)
                rho_B = self.env.total_defaults_B / max(self.env.total_loans_B, 1)
                r_R = self.env.interest_rate * (1 - rho_R) - rho_R
                r_B = self.env.interest_rate * (1 - rho_B) - rho_B
                profit_rate_gap = abs(r_R - r_B)
                base = self._baseline("dm", profit_rate_gap)
                lw = ll.lambda_wealth.item()
                lw_new = max(lw + lr * (profit_rate_gap - base), eps)
                ll.log_lambda_wealth.copy_(torch.log(torch.tensor(lw_new)))

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
        desc = f"PG {self.reward_func_name}/{self.constraint_type}"
        pbar = tqdm(range(num_episodes), desc=desc, unit="ep")
        for episode in pbar:
            episode_reward = self.train_episode()

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
            wealth_gap = (
                self.episode_metrics["wealth_gap"][-1]
                if self.episode_metrics["wealth_gap"]
                else 0
            )
            pbar.set_postfix(
                R=f"{episode_reward:.1f}",
                avgR=f"{avg_reward:.1f}",
                rho=f"{rho:.2f}",
                gap=f"{wealth_gap:.2f}",
                lw=f"{lambda_w:.3f}",
            )

        return self.episode_rewards
