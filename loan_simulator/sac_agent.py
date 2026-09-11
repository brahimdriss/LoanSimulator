"""
Soft Actor-Critic agent for the loan simulator -- a plain (non-performative)
RL baseline alongside PolicyGradientAgent (REINFORCE), sharing everything
that is not the learning rule:

  * the SAME Beta actor (loan_simulator/policy_net.py BetaPolicyNet: fixed
    input scaling, pre-activation guard, concentration cap, AdamW head
    decay), so cluster/probe_policies.py and the deploy-artifact layout work
    unchanged and the two agents' policies are directly comparable;
  * the SAME per-applicant reward (reward.compute_batched_rewards) and the
    SAME Table 1 dual-ascent lambda update (PolicyGradientAgent._update_
    lambdas), inherited as-is;
  * the SAME episode bookkeeping (_finish_episode), so episode_rewards /
    lambda_history / episode_metrics and save_model's checkpoint keys mean
    the same thing as for PG.

What is different is the update: off-policy soft actor-critic with twin Q
critics + Polyak targets + automatic temperature tuning, trained from a
replay buffer instead of on-policy returns-to-go.

MDP structure. The environment is stepped per COHORT (all applicants
arriving at one timestep decided in one step_cohort call). Each applicant
decision is one (state, action, reward): state = that applicant's 12-dim
observation (own features + shared population state), action = approval
probability in (0,1), reward = that applicant's compute_batched_rewards
value. There is no per-applicant "next state" (the next cohort is a
different set of people), so the bootstrap uses the mean soft value of the
NEXT cohort, V(s') = mean_j [ min Q_targ(s'_j, a'_j) - alpha log pi(a'_j|s'_j) ],
shared by every applicant of the current cohort -- a mean-field TD target.
Empty cohorts carry no decisions and are skipped; a cohort whose next
non-empty cohort never arrives before the episode ends bootstraps to 0.
Rewards are Welford-normalised per applicant (mean and std over everything
seen so far); a constant shift only offsets every Q by c/(1-gamma) and does
not change the policy, the scale is what matters for a critic.

Not performative: like PG, this agent never differentiates through the
population's response. pg_adapt.py's pipeline (static IncomeEnvironment
for Phase 1, performative TestingIncomeEnvironment for deploy) is reused
via sac_adapt.py, which just points pg_adapt at this class.
"""

from collections import deque
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from .agent import PolicyGradientAgent
from .policy_net import OBS_SCALE
from .reward import compute_batched_rewards

ACT_EPS = 1e-4   # keep Beta log_prob finite at the (0,1) boundary


class QNet(nn.Module):
    """Q(s, a): the same fixed input scaling as BetaPolicyNet, action
    concatenated after scaling, two hidden layers, scalar output."""

    def __init__(self, input_dim: int = 12, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.register_buffer("obs_scale", torch.tensor(OBS_SCALE, dtype=torch.float32))

    def forward(self, obs, act):
        x = torch.cat([obs / self.obs_scale, act.reshape(-1, 1)], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(-1)


class CohortReplayBuffer:
    """Cohort-level transitions: (obs [n,12], act [n], rew [n], next_obs
    [m,12] or None, done). Sampling returns flat per-applicant tensors plus
    the cohort index of each applicant, so the next-cohort mean value can be
    scattered back onto the applicants it bootstraps."""

    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buf)

    def push(self, obs, act, rew, next_obs, done):
        self.buf.append((
            np.asarray(obs, dtype=np.float32),
            np.asarray(act, dtype=np.float32),
            np.asarray(rew, dtype=np.float32),
            None if next_obs is None or len(next_obs) == 0 else np.asarray(next_obs, dtype=np.float32),
            bool(done),
        ))

    def sample(self, k, rng, device):
        idx = rng.choice(len(self.buf), size=min(k, len(self.buf)), replace=False)
        obs, act, rew, cid = [], [], [], []
        nobs, ncid, done = [], [], []
        for c, i in enumerate(idx):
            o, a, r, no, d = self.buf[i]
            obs.append(o); act.append(a); rew.append(r)
            cid.append(np.full(len(o), c, dtype=np.int64))
            terminal = d or no is None
            done.append(1.0 if terminal else 0.0)
            if not terminal:
                nobs.append(no); ncid.append(np.full(len(no), c, dtype=np.int64))
        t = lambda arrs, dt=torch.float32: torch.from_numpy(np.concatenate(arrs)).to(device=device, dtype=dt)
        batch = {
            "obs": t(obs), "act": t(act), "rew": t(rew),
            "cid": t(cid, torch.int64),
            "done": torch.tensor(done, dtype=torch.float32, device=device),
            "n_cohorts": len(idx),
        }
        if nobs:
            batch["nobs"] = t(nobs); batch["ncid"] = t(ncid, torch.int64)
        else:
            batch["nobs"] = None; batch["ncid"] = None
        return batch


class SACAgent(PolicyGradientAgent):
    """Soft actor-critic on the shared Beta actor. See module docstring."""

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
        entropy_coef=1e-3,       # accepted for signature parity; SAC's temperature is sac_alpha below
        use_amp=True,
        # --- SAC-specific ---
        gamma=0.99,
        tau=0.005,
        critic_lr=None,
        sac_alpha=0.2,
        auto_alpha=True,
        target_entropy=-1.0,     # Beta on (0,1): uniform has entropy 0, Beta(5,5) ~ -0.6, Beta(50,50) ~ -1.7
        buffer_size=10000,       # cohorts (~50 episodes at ~200 non-empty cohorts each)
        batch_cohorts=32,
        updates_per_episode=50,
        warmup_cohorts=100,
    ):
        super().__init__(
            env, hidden_dim=hidden_dim, lr=lr, reward_function=reward_function,
            constraint_type=constraint_type, lambda_wealth=lambda_wealth,
            lambda_approval=lambda_approval, lambda_lr=lambda_lr, alpha_lr=alpha_lr,
            entropy_coef=entropy_coef, use_amp=use_amp,
        )
        self.agent_label = "SAC"
        self.gamma = gamma
        self.tau = tau
        self.batch_cohorts = batch_cohorts
        self.updates_per_episode = updates_per_episode
        self.warmup_cohorts = warmup_cohorts
        self.target_entropy = float(target_entropy)
        self.auto_alpha = auto_alpha

        self.q1 = QNet(12, hidden_dim).to(self.device)
        self.q2 = QNet(12, hidden_dim).to(self.device)
        self.q1_targ = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_targ = copy.deepcopy(self.q2).requires_grad_(False)
        clr = critic_lr if critic_lr is not None else lr
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=clr
        )
        self.log_alpha = torch.tensor(math.log(sac_alpha), device=self.device,
                                      requires_grad=bool(auto_alpha))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=clr) if auto_alpha else None

        self.buffer = CohortReplayBuffer(buffer_size)
        self._rng = np.random.default_rng(int(torch.initial_seed()) % (2 ** 32))
        # Per-applicant reward Welford stats (mean, M2, n) -- see module docstring.
        self._sac_rew_n = 0
        self._sac_rew_mean = 0.0
        self._sac_rew_M2 = 0.0
        self.sac_stats = {"critic_loss": [], "actor_loss": [], "alpha": [], "q_mean": []}

    # ------------------------------------------------------------------ utils
    @property
    def sac_alpha(self):
        return float(self.log_alpha.exp().item())

    def _norm_rewards(self, r):
        """Welford-merge this cohort's rewards into the running stats (Chan's
        parallel formula), then return (r - mean) / std."""
        r = np.asarray(r, dtype=np.float64)
        n_b = len(r)
        if n_b:
            mean_b = float(r.mean())
            M2_b = float(((r - mean_b) ** 2).sum())
            n_a = self._sac_rew_n
            delta = mean_b - self._sac_rew_mean
            n = n_a + n_b
            self._sac_rew_mean += delta * n_b / n
            self._sac_rew_M2 += M2_b + delta * delta * n_a * n_b / n
            self._sac_rew_n = n
        std = math.sqrt(self._sac_rew_M2 / max(self._sac_rew_n - 1, 1)) if self._sac_rew_n > 1 else 1.0
        return ((r - self._sac_rew_mean) / max(std, 1e-8)).astype(np.float32)

    def _dist(self, obs_t):
        alpha, beta = self.policy_net(obs_t)
        return Beta(alpha.squeeze(-1), beta.squeeze(-1))

    # --------------------------------------------------------------- rollout
    def train_episode(self):
        obs, _ = self.env.reset_cohort()
        mu_M_start, mu_F_start = self.env.mu_R, self.env.mu_B
        if self.initial_mu_M is None:
            self.initial_mu_M, self.initial_mu_F = mu_M_start, mu_F_start

        lambda_w, lambda_a = self._get_current_lambdas()
        raw_rewards = []
        pending = None   # (obs, act, rew_norm) of the last non-empty cohort, awaiting its next state
        done = False

        while not done:
            n = obs.shape[0]
            if n == 0:
                obs, terminated, truncated, _ = self.env.step_cohort(np.zeros(0))
                done = terminated or truncated
                continue

            if pending is not None:            # this non-empty cohort is the previous one's s'
                self.buffer.push(*pending, obs, False)
                pending = None

            obs_t = torch.from_numpy(obs).float().to(self.device)
            with torch.no_grad():
                action = self._dist(obs_t).sample().clamp(ACT_EPS, 1 - ACT_EPS)
            act_np = action.cpu().numpy()

            next_obs, terminated, truncated, info = self.env.step_cohort(act_np)
            done = terminated or truncated

            reward_arr = compute_batched_rewards(
                self.reward_func_name, info["reward_snapshot"], info["actions"],
                info["default_probs"], info["loan_amounts"],
                constraint_type=self.constraint_type,
                lambda_wealth=lambda_w, lambda_approval=lambda_a,
                groups=info["groups"], wealth_gains=info["wealth_gains"],
            )
            cohort_reward = float(np.sum(reward_arr))
            raw_rewards.append(cohort_reward)
            self.per_step_rewards.append(cohort_reward)

            pending = (obs, act_np, self._norm_rewards(reward_arr))
            obs = next_obs

        if pending is not None:                # episode ended before another non-empty cohort
            self.buffer.push(*pending, None, True)

        # ------------------------------------------------------------ learn
        if len(self.buffer) >= self.warmup_cohorts:
            for _ in range(self.updates_per_episode):
                self._update()

        if self.learnable_lambdas is not None:
            self._update_lambdas(
                dW_R=self.env.N_male * (self.env.mu_R - mu_M_start),
                dW_B=self.env.N_female * (self.env.mu_B - mu_F_start),
            )
        return self._finish_episode(mu_M_start, mu_F_start, raw_rewards)

    # ---------------------------------------------------------------- update
    def _update(self):
        b = self.buffer.sample(self.batch_cohorts, self._rng, self.device)
        alpha = self.log_alpha.exp().detach()

        # --- TD target: mean soft value of the next cohort, per cohort ---
        with torch.no_grad():
            v_cohort = torch.zeros(b["n_cohorts"], device=self.device)
            if b["nobs"] is not None:
                ndist = self._dist(b["nobs"])
                na = ndist.sample().clamp(ACT_EPS, 1 - ACT_EPS)
                nlogp = ndist.log_prob(na)
                nq = torch.min(self.q1_targ(b["nobs"], na), self.q2_targ(b["nobs"], na))
                v = nq - alpha * nlogp
                counts = torch.zeros(b["n_cohorts"], device=self.device).index_add_(
                    0, b["ncid"], torch.ones_like(v))
                v_cohort.index_add_(0, b["ncid"], v)
                v_cohort = v_cohort / counts.clamp(min=1.0)
            y = b["rew"] + self.gamma * (1.0 - b["done"][b["cid"]]) * v_cohort[b["cid"]]

        # --- critics ---
        q1 = self.q1(b["obs"], b["act"])
        q2 = self.q2(b["obs"], b["act"])
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 1.0)
        self.critic_optimizer.step()

        # --- actor (reparameterised Beta sample) ---
        dist = self._dist(b["obs"])
        a = dist.rsample().clamp(ACT_EPS, 1 - ACT_EPS)
        logp = dist.log_prob(a)
        q_pi = torch.min(self.q1(b["obs"], a), self.q2(b["obs"], a))
        actor_loss = (alpha * logp - q_pi).mean()
        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # --- temperature ---
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # --- Polyak targets ---
        with torch.no_grad():
            for net, targ in ((self.q1, self.q1_targ), (self.q2, self.q2_targ)):
                for p, pt in zip(net.parameters(), targ.parameters()):
                    pt.mul_(1.0 - self.tau).add_(self.tau * p)

        self.sac_stats["critic_loss"].append(float(critic_loss.item()))
        self.sac_stats["actor_loss"].append(float(actor_loss.item()))
        self.sac_stats["alpha"].append(self.sac_alpha)
        self.sac_stats["q_mean"].append(float(q1.mean().item()))

    # ------------------------------------------------------------ persistence
    def _extra_state(self):
        return {
            "sac_q1_state_dict": self.q1.state_dict(),
            "sac_q2_state_dict": self.q2.state_dict(),
            "sac_q1_targ_state_dict": self.q1_targ.state_dict(),
            "sac_q2_targ_state_dict": self.q2_targ.state_dict(),
            "sac_critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "sac_log_alpha": float(self.log_alpha.item()),
            "sac_alpha_optimizer_state_dict": (
                self.alpha_optimizer.state_dict() if self.alpha_optimizer is not None else None),
            "sac_reward_stats": (self._sac_rew_n, self._sac_rew_mean, self._sac_rew_M2),
            "sac_stats": self.sac_stats,
            "agent_kind": "sac",
        }

    def save_model(self, filepath):
        """PolicyGradientAgent.save_model's checkpoint (same keys, so every
        downstream reader works unchanged) plus the critics / targets /
        temperature / reward-normaliser state."""
        super().save_model(filepath)
        ck = torch.load(filepath, map_location=self.device, weights_only=False)
        ck.update(self._extra_state())
        torch.save(ck, filepath)
        return filepath

    def load_extra_state(self, ck):
        """Restore what PolicyGradientAgent.load_model / pg_adapt's deploy
        loader don't know about. Safe no-op on a checkpoint without SAC keys
        (e.g. a PG one), so the deploy hook can call it unconditionally."""
        if "sac_q1_state_dict" not in ck:
            return False
        self.q1.load_state_dict(ck["sac_q1_state_dict"])
        self.q2.load_state_dict(ck["sac_q2_state_dict"])
        self.q1_targ.load_state_dict(ck["sac_q1_targ_state_dict"])
        self.q2_targ.load_state_dict(ck["sac_q2_targ_state_dict"])
        self.critic_optimizer.load_state_dict(ck["sac_critic_optimizer_state_dict"])
        with torch.no_grad():
            self.log_alpha.fill_(float(ck["sac_log_alpha"]))
        if self.alpha_optimizer is not None and ck.get("sac_alpha_optimizer_state_dict"):
            self.alpha_optimizer.load_state_dict(ck["sac_alpha_optimizer_state_dict"])
        if "sac_reward_stats" in ck:
            self._sac_rew_n, self._sac_rew_mean, self._sac_rew_M2 = ck["sac_reward_stats"]
        return True

    def load_model(self, filepath):
        ck = super().load_model(filepath)
        self.load_extra_state(ck)
        return ck
