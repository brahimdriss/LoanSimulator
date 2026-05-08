#!/usr/bin/env python3
"""
individual_loan_frequency.py

Tracks which individuals receive loans across performative episodes for PG and PEPG.
Agents continue fine-tuning on TestingIncomeEnvironment throughout (same as pg_adapt /
pepg_adapt deploy phase).

Per-individual unique identifier: array index (0 .. N-1) within each gender group.

Saves for each seed / policy / group:
  results_individual_freq/{policy}/seed{s}/{group}/lorenz/episode_{ep:04d}.csv
  results_individual_freq/{policy}/seed{s}/{group}/metrics.csv
  results_individual_freq/{policy}/seed{s}/{group}/final_loan_counts.csv

Generates plots (averaged across seeds):
  {policy}_{group}_lorenz.png          -- Lorenz curves at ep 100,200,...,1000
  {policy}_{group}_entropy.png         -- Normalised Shannon entropy over episodes
  {policy}_{group}_reach_rate.png          -- unique loan recipients per episode / N
  {policy}_{group}_histogram.png       -- total loans by binned individual index
"""

import argparse
import multiprocessing as mp
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from loan_simulator.testing.data_loader import TestingAdultIncomeDataLoader
from loan_simulator.testing.environment import TestingIncomeEnvironment
from loan_simulator.transition_learner import TransitionParameterLearner
from loan_simulator.agent import PolicyGradientAgent
from loan_simulator.pepg import PePGAgentV2


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------

class TrackingEnvironment(TestingIncomeEnvironment):
    """
    Thin subclass that intercepts step() to record which individual gets each
    approved loan, without changing any environment dynamics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cumulative loan counts per individual — persist across all episodes
        self.loan_counts_M = np.zeros(self.N_male,  dtype=np.int64)
        self.loan_counts_F = np.zeros(self.N_female, dtype=np.int64)
        # Per-episode unique recipient sets — reset each episode
        self._ep_unique_M = set()
        self._ep_unique_F = set()

    def reset(self, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self._ep_unique_M = set()
        self._ep_unique_F = set()
        return result

    def step(self, action):
        applicant      = self.current_applicant
        loans_M_before = self.episode_loans_M
        loans_F_before = self.episode_loans_F

        result = super().step(action)

        if applicant is not None:
            idx = applicant["individual_id"]
            if applicant["S"] == 1 and self.episode_loans_M > loans_M_before:
                self.loan_counts_M[idx] += 1
                self._ep_unique_M.add(idx)
            elif applicant["S"] == 0 and self.episode_loans_F > loans_F_before:
                self.loan_counts_F[idx] += 1
                self._ep_unique_F.add(idx)

        return result


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def compute_lorenz(counts, n_points=100):
    """
    Lorenz curve sampled at n_points evenly spaced percentile positions.

    Returns (cum_pop_fraction, cum_loan_fraction), both length n_points+1
    with a (0, 0) origin prepended.
    """
    sorted_c = np.sort(counts)
    cum      = np.cumsum(sorted_c)
    total    = cum[-1]
    n        = len(counts)

    idx      = np.linspace(0, n - 1, n_points, dtype=int)
    cum_pop  = np.concatenate([[0.0], (idx + 1) / n])
    if total > 0:
        cum_loan = np.concatenate([[0.0], cum[idx] / total])
    else:
        cum_loan = np.zeros(n_points + 1)
    return cum_pop, cum_loan


def compute_entropy(counts):
    """
    Normalised Shannon entropy of the loan-count distribution.

    Returns H / log2(N) in [0, 1].  0 = all loans to one individual,
    1 = perfectly uniform distribution.
    """
    total = counts.sum()
    if total == 0:
        return 0.0
    p     = counts[counts > 0] / total
    H     = float(-np.sum(p * np.log2(p)))
    H_max = float(np.log2(len(counts)))
    return H / H_max if H_max > 0 else 0.0


def save_lorenz_csv(cum_pop, cum_loan, directory, episode):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"episode_{episode:04d}.csv")
    pd.DataFrame({
        "cum_pop_fraction":  cum_pop,
        "cum_loan_fraction": cum_loan,
    }).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Default lambda hyperparameters (matches shell scripts)
# ---------------------------------------------------------------------------

def _default_lambdas(reward, constraint):
    if reward == "utilitarian_profit":
        lw = 0.5 if constraint == "two_sided" else 0.0
        return lw, 0.0
    lw = 0.5 if constraint == "two_sided" else (
        5.0 if reward == "rawlsian_maximin" else 2.0
    )
    la = 5.0 if reward == "rawlsian_maximin" else 2.0
    return lw, la


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(path, env, agent, ep_metrics_M, ep_metrics_F,
                     completed_ep, policy_name):
    from collections import deque as _deque
    ckpt = {
        # Progress
        "completed_episodes": completed_ep,
        "ep_metrics_M":       ep_metrics_M,
        "ep_metrics_F":       ep_metrics_F,
        # Wealth + time
        "current_X_male":         env.current_X_male.copy(),
        "current_X_female":       env.current_X_female.copy(),
        "mu_M":                   env.mu_M,
        "mu_F":                   env.mu_F,
        "prev_mu_M":              env.prev_mu_M,
        "prev_mu_F":              env.prev_mu_F,
        "mu_M_0":                 env.mu_M_0,
        "mu_F_0":                 env.mu_F_0,
        "current_time":           env.current_time,
        "total_episodes":         env.total_episodes,
        # Cumulative counters
        "total_loans_M":          env.total_loans_M,
        "total_loans_F":          env.total_loans_F,
        "total_applications_M":   env.total_applications_M,
        "total_applications_F":   env.total_applications_F,
        "total_defaults_M":       env.total_defaults_M,
        "total_defaults_F":       env.total_defaults_F,
        "cumulative_profit":      env.cumulative_profit,
        # Confusion matrix
        "tp_M": env.tp_M, "fp_M": env.fp_M,
        "tn_M": env.tn_M, "fn_M": env.fn_M,
        "tp_F": env.tp_F, "fp_F": env.fp_F,
        "tn_F": env.tn_F, "fn_F": env.fn_F,
        # Hawkes history (deque → list for pickling)
        "event_times_R": list(env.event_times_R),
        "event_times_B": list(env.event_times_B),
        # Individual tracking (TrackingEnvironment)
        "loan_counts_M":    env.loan_counts_M.copy(),
        "loan_counts_F":    env.loan_counts_F.copy(),
        # Agent — policy + optimizer
        "policy_net_state_dict":    agent.policy_net.state_dict(),
        "optimizer_state_dict":     agent.optimizer.state_dict(),
        "initial_mu_M":             agent.initial_mu_M,
        "initial_mu_F":             agent.initial_mu_F,
        "cumulative_time":          agent.cumulative_time,
        "total_episodes_completed": agent.total_episodes_completed,
    }
    # PG-only EMA baseline attributes
    if policy_name == "pg":
        ckpt["rew_ema_mean"] = agent._rew_ema_mean
        ckpt["rew_ema_var"]  = agent._rew_ema_var
        ckpt["rew_ema_n"]    = agent._rew_ema_n
    if agent.learnable_lambdas is not None:
        ckpt["learnable_lambdas_state_dict"] = agent.learnable_lambdas.state_dict()
        ckpt["lambda_optimizer_state_dict"]  = agent.lambda_optimizer.state_dict()
    if policy_name == "pepg" and hasattr(agent, "replay_buffer"):
        ckpt["replay_buffer_data"] = list(agent.replay_buffer.buffer)

    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)   # atomic replace — avoids corrupt file on crash


def _restore_from_checkpoint(path, env, agent, policy_name):
    from collections import deque as _deque
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Env state
    env.current_X_male       = ckpt["current_X_male"].copy()
    env.current_X_female     = ckpt["current_X_female"].copy()
    env.mu_M                 = ckpt["mu_M"]
    env.mu_F                 = ckpt["mu_F"]
    env.prev_mu_M            = ckpt["prev_mu_M"]
    env.prev_mu_F            = ckpt["prev_mu_F"]
    env.mu_M_0               = ckpt["mu_M_0"]
    env.mu_F_0               = ckpt["mu_F_0"]
    env.current_time         = ckpt["current_time"]
    env.total_episodes       = ckpt["total_episodes"]
    env.total_loans_M        = ckpt["total_loans_M"]
    env.total_loans_F        = ckpt["total_loans_F"]
    env.total_applications_M = ckpt["total_applications_M"]
    env.total_applications_F = ckpt["total_applications_F"]
    env.total_defaults_M     = ckpt["total_defaults_M"]
    env.total_defaults_F     = ckpt["total_defaults_F"]
    env.cumulative_profit    = ckpt["cumulative_profit"]
    env.tp_M = ckpt["tp_M"]; env.fp_M = ckpt["fp_M"]
    env.tn_M = ckpt["tn_M"]; env.fn_M = ckpt["fn_M"]
    env.tp_F = ckpt["tp_F"]; env.fp_F = ckpt["fp_F"]
    env.tn_F = ckpt["tn_F"]; env.fn_F = ckpt["fn_F"]
    env.event_times_R = _deque(ckpt["event_times_R"])
    env.event_times_B = _deque(ckpt["event_times_B"])
    # TrackingEnvironment
    env.loan_counts_M   = ckpt["loan_counts_M"].copy()
    env.loan_counts_F   = ckpt["loan_counts_F"].copy()

    # Agent state
    agent.policy_net.load_state_dict(ckpt["policy_net_state_dict"])
    agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if policy_name == "pg":
        agent._rew_ema_mean = ckpt.get("rew_ema_mean", 0.0)
        agent._rew_ema_var  = ckpt.get("rew_ema_var",  1.0)
        agent._rew_ema_n    = ckpt.get("rew_ema_n",    0)
    agent.initial_mu_M             = ckpt.get("initial_mu_M")
    agent.initial_mu_F             = ckpt.get("initial_mu_F")
    agent.cumulative_time          = ckpt.get("cumulative_time", 0.0)
    agent.total_episodes_completed = ckpt.get("total_episodes_completed", 0)
    if agent.learnable_lambdas is not None and "learnable_lambdas_state_dict" in ckpt:
        agent.learnable_lambdas.load_state_dict(ckpt["learnable_lambdas_state_dict"])
        if "lambda_optimizer_state_dict" in ckpt:
            agent.lambda_optimizer.load_state_dict(ckpt["lambda_optimizer_state_dict"])
    if policy_name == "pepg" and "replay_buffer_data" in ckpt:
        agent.replay_buffer.buffer = _deque(
            ckpt["replay_buffer_data"], maxlen=agent.buffer_capacity
        )

    return ckpt["ep_metrics_M"], ckpt["ep_metrics_F"], ckpt["completed_episodes"]


# ---------------------------------------------------------------------------
# Per-seed worker (module-level so it can be pickled by mp.Pool)
# ---------------------------------------------------------------------------

def _seed_worker(cfg):
    seed           = cfg["seed"]
    policy_name    = cfg["policy_name"]
    weights_path   = cfg["weights_path"]
    deploy_episodes = cfg["deploy_episodes"]
    seed_dir       = cfg["seed_dir"]

    checkpoint_freq = cfg.get("checkpoint_freq", 100)
    checkpoint_path = os.path.join(cfg["seed_dir"], "checkpoint.pt")

    try:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        if not os.path.exists(weights_path):
            print(f"  MISSING weights: {weights_path} — skipping seed {seed}")
            return None

        for grp in ("male", "female"):
            os.makedirs(os.path.join(seed_dir, grp, "lorenz"), exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env    = TrackingEnvironment(
            theta_params          = cfg["theta"],
            initial_wealth_male   = cfg["male_X"],
            initial_wealth_female = cfg["female_X"],
            ground_truth_male     = cfg["gt_male"],
            ground_truth_female   = cfg["gt_female"],
            N_male                = cfg["N_male"],
            N_female              = cfg["N_female"],
            T                     = cfg["T"],
            dt                    = cfg["dt"],
            seed                  = seed,
        )

        lw, la = _default_lambdas(cfg["reward_function"], cfg["constraint_type"])

        if policy_name == "pg":
            agent = PolicyGradientAgent(
                env,
                hidden_dim       = cfg["hidden_dim"],
                lr               = cfg["lr"],
                reward_function  = cfg["reward_function"],
                constraint_type  = cfg["constraint_type"],
                lambda_wealth    = lw,
                lambda_approval  = la,
                lambda_lr        = cfg["lambda_lr"],
                entropy_coef     = cfg["entropy_coef"],
            )
        else:
            agent = PePGAgentV2(
                env,
                hidden_dim        = cfg["hidden_dim"],
                lr                = cfg["lr"],
                reward_function   = cfg["reward_function"],
                constraint_type   = cfg["constraint_type"],
                lambda_wealth     = lw,
                lambda_approval   = la,
                lambda_lr         = cfg["lambda_lr"],
                buffer_capacity   = cfg["buffer_capacity"],
                warmup_episodes   = 0,
                alpha_R           = env.alpha_R,
                alpha_B           = env.alpha_B,
                beta_R            = env.beta_R,
                beta_B            = env.beta_B,
                hawkes_weight     = 1.0,
                wealth_weight     = 1.0,
                transition_weight = 1.0,
                reward_weight     = 1.0,
            )

        # Resume from checkpoint if one exists, otherwise load pre-trained weights
        if os.path.exists(checkpoint_path):
            print(f"  [seed {seed}] Resuming from checkpoint: {checkpoint_path}")
            ep_metrics_M, ep_metrics_F, start_ep = _restore_from_checkpoint(
                checkpoint_path, env, agent, policy_name
            )
            start_ep += 1
            print(f"  [seed {seed}] Continuing from episode {start_ep}")
        else:
            saved = torch.load(weights_path, map_location=device, weights_only=False)
            if policy_name == "pg":
                agent.policy_net.load_state_dict(saved["policy_net_state_dict"])
                if "lambda_state_dict" in saved and agent.learnable_lambdas is not None:
                    agent.learnable_lambdas.load_state_dict(saved["lambda_state_dict"])
            else:
                agent.policy_net.load_state_dict(saved["policy_net_state_dict"])
                if "learnable_lambdas_state_dict" in saved and agent.learnable_lambdas is not None:
                    agent.learnable_lambdas.load_state_dict(saved["learnable_lambdas_state_dict"])
                if "replay_buffer_data" in saved:
                    from collections import deque as _deque
                    agent.replay_buffer.buffer = _deque(
                        saved["replay_buffer_data"], maxlen=agent.buffer_capacity
                    )
            ep_metrics_M, ep_metrics_F = [], []
            start_ep = 1
            print(f"  [seed {seed}] Loaded weights from {weights_path}")

        for ep in range(start_ep, deploy_episodes + 1):
            if policy_name == "pepg":
                agent.train_episode(use_performative=True)
            else:
                agent.train_episode()

            counts_M   = env.loan_counts_M.copy()
            counts_F   = env.loan_counts_F.copy()
            entropy_M  = compute_entropy(counts_M)
            entropy_F  = compute_entropy(counts_F)
            # Reach rate: unique individuals who received a loan this episode / N
            reach_M = len(env._ep_unique_M) / env.N_male
            reach_F = len(env._ep_unique_F) / env.N_female

            ep_metrics_M.append({
                "episode":    ep,
                "entropy":    entropy_M,
                "reach_rate": reach_M,
            })
            ep_metrics_F.append({
                "episode":    ep,
                "entropy":    entropy_F,
                "reach_rate": reach_F,
            })

            pop_M, loan_M = compute_lorenz(counts_M)
            pop_F, loan_F = compute_lorenz(counts_F)
            save_lorenz_csv(pop_M, loan_M, os.path.join(seed_dir, "male",   "lorenz"), ep)
            save_lorenz_csv(pop_F, loan_F, os.path.join(seed_dir, "female", "lorenz"), ep)

            if ep % checkpoint_freq == 0:
                _save_checkpoint(
                    checkpoint_path, env, agent,
                    ep_metrics_M, ep_metrics_F, ep, policy_name,
                )
                # Save intermediate loan counts for post-hoc histogram analysis
                for grp, counts in [("male", env.loan_counts_M), ("female", env.loan_counts_F)]:
                    counts_dir = os.path.join(seed_dir, grp, "counts")
                    os.makedirs(counts_dir, exist_ok=True)
                    pd.DataFrame({
                        "individual_id": np.arange(len(counts)),
                        "loan_count":    counts,
                    }).to_csv(os.path.join(counts_dir, f"loan_counts_{ep:04d}.csv"), index=False)
                print(
                    f"  [seed {seed}] ep {ep:4d}/{deploy_episodes}"
                    f"  H_M={entropy_M:.4f}  H_F={entropy_F:.4f}"
                    f"  reach_M={reach_M:.4f}  reach_F={reach_F:.4f}"
                    f"  [checkpointed]"
                )

        for grp, metrics in [("male", ep_metrics_M), ("female", ep_metrics_F)]:
            pd.DataFrame(metrics).to_csv(
                os.path.join(seed_dir, grp, "metrics.csv"), index=False
            )

        for grp, counts in [
            ("male",   env.loan_counts_M),
            ("female", env.loan_counts_F),
        ]:
            pd.DataFrame({
                "individual_id": np.arange(len(counts)),
                "loan_count":    counts,
            }).to_csv(
                os.path.join(seed_dir, grp, "final_loan_counts.csv"), index=False
            )

        # Remove checkpoint once fully complete
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        print(f"  [seed {seed}] Done.")
        return {
            "seed":           seed,
            "metrics_M":      pd.DataFrame(ep_metrics_M),
            "metrics_F":      pd.DataFrame(ep_metrics_F),
            "final_counts_M": env.loan_counts_M.copy(),
            "final_counts_F": env.loan_counts_F.copy(),
        }

    except Exception as exc:
        import traceback
        print(f"  [seed {seed}] FAILED: {exc}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Run one policy across all seeds in parallel
# ---------------------------------------------------------------------------

def run_policy(
    policy_name,
    weights_dir,
    seeds,
    env_kwargs,
    deploy_episodes,
    results_dir,
    reward_function,
    constraint_type,
    hidden_dim,
    lr,
    lambda_lr,
    entropy_coef,
    buffer_capacity,
    workers,
    checkpoint_freq,
    checkpoint_dir=None,
):
    policy_dir = os.path.join(results_dir, policy_name)
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    def _weights_path(seed):
        if policy_name == "pepg":
            fname = f"pepg_{reward_function}__{constraint_type}__seed{seed}.pt"
        else:
            fname = f"{reward_function}__{constraint_type}__seed{seed}.pt"
        return os.path.join(weights_dir, fname)

    # pepg_adapt style: load existing per-seed CSV checkpoints, skip those seeds
    all_metrics      = {"male": [], "female": []}
    all_final_counts = {"male": [], "female": []}
    seeds_to_run     = []
    n_loaded         = 0

    for seed in seeds:
        ckpt_M        = os.path.join(checkpoint_dir, f"{policy_name}_seed{seed}_metrics_M.csv")
        ckpt_F        = os.path.join(checkpoint_dir, f"{policy_name}_seed{seed}_metrics_F.csv")
        ckpt_counts_M = os.path.join(checkpoint_dir, f"{policy_name}_seed{seed}_counts_M.csv")
        ckpt_counts_F = os.path.join(checkpoint_dir, f"{policy_name}_seed{seed}_counts_F.csv")

        if all(os.path.exists(p) for p in [ckpt_M, ckpt_F, ckpt_counts_M, ckpt_counts_F]):
            try:
                all_metrics["male"].append(pd.read_csv(ckpt_M))
                all_metrics["female"].append(pd.read_csv(ckpt_F))
                all_final_counts["male"].append(pd.read_csv(ckpt_counts_M)["loan_count"].values)
                all_final_counts["female"].append(pd.read_csv(ckpt_counts_F)["loan_count"].values)
                n_loaded += 1
                print(f"  [{policy_name} seed {seed}] Loaded from CSV checkpoint — skipping.")
            except Exception as exc:
                print(f"  [{policy_name} seed {seed}] CSV checkpoint corrupt ({exc}) — re-running.")
                seeds_to_run.append(seed)
        else:
            seeds_to_run.append(seed)

    if n_loaded:
        print(f"  Loaded {n_loaded} CSV checkpoint(s) — skipping those seeds.")

    # Fallback: if a seed's weights are missing, substitute the next available seed
    already_allocated = set(seeds)
    next_candidate    = max(seeds) + 1 if seeds else 0
    resolved_to_run   = []

    for seed in seeds_to_run:
        if os.path.exists(_weights_path(seed)):
            resolved_to_run.append(seed)
        else:
            found = False
            while next_candidate <= max(seeds) + 200:
                if next_candidate not in already_allocated and \
                        os.path.exists(_weights_path(next_candidate)):
                    print(f"  [{policy_name} seed {seed}] weights missing "
                          f"— substituting seed {next_candidate}")
                    resolved_to_run.append(next_candidate)
                    already_allocated.add(next_candidate)
                    next_candidate += 1
                    found = True
                    break
                next_candidate += 1
            if not found:
                print(f"  [{policy_name} seed {seed}] weights missing — no substitute found, skipping")

    seeds_to_run = resolved_to_run

    if seeds_to_run:
        seed_configs = []
        for seed in seeds_to_run:
            seed_configs.append({
                "seed":            seed,
                "policy_name":     policy_name,
                "weights_path":    _weights_path(seed),
                "deploy_episodes": deploy_episodes,
                "seed_dir":        os.path.join(policy_dir, f"seed{seed}"),
                "theta":           env_kwargs["theta_params"],
                "male_X":          env_kwargs["initial_wealth_male"],
                "female_X":        env_kwargs["initial_wealth_female"],
                "gt_male":         env_kwargs["ground_truth_male"],
                "gt_female":       env_kwargs["ground_truth_female"],
                "N_male":          env_kwargs["N_male"],
                "N_female":        env_kwargs["N_female"],
                "T":               env_kwargs["T"],
                "dt":              env_kwargs["dt"],
                "reward_function": reward_function,
                "constraint_type": constraint_type,
                "hidden_dim":      hidden_dim,
                "lr":              lr,
                "lambda_lr":       lambda_lr,
                "entropy_coef":    entropy_coef,
                "buffer_capacity": buffer_capacity,
                "checkpoint_freq": checkpoint_freq,
            })

        # imap_unordered: save each seed's CSV checkpoint as soon as it finishes
        with mp.Pool(processes=min(workers, len(seeds_to_run))) as pool:
            for res in pool.imap_unordered(_seed_worker, seed_configs):
                if res is None:
                    continue
                seed = res["seed"]

                ckpt_M        = os.path.join(checkpoint_dir, f"{policy_name}_seed{seed}_metrics_M.csv")
                ckpt_F        = os.path.join(checkpoint_dir, f"{policy_name}_seed{seed}_metrics_F.csv")
                ckpt_counts_M = os.path.join(checkpoint_dir, f"{policy_name}_seed{seed}_counts_M.csv")
                ckpt_counts_F = os.path.join(checkpoint_dir, f"{policy_name}_seed{seed}_counts_F.csv")

                res["metrics_M"].to_csv(ckpt_M, index=False)
                res["metrics_F"].to_csv(ckpt_F, index=False)
                pd.DataFrame({
                    "individual_id": np.arange(len(res["final_counts_M"])),
                    "loan_count":    res["final_counts_M"],
                }).to_csv(ckpt_counts_M, index=False)
                pd.DataFrame({
                    "individual_id": np.arange(len(res["final_counts_F"])),
                    "loan_count":    res["final_counts_F"],
                }).to_csv(ckpt_counts_F, index=False)
                print(f"  Checkpointed: {policy_name} seed{seed}")

                all_metrics["male"].append(res["metrics_M"])
                all_metrics["female"].append(res["metrics_F"])
                all_final_counts["male"].append(res["final_counts_M"])
                all_final_counts["female"].append(res["final_counts_F"])

    return all_metrics, all_final_counts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

GROUP_COLORS = {"male": "#C0392B", "female": "#2980B9"}
GROUP_LABELS = {"male": "Male (Red Group)", "female": "Female (Blue Group)"}

_GRID_KW = dict(lw=0.4, color="gray", alpha=0.35, zorder=0)


def _load_lorenz_checkpoint(policy_dir, seeds, group, ep):
    loan_curves = []
    pop_ref = None
    for seed in seeds:
        path = os.path.join(
            policy_dir, f"seed{seed}", group, "lorenz", f"episode_{ep:04d}.csv"
        )
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if pop_ref is None:
            pop_ref = df["cum_pop_fraction"].values
        loan_curves.append(df["cum_loan_fraction"].values)
    if not loan_curves:
        return None, None
    return pop_ref, np.mean(loan_curves, axis=0)


def _load_counts_episode(policy_dir, seeds, group, ep):
    """Load and average per-individual loan counts for a given episode across seeds.
    ep can be an int (intermediate) or 'final'."""
    all_counts = []
    for seed in seeds:
        if ep == "final":
            path = os.path.join(policy_dir, f"seed{seed}", group, "final_loan_counts.csv")
        else:
            path = os.path.join(
                policy_dir, f"seed{seed}", group, "counts", f"loan_counts_{ep:04d}.csv"
            )
        if not os.path.exists(path):
            continue
        all_counts.append(pd.read_csv(path)["loan_count"].values)
    if not all_counts:
        return None
    return np.mean(all_counts, axis=0)


def _bin_counts(counts, n_bins, N):
    bin_width  = N // n_bins
    edges      = [i * bin_width for i in range(n_bins)] + [N]
    totals     = np.array([counts[edges[i]:edges[i + 1]].sum() for i in range(n_bins)])
    centers    = np.array([(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)])
    width      = bin_width * 0.85
    return centers, totals, width, edges


def plot_lorenz(policy_dir, seeds, group, deploy_episodes, plots_dir, policy_name):
    checkpoints = [ep for ep in [100, 400, 800, 1000] if ep <= deploy_episodes]
    colors      = cm.Blues(np.linspace(0.35, 0.95, len(checkpoints)))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.grid(True, **_GRID_KW)

    for i, ep in enumerate(checkpoints):
        pop, loan = _load_lorenz_checkpoint(policy_dir, seeds, group, ep)
        if pop is None:
            continue
        ax.plot(pop, loan, color=colors[i], lw=1.6, label=f"ep {ep}")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect equality")
    ax.set_xlabel("Cumulative fraction of individuals")
    ax.set_ylabel("Cumulative fraction of loans received")
    ax.set_title(
        f"{policy_name.upper()} — {GROUP_LABELS[group]}\nLoan Lorenz Curves"
        f" (avg over {len(seeds)} seeds)"
    )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(plots_dir, f"{policy_name}_{group}_lorenz.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_scalar_metric(all_metrics, metric_col, ylabel, title_suffix, plots_dir, policy_name, group):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(True, **_GRID_KW)
    color = GROUP_COLORS[group]

    if all_metrics:
        episodes = all_metrics[0]["episode"].values
        values   = np.stack([df[metric_col].values for df in all_metrics])
        mean, std = values.mean(axis=0), values.std(axis=0)
        ax.plot(episodes, mean, color=color, lw=1.8, label="mean")
        ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.25, label="±1 std")

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{policy_name.upper()} — {GROUP_LABELS[group]}\n{title_suffix}"
        f" (avg over {len(all_metrics)} seeds)"
    )
    ax.legend()
    plt.tight_layout()
    out = os.path.join(plots_dir, f"{policy_name}_{group}_{metric_col}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_reach_rate_combined(all_metrics_M, all_metrics_F, plots_dir, policy_name):
    """Males and females superimposed on a single reach-rate plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(True, **_GRID_KW)

    for all_metrics, group in [(all_metrics_M, "male"), (all_metrics_F, "female")]:
        color = GROUP_COLORS[group]
        if not all_metrics:
            continue
        episodes = all_metrics[0]["episode"].values
        values   = np.stack([df["reach_rate"].values for df in all_metrics])
        mean, std = values.mean(axis=0), values.std(axis=0)
        ax.plot(episodes, mean, color=color, lw=1.8, label=GROUP_LABELS[group])
        ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reach rate  (unique recipients this episode / N)")
    n_seeds = len(all_metrics_M) or len(all_metrics_F)
    ax.set_title(f"{policy_name.upper()} — Episode Reach Rate  (avg over {n_seeds} seeds)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(plots_dir, f"{policy_name}_reach_rate.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_histogram(final_counts_list, n_bins, plots_dir, policy_name, group, N):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.grid(True, axis="y", **_GRID_KW)
    color = GROUP_COLORS[group]

    if final_counts_list:
        counts_mean        = np.mean(final_counts_list, axis=0)
        centers, totals, width, _ = _bin_counts(counts_mean, n_bins, N)
        ax.bar(centers, totals, width=width, color=color, alpha=0.75, edgecolor="none")

    ax.set_xlabel(f"Individual index (binned into {n_bins} groups of {N // n_bins} each)")
    ax.set_ylabel("Total loans received (cumulative)")
    ax.set_title(
        f"{policy_name.upper()} — {GROUP_LABELS[group]}\n"
        f"Loan Count by Individual Index  (avg over {len(final_counts_list)} seeds)"
    )
    plt.tight_layout()
    out = os.path.join(plots_dir, f"{policy_name}_{group}_histogram.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_histogram_comparison(policy_dir, seeds, n_bins, plots_dir, policy_name, group, N):
    """Episode 100 vs final histogram superimposed, single group."""
    counts_100   = _load_counts_episode(policy_dir, seeds, group, 100)
    counts_final = _load_counts_episode(policy_dir, seeds, group, "final")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.grid(True, axis="y", **_GRID_KW)
    color = GROUP_COLORS[group]

    if counts_100 is not None:
        centers, totals, width, _ = _bin_counts(counts_100, n_bins, N)
        ax.bar(centers, totals, width=width, color=color, alpha=0.35,
               edgecolor="none", label="Episode 100")
    if counts_final is not None:
        centers, totals, width, _ = _bin_counts(counts_final, n_bins, N)
        ax.bar(centers, totals, width=width, color=color, alpha=0.80,
               edgecolor="none", label="Final episode")

    ax.set_xlabel(f"Individual index (binned, {N // n_bins} per bin)")
    ax.set_ylabel("Total loans received (cumulative)")
    ax.set_title(
        f"{policy_name.upper()} — {GROUP_LABELS[group]}\n"
        f"Episode 100 vs Final  (avg over {len(seeds)} seeds)"
    )
    ax.legend()
    plt.tight_layout()
    out = os.path.join(plots_dir, f"{policy_name}_{group}_histogram_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_histogram_subplots(policy_dir, seeds, n_bins, plots_dir, policy_name, N_male, N_female):
    """Episode 100 vs final, two subplots: males (top) and females (bottom)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=False)

    for ax, group, N in zip(axes, ("male", "female"), (N_male, N_female)):
        ax.grid(True, axis="y", **_GRID_KW)
        color        = GROUP_COLORS[group]
        counts_100   = _load_counts_episode(policy_dir, seeds, group, 100)
        counts_final = _load_counts_episode(policy_dir, seeds, group, "final")

        if counts_100 is not None:
            centers, totals, width, _ = _bin_counts(counts_100, n_bins, N)
            ax.bar(centers, totals, width=width, color=color, alpha=0.35,
                   edgecolor="none", label="Episode 100")
        if counts_final is not None:
            centers, totals, width, _ = _bin_counts(counts_final, n_bins, N)
            ax.bar(centers, totals, width=width, color=color, alpha=0.80,
                   edgecolor="none", label="Final episode")

        ax.set_ylabel("Total loans received")
        ax.set_xlabel(f"Individual index ({N // n_bins} per bin)")
        ax.set_title(GROUP_LABELS[group])
        ax.legend(fontsize=9)

    fig.suptitle(
        f"{policy_name.upper()} — Loan Distribution: Episode 100 vs Final"
        f"\n(avg over {len(seeds)} seeds)",
        fontsize=12,
    )
    plt.tight_layout()
    out = os.path.join(plots_dir, f"{policy_name}_histogram_subplots.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Track per-individual loan frequency across performative episodes."
    )
    parser.add_argument("--data",              type=str,   default="adult.csv")
    parser.add_argument("--N-male",            type=int,   default=3000)
    parser.add_argument("--N-female",          type=int,   default=3000)
    parser.add_argument("--T",                 type=int,   default=100)
    parser.add_argument("--dt",                type=float, default=0.5)
    parser.add_argument("--deploy-episodes",   type=int,   default=1000)
    parser.add_argument("--seeds",             type=int,   default=3)
    parser.add_argument("--reward",            type=str,   default="utilitarian_profit")
    parser.add_argument("--constraint",        type=str,   default="two_sided")
    parser.add_argument("--hidden-dim",        type=int,   default=128)
    parser.add_argument("--lr",                type=float, default=1e-3)
    parser.add_argument("--lambda-lr",         type=float, default=1e-2)
    parser.add_argument("--entropy-coef",      type=float, default=0.01)
    parser.add_argument("--buffer-capacity",   type=int,   default=50)
    parser.add_argument("--credit-threshold",  type=float, default=0.5)
    parser.add_argument("--pg-weights-dir",    type=str,   default="weights_pg_adapt")
    parser.add_argument("--pepg-weights-dir",  type=str,   default="weights_pepg_adapt")
    parser.add_argument("--results-dir",       type=str,
                        default="/content/drive/MyDrive/LoanSimulator/results_individual_freq")
    parser.add_argument("--n-bins",            type=int,   default=30)
    parser.add_argument("--workers",           type=int,   default=None)
    parser.add_argument("--checkpoint-freq",   type=int,   default=100)
    parser.add_argument("--checkpoint-dir",    type=str,   default=None)
    parser.add_argument("--skip-pg",           action="store_true")
    parser.add_argument("--skip-pepg",         action="store_true")
    args = parser.parse_args()

    if args.workers is None:
        args.workers = min(int(mp.cpu_count() * 0.8), 20)

    seeds     = list(range(args.seeds))
    plots_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("Loading test data...")
    loader = TestingAdultIncomeDataLoader(
        filepath=args.data,
        sample_size=20000,
        credit_threshold=args.credit_threshold,
    )
    loader.load_data()
    loader.preprocess()

    theta = TransitionParameterLearner(default_rate_min=0.14, default_rate_max=0.16)
    theta.fit(loader.data)

    env_kwargs = dict(
        theta_params          = theta,
        initial_wealth_male   = loader.male_data["X"].values,
        initial_wealth_female = loader.female_data["X"].values,
        ground_truth_male     = loader.male_data["ground_truth_approval"].values,
        ground_truth_female   = loader.female_data["ground_truth_approval"].values,
        N_male                = args.N_male,
        N_female              = args.N_female,
        T                     = args.T,
        dt                    = args.dt,
    )

    checkpoint_dir = args.checkpoint_dir or os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    common_kwargs = dict(
        seeds            = seeds,
        env_kwargs       = env_kwargs,
        deploy_episodes  = args.deploy_episodes,
        results_dir      = args.results_dir,
        reward_function  = args.reward,
        constraint_type  = args.constraint,
        hidden_dim       = args.hidden_dim,
        lr               = args.lr,
        lambda_lr        = args.lambda_lr,
        entropy_coef     = args.entropy_coef,
        buffer_capacity  = args.buffer_capacity,
        workers          = args.workers,
        checkpoint_freq  = args.checkpoint_freq,
        checkpoint_dir   = checkpoint_dir,
    )

    for policy_name, weights_dir, skip in [
        ("pg",   args.pg_weights_dir,   args.skip_pg),
        ("pepg", args.pepg_weights_dir, args.skip_pepg),
    ]:
        if skip:
            continue

        print(f"\n{'='*60}")
        print(f"  {policy_name.upper()}  |  {args.reward} / {args.constraint}")
        print(f"{'='*60}")

        all_metrics, all_final_counts = run_policy(
            policy_name=policy_name,
            weights_dir=weights_dir,
            **common_kwargs,
        )

        policy_dir = os.path.join(args.results_dir, policy_name)

        # Reach rate — males and females superimposed
        print(f"\n  Plotting {policy_name} / reach rate (combined)...")
        plot_reach_rate_combined(
            all_metrics["male"], all_metrics["female"], plots_dir, policy_name
        )

        # Histogram: ep100 vs final, two subplots
        print(f"\n  Plotting {policy_name} / histogram subplots...")
        plot_histogram_subplots(
            policy_dir, seeds, args.n_bins, plots_dir, policy_name,
            args.N_male, args.N_female,
        )

        for group in ("male", "female"):
            N = args.N_male if group == "male" else args.N_female
            print(f"\n  Plotting {policy_name} / {group}...")

            plot_lorenz(
                policy_dir, seeds, group, args.deploy_episodes, plots_dir, policy_name
            )
            plot_scalar_metric(
                all_metrics[group], "entropy",
                "Normalised Shannon Entropy  H / log₂(N)",
                "Loan Distribution Entropy over Episodes",
                plots_dir, policy_name, group,
            )
            plot_histogram(
                all_final_counts[group], args.n_bins, plots_dir, policy_name, group, N
            )
            plot_histogram_comparison(
                policy_dir, seeds, args.n_bins, plots_dir, policy_name, group, N
            )

    print("\nDone.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
