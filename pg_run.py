#!/usr/bin/env python3
"""
Policy gradient performative training + frozen deployment on TestingIncomeEnvironment.

Trains PG agents on TestingIncomeEnvironment (performative, state preserved across
episodes) for all valid (reward_function, constraint_type) combos across multiple
seeds, then deploys trained weights frozen (no gradient updates) on the same env.

Valid combos (14 of 16 — skips undefined cells):
  Skip: (utilitarian_profit, social), (social_welfare, dm)

Plots use colour = reward_function, linestyle = constraint_type.

Usage
-----
  python pg_run.py --seeds 3 --train-episodes 100 --test-episodes 50
  python pg_run.py --seed-list 0 7 42 --train-episodes 200 --test-episodes 100
  python pg_run.py --reward social_welfare --constraint social --seeds 3
"""

import argparse
import multiprocessing as mp
import os
import random
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from loan_simulator.testing.data_loader import TestingAdultIncomeDataLoader
from loan_simulator.testing.environment import TestingIncomeEnvironment
from loan_simulator.transition_learner import TransitionParameterLearner
from loan_simulator.pepg import PePGAgentV2
from run_multi_seed import (
    add_derived_columns,
    aggregate_across_seeds,
    _band,
    _finish_axes,
    _save,
)
from test_rule_based_policies import setup_plot_style


# ---------------------------------------------------------------------------
# Valid (reward_function, constraint_type) combos
# ---------------------------------------------------------------------------

VALID_COMBOS = [
    ("utilitarian_profit",  "predictive"),
    ("utilitarian_profit",  "dm"),
    ("utilitarian_profit",  "two_sided"),
    ("social_welfare",      "predictive"),
    ("social_welfare",      "social"),
    ("social_welfare",      "two_sided"),
    ("rawlsian_maximin",    "predictive"),
    ("rawlsian_maximin",    "social"),
    ("rawlsian_maximin",    "dm"),
    ("rawlsian_maximin",    "two_sided"),
    ("fairness_lagrangian", "predictive"),
    ("fairness_lagrangian", "social"),
    ("fairness_lagrangian", "dm"),
    ("fairness_lagrangian", "two_sided"),
]

# colour = reward function
REWARD_COLORS = {
    "utilitarian_profit":  "#1f77b4",
    "social_welfare":      "#2ca02c",
    "rawlsian_maximin":    "#ff7f0e",
    "fairness_lagrangian": "#9467bd",
}

# linestyle = constraint type
CONSTRAINT_STYLES = {
    "predictive": "-",
    "social":     "--",
    "dm":         ":",
    "two_sided":  "-.",
}

REWARD_LABELS = {
    "utilitarian_profit":  "Util. Profit",
    "social_welfare":      "Social Welfare",
    "rawlsian_maximin":    "Rawlsian Max-Min",
    "fairness_lagrangian": "Fairness Lagrangian",
}

CONSTRAINT_LABELS = {
    "predictive": "Predictive",
    "social":     "Social",
    "dm":         "DM",
    "two_sided":  "Two-Sided",
}


def _combo_label(reward, constraint):
    return f"{REWARD_LABELS[reward]} / {CONSTRAINT_LABELS[constraint]}"


def _combo_key(reward, constraint):
    return f"{reward}__{constraint}"


# ---------------------------------------------------------------------------
# Standalone policy network (mirrors agent.py _build_network exactly)
# ---------------------------------------------------------------------------

class _PolicyNet(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head  = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x     = F.relu(self.fc1(x))
        x     = F.relu(self.fc2(x))
        alpha = F.softplus(self.alpha_head(x)) + 1.0
        beta  = F.softplus(self.beta_head(x))  + 1.0
        return alpha, beta


def _get_action(policy_net, device, obs):
    """Sample a single action from the policy (no gradient)."""
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        alpha, beta = policy_net(obs_t)
    dist = Beta(alpha, beta)
    return dist.sample().cpu().item()


# ---------------------------------------------------------------------------
# Training worker
# ---------------------------------------------------------------------------

def _train_worker(cfg):
    """
    Train one PG agent for (seed, reward_function, constraint_type).
    Returns a result dict.
    """
    seed       = cfg["seed"]
    reward     = cfg["reward_function"]
    constraint = cfg["constraint_type"]
    run_id     = cfg.get("run_id", 0)
    total      = cfg.get("total_runs", 1)

    try:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        loader = TestingAdultIncomeDataLoader(
            filepath=cfg["data_filepath"],
            sample_size=20000,
            credit_threshold=cfg.get("credit_threshold", 0.5),
        )
        loader.load_data()
        loader.preprocess()

        theta = TransitionParameterLearner(
            default_rate_min=cfg.get("default_rate_min", 0.14),
            default_rate_max=cfg.get("default_rate_max", 0.16),
        )
        theta.fit(loader.data)

        # Skip if weights already exist (allows partial restart)
        weights_path = cfg.get("weights_path")
        if weights_path and os.path.exists(weights_path):
            print(
                f"  [{run_id:3d}/{total}] TRAIN SKIP (weights exist)  "
                f"seed={seed}  {reward}/{constraint}"
            )
            train_df = None
            tmp = cfg.get("train_metrics_path")
            if tmp and os.path.exists(tmp):
                try:
                    train_df = pd.read_csv(tmp)
                except Exception:
                    pass
            return {"success": True, "seed": seed, "reward": reward,
                    "constraint": constraint, "weights_path": weights_path,
                    "train_df": train_df}

        env = TestingIncomeEnvironment(
            theta_params=theta,
            initial_wealth_male=loader.male_data["X"].values,
            initial_wealth_female=loader.female_data["X"].values,
            ground_truth_male=loader.male_data["ground_truth_approval"].values,
            ground_truth_female=loader.female_data["ground_truth_approval"].values,
            N_male=cfg["N_male"],
            N_female=cfg["N_female"],
            T=cfg["T"],
            dt=cfg["dt"],
            seed=seed,
        )

        agent = PePGAgentV2(
            env,
            hidden_dim=cfg.get("hidden_dim", 128),
            lr=cfg.get("lr", 1e-3),
            reward_function=reward,
            constraint_type=constraint,
            lambda_wealth=cfg.get("lambda_wealth", 0.5 if constraint == "two_sided" else 2.0),
            lambda_approval=cfg.get("lambda_approval", 2.0),
            lambda_lr=cfg.get("lambda_lr", 1e-2),
            buffer_capacity=cfg.get("buffer_capacity", 50),
            warmup_episodes=cfg.get("warmup_episodes", 0),
            alpha_R=env.alpha_R,
            alpha_B=env.alpha_B,
            beta_R=env.beta_R,
            beta_B=env.beta_B,
        )

        agent.train(num_episodes=cfg["train_episodes"], use_performative=True)

        train_df = agent.get_episode_metrics_dataframe()
        train_metrics_path = cfg.get("train_metrics_path")
        if train_metrics_path:
            train_df.to_csv(train_metrics_path, index=False)

        weights_path = cfg.get("weights_path")
        if weights_path:
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            agent.save_model(weights_path)

        print(
            f"  [{run_id:3d}/{total}] TRAIN OK  "
            f"seed={seed}  {reward}/{constraint}  "
            f"eps={cfg['train_episodes']}"
        )
        return {
            "success": True,
            "seed": seed,
            "reward": reward,
            "constraint": constraint,
            "weights_path": weights_path,
            "train_df": train_df,
        }

    except Exception as exc:
        import traceback
        print(
            f"  [{run_id}/{total}] TRAIN FAIL  "
            f"seed={seed}  {reward}/{constraint}: {exc}"
        )
        traceback.print_exc()
        return {
            "success": False,
            "seed": seed,
            "reward": reward,
            "constraint": constraint,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Testing worker
# ---------------------------------------------------------------------------

def _test_worker(cfg):
    """
    Test a trained PG agent for (seed, reward_function, constraint_type).
    Returns (combo_key, DataFrame) or None on failure.
    """
    seed         = cfg["seed"]
    reward       = cfg["reward_function"]
    constraint   = cfg["constraint_type"]
    weights_path = cfg["weights_path"]
    run_id       = cfg.get("run_id", 0)
    total        = cfg.get("total_runs", 1)

    try:
        np.random.seed(seed)

        # Data arrays and theta are pre-loaded in main() and passed via cfg
        # to avoid redundant I/O across N_combos × N_seeds workers.
        theta         = cfg["theta"]
        male_X        = cfg["male_X"]
        female_X      = cfg["female_X"]
        gt_male       = cfg["gt_male"]
        gt_female     = cfg["gt_female"]

        env = TestingIncomeEnvironment(
            theta_params=theta,
            initial_wealth_male=male_X,
            initial_wealth_female=female_X,
            ground_truth_male=gt_male,
            ground_truth_female=gt_female,
            N_male=cfg["N_male"],
            N_female=cfg["N_female"],
            T=cfg["T"],
            dt=cfg["dt"],
            seed=seed,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_net = _PolicyNet(
            input_dim=12, hidden_dim=cfg.get("hidden_dim", 128)
        ).to(device)
        checkpoint = torch.load(
            weights_path, map_location=device, weights_only=False
        )
        policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        policy_net.eval()

        test_eps = cfg["test_episodes"]
        for ep in range(test_eps):
            obs, _ = env.reset()
            done = False
            while not done:
                action = _get_action(policy_net, device, obs)
                obs, _, term, trunc, _ = env.step(np.array([action]))
                done = term or trunc

            if (ep + 1) % max(1, test_eps // 5) == 0 or ep + 1 == test_eps:
                app_M = env.episode_loans_M / max(env.episode_applications_M, 1)
                app_F = env.episode_loans_F / max(env.episode_applications_F, 1)
                print(
                    f"  [{run_id:3d}/{total}] TEST  "
                    f"seed={seed}  {reward[:4]}/{constraint[:4]}  "
                    f"ep {ep+1}/{test_eps}  "
                    f"μ_M={env.mu_M:.1f}  μ_F={env.mu_F:.1f}  "
                    f"appM={app_M:.3f} appF={app_F:.3f}"
                )

        env.finalize_episode_metrics()
        df = env.get_episode_metrics_dataframe()
        key = _combo_key(reward, constraint)
        print(f"  [{run_id:3d}/{total}] TEST OK  seed={seed}  {reward}/{constraint}")
        return key, df

    except Exception as exc:
        import traceback
        print(
            f"  [{run_id}/{total}] TEST FAIL  "
            f"seed={seed}  {reward}/{constraint}: {exc}"
        )
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Aggregated plots  (same structure as run_multi_seed.py)
# ---------------------------------------------------------------------------

def _iter_combos(aggregated, constraint_filter=None):
    """Yield (reward, constraint, mean_df, std_df) for each entry."""
    for key, (mdf, sdf) in aggregated.items():
        reward, constraint = key.split("__", 1)
        if constraint_filter is not None and constraint != constraint_filter:
            continue
        yield reward, constraint, mdf, sdf


def _build_combined_df(train_df, test_df):
    """Concatenate train and test episode metrics, relabeling test episode numbers."""
    n_train = len(train_df)
    train_adj = train_df.copy()
    train_adj["phase"] = "train"
    test_adj = test_df.copy()
    test_adj["episode"] = test_df["episode"] + n_train
    test_adj["phase"] = "test"
    return pd.concat([train_adj, test_adj], ignore_index=True)


def plot_comparison_agg(aggregated, results_dir, timestamp, n_seeds, constraint_filter=None, prefix="", boundary_episode=None):
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ctitle = CONSTRAINT_LABELS[constraint_filter] if constraint_filter else "All Constraints"
    fig.suptitle(
        f"PG Policies — {ctitle}  (mean ± std, {n_seeds} seeds)",
        fontsize=13, fontweight="bold",
    )

    for reward, _, mdf, sdf in _iter_combos(aggregated, constraint_filter):
        color = REWARD_COLORS[reward]
        label = REWARD_LABELS[reward]
        ep    = mdf["episode"]

        _band(axes[0, 0], ep, mdf["wealth_gap"],        sdf["wealth_gap"],        label, color, lw=1.5)
        axes[0, 0].collections[-1].set_alpha(0.12)

        _band(axes[0, 1], ep, mdf["approval_disparity"], sdf["approval_disparity"], label, color, lw=1.5)
        axes[0, 1].collections[-1].set_alpha(0.12)

        _band(axes[1, 0], ep, mdf["profit_episode"],    sdf["profit_episode"],    label, color, lw=1.5)
        axes[1, 0].collections[-1].set_alpha(0.12)

        axes[1, 1].plot(ep, mdf["R_M"], label=f"{label} (M)", color=color, lw=1.5)
        axes[1, 1].fill_between(ep, mdf["R_M"] - sdf["R_M"], mdf["R_M"] + sdf["R_M"], color=color, alpha=0.1)
        axes[1, 1].plot(ep, mdf["R_F"], color=color, lw=1.2, alpha=0.6)
        axes[1, 1].fill_between(ep, mdf["R_F"] - sdf["R_F"], mdf["R_F"] + sdf["R_F"], color=color, alpha=0.07)

    for ax in axes.flat:
        ax.axhline(0, color="k", linewidth=0.8, ls="-")

    if boundary_episode is not None:
        for ax in axes.flat:
            ax.axvline(boundary_episode, color="gray", ls="--", lw=1.0, alpha=0.6, label="Train/Test")

    axes[0, 0].set_ylabel("μ_M − μ_F  ($k)");   axes[0, 0].set_title("(a) Wealth Gap")
    axes[0, 1].set_ylabel("Approval M − F");     axes[0, 1].set_title("(b) Approval Rate Disparity")
    axes[1, 0].set_ylabel("Episode profit ($k)"); axes[1, 0].set_title("(c) Episode Profit")
    axes[1, 1].set_ylabel("R_g = (μ_g,t − μ_g,0) / t")
    axes[1, 1].set_title("(d) Long-term Social Welfare R_g(t)")

    _finish_axes(axes.flat)
    plt.tight_layout()
    suffix = f"_{constraint_filter}" if constraint_filter else ""
    _save(fig, os.path.join(results_dir, f"{prefix}pg_comparison{suffix}_{timestamp}.png"))


def plot_wealth_agg(aggregated, results_dir, timestamp, n_seeds, constraint_filter=None, prefix="", boundary_episode=None):
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ctitle = CONSTRAINT_LABELS[constraint_filter] if constraint_filter else "All Constraints"
    fig.suptitle(
        f"PG Metric Trajectories — {ctitle}  (mean ± std, {n_seeds} seeds)",
        fontsize=13, fontweight="bold",
    )

    for reward, _, mdf, sdf in _iter_combos(aggregated, constraint_filter):
        color = REWARD_COLORS[reward]
        label = REWARD_LABELS[reward]
        ep    = mdf["episode"]

        for ax, col in zip(
            [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]],
            ["mu_M_end", "mu_F_end", "cumulative_profit", "rho_episode"],
        ):
            _band(ax, ep, mdf[col], sdf[col], label, color, lw=1.5)
            ax.collections[-1].set_alpha(0.12)

    axes[1, 0].axhline(0, color="k", linewidth=0.8, ls="-")
    axes[1, 1].axhline(1, color="k", linewidth=0.8, ls="-", label="ρ = 1 (equal growth)")
    axes[1, 1].set_ylim(-5, 10)

    if boundary_episode is not None:
        for ax in axes.flat:
            ax.axvline(boundary_episode, color="gray", ls="--", lw=1.0, alpha=0.6, label="Train/Test")

    axes[0, 0].set_ylabel("Mean wealth ($k)");      axes[0, 0].set_title("(a) μ_M — Male Mean Wealth")
    axes[0, 1].set_ylabel("Mean wealth ($k)");      axes[0, 1].set_title("(b) μ_F — Female Mean Wealth")
    axes[1, 0].set_ylabel("Cumulative profit ($k)"); axes[1, 0].set_title("(c) Cumulative Bank Profit")
    axes[1, 1].set_ylabel("ρ = Δμ_M / Δμ_F");      axes[1, 1].set_title("(d) Inequality Ratio ρ (per episode)")

    _finish_axes(axes.flat)
    plt.tight_layout()
    suffix = f"_{constraint_filter}" if constraint_filter else ""
    _save(fig, os.path.join(results_dir, f"{prefix}pg_metric_trajectories{suffix}_{timestamp}.png"))


def plot_social_welfare_agg(aggregated, results_dir, timestamp, n_seeds, constraint_filter=None, prefix="", boundary_episode=None):
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ctitle = CONSTRAINT_LABELS[constraint_filter] if constraint_filter else "All Constraints"
    fig.suptitle(
        f"PG Long-term Social Welfare Trajectories — {ctitle}  (mean ± std, {n_seeds} seeds)",
        fontsize=13, fontweight="bold",
    )

    # First pass: collect y extents for shared range
    y_min, y_max = np.inf, -np.inf
    plot_data = []
    for reward, constraint, mdf, sdf in _iter_combos(aggregated, constraint_filter):
        N_M    = mdf["total_applications_M"]
        N_F    = mdf["total_applications_F"]
        total_N = (N_M + N_F).replace(0, np.nan)
        R_bar_mean = (N_M * mdf["R_M"] + N_F * mdf["R_F"]) / total_N
        R_bar_std  = np.sqrt((N_M * sdf["R_M"])**2 + (N_F * sdf["R_F"])**2) / total_N
        plot_data.append((reward, constraint, mdf, sdf, R_bar_mean, R_bar_std))
        for mean, std in [
            (mdf["R_M"], sdf["R_M"]),
            (mdf["R_F"], sdf["R_F"]),
            (R_bar_mean, R_bar_std),
        ]:
            y_min = min(y_min, (mean - std).min())
            y_max = max(y_max, (mean + std).max())

    margin = (y_max - y_min) * 0.05
    ylim = (y_min - margin, y_max + margin)

    # Second pass: draw
    for reward, _, mdf, sdf, R_bar_mean, R_bar_std in plot_data:
        color = REWARD_COLORS[reward]
        label = REWARD_LABELS[reward]
        ep    = mdf["episode"]

        for ax, mean, std in zip(
            axes,
            [mdf["R_M"],  mdf["R_F"],  R_bar_mean],
            [sdf["R_M"],  sdf["R_F"],  R_bar_std],
        ):
            _band(ax, ep, mean, std, label, color, lw=1.5)
            ax.collections[-1].set_alpha(0.12)

    for ax in axes:
        ax.axhline(0, color="k", linewidth=0.8, ls="-")
        ax.set_ylim(ylim)

    if boundary_episode is not None:
        for ax in axes:
            ax.axvline(boundary_episode, color="gray", ls="--", lw=1.0, alpha=0.6, label="Train/Test")

    axes[0].set_ylabel(r"$R_M$  (welfare / episode)")
    axes[1].set_ylabel(r"$R_F$  (welfare / episode)")
    axes[2].set_ylabel(r"$\bar{R}$  (welfare / episode)")
    axes[0].set_title("(a) Male Long-term Social Welfare $R_M$")
    axes[1].set_title("(b) Female Long-term Social Welfare $R_F$")
    axes[2].set_title(r"(c) Weighted Average $\bar{R}$")

    _finish_axes(axes)
    plt.tight_layout()
    suffix = f"_{constraint_filter}" if constraint_filter else ""
    _save(fig, os.path.join(results_dir, f"{prefix}pg_social_welfare{suffix}_{timestamp}.png"))


def plot_inequality_agg(aggregated, results_dir, timestamp, n_seeds, constraint_filter=None, prefix="", boundary_episode=None):
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ctitle = CONSTRAINT_LABELS[constraint_filter] if constraint_filter else "All Constraints"
    fig.suptitle(
        f"PG Wealth Inequality & Disparity Trajectories — {ctitle}  (mean ± std, {n_seeds} seeds)",
        fontsize=13, fontweight="bold",
    )

    for reward, _, mdf, sdf in _iter_combos(aggregated, constraint_filter):
        color = REWARD_COLORS[reward]
        label = REWARD_LABELS[reward]
        ep    = mdf["episode"]

        for ax, col in zip(
            axes,
            ["wealth_gap", "approval_disparity", "rho_cumulative"],
        ):
            _band(ax, ep, mdf[col], sdf[col], label, color, lw=1.5)
            ax.collections[-1].set_alpha(0.12)

    axes[0].axhline(0, color="k", linewidth=0.8, ls="-")
    axes[1].axhline(0, color="k", linewidth=0.8, ls="-")
    axes[2].axhline(1, color="red", linewidth=1.0, ls="--", alpha=0.7, label="ρ = 1 (equal growth)")

    if boundary_episode is not None:
        for ax in axes:
            ax.axvline(boundary_episode, color="gray", ls="--", lw=1.0, alpha=0.6, label="Train/Test")

    axes[0].set_ylabel("μ_M − μ_F  ($k)")
    axes[1].set_ylabel("Approval rate  M − F")
    axes[2].set_ylabel(r"$\rho(t)$")
    axes[0].set_title("(a) Wealth Gap Trajectory")
    axes[1].set_title("(b) Approval Rate Disparity Trajectory")
    axes[2].set_title(r"(c) Inequality Ratio $\rho(t)$")

    _finish_axes(axes)
    plt.tight_layout()
    suffix = f"_{constraint_filter}" if constraint_filter else ""
    _save(fig, os.path.join(results_dir, f"{prefix}pg_inequality{suffix}_{timestamp}.png"))


def print_summary_table(aggregated, n_seeds):
    print(f"\n{'=' * 90}")
    print(
        f"{'Combo':<44} {'R_M mean±std':>16} {'R_F mean±std':>16} {'profit ($k)':>12}"
    )
    print(f"{'—' * 90}")
    for reward, constraint, mdf, sdf in _iter_combos(aggregated):
        label  = _combo_label(reward, constraint)
        R_M_m  = mdf["R_M"].iloc[-1];     R_M_s  = sdf["R_M"].iloc[-1]
        R_F_m  = mdf["R_F"].iloc[-1];     R_F_s  = sdf["R_F"].iloc[-1]
        pr_m   = mdf["cumulative_profit"].iloc[-1]
        pr_s   = sdf["cumulative_profit"].iloc[-1]
        print(
            f"  {label:<42} "
            f"{R_M_m:+.4f}±{R_M_s:.4f}  "
            f"{R_F_m:+.4f}±{R_F_s:.4f}  "
            f"{pr_m:>8.1f}±{pr_s:.1f}"
        )
    print(f"{'=' * 90}")
    print(f"  ({n_seeds} seeds)\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed PG training + testing on performative environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Seed spec
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seeds", type=int, default=3,
        help="Number of seeds (0 … N-1).  Default: 3",
    )
    seed_group.add_argument(
        "--seed-list", type=int, nargs="+", metavar="S",
        help="Explicit list of seeds, e.g. --seed-list 0 7 42",
    )

    # Episode counts
    parser.add_argument("--train-episodes", type=int, default=100,
                        help="Training episodes per combo (default: 100)")
    parser.add_argument("--test-episodes",  type=int, default=50,
                        help="Testing episodes per combo (default: 50)")
    parser.add_argument("--warmup",         type=int, default=0,
                        help="Random warmup episodes before training (default: 0)")

    # Environment
    parser.add_argument("--N-male",   type=int,   default=3000)
    parser.add_argument("--N-female", type=int,   default=3000)
    parser.add_argument("--T",        type=int,   default=100)
    parser.add_argument("--dt",       type=float, default=0.5)
    parser.add_argument("--data",     type=str,   default=None,
                        help="Path to adult.csv (None = auto-download)")

    # Agent hyperparams
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--hidden-dim",     type=int,   default=128)
    parser.add_argument("--lambda-wealth",  type=float, default=2.0)
    parser.add_argument("--lambda-approval",type=float, default=2.0)
    parser.add_argument("--lambda-lr",      type=float, default=1e-2)
    parser.add_argument("--entropy-coef",   type=float, default=0.01,
                        help="Entropy bonus coefficient (default: 0.01)")
    parser.add_argument("--buffer-capacity", type=int, default=50,
                        help="PePG replay buffer capacity in episodes (default: 50)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip Phase 1 — use existing weights in --weights-dir")

    # Combo filter (optional)
    parser.add_argument(
        "--reward", type=str, default="all",
        choices=["all", "utilitarian_profit", "social_welfare",
                 "rawlsian_maximin", "fairness_lagrangian"],
        help="Restrict to a single reward function (default: all)",
    )
    parser.add_argument(
        "--constraint", type=str, default="all",
        choices=["all", "predictive", "social", "dm", "two_sided"],
        help="Restrict to a single constraint type (default: all)",
    )

    # Workers
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel workers (default: 80%% of CPUs, max 20)",
    )

    # Output
    parser.add_argument("--weights-dir",  type=str, default="./weights_pg")
    parser.add_argument("--results-dir",  type=str, default="./pg_results")
    parser.add_argument("--no-plots",     action="store_true")
    parser.add_argument("--save-per-seed-csv", action="store_true",
                        help="Also save one CSV per combo per seed")
    parser.add_argument("--credit-threshold", type=float, default=0.5)

    args = parser.parse_args()

    seeds   = args.seed_list if args.seed_list is not None else list(range(args.seeds))
    n_seeds = len(seeds)

    # Apply combo filter
    combos = VALID_COMBOS
    if args.reward != "all":
        combos = [(r, c) for r, c in combos if r == args.reward]
    if args.constraint != "all":
        combos = [(r, c) for r, c in combos if c == args.constraint]

    if not combos:
        print("No valid combos match the requested filter. Exiting.")
        return

    if args.workers is None:
        args.workers = min(int(mp.cpu_count() * 0.8), 20)

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("PG MULTI-SEED TRAINING + TESTING")
    print("=" * 70)
    print(f"  Seeds          : {seeds}")
    print(f"  Combos         : {len(combos)}")
    print(f"  Train episodes : {args.train_episodes}")
    print(f"  Test episodes  : {args.test_episodes}")
    print(f"  Warmup         : {args.warmup}")
    print(f"  Workers        : {args.workers}")
    print(f"  Population     : {args.N_male}M + {args.N_female}F")
    print(f"  Results dir    : {args.results_dir}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 1: Training
    # ------------------------------------------------------------------
    if not args.skip_train:
        print(f"\n[1/3] Training {len(seeds) * len(combos)} runs in parallel…")

        train_configs = []
        for seed in seeds:
            for reward, constraint in combos:
                weights_path = os.path.join(
                    args.weights_dir,
                    f"{reward}__{constraint}__seed{seed}.pt",
                )
                cfg = {
                    "seed":            seed,
                    "reward_function": reward,
                    "constraint_type": constraint,
                    "data_filepath":   args.data,
                    "train_episodes":  args.train_episodes,
                    "warmup_episodes": args.warmup,
                    "N_male":          args.N_male,
                    "N_female":        args.N_female,
                    "T":               args.T,
                    "dt":              args.dt,
                    "lr":              args.lr,
                    "hidden_dim":      args.hidden_dim,
                    "lambda_wealth":   args.lambda_wealth,
                    "lambda_approval": args.lambda_approval,
                    "lambda_lr":       args.lambda_lr,
                    "entropy_coef":    args.entropy_coef,
                    "buffer_capacity": args.buffer_capacity,
                    "credit_threshold": args.credit_threshold,
                    "weights_path":    weights_path,
                    "train_metrics_path": os.path.join(
                        args.weights_dir,
                        f"train_metrics_{reward}__{constraint}__seed{seed}.csv"
                    ),
                    "run_id":          len(train_configs) + 1,
                    "total_runs":      len(seeds) * len(combos),
                }
                train_configs.append(cfg)

        with mp.Pool(processes=args.workers) as pool:
            train_results = pool.map(_train_worker, train_configs)

        n_ok = sum(1 for r in train_results if r["success"])
        print(f"\n  Training done: {n_ok}/{len(train_configs)} successful")

        seed_to_train = defaultdict(dict)
        for tr in train_results:
            if tr["success"] and tr.get("train_df") is not None:
                key = _combo_key(tr["reward"], tr["constraint"])
                seed_to_train[tr["seed"]][key] = add_derived_columns(tr["train_df"])
    else:
        print(f"\n[1/3] --skip-train: using existing weights in {args.weights_dir}")
        train_results = []
        for seed in seeds:
            for reward, constraint in combos:
                wp = os.path.join(args.weights_dir, f"{reward}__{constraint}__seed{seed}.pt")
                exists = os.path.exists(wp)
                if not exists:
                    print(f"  MISSING: {wp}")
                train_results.append({
                    "success":      exists,
                    "seed":         seed,
                    "reward":       reward,
                    "constraint":   constraint,
                    "weights_path": wp,
                })
        n_ok = sum(1 for r in train_results if r["success"])
        print(f"  Found {n_ok}/{len(train_results)} weight files")

        seed_to_train = defaultdict(dict)
        for seed in seeds:
            for reward, constraint in combos:
                tmp = os.path.join(
                    args.weights_dir,
                    f"train_metrics_{reward}__{constraint}__seed{seed}.csv"
                )
                if os.path.exists(tmp):
                    try:
                        key = _combo_key(reward, constraint)
                        seed_to_train[seed][key] = add_derived_columns(pd.read_csv(tmp))
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Phase 2: Testing
    # ------------------------------------------------------------------
    print(f"\n[2/3] Testing {n_ok} trained agents…")

    # Pre-load test data once — passed to every worker to avoid redundant I/O
    print("  Loading test data (once)…")
    test_loader = TestingAdultIncomeDataLoader(
        filepath=args.data,
        sample_size=20000,
        credit_threshold=args.credit_threshold,
    )
    test_loader.load_data()
    test_loader.preprocess()
    test_theta = TransitionParameterLearner(
        default_rate_min=0.14, default_rate_max=0.16
    )
    test_theta.fit(test_loader.data)
    _male_X  = test_loader.male_data["X"].values
    _female_X = test_loader.female_data["X"].values
    _gt_male  = test_loader.male_data["ground_truth_approval"].values
    _gt_female = test_loader.female_data["ground_truth_approval"].values

    test_configs = []
    for tr in train_results:
        if not tr["success"]:
            continue
        cfg = {
            "seed":            tr["seed"],
            "reward_function": tr["reward"],
            "constraint_type": tr["constraint"],
            "weights_path":    tr["weights_path"],
            "test_episodes":   args.test_episodes,
            "N_male":          args.N_male,
            "N_female":        args.N_female,
            "T":               args.T,
            "dt":              args.dt,
            "hidden_dim":      args.hidden_dim,
            # Pre-loaded arrays — no I/O inside worker
            "theta":           test_theta,
            "male_X":          _male_X,
            "female_X":        _female_X,
            "gt_male":         _gt_male,
            "gt_female":       _gt_female,
            "run_id":          len(test_configs) + 1,
            "total_runs":      n_ok,
        }
        test_configs.append(cfg)

    with mp.Pool(processes=args.workers) as pool:
        test_results_raw = pool.map(_test_worker, test_configs)

    # Organise test results: seed_results[i] = dict[combo_key -> df]
    # We'll build per-seed dicts
    seed_to_results = defaultdict(dict)
    for i, raw in enumerate(test_results_raw):
        if raw is None:
            continue
        key, df = raw
        seed = test_configs[i]["seed"]
        df = add_derived_columns(df)
        seed_to_results[seed][key] = df

        if args.save_per_seed_csv:
            path = os.path.join(
                args.results_dir, f"seed{seed}_{key}_{timestamp}.csv"
            )
            df.to_csv(path, index=False)

    # Keep only seeds that have all combos
    expected_keys = {_combo_key(r, c) for r, c in combos}
    seed_results = [
        seed_to_results[s]
        for s in seeds
        if expected_keys.issubset(seed_to_results[s].keys())
    ]

    if not seed_results:
        print("No complete seed results to aggregate. Exiting.")
        return

    n_complete = len(seed_results)
    print(f"\n  {n_complete}/{n_seeds} seeds complete with all combos")

    # ------------------------------------------------------------------
    # Phase 3: Aggregate + Plot
    # ------------------------------------------------------------------
    print(f"\n[3/3] Aggregating mean ± std across {n_complete} seeds…")
    aggregated = aggregate_across_seeds(seed_results)

    # Save aggregated CSVs
    for key, (mdf, sdf) in aggregated.items():
        mdf.to_csv(
            os.path.join(args.results_dir, f"mean_{key}_{timestamp}.csv"),
            index=False,
        )
        sdf.to_csv(
            os.path.join(args.results_dir, f"std_{key}_{timestamp}.csv"),
            index=False,
        )
    print(f"  Aggregated CSVs saved to {args.results_dir}/")

    print_summary_table(aggregated, n_complete)

    if not args.no_plots:
        print("  Generating aggregated plots…")
        for ct in ["predictive", "social", "dm", "two_sided"]:
            plot_comparison_agg(aggregated, args.results_dir, timestamp, n_complete, ct)
            plot_wealth_agg(aggregated, args.results_dir, timestamp, n_complete, ct)
            plot_social_welfare_agg(aggregated, args.results_dir, timestamp, n_complete, ct)
            plot_inequality_agg(aggregated, args.results_dir, timestamp, n_complete, ct)

    # --- Training plots ---
    train_seed_results = [
        seed_to_train[s]
        for s in seeds
        if expected_keys.issubset(seed_to_train[s].keys())
    ]
    if train_seed_results:
        n_train_complete = len(train_seed_results)
        print(f"  Generating training plots ({n_train_complete} seeds)…")
        train_aggregated = aggregate_across_seeds(train_seed_results)
        for key, (mdf, sdf) in train_aggregated.items():
            mdf.to_csv(os.path.join(args.results_dir, f"train_mean_{key}_{timestamp}.csv"), index=False)
            sdf.to_csv(os.path.join(args.results_dir, f"train_std_{key}_{timestamp}.csv"), index=False)
        if not args.no_plots:
            for ct in ["predictive", "social", "dm", "two_sided"]:
                plot_comparison_agg(train_aggregated, args.results_dir, timestamp, n_train_complete, ct, prefix="train_")
                plot_wealth_agg(train_aggregated, args.results_dir, timestamp, n_train_complete, ct, prefix="train_")
                plot_social_welfare_agg(train_aggregated, args.results_dir, timestamp, n_train_complete, ct, prefix="train_")
                plot_inequality_agg(train_aggregated, args.results_dir, timestamp, n_train_complete, ct, prefix="train_")

    # --- Combined plots ---
    combined_seed_results = []
    for s in seeds:
        td = seed_to_train.get(s, {})
        xd = seed_to_results.get(s, {})
        if not (expected_keys.issubset(td.keys()) and expected_keys.issubset(xd.keys())):
            continue
        combined_seed_results.append(
            {key: _build_combined_df(td[key], xd[key]) for key in expected_keys}
        )
    if combined_seed_results:
        n_combined = len(combined_seed_results)
        print(f"  Generating combined (train+test) plots ({n_combined} seeds)…")
        combined_aggregated = aggregate_across_seeds(combined_seed_results)
        for key, (mdf, sdf) in combined_aggregated.items():
            mdf.to_csv(os.path.join(args.results_dir, f"combined_mean_{key}_{timestamp}.csv"), index=False)
            sdf.to_csv(os.path.join(args.results_dir, f"combined_std_{key}_{timestamp}.csv"), index=False)
        if not args.no_plots:
            for ct in ["predictive", "social", "dm", "two_sided"]:
                plot_comparison_agg(combined_aggregated, args.results_dir, timestamp, n_combined, ct, prefix="combined_", boundary_episode=args.train_episodes)
                plot_wealth_agg(combined_aggregated, args.results_dir, timestamp, n_combined, ct, prefix="combined_", boundary_episode=args.train_episodes)
                plot_social_welfare_agg(combined_aggregated, args.results_dir, timestamp, n_combined, ct, prefix="combined_", boundary_episode=args.train_episodes)
                plot_inequality_agg(combined_aggregated, args.results_dir, timestamp, n_combined, ct, prefix="combined_", boundary_episode=args.train_episodes)

    print("\nDone.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
