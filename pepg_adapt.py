"""Entry point. See cluster/ for HTCondor submission."""

# --- BLAS thread guard -----------------------------------------------------
# MUST run before numpy/torch/matplotlib are imported: the BLAS backend reads
# these at load time and cannot be reconfigured afterwards.
#
# OpenBLAS defaults to one thread per visible core (32 on the cluster nodes).
# With multiprocessing's "spawn" start method every worker re-imports this
# module and does the same, so N workers try to create 32*N threads. On the
# MPI-IS login node -- capped at ~100 processes/threads per user -- that fails
# immediately with "blas_thread_init: pthread_create failed ... Resource
# temporarily unavailable". On an exec node it silently oversubscribes the
# slot, and CPU limits there are HARD-enforced, so it runs slower rather than
# faster.
#
# One BLAS thread per worker is right for this workload: parallelism comes
# from the (seed, combo) process pool, and the per-cohort tensors are small.
# Set these in the environment beforehand to override.
import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

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

# Cap torch intra-op threads too, honouring the guard set at the top.
torch.set_num_threads(int(_os.environ["OMP_NUM_THREADS"]))
from tqdm import tqdm

from loan_simulator.testing.data_loader import TestingAdultIncomeDataLoader
from loan_simulator.testing.environment import TestingIncomeEnvironment
from loan_simulator.transition_learner import TransitionParameterLearner
from loan_simulator.pepg import PePGAgentV2
AGENT_TAG = "pepg"

from run_multi_seed import add_derived_columns, aggregate_across_seeds
from pg_run import (
    REWARD_COLORS,
    REWARD_LABELS,
    _combo_key,
    setup_plot_style,
    _band,
    _finish_axes,
    _save,
    _build_combined_df,
    plot_comparison_agg,
    plot_wealth_agg,
    plot_social_welfare_agg,
    plot_inequality_agg,
)


# ---------------------------------------------------------------------------
# PEPG-specific combos and labels
# ---------------------------------------------------------------------------

PEPG_COMBOS = [
    ("utilitarian_profit",  "dm"),
    ("utilitarian_profit",  "two_sided"),
    ("social_welfare",      "social"),
    ("social_welfare",      "two_sided"),
    ("rawlsian_maximin",    "social"),
    ("rawlsian_maximin",    "dm"),
    ("rawlsian_maximin",    "two_sided"),
    ("fairness_lagrangian", "social"),
    ("fairness_lagrangian", "dm"),
    ("fairness_lagrangian", "two_sided"),
]

PEPG_CONSTRAINT_LABELS = {
    "social":     "Social",
    "dm":         "DM",
    "two_sided":  "Two-Sided",
}


def _pepg_combo_label(reward, constraint):
    return f"{REWARD_LABELS[reward]} / {PEPG_CONSTRAINT_LABELS[constraint]}"


def _default_lambdas(reward, constraint):
    """Initial lambda / alpha per combo.

    MUST stay byte-for-byte equivalent to pg_adapt._default_lambdas: these
    are the STARTING values of the dual variables, and PG and PePG have to
    begin from the same point or a cross-agent comparison is confounded by
    initialisation rather than measuring the learning algorithm. A previous
    version of this function returned 2.0 for all three fairness_lagrangian
    combos while pg_adapt returned 10.0 -- a 5x difference in the fairness
    penalty weight at episode 0. The values below match pg_adapt and also
    match reward.RewardFunction's own per-reward signature defaults
    (social_welfare 2.0, rawlsian_maximin 5.0, fairness_lagrangian 10.0).

    two_sided's lw is alpha, a blend weight in (0,1), initialised at 0.5.
    """
    lw = 0.5 if constraint == "two_sided" else (
        0.0 if reward == "utilitarian_profit" else
        2.0 if reward == "social_welfare" else
        5.0 if reward == "rawlsian_maximin" else
        10.0  # fairness_lagrangian
    )
    la = (
        0.0 if reward == "utilitarian_profit" else
        2.0 if reward == "social_welfare" else
        5.0 if reward == "rawlsian_maximin" else
        10.0
    )
    return lw, la


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _iter_combos(aggregated, constraint_filter=None):
    for key, (mdf, sdf) in aggregated.items():
        reward, constraint = key.split("__", 1)
        if constraint_filter is not None and constraint != constraint_filter:
            continue
        yield reward, constraint, mdf, sdf


def _plot_comparison(aggregated, results_dir, timestamp, n_seeds, constraint_filter=None):
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ctitle = PEPG_CONSTRAINT_LABELS.get(constraint_filter, "All") if constraint_filter else "All Constraints"
    fig.suptitle(
        f"PePG Adapt — {ctitle}  (mean ± std, {n_seeds} seeds)",
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
    axes[0, 0].set_ylabel("μ_M − μ_F  ($k)");    axes[0, 0].set_title("(a) Wealth Gap")
    axes[0, 1].set_ylabel("Approval M − F");      axes[0, 1].set_title("(b) Approval Rate Disparity")
    axes[1, 0].set_ylabel("Episode profit ($k)"); axes[1, 0].set_title("(c) Episode Profit")
    axes[1, 1].set_ylabel("R_g");                 axes[1, 1].set_title("(d) Long-term Social Welfare R_g(t)")
    _finish_axes(axes.flat)
    plt.tight_layout()
    suffix = f"_{constraint_filter}" if constraint_filter else ""
    _save(fig, os.path.join(results_dir, f"pepg_adapt_comparison{suffix}_{timestamp}.png"))


def _plot_wealth(aggregated, results_dir, timestamp, n_seeds, constraint_filter=None):
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ctitle = PEPG_CONSTRAINT_LABELS.get(constraint_filter, "All") if constraint_filter else "All Constraints"
    fig.suptitle(
        f"PePG Adapt Metric Trajectories — {ctitle}  (mean ± std, {n_seeds} seeds)",
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
    axes[0, 0].set_ylabel("Mean wealth ($k)");       axes[0, 0].set_title("(a) μ_M — Male Mean Wealth")
    axes[0, 1].set_ylabel("Mean wealth ($k)");       axes[0, 1].set_title("(b) μ_F — Female Mean Wealth")
    axes[1, 0].set_ylabel("Cumulative profit ($k)"); axes[1, 0].set_title("(c) Cumulative Bank Profit")
    axes[1, 1].set_ylabel("ρ = Δμ_M / Δμ_F");       axes[1, 1].set_title("(d) Inequality Ratio ρ (per episode)")
    _finish_axes(axes.flat)
    plt.tight_layout()
    suffix = f"_{constraint_filter}" if constraint_filter else ""
    _save(fig, os.path.join(results_dir, f"pepg_adapt_wealth{suffix}_{timestamp}.png"))


def _plot_social_welfare(aggregated, results_dir, timestamp, n_seeds, constraint_filter=None):
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ctitle = PEPG_CONSTRAINT_LABELS.get(constraint_filter, "All") if constraint_filter else "All Constraints"
    fig.suptitle(
        f"PePG Adapt Long-term Social Welfare — {ctitle}  (mean ± std, {n_seeds} seeds)",
        fontsize=13, fontweight="bold",
    )
    y_min, y_max = np.inf, -np.inf
    plot_data = []
    for reward, constraint, mdf, sdf in _iter_combos(aggregated, constraint_filter):
        N_M = mdf["total_applications_M"]
        N_F = mdf["total_applications_F"]
        total_N = (N_M + N_F).replace(0, np.nan)
        R_bar_mean = (N_M * mdf["R_M"] + N_F * mdf["R_F"]) / total_N
        R_bar_std  = np.sqrt((N_M * sdf["R_M"])**2 + (N_F * sdf["R_F"])**2) / total_N
        plot_data.append((reward, constraint, mdf, sdf, R_bar_mean, R_bar_std))
        for mean, std in [(mdf["R_M"], sdf["R_M"]), (mdf["R_F"], sdf["R_F"]), (R_bar_mean, R_bar_std)]:
            y_min = min(y_min, (mean - std).min())
            y_max = max(y_max, (mean + std).max())
    margin = (y_max - y_min) * 0.05
    ylim = (y_min - margin, y_max + margin)
    for reward, _, mdf, sdf, R_bar_mean, R_bar_std in plot_data:
        color = REWARD_COLORS[reward]
        label = REWARD_LABELS[reward]
        ep    = mdf["episode"]
        for ax, mean, std in zip(axes, [mdf["R_M"], mdf["R_F"], R_bar_mean], [sdf["R_M"], sdf["R_F"], R_bar_std]):
            _band(ax, ep, mean, std, label, color, lw=1.5)
            ax.collections[-1].set_alpha(0.12)
    for ax in axes:
        ax.axhline(0, color="k", linewidth=0.8, ls="-")
        ax.set_ylim(ylim)
    axes[0].set_ylabel(r"$R_M$"); axes[0].set_title("(a) Male Long-term Social Welfare $R_M$")
    axes[1].set_ylabel(r"$R_F$"); axes[1].set_title("(b) Female Long-term Social Welfare $R_F$")
    axes[2].set_ylabel(r"$\bar{R}$"); axes[2].set_title(r"(c) Weighted Average $\bar{R}$")
    _finish_axes(axes)
    plt.tight_layout()
    suffix = f"_{constraint_filter}" if constraint_filter else ""
    _save(fig, os.path.join(results_dir, f"pepg_adapt_social_welfare{suffix}_{timestamp}.png"))


def _plot_inequality(aggregated, results_dir, timestamp, n_seeds, constraint_filter=None):
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ctitle = PEPG_CONSTRAINT_LABELS.get(constraint_filter, "All") if constraint_filter else "All Constraints"
    fig.suptitle(
        f"PePG Adapt Wealth Inequality & Disparity — {ctitle}  (mean ± std, {n_seeds} seeds)",
        fontsize=13, fontweight="bold",
    )
    for reward, _, mdf, sdf in _iter_combos(aggregated, constraint_filter):
        color = REWARD_COLORS[reward]
        label = REWARD_LABELS[reward]
        ep    = mdf["episode"]
        for ax, col in zip(axes, ["wealth_gap", "approval_disparity", "rho_cumulative"]):
            _band(ax, ep, mdf[col], sdf[col], label, color, lw=1.5)
            ax.collections[-1].set_alpha(0.12)
    axes[0].axhline(0, color="k", linewidth=0.8, ls="-")
    axes[1].axhline(0, color="k", linewidth=0.8, ls="-")
    axes[2].axhline(1, color="red", linewidth=1.0, ls="--", alpha=0.7, label="ρ = 1 (equal growth)")
    axes[0].set_ylabel("μ_M − μ_F  ($k)");     axes[0].set_title("(a) Wealth Gap Trajectory")
    axes[1].set_ylabel("Approval rate M − F"); axes[1].set_title("(b) Approval Rate Disparity")
    axes[2].set_ylabel(r"$\rho(t)$");           axes[2].set_title(r"(c) Inequality Ratio $\rho(t)$")
    _finish_axes(axes)
    plt.tight_layout()
    suffix = f"_{constraint_filter}" if constraint_filter else ""
    _save(fig, os.path.join(results_dir, f"pepg_adapt_inequality{suffix}_{timestamp}.png"))


def print_summary_table(aggregated, n_seeds):
    print(f"\n{'=' * 90}")
    print(f"{'Combo':<44} {'R_M mean±std':>16} {'R_F mean±std':>16} {'profit ($k)':>12}")
    print(f"{'—' * 90}")
    for reward, constraint, mdf, sdf in _iter_combos(aggregated):
        label = _pepg_combo_label(reward, constraint)
        R_M_m = mdf["R_M"].iloc[-1];     R_M_s = sdf["R_M"].iloc[-1]
        R_F_m = mdf["R_F"].iloc[-1];     R_F_s = sdf["R_F"].iloc[-1]
        pr_m  = mdf["cumulative_profit"].iloc[-1]; pr_s = sdf["cumulative_profit"].iloc[-1]
        print(
            f"  {label:<42} "
            f"{R_M_m:+.4f}±{R_M_s:.4f}  "
            f"{R_F_m:+.4f}±{R_F_s:.4f}  "
            f"{pr_m:>8.1f}±{pr_s:.1f}"
        )
    print(f"{'=' * 90}")
    print(f"  ({n_seeds} seeds)\n")


# ---------------------------------------------------------------------------
# Phase 1 worker — TestingIncomeEnvironment (performative, state preserved)
# ---------------------------------------------------------------------------

def _train_worker(cfg):
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

        theta = TransitionParameterLearner(default_rate_min=0.05, default_rate_max=0.25)
        theta.fit(loader.data)

        # Skip if weights already exist (allows partial restart)
        weights_path = cfg["weights_path"]
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        if os.path.exists(weights_path):
            print(f"  [{run_id:3d}/{total}] TRAIN SKIP (weights exist)  seed={seed}  {reward}/{constraint}")
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

        lw, la = _default_lambdas(reward, constraint)
        agent = PePGAgentV2(
            env,
            hidden_dim=cfg.get("hidden_dim", 128),
            lr=cfg.get("lr", 1e-3),
            reward_function=reward,
            constraint_type=constraint,
            lambda_wealth=lw,
            lambda_approval=la,
            lambda_lr=cfg.get("lambda_lr", 1e-3),
            alpha_lr=cfg.get("alpha_lr", None),
            buffer_capacity=cfg.get("buffer_capacity", 50),
            warmup_episodes=cfg.get("warmup_episodes", 0),
            alpha_R=env.alpha_R,
            alpha_B=env.alpha_B,
            beta_R=env.beta_R,
            beta_B=env.beta_B,
            hawkes_weight=cfg.get("hawkes_weight", 1.0),
            wealth_weight=cfg.get("wealth_weight", 1.0),
            transition_weight=cfg.get("transition_weight", 1.0),
            reward_weight=cfg.get("reward_weight", 1.0),
            entropy_coef=cfg.get("entropy_coef", 0.01),
        )

        agent.train_reparam(num_episodes=cfg["train_episodes"])

        train_df = agent.get_episode_metrics_dataframe()
        train_metrics_path = cfg.get("train_metrics_path")
        if train_metrics_path:
            train_df.to_csv(train_metrics_path, index=False)

        agent.save_model(weights_path)

        print(
            f"  [{run_id:3d}/{total}] TRAIN OK  "
            f"seed={seed}  {reward}/{constraint}  eps={cfg['train_episodes']}"
        )
        return {"success": True, "seed": seed, "reward": reward,
                "constraint": constraint, "weights_path": weights_path,
                "train_df": train_df}

    except Exception as exc:
        import traceback
        print(f"  [{run_id}/{total}] TRAIN FAIL  seed={seed}  {reward}/{constraint}: {exc}")
        traceback.print_exc()
        return {"success": False, "seed": seed, "reward": reward, "constraint": constraint,
                "train_df": None}


# ---------------------------------------------------------------------------
# Phase 2 worker — continued performative training on TestingIncomeEnvironment
# ---------------------------------------------------------------------------

def _deploy_worker(cfg):
    seed         = cfg["seed"]
    reward       = cfg["reward_function"]
    constraint   = cfg["constraint_type"]
    weights_path = cfg["weights_path"]
    run_id       = cfg.get("run_id", 0)
    total        = cfg.get("total_runs", 1)

    try:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        env = TestingIncomeEnvironment(
            theta_params=cfg["theta"],
            initial_wealth_male=cfg["male_X"],
            initial_wealth_female=cfg["female_X"],
            ground_truth_male=cfg["gt_male"],
            ground_truth_female=cfg["gt_female"],
            N_male=cfg["N_male"],
            N_female=cfg["N_female"],
            T=cfg["T"],
            dt=cfg["dt"],
            seed=seed,
        )

        lw, la = _default_lambdas(reward, constraint)
        agent = PePGAgentV2(
            env,
            hidden_dim=cfg.get("hidden_dim", 128),
            lr=cfg.get("lr", 1e-3),
            reward_function=reward,
            constraint_type=constraint,
            lambda_wealth=lw,
            lambda_approval=la,
            lambda_lr=cfg.get("lambda_lr", 1e-3),
            alpha_lr=cfg.get("alpha_lr", None),
            buffer_capacity=cfg.get("buffer_capacity", 50),
            warmup_episodes=0,
            alpha_R=env.alpha_R,
            alpha_B=env.alpha_B,
            beta_R=env.beta_R,
            beta_B=env.beta_B,
            hawkes_weight=cfg.get("hawkes_weight", 1.0),
            wealth_weight=cfg.get("wealth_weight", 1.0),
            transition_weight=cfg.get("transition_weight", 1.0),
            reward_weight=cfg.get("reward_weight", 1.0),
            entropy_coef=cfg.get("entropy_coef", 0.01),
        )

        # Load pre-trained weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        saved  = torch.load(weights_path, map_location=device, weights_only=False)
        agent.policy_net.load_state_dict(saved["policy_net_state_dict"])
        if "learnable_lambdas_state_dict" in saved and agent.learnable_lambdas is not None:
            agent.learnable_lambdas.load_state_dict(saved["learnable_lambdas_state_dict"])
        if "replay_buffer_data" in saved:
            from collections import deque
            agent.replay_buffer.buffer = deque(
                saved["replay_buffer_data"], maxlen=agent.buffer_capacity
            )

        # Continued performative training on the testing environment.
        # train_episode_reparam(): the fixed, reparameterized gradient path
        # (see differentiable_gradient.py) -- not train_episode(), which is
        # the score-function path with structurally-dead performative terms
        # for this environment's dynamics.
        deploy_eps = cfg["deploy_episodes"]
        for ep in range(deploy_eps):
            agent.train_episode_reparam()
            if (ep + 1) % max(1, deploy_eps // 5) == 0 or ep + 1 == deploy_eps:
                app_M = env.episode_loans_M / max(env.episode_applications_M, 1)
                app_F = env.episode_loans_F / max(env.episode_applications_F, 1)
                print(
                    f"  [{run_id:3d}/{total}] DEPLOY  "
                    f"seed={seed}  {reward[:4]}/{constraint[:4]}  "
                    f"ep {ep+1}/{deploy_eps}  "
                    f"μ_M={env.mu_M:.1f}  μ_F={env.mu_F:.1f}  "
                    f"appM={app_M:.3f} appF={app_F:.3f}"
                )

        env.finalize_episode_metrics()
        df  = env.get_episode_metrics_dataframe()
        key = _combo_key(reward, constraint)

        # Persist the POST-DEPLOY artefacts. Previously only the train phase
        # saved weights, so the policy that actually produced every reported
        # result -- after `deploy_episodes` further updates -- was discarded.
        deploy_dir = cfg.get("deploy_artifacts_dir")
        if deploy_dir:
            os.makedirs(deploy_dir, exist_ok=True)
            stem = f"{AGENT_TAG}_{reward}__{constraint}__seed{seed}"
            agent.save_model(os.path.join(deploy_dir, f"{stem}_deployed.pt"))
            df.to_csv(os.path.join(deploy_dir, f"{stem}_episodes.csv"), index=False)
            pd.DataFrame({
                "episode": range(1, len(agent.episode_rewards) + 1),
                "episode_reward": agent.episode_rewards,
                "lambda_wealth": agent.lambda_history["wealth"],
                "lambda_approval": agent.lambda_history["approval"],
            }).to_csv(os.path.join(deploy_dir, f"{stem}_training_trace.csv"), index=False)
            np.savez_compressed(
                os.path.join(deploy_dir, f"{stem}_population.npz"),
                X_male=env.current_X_male, X_female=env.current_X_female,
                loan_counts_M=env.loan_counts_M, loan_counts_F=env.loan_counts_F,
            )

        print(f"  [{run_id:3d}/{total}] DEPLOY OK  seed={seed}  {reward}/{constraint}")
        return seed, key, df

    except Exception as exc:
        import traceback
        print(f"  [{run_id}/{total}] DEPLOY FAIL  seed={seed}  {reward}/{constraint}: {exc}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PePG performative pre-train → performative deploy on testing env"
    )

    # Seeds / combos
    parser.add_argument("--seeds",      type=int, default=5)  # paper: 5 seeds
    parser.add_argument("--seed-list",  type=int, nargs="+", default=None)
    parser.add_argument("--reward",     type=str, default="all",
                        choices=["all", "utilitarian_profit", "social_welfare",
                                 "rawlsian_maximin", "fairness_lagrangian"])
    parser.add_argument("--constraint", type=str, default="all",
                        choices=["all", "predictive", "social", "dm", "two_sided"])

    # Training (performative env — IncomeEnvironment with PePG gradients)
    parser.add_argument("--train-episodes",  type=int,   default=500)
    parser.add_argument("--warmup",           type=int,   default=0)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--lambda-lr",        type=float, default=1e-3)
    parser.add_argument("--alpha-lr",         type=float, default=None,
                        help="LR for the two_sided alpha blend weight. "
                             "Defaults to lambda_lr/4 -- alpha is bounded in (0,1) "
                             "and takes a normalised signal, so the rate that suits "
                             "the unbounded lambdas saturates it.")
    parser.add_argument("--entropy-coef",     type=float, default=0.01,
                        help="Exploration pressure on the shadow-rollout gradient. "
                             "Matches pg_adapt.py's default so PG/PePG aren't "
                             "compared under a hidden asymmetry.")
    parser.add_argument("--buffer-capacity",  type=int,   default=50)
    parser.add_argument("--hawkes-weight",    type=float, default=1.0)
    parser.add_argument("--wealth-weight",    type=float, default=1.0)
    parser.add_argument("--transition-weight",type=float, default=1.0)
    parser.add_argument("--reward-weight",    type=float, default=1.0)

    # Deployment (TestingIncomeEnvironment)
    parser.add_argument("--deploy-episodes",  type=int,   default=1000)  # paper: 1000-episode axis
    parser.add_argument("--credit-threshold", type=float, default=0.5)

    # Architecture
    parser.add_argument("--hidden-dim", type=int, default=128)

    # Environment
    parser.add_argument("--N-male",   type=int,   default=12000)
    parser.add_argument("--N-female", type=int,   default=12000)
    parser.add_argument("--T",        type=int,   default=100)
    parser.add_argument("--dt",       type=float, default=0.5)

    # I/O
    parser.add_argument("--data",              type=str, default=None)
    parser.add_argument("--weights-dir",       type=str, default="./weights_pepg_adapt")
    parser.add_argument("--results-dir",       type=str, default="./results_pepg_adapt")
    parser.add_argument("--checkpoint-dir",    type=str, default=None)
    parser.add_argument("--workers",           type=int, default=None)
    parser.add_argument("--no-plots",          action="store_true")
    parser.add_argument("--save-per-seed-csv", action="store_true")
    parser.add_argument("--skip-train",        action="store_true",
                        help="Skip Phase 1 — use existing weights in --weights-dir")

    args = parser.parse_args()

    seeds   = args.seed_list if args.seed_list is not None else list(range(args.seeds))
    n_seeds = len(seeds)

    combos = PEPG_COMBOS
    if args.reward != "all":
        combos = [(r, c) for r, c in combos if r == args.reward]
    if args.constraint != "all":
        combos = [(r, c) for r, c in combos if c == args.constraint]

    if not combos:
        print("No valid combos match filter. Exiting.")
        return

    if args.workers is None:
        args.workers = min(int(mp.cpu_count() * 0.8), 20)

    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("PEPG PERFORMATIVE PRE-TRAIN → PERFORMATIVE DEPLOY")
    print("=" * 70)
    print(f"  Seeds           : {seeds}")
    print(f"  Combos          : {len(combos)}")
    print(f"  Train episodes  : {args.train_episodes}  (IncomeEnvironment, performative gradients)")
    print(f"  Deploy episodes : {args.deploy_episodes} (TestingIncomeEnvironment, state preserved)")
    print(f"  Workers         : {args.workers}")
    print(f"  Population      : {args.N_male}M + {args.N_female}F")
    print(f"  Buffer capacity : {args.buffer_capacity}")
    print(f"  Weights dir     : {args.weights_dir}")
    print(f"  Results dir     : {args.results_dir}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 1: Train on IncomeEnvironment with performative PePG
    # ------------------------------------------------------------------
    if not args.skip_train:
        print(f"\n[1/3] Training {len(seeds) * len(combos)} PePG agents…")
        train_configs = []
        for seed in seeds:
            for reward, constraint in combos:
                train_configs.append({
                    "seed":             seed,
                    "reward_function":  reward,
                    "constraint_type":  constraint,
                    "weights_path":     os.path.join(
                        args.weights_dir,
                        f"pepg_{reward}__{constraint}__seed{seed}.pt"
                    ),
                    "train_metrics_path": os.path.join(
                        args.weights_dir,
                        f"train_metrics_{reward}__{constraint}__seed{seed}.csv"
                    ),
                    "train_episodes":   args.train_episodes,
                    "warmup_episodes":  args.warmup,
                    "N_male":           args.N_male,
                    "N_female":         args.N_female,
                    "T":                args.T,
                    "dt":               args.dt,
                    "hidden_dim":       args.hidden_dim,
                    "lr":               args.lr,
                    "lambda_lr":        args.lambda_lr,
                    "alpha_lr":         args.alpha_lr,
                    "entropy_coef":     args.entropy_coef,
                    "buffer_capacity":  args.buffer_capacity,
                    "hawkes_weight":    args.hawkes_weight,
                    "wealth_weight":    args.wealth_weight,
                    "transition_weight":args.transition_weight,
                    "reward_weight":    args.reward_weight,
                    "data_filepath":    args.data,
                    "run_id":           len(train_configs) + 1,
                    "total_runs":       len(seeds) * len(combos),
                })

        with mp.Pool(processes=args.workers) as pool:
            train_results = list(
                tqdm(
                    pool.imap(_train_worker, train_configs),
                    total=len(train_configs),
                    desc="Phase 1: training",
                    unit="run",
                )
            )

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
                wp = os.path.join(
                    args.weights_dir,
                    f"pepg_{reward}__{constraint}__seed{seed}.pt"
                )
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
    # Phase 2: Deploy — continued performative training on TestingIncomeEnvironment
    # ------------------------------------------------------------------
    print(f"\n[2/3] Deploying {n_ok} agents on TestingIncomeEnvironment…")

    print("  Loading test data (once)…")
    test_loader = TestingAdultIncomeDataLoader(
        filepath=args.data,
        sample_size=20000,
        credit_threshold=args.credit_threshold,
    )
    test_loader.load_data()
    test_loader.preprocess()
    test_theta = TransitionParameterLearner(default_rate_min=0.05, default_rate_max=0.25)
    test_theta.fit(test_loader.data)
    _male_X   = test_loader.male_data["X"].values
    _female_X = test_loader.female_data["X"].values
    _gt_male  = test_loader.male_data["ground_truth_approval"].values
    _gt_female= test_loader.female_data["ground_truth_approval"].values

    # Load existing deploy checkpoints
    seed_to_results = defaultdict(dict)
    n_loaded = 0
    for tr in train_results:
        if not tr["success"]:
            continue
        seed      = tr["seed"]
        key       = _combo_key(tr["reward"], tr["constraint"])
        ckpt_path = os.path.join(checkpoint_dir, f"seed{seed}_{key}.csv")
        if os.path.exists(ckpt_path):
            try:
                df = add_derived_columns(pd.read_csv(ckpt_path))
                seed_to_results[seed][key] = df
                n_loaded += 1
            except Exception:
                pass  # corrupt — re-run

    if n_loaded:
        print(f"\n  Loaded {n_loaded} checkpoint(s) — skipping those combos.")

    deploy_configs = []
    for tr in train_results:
        if not tr["success"]:
            continue
        seed       = tr["seed"]
        reward     = tr["reward"]
        constraint = tr["constraint"]
        key        = _combo_key(reward, constraint)
        if key in seed_to_results.get(seed, {}):
            continue
        deploy_configs.append({
            "seed":             seed,
            "reward_function":  reward,
            "constraint_type":  constraint,
            "weights_path":     tr["weights_path"],
            "deploy_episodes":  args.deploy_episodes,
            "N_male":           args.N_male,
            "N_female":         args.N_female,
            "T":                args.T,
            "dt":               args.dt,
            "hidden_dim":       args.hidden_dim,
            "lr":               args.lr,
            "lambda_lr":        args.lambda_lr,
            "alpha_lr":         args.alpha_lr,
            "entropy_coef":     args.entropy_coef,
            "buffer_capacity":  args.buffer_capacity,
            "hawkes_weight":    args.hawkes_weight,
            "wealth_weight":    args.wealth_weight,
            "transition_weight":args.transition_weight,
            "reward_weight":    args.reward_weight,
            "deploy_artifacts_dir": os.path.join(args.results_dir, "deploy_artifacts"),
            "theta":            test_theta,
            "male_X":           _male_X,
            "female_X":         _female_X,
            "gt_male":          _gt_male,
            "gt_female":        _gt_female,
            "run_id":           len(deploy_configs) + 1,
            "total_runs":       n_ok - n_loaded,
        })

    if not deploy_configs:
        print("\n  All combos already checkpointed — skipping to aggregation.")
    else:
        print(f"\n  Running {len(deploy_configs)} deploy workers ({n_loaded} already done)…")
        with mp.Pool(processes=args.workers) as pool:
            deploy_iter = tqdm(
                pool.imap_unordered(_deploy_worker, deploy_configs),
                total=len(deploy_configs),
                desc="Phase 2: deploying",
                unit="run",
            )
            for raw in deploy_iter:
                if raw is None:
                    continue
                seed, key, df = raw
                df = add_derived_columns(df)
                seed_to_results[seed][key] = df

                ckpt_path = os.path.join(checkpoint_dir, f"seed{seed}_{key}.csv")
                df.to_csv(ckpt_path, index=False)
                print(f"  Checkpointed: seed{seed}_{key}")

                if args.save_per_seed_csv:
                    df.to_csv(
                        os.path.join(args.results_dir, f"seed{seed}_{key}_{timestamp}.csv"),
                        index=False,
                    )

    # ------------------------------------------------------------------
    # Phase 3: Aggregate + Plot
    # ------------------------------------------------------------------
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
    print(f"\n[3/3] Aggregating mean ± std across {n_complete} seeds…")
    aggregated = aggregate_across_seeds(seed_results)

    for key, (mdf, sdf) in aggregated.items():
        mdf.to_csv(os.path.join(args.results_dir, f"mean_{key}_{timestamp}.csv"), index=False)
        sdf.to_csv(os.path.join(args.results_dir, f"std_{key}_{timestamp}.csv"),  index=False)
    print(f"  Aggregated CSVs saved to {args.results_dir}/")

    print_summary_table(aggregated, n_complete)

    if not args.no_plots:
        print("  Generating plots…")
        for ct in ["predictive", "social", "dm", "two_sided"]:
            _plot_comparison(aggregated,     args.results_dir, timestamp, n_complete, ct)
            _plot_wealth(aggregated,         args.results_dir, timestamp, n_complete, ct)
            _plot_social_welfare(aggregated, args.results_dir, timestamp, n_complete, ct)
            _plot_inequality(aggregated,     args.results_dir, timestamp, n_complete, ct)

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
            for ct in ["social", "dm", "two_sided"]:
                plot_comparison_agg(combined_aggregated, args.results_dir, timestamp, n_combined, ct, prefix="combined_", boundary_episode=args.deploy_episodes)
                plot_wealth_agg(combined_aggregated, args.results_dir, timestamp, n_combined, ct, prefix="combined_", boundary_episode=args.deploy_episodes)
                plot_social_welfare_agg(combined_aggregated, args.results_dir, timestamp, n_combined, ct, prefix="combined_", boundary_episode=args.deploy_episodes)
                plot_inequality_agg(combined_aggregated, args.results_dir, timestamp, n_combined, ct, prefix="combined_", boundary_episode=args.deploy_episodes)

    print("\nDone.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
