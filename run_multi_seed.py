#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loan_simulator.testing.data_loader import TestingAdultIncomeDataLoader
from loan_simulator.transition_learner import TransitionParameterLearner
from test_rule_based_policies import (
    POLICY_COLORS,
    POLICY_LABELS,
    AlwaysApprovePolicy,
    AlwaysRejectPolicy,
    OraclePolicy,
    PatternPredictionPolicy,
    ReverseRichBecomesRicherPolicy,
    RichBecomesRicherPolicy,
    UniformAcceptancePolicy,
    _policy_worker,
    setup_plot_style,
)


# ---------------------------------------------------------------------------
# Derived-column helper
# ---------------------------------------------------------------------------

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived metrics to an episode-metrics DataFrame before aggregation."""
    df = df.copy()
    df["approval_disparity"] = (
        df["approval_rate_M_episode"] - df["approval_rate_F_episode"]
    )
    mu_M_0 = df["mu_M_start"].iloc[0]
    mu_F_0 = df["mu_F_start"].iloc[0]
    delta_M = df["mu_M_end"] - mu_M_0
    delta_F = (df["mu_F_end"] - mu_F_0).replace(0.0, np.nan)
    df["rho_cumulative"] = delta_M / delta_F
    return df


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------

def run_one_seed(
    seed: int,
    policies: list,
    loader: TestingAdultIncomeDataLoader,
    theta_learner: TransitionParameterLearner,
    args: argparse.Namespace,
) -> dict:
    """
    Run all policies for a single seed in parallel (one process per policy).
    Returns dict[policy_name -> DataFrame].
    """
    print(f"  [seed {seed:4d}] launching {len(policies)} policy workers…")
    worker_args = [
        (
            policy, loader, theta_learner,
            args.N_male, args.N_female, args.T, args.dt,
            seed, args.episodes,
        )
        for policy in policies
    ]
    n_workers = min(len(policies), mp.cpu_count())
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_policy_worker, worker_args)
    seed_metrics = {name: add_derived_columns(df) for name, df in results}
    print(f"  [seed {seed:4d}] done.")
    return seed_metrics


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------

def aggregate_across_seeds(seed_results: list) -> dict:
    """
    Aggregate per-seed DataFrames into mean ± std.

    Parameters
    ----------
    seed_results : list of dict[policy_name -> DataFrame]
        One dict per seed.

    Returns
    -------
    dict[policy_name -> (mean_df, std_df)]
        Both DataFrames are indexed by episode.
    """
    policy_names = list(seed_results[0].keys())
    aggregated = {}

    for name in policy_names:
        dfs = [sr[name] for sr in seed_results]
        numeric_cols = dfs[0].select_dtypes(include=np.number).columns.tolist()

        # Align on shortest run (should all be equal, but guard against edge cases)
        n_ep = min(len(df) for df in dfs)
        stacked = np.stack(
            [df[numeric_cols].iloc[:n_ep].values for df in dfs],
            axis=0,
        )  # shape: (n_seeds, n_episodes, n_cols)

        ddof = 1 if len(seed_results) > 1 else 0
        mean_vals = stacked.mean(axis=0)
        std_vals  = stacked.std(axis=0, ddof=ddof)

        mean_df = pd.DataFrame(mean_vals, columns=numeric_cols)
        std_df  = pd.DataFrame(std_vals,  columns=numeric_cols)
        mean_df["episode"] = dfs[0]["episode"].values[:n_ep]
        std_df["episode"]  = dfs[0]["episode"].values[:n_ep]

        aggregated[name] = (mean_df, std_df)

    return aggregated


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _band(ax, x, mean, std, label, color, lw=1.5, alpha=0.2):
    """Plot a mean line with a ±std shaded band."""
    ax.plot(x, mean, label=label, color=color, linewidth=lw)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha)


def _finish_axes(axes_flat, xlabel="Episode"):
    for ax in axes_flat:
        ax.set_xlabel(xlabel)
        ax.legend(loc="best", fontsize=7)


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Aggregated plot 1 — 2×2 comparison
# ---------------------------------------------------------------------------

def plot_comparison_agg(aggregated, results_dir, timestamp, n_seeds):
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Rule-Based Policies — Performative Environment  (mean ± std, {n_seeds} seeds)",
        fontsize=13, fontweight="bold",
    )

    for name, (mdf, sdf) in aggregated.items():
        color = POLICY_COLORS.get(name)
        label = POLICY_LABELS.get(name, name)
        ep = mdf["episode"]

        # (a) Wealth gap
        _band(axes[0, 0], ep, mdf["wealth_gap"], sdf["wealth_gap"], label, color)

        # (b) Approval rate disparity  (propagate std: sqrt(σ_M² + σ_F²))
        m_disp = mdf["approval_disparity"]
        s_disp = sdf["approval_disparity"]
        _band(axes[0, 1], ep, m_disp, s_disp, label, color)

        # (c) Episode profit
        _band(axes[1, 0], ep, mdf["profit_episode"], sdf["profit_episode"], label, color)

        # (d) R_M (solid) / R_F (dashed)
        axes[1, 1].plot(ep, mdf["R_M"], label=f"{label} (M)", color=color, lw=1.5, ls="-")
        axes[1, 1].fill_between(
            ep, mdf["R_M"] - sdf["R_M"], mdf["R_M"] + sdf["R_M"], color=color, alpha=0.15,
        )
        axes[1, 1].plot(ep, mdf["R_F"], color=color, lw=1.5, ls="--", alpha=0.7)
        axes[1, 1].fill_between(
            ep, mdf["R_F"] - sdf["R_F"], mdf["R_F"] + sdf["R_F"], color=color, alpha=0.1,
        )

    for ax in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.axhline(0, color="k", linewidth=0.8, ls="-")

    axes[0, 0].set_ylabel("μ_M − μ_F  ($k)");   axes[0, 0].set_title("(a) Wealth Gap")
    axes[0, 1].set_ylabel("Approval M − F");     axes[0, 1].set_title("(b) Approval Rate Disparity")
    axes[1, 0].set_ylabel("Episode profit ($k)"); axes[1, 0].set_title("(c) Episode Profit")
    axes[1, 1].set_ylabel("R_g = (μ_g,t − μ_g,0) / t")
    axes[1, 1].set_title("(d) Long-term Social Welfare R_g(t)")

    _finish_axes(axes.flat)
    plt.tight_layout()
    _save(fig, os.path.join(results_dir, f"agg_comparison_{timestamp}.png"))


# ---------------------------------------------------------------------------
# Aggregated plot 2 — metric trajectories (2×2)
# ---------------------------------------------------------------------------

def plot_wealth_agg(aggregated, results_dir, timestamp, n_seeds):
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Metric Trajectories — Performative Environment  (mean ± std, {n_seeds} seeds)",
        fontsize=13, fontweight="bold",
    )

    for name, (mdf, sdf) in aggregated.items():
        color = POLICY_COLORS.get(name)
        label = POLICY_LABELS.get(name, name)
        ep = mdf["episode"]

        _band(axes[0, 0], ep, mdf["mu_M_end"],         sdf["mu_M_end"],         label, color)
        _band(axes[0, 1], ep, mdf["mu_F_end"],         sdf["mu_F_end"],         label, color)
        _band(axes[1, 0], ep, mdf["cumulative_profit"], sdf["cumulative_profit"], label, color)
        _band(axes[1, 1], ep, mdf["rho_episode"],       sdf["rho_episode"],       label, color)

    axes[1, 0].axhline(0, color="k", linewidth=0.8, ls="-")
    axes[1, 1].axhline(1, color="k", linewidth=0.8, ls="-", label="ρ = 1 (equal growth)")
    axes[1, 1].set_ylim(-5, 10)

    axes[0, 0].set_ylabel("Mean wealth ($k)");      axes[0, 0].set_title("(a) μ_M — Male Mean Wealth")
    axes[0, 1].set_ylabel("Mean wealth ($k)");      axes[0, 1].set_title("(b) μ_F — Female Mean Wealth")
    axes[1, 0].set_ylabel("Cumulative profit ($k)"); axes[1, 0].set_title("(c) Cumulative Bank Profit")
    axes[1, 1].set_ylabel("ρ = Δμ_M / Δμ_F");      axes[1, 1].set_title("(d) Inequality Ratio ρ (per episode)")

    _finish_axes(axes.flat)
    plt.tight_layout()
    _save(fig, os.path.join(results_dir, f"agg_metric_trajectories_{timestamp}.png"))


# ---------------------------------------------------------------------------
# Aggregated plot 3 — social welfare (1×3)
# ---------------------------------------------------------------------------

def plot_social_welfare_agg(aggregated, results_dir, timestamp, n_seeds):
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"Long-term Social Welfare Trajectories  (mean ± std, {n_seeds} seeds)",
        fontsize=13, fontweight="bold",
    )

    # First pass: collect all band extents to compute a shared y range
    y_min, y_max = np.inf, -np.inf
    plot_data = []
    for name, (mdf, sdf) in aggregated.items():
        N_M = mdf["total_applications_M"]
        N_F = mdf["total_applications_F"]
        total_N = (N_M + N_F).replace(0, np.nan)
        R_bar_mean = (N_M * mdf["R_M"] + N_F * mdf["R_F"]) / total_N
        R_bar_std  = np.sqrt(
            (N_M * sdf["R_M"]) ** 2 + (N_F * sdf["R_F"]) ** 2
        ) / total_N
        plot_data.append((name, mdf, sdf, R_bar_mean, R_bar_std))

        for mean, std in [
            (mdf["R_M"], sdf["R_M"]),
            (mdf["R_F"], sdf["R_F"]),
            (R_bar_mean, R_bar_std),
        ]:
            y_min = min(y_min, (mean - std).min())
            y_max = max(y_max, (mean + std).max())

    margin = (y_max - y_min) * 0.05
    ylim = (y_min - margin, y_max + margin)

    # Second pass: draw with shared y range
    for name, mdf, sdf, R_bar_mean, R_bar_std in plot_data:
        color = POLICY_COLORS.get(name)
        label = POLICY_LABELS.get(name, name)
        ep = mdf["episode"]

        _band(axes[0], ep, mdf["R_M"],  sdf["R_M"],  label, color)
        _band(axes[1], ep, mdf["R_F"],  sdf["R_F"],  label, color)
        _band(axes[2], ep, R_bar_mean,  R_bar_std,   label, color)

    for ax in axes:
        ax.axhline(0, color="k", linewidth=0.8, ls="-")
        ax.set_ylim(ylim)

    axes[0].set_ylabel(r"$R_M$  (welfare / episode)")
    axes[1].set_ylabel(r"$R_F$  (welfare / episode)")
    axes[2].set_ylabel(r"$\bar{R}$  (welfare / episode)")
    axes[0].set_title("(a) Male Long-term Social Welfare $R_M$")
    axes[1].set_title("(b) Female Long-term Social Welfare $R_F$")
    axes[2].set_title(r"(c) Weighted Average $\bar{R}$")

    _finish_axes(axes)
    plt.tight_layout()
    _save(fig, os.path.join(results_dir, f"agg_social_welfare_{timestamp}.png"))


# ---------------------------------------------------------------------------
# Aggregated plot 4 — inequality & disparity (1×3)
# ---------------------------------------------------------------------------

def plot_inequality_agg(aggregated, results_dir, timestamp, n_seeds):
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"Wealth Inequality & Disparity Trajectories  (mean ± std, {n_seeds} seeds)",
        fontsize=13, fontweight="bold",
    )

    for name, (mdf, sdf) in aggregated.items():
        color = POLICY_COLORS.get(name)
        label = POLICY_LABELS.get(name, name)
        ep = mdf["episode"]

        # (a) Wealth gap
        _band(axes[0], ep, mdf["wealth_gap"],       sdf["wealth_gap"],       label, color)
        # (b) Approval disparity
        _band(axes[1], ep, mdf["approval_disparity"], sdf["approval_disparity"], label, color)
        # (c) Cumulative ρ(t) — pre-computed in add_derived_columns
        _band(axes[2], ep, mdf["rho_cumulative"],   sdf["rho_cumulative"],   label, color)

    axes[0].axhline(0, color="k", linewidth=0.8, ls="-")
    axes[1].axhline(0, color="k", linewidth=0.8, ls="-")
    axes[2].axhline(1, color="red", linewidth=1.0, ls="--", alpha=0.7, label="ρ = 1 (equal growth)")

    axes[0].set_ylabel("μ_M − μ_F  ($k)")
    axes[1].set_ylabel("Approval rate  M − F")
    axes[2].set_ylabel(r"$\rho(t)$")
    axes[0].set_title("(a) Wealth Gap Trajectory")
    axes[1].set_title("(b) Approval Rate Disparity Trajectory")
    axes[2].set_title(r"(c) Inequality Ratio $\rho(t)$")

    _finish_axes(axes)
    plt.tight_layout()
    _save(fig, os.path.join(results_dir, f"agg_inequality_{timestamp}.png"))


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(aggregated, n_seeds):
    print(f"\n{'=' * 82}")
    print(
        f"{'Policy':<32} {'R_M mean±std':>16} {'R_F mean±std':>16} "
        f"{'profit ($k)':>14}"
    )
    print(f"{'—' * 82}")
    for name, (mdf, sdf) in aggregated.items():
        label = POLICY_LABELS.get(name, name)
        R_M_m = mdf["R_M"].iloc[-1];     R_M_s = sdf["R_M"].iloc[-1]
        R_F_m = mdf["R_F"].iloc[-1];     R_F_s = sdf["R_F"].iloc[-1]
        pr_m  = mdf["cumulative_profit"].iloc[-1]
        pr_s  = sdf["cumulative_profit"].iloc[-1]
        print(
            f"  {label:<30} "
            f"{R_M_m:+.4f}±{R_M_s:.4f}  "
            f"{R_F_m:+.4f}±{R_F_s:.4f}  "
            f"{pr_m:>8.1f}±{pr_s:.1f}"
        )
    print(f"{'=' * 82}")
    print(f"  ({n_seeds} seeds)\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed scheduler for rule-based policy evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Seed specification (mutually exclusive)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seeds", type=int, default=5,
        help="Number of seeds (uses 0, 1, …, N-1).  Default: 5",
    )
    seed_group.add_argument(
        "--seed-list", type=int, nargs="+", metavar="S",
        help="Explicit list of seeds, e.g. --seed-list 0 7 42 99 123",
    )

    parser.add_argument("--episodes",   type=int,   default=200)
    parser.add_argument("--N-male",     type=int,   default=3000)
    parser.add_argument("--N-female",   type=int,   default=3000)
    parser.add_argument("--T",          type=int,   default=100)
    parser.add_argument("--dt",         type=float, default=0.5)
    parser.add_argument("--data",       type=str,   default=None,
                        help="Path to adult.csv (None = auto-download)")
    parser.add_argument("--credit-threshold", type=float, default=0.5)
    parser.add_argument("--results-dir", type=str,  default="./multi_seed_results")
    parser.add_argument("--no-plots",    action="store_true")
    parser.add_argument("--save-per-seed-csv", action="store_true",
                        help="Also save one CSV per policy per seed")
    args = parser.parse_args()

    seeds = args.seed_list if args.seed_list is not None else list(range(args.seeds))
    n_seeds = len(seeds)

    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("MULTI-SEED RULE-BASED POLICY EVALUATION")
    print("=" * 70)
    print(f"  Seeds             : {seeds}")
    print(f"  Episodes / policy : {args.episodes}")
    print(f"  Population        : {args.N_male}M + {args.N_female}F")
    print(f"  T={args.T}  dt={args.dt}")
    print(f"  Results dir       : {args.results_dir}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Data + theta learner (shared across seeds — deterministic given data)
    # ------------------------------------------------------------------
    print("\n[1] Loading and preprocessing data…")
    loader = TestingAdultIncomeDataLoader(
        filepath=args.data,
        sample_size=20000,
        credit_threshold=args.credit_threshold,
    )
    loader.load_data()
    loader.preprocess()

    print("\n[2] Fitting transition parameters…")
    theta_learner = TransitionParameterLearner(
        default_rate_min=0.05, default_rate_max=0.25
    )
    theta_learner.fit(loader.data)

    policies = [
        AlwaysApprovePolicy(),
        AlwaysRejectPolicy(),
        UniformAcceptancePolicy(),
        OraclePolicy(),
        PatternPredictionPolicy(),
        RichBecomesRicherPolicy(),
        ReverseRichBecomesRicherPolicy(),
    ]

    # ------------------------------------------------------------------
    # Run across seeds (sequential seeds, parallel policies within each)
    # ------------------------------------------------------------------
    print(f"\n[3] Running {n_seeds} seeds × {len(policies)} policies…")
    seed_results = []
    for i, seed in enumerate(seeds):
        print(f"\n  Seed {i+1}/{n_seeds}  (seed={seed})")
        sr = run_one_seed(seed, policies, loader, theta_learner, args)
        seed_results.append(sr)

        if args.save_per_seed_csv:
            for name, df in sr.items():
                path = os.path.join(
                    args.results_dir,
                    f"seed{seed}_{name}_{timestamp}.csv",
                )
                df.to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    print("\n[4] Aggregating mean ± std across seeds…")
    aggregated = aggregate_across_seeds(seed_results)

    # Always save aggregated mean/std CSVs
    for name, (mdf, sdf) in aggregated.items():
        mdf.to_csv(
            os.path.join(args.results_dir, f"mean_{name}_{timestamp}.csv"),
            index=False,
        )
        sdf.to_csv(
            os.path.join(args.results_dir, f"std_{name}_{timestamp}.csv"),
            index=False,
        )
    print(f"  Aggregated CSVs saved to {args.results_dir}/")

    print_summary_table(aggregated, n_seeds)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        print("[5] Generating aggregated plots…")
        plot_comparison_agg(aggregated, args.results_dir, timestamp, n_seeds)
        plot_wealth_agg(aggregated, args.results_dir, timestamp, n_seeds)
        plot_social_welfare_agg(aggregated, args.results_dir, timestamp, n_seeds)
        plot_inequality_agg(aggregated, args.results_dir, timestamp, n_seeds)

    print("\nDone.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
