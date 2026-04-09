#!/usr/bin/env python3
"""
Generate plots grouped by reward function, with constraints as the coloured lines.
Produces both short-term (ep 1-100) and long-term (ep 1-1000) versions.

Saved in the same results directory with prefix:
  short_term_bycst_   (ep 1-100)
  long_term_bycst_    (ep 1-1000)

Four plot types per (reward, episode-range, constraint-set):
  *_comparison_*     2×2: wealth gap, approval disparity, profit, R_g
  *_wealth_*         2×2: μ_M, μ_F, cumulative profit, ρ
  *_social_welfare_* 1×3: R_M, R_F, R̄
  *_inequality_*     1×3: wealth gap, approval disparity, ρ(t)
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_multi_seed import _band, _finish_axes, _save
from pg_run import setup_plot_style, REWARD_LABELS

# ---------------------------------------------------------------------------
# Colours and labels for constraints  (colour now encodes constraint)
# ---------------------------------------------------------------------------

CONSTRAINT_COLORS = {
    "predictive": "#1f77b4",   # blue
    "social":     "#2ca02c",   # green
    "dm":         "#ff7f0e",   # orange
    "two_sided":  "#9467bd",   # purple
}

CONSTRAINT_LINESTYLES = {
    "predictive": "-",
    "social":     "--",
    "dm":         "-.",
    "two_sided":  ":",
}

# Small vertical offsets so overlapping lines remain distinguishable
CONSTRAINT_OFFSETS = {
    "predictive":  0.0,
    "social":      0.0,
    "dm":          0.0,
    "two_sided":   0.0,
}

CONSTRAINT_LABELS = {
    "predictive": "Predictive",
    "social":     "Social",
    "dm":         "DM",
    "two_sided":  "Two-Sided",
}

REWARDS = [
    "utilitarian_profit",
    "social_welfare",
    "rawlsian_maximin",
    "fairness_lagrangian",
]

CONSTRAINTS = ["predictive", "social", "dm", "two_sided"]

# ---------------------------------------------------------------------------
# Result directories
# ---------------------------------------------------------------------------

RESULT_SETS = [
    {
        "dir":       "results_pepg_adapt",
        "timestamp": "20260325_113301",
        "agent":     "PePG Adapt",
        "tag":       "pepg_adapt",
    },
    {
        "dir":       "results_pg",
        "timestamp": "20260329_162259",
        "agent":     "PG (Performative)",
        "tag":       "pg",
    },
    {
        "dir":       "results_static",
        "timestamp": "20260403_151304",
        "agent":     "Static",
        "tag":       "static",
    },
]

EP_RANGES = [
    ("short_term_bycst", 100),
    ("long_term_bycst",  1000),
]

N_SEEDS = 3

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_aggregated(results_dir, timestamp):
    """Return dict keyed by '{reward}__{constraint}' → (mean_df, std_df)."""
    aggregated = {}
    for mean_path in sorted(glob.glob(
        os.path.join(results_dir, f"mean_*_{timestamp}.csv")
    )):
        fname = os.path.basename(mean_path)
        m = re.match(r"mean_(.+)__(.+)_\d{8}_\d{6}\.csv$", fname)
        if not m:
            continue
        reward, constraint = m.group(1), m.group(2)
        std_path = mean_path.replace("mean_", "std_")
        if not os.path.exists(std_path):
            continue
        mdf = pd.read_csv(mean_path)
        sdf = pd.read_csv(std_path)
        aggregated[f"{reward}__{constraint}"] = (mdf, sdf)
    return aggregated


def slice_to(aggregated, n_eps):
    return {
        k: (
            mdf[mdf["episode"] <= n_eps].reset_index(drop=True),
            sdf[sdf["episode"] <= n_eps].reset_index(drop=True),
        )
        for k, (mdf, sdf) in aggregated.items()
    }


def _iter_by_reward(aggregated, reward_filter):
    """Yield (constraint, mean_df, std_df) for a given reward function."""
    for ct in CONSTRAINTS:
        key = f"{reward_filter}__{ct}"
        if key in aggregated:
            yield ct, aggregated[key][0], aggregated[key][1]

# ---------------------------------------------------------------------------
# Four plot types (colour = constraint, one plot per reward function)
# ---------------------------------------------------------------------------

def _plot_comparison(aggregated, results_dir, timestamp, reward, prefix,
                     agent_title, n_eps):
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"{agent_title} — {REWARD_LABELS[reward]}"
        f"  (mean ± std, {N_SEEDS} seeds, ep 1–{n_eps})",
        fontsize=13, fontweight="bold",
    )
    for ct, mdf, sdf in _iter_by_reward(aggregated, reward):
        color = CONSTRAINT_COLORS[ct]
        ls    = CONSTRAINT_LINESTYLES[ct]
        label = CONSTRAINT_LABELS[ct]
        ep    = mdf["episode"]
        _band(axes[0, 0], ep, mdf["wealth_gap"],         sdf["wealth_gap"],         label, color, lw=1.8)
        axes[0, 0].lines[-1].set_linestyle(ls); axes[0, 0].collections[-1].set_alpha(0.10)
        _band(axes[0, 1], ep, mdf["approval_disparity"],  sdf["approval_disparity"],  label, color, lw=1.8)
        axes[0, 1].lines[-1].set_linestyle(ls); axes[0, 1].collections[-1].set_alpha(0.10)
        _band(axes[1, 0], ep, mdf["profit_episode"],      sdf["profit_episode"],      label, color, lw=1.8)
        axes[1, 0].lines[-1].set_linestyle(ls); axes[1, 0].collections[-1].set_alpha(0.10)
        axes[1, 1].plot(ep, mdf["R_M"], label=f"{label} (M)", color=color, lw=1.8, ls=ls)
        axes[1, 1].fill_between(ep, mdf["R_M"] - sdf["R_M"], mdf["R_M"] + sdf["R_M"],
                                color=color, alpha=0.08)
        axes[1, 1].plot(ep, mdf["R_F"], color=color, lw=1.2, alpha=0.6, ls=ls)
        axes[1, 1].fill_between(ep, mdf["R_F"] - sdf["R_F"], mdf["R_F"] + sdf["R_F"],
                                color=color, alpha=0.06)
    for ax in axes.flat:
        ax.axhline(0, color="k", linewidth=0.8, ls="-")
    axes[0, 0].set_ylabel("μ_M − μ_F  ($k)");    axes[0, 0].set_title("(a) Wealth Gap")
    axes[0, 1].set_ylabel("Approval M − F");      axes[0, 1].set_title("(b) Approval Rate Disparity")
    axes[1, 0].set_ylabel("Episode profit ($k)"); axes[1, 0].set_title("(c) Episode Profit")
    axes[1, 1].set_ylabel("R_g");                 axes[1, 1].set_title("(d) Long-term Social Welfare R_g(t)")
    _finish_axes(axes.flat)
    plt.tight_layout()
    _save(fig, os.path.join(results_dir,
          f"{prefix}_{reward}_comparison_{timestamp}.png"))


def _plot_wealth(aggregated, results_dir, timestamp, reward, prefix,
                 agent_title, n_eps):
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"{agent_title} Metric Trajectories — {REWARD_LABELS[reward]}"
        f"  (mean ± std, {N_SEEDS} seeds, ep 1–{n_eps})",
        fontsize=13, fontweight="bold",
    )
    for ct, mdf, sdf in _iter_by_reward(aggregated, reward):
        color = CONSTRAINT_COLORS[ct]
        ls    = CONSTRAINT_LINESTYLES[ct]
        label = CONSTRAINT_LABELS[ct]
        ep    = mdf["episode"]
        for ax, col in zip(
            [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]],
            ["mu_M_end", "mu_F_end", "cumulative_profit", "rho_episode"],
        ):
            _band(ax, ep, mdf[col], sdf[col], label, color, lw=1.8)
            ax.lines[-1].set_linestyle(ls)
            ax.collections[-1].set_alpha(0.10)
    axes[1, 0].axhline(0, color="k", linewidth=0.8, ls="-")
    axes[1, 1].axhline(1, color="k", linewidth=0.8, ls="-", label="ρ = 1 (equal growth)")
    axes[1, 1].set_ylim(-5, 10)
    axes[0, 0].set_ylabel("Mean wealth ($k)");       axes[0, 0].set_title("(a) μ_M — Male Mean Wealth")
    axes[0, 1].set_ylabel("Mean wealth ($k)");       axes[0, 1].set_title("(b) μ_F — Female Mean Wealth")
    axes[1, 0].set_ylabel("Cumulative profit ($k)"); axes[1, 0].set_title("(c) Cumulative Bank Profit")
    axes[1, 1].set_ylabel("ρ = Δμ_M / Δμ_F");       axes[1, 1].set_title("(d) Inequality Ratio ρ (per episode)")
    _finish_axes(axes.flat)
    plt.tight_layout()
    _save(fig, os.path.join(results_dir,
          f"{prefix}_{reward}_wealth_{timestamp}.png"))


def _plot_social_welfare(aggregated, results_dir, timestamp, reward, prefix,
                         agent_title, n_eps):
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"{agent_title} Long-term Social Welfare — {REWARD_LABELS[reward]}"
        f"  (mean ± std, {N_SEEDS} seeds, ep 1–{n_eps})",
        fontsize=13, fontweight="bold",
    )
    y_min, y_max = np.inf, -np.inf
    plot_data = []
    for ct, mdf, sdf in _iter_by_reward(aggregated, reward):
        N_M = mdf["total_applications_M"]
        N_F = mdf["total_applications_F"]
        total_N = (N_M + N_F).replace(0, np.nan)
        R_bar_mean = (N_M * mdf["R_M"] + N_F * mdf["R_F"]) / total_N
        R_bar_std  = np.sqrt((N_M * sdf["R_M"])**2 + (N_F * sdf["R_F"])**2) / total_N
        plot_data.append((ct, mdf, sdf, R_bar_mean, R_bar_std))
        for mean, std in [
            (mdf["R_M"], sdf["R_M"]),
            (mdf["R_F"], sdf["R_F"]),
            (R_bar_mean, R_bar_std),
        ]:
            y_min = min(y_min, (mean - std).min())
            y_max = max(y_max, (mean + std).max())
    margin = (y_max - y_min) * 0.05
    ylim   = (y_min - margin, y_max + margin)
    for ct, mdf, sdf, R_bar_mean, R_bar_std in plot_data:
        color = CONSTRAINT_COLORS[ct]
        ls    = CONSTRAINT_LINESTYLES[ct]
        label = CONSTRAINT_LABELS[ct]
        ep    = mdf["episode"]
        for ax, mean, std in zip(
            axes,
            [mdf["R_M"],  mdf["R_F"],  R_bar_mean],
            [sdf["R_M"],  sdf["R_F"],  R_bar_std],
        ):
            _band(ax, ep, mean, std, label, color, lw=1.8)
            ax.lines[-1].set_linestyle(ls)
            ax.collections[-1].set_alpha(0.10)
    for ax in axes:
        ax.axhline(0, color="k", linewidth=0.8, ls="-")
        ax.set_ylim(ylim)
    axes[0].set_ylabel(r"$R_M$"); axes[0].set_title("(a) Male Long-term Social Welfare $R_M$")
    axes[1].set_ylabel(r"$R_F$"); axes[1].set_title("(b) Female Long-term Social Welfare $R_F$")
    axes[2].set_ylabel(r"$\bar{R}$"); axes[2].set_title(r"(c) Weighted Average $\bar{R}$")
    _finish_axes(axes)
    plt.tight_layout()
    _save(fig, os.path.join(results_dir,
          f"{prefix}_{reward}_social_welfare_{timestamp}.png"))


def _plot_inequality(aggregated, results_dir, timestamp, reward, prefix,
                     agent_title, n_eps):
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"{agent_title} Wealth Inequality & Disparity — {REWARD_LABELS[reward]}"
        f"  (mean ± std, {N_SEEDS} seeds, ep 1–{n_eps})",
        fontsize=13, fontweight="bold",
    )
    for ct, mdf, sdf in _iter_by_reward(aggregated, reward):
        color = CONSTRAINT_COLORS[ct]
        ls    = CONSTRAINT_LINESTYLES[ct]
        label = CONSTRAINT_LABELS[ct]
        ep    = mdf["episode"]
        for ax, col in zip(axes, ["wealth_gap", "approval_disparity", "rho_cumulative"]):
            _band(ax, ep, mdf[col], sdf[col], label, color, lw=1.8)
            ax.lines[-1].set_linestyle(ls)
            ax.collections[-1].set_alpha(0.10)
    axes[0].axhline(0, color="k", linewidth=0.8, ls="-")
    axes[1].axhline(0, color="k", linewidth=0.8, ls="-")
    axes[2].axhline(1, color="red", linewidth=1.0, ls="--", alpha=0.7,
                    label="ρ = 1 (equal growth)")
    axes[0].set_ylabel("μ_M − μ_F  ($k)");     axes[0].set_title("(a) Wealth Gap Trajectory")
    axes[1].set_ylabel("Approval rate M − F"); axes[1].set_title("(b) Approval Rate Disparity")
    axes[2].set_ylabel(r"$\rho(t)$");           axes[2].set_title(r"(c) Inequality Ratio $\rho(t)$")
    _finish_axes(axes)
    plt.tight_layout()
    _save(fig, os.path.join(results_dir,
          f"{prefix}_{reward}_inequality_{timestamp}.png"))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for cfg in RESULT_SETS:
        results_dir = cfg["dir"]
        timestamp   = cfg["timestamp"]
        agent_title = cfg["agent"]
        tag         = cfg["tag"]

        print(f"\n── {agent_title}  ({results_dir}) ──")

        full_agg = load_aggregated(results_dir, timestamp)
        if not full_agg:
            print("  No CSVs found — skipping.")
            continue

        for ep_prefix, n_eps in EP_RANGES:
            agg = slice_to(full_agg, n_eps)
            prefix = f"{ep_prefix}_{tag}"

            for reward in REWARDS:
                # Skip if no data exists for this reward at all
                if not any(k.startswith(f"{reward}__") for k in agg):
                    continue

                _plot_comparison(   agg, results_dir, timestamp, reward, prefix, agent_title, n_eps)
                _plot_wealth(       agg, results_dir, timestamp, reward, prefix, agent_title, n_eps)
                _plot_social_welfare(agg, results_dir, timestamp, reward, prefix, agent_title, n_eps)
                _plot_inequality(   agg, results_dir, timestamp, reward, prefix, agent_title, n_eps)

    print("\nDone.")
