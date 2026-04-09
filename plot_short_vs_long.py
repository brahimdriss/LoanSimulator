#!/usr/bin/env python3
"""
Generate short-term (ep 1-100) versions of the standard comparison, wealth,
social-welfare, and inequality plots for pepg_adapt, pg, and static results.

Plots are saved with a "short_term_" prefix in the same results directory,
using the exact same format as the originals.
"""

import os
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_multi_seed import _band, _finish_axes, _save
from pg_run import (
    REWARD_COLORS,
    REWARD_LABELS,
    CONSTRAINT_LABELS,
    setup_plot_style,
)

SHORT_TERM = 100   # episodes to keep

# ---------------------------------------------------------------------------
# Result-directory configurations
# ---------------------------------------------------------------------------

RESULT_SETS = [
    {
        "dir":       "results_pepg_adapt",
        "timestamp": "20260325_113301",
        "prefix":    "short_term_pepg_adapt",
        "agent":     "pepg_adapt",
    },
    {
        "dir":       "results_pg",
        "timestamp": "20260329_162259",
        "prefix":    "short_term_pg",
        "agent":     "pg",
    },
    {
        "dir":       "results_static",
        "timestamp": "20260403_151304",
        "prefix":    "short_term_static",
        "agent":     "static",
    },
]

CONSTRAINTS = ["predictive", "social", "dm", "two_sided"]

AGENT_TITLES = {
    "pepg_adapt": "PePG Adapt",
    "pg":         "PG (Performative)",
    "static":     "Static",
}

# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_aggregated(results_dir, timestamp):
    """
    Load all mean_*__{constraint}_{timestamp}.csv and std_*__{constraint}_{timestamp}.csv
    from results_dir and return aggregated dict keyed by "{reward}__{constraint}".
    """
    aggregated = {}
    pattern = os.path.join(results_dir, f"mean_*_{timestamp}.csv")
    for mean_path in sorted(glob.glob(pattern)):
        fname = os.path.basename(mean_path)
        # fname: mean_{reward}__{constraint}_{timestamp}.csv
        m = re.match(r"mean_(.+)__(.+)_\d{8}_\d{6}\.csv$", fname)
        if not m:
            continue
        reward, constraint = m.group(1), m.group(2)
        std_path = mean_path.replace("mean_", "std_")
        if not os.path.exists(std_path):
            continue
        mdf = pd.read_csv(mean_path)
        sdf = pd.read_csv(std_path)
        key = f"{reward}__{constraint}"
        aggregated[key] = (mdf, sdf)
    return aggregated


def slice_to(aggregated, n_eps):
    """Return a new aggregated dict with DataFrames sliced to the first n_eps episodes."""
    out = {}
    for key, (mdf, sdf) in aggregated.items():
        out[key] = (
            mdf[mdf["episode"] <= n_eps].reset_index(drop=True),
            sdf[sdf["episode"] <= n_eps].reset_index(drop=True),
        )
    return out


def _iter_combos(aggregated, constraint_filter=None):
    for key, (mdf, sdf) in aggregated.items():
        reward, constraint = key.split("__", 1)
        if constraint_filter is not None and constraint != constraint_filter:
            continue
        yield reward, constraint, mdf, sdf

# ---------------------------------------------------------------------------
# Plot functions — identical logic to pepg_adapt.py / pg_run.py
# ---------------------------------------------------------------------------

def _plot_comparison(aggregated, results_dir, timestamp, n_seeds, constraint_filter,
                     prefix, agent_title):
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ctitle = CONSTRAINT_LABELS.get(constraint_filter, constraint_filter)
    fig.suptitle(
        f"{agent_title} — {ctitle}  (mean ± std, {n_seeds} seeds, ep 1–{SHORT_TERM})",
        fontsize=13, fontweight="bold",
    )
    for reward, _, mdf, sdf in _iter_combos(aggregated, constraint_filter):
        color = REWARD_COLORS[reward]
        label = REWARD_LABELS[reward]
        ep = mdf["episode"]
        _band(axes[0, 0], ep, mdf["wealth_gap"],         sdf["wealth_gap"],         label, color, lw=1.5)
        axes[0, 0].collections[-1].set_alpha(0.12)
        _band(axes[0, 1], ep, mdf["approval_disparity"],  sdf["approval_disparity"],  label, color, lw=1.5)
        axes[0, 1].collections[-1].set_alpha(0.12)
        _band(axes[1, 0], ep, mdf["profit_episode"],      sdf["profit_episode"],      label, color, lw=1.5)
        axes[1, 0].collections[-1].set_alpha(0.12)
        axes[1, 1].plot(ep, mdf["R_M"], label=f"{label} (M)", color=color, lw=1.5)
        axes[1, 1].fill_between(ep, mdf["R_M"] - sdf["R_M"], mdf["R_M"] + sdf["R_M"],
                                color=color, alpha=0.10)
        axes[1, 1].plot(ep, mdf["R_F"], color=color, lw=1.2, alpha=0.6)
        axes[1, 1].fill_between(ep, mdf["R_F"] - sdf["R_F"], mdf["R_F"] + sdf["R_F"],
                                color=color, alpha=0.07)
    for ax in axes.flat:
        ax.axhline(0, color="k", linewidth=0.8, ls="-")
    axes[0, 0].set_ylabel("μ_M − μ_F  ($k)");    axes[0, 0].set_title("(a) Wealth Gap")
    axes[0, 1].set_ylabel("Approval M − F");      axes[0, 1].set_title("(b) Approval Rate Disparity")
    axes[1, 0].set_ylabel("Episode profit ($k)"); axes[1, 0].set_title("(c) Episode Profit")
    axes[1, 1].set_ylabel("R_g");                 axes[1, 1].set_title("(d) Long-term Social Welfare R_g(t)")
    _finish_axes(axes.flat)
    plt.tight_layout()
    _save(fig, os.path.join(results_dir, f"{prefix}_comparison_{constraint_filter}_{timestamp}.png"))


def _plot_wealth(aggregated, results_dir, timestamp, n_seeds, constraint_filter,
                 prefix, agent_title):
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ctitle = CONSTRAINT_LABELS.get(constraint_filter, constraint_filter)
    fig.suptitle(
        f"{agent_title} Metric Trajectories — {ctitle}  (mean ± std, {n_seeds} seeds, ep 1–{SHORT_TERM})",
        fontsize=13, fontweight="bold",
    )
    for reward, _, mdf, sdf in _iter_combos(aggregated, constraint_filter):
        color = REWARD_COLORS[reward]
        label = REWARD_LABELS[reward]
        ep = mdf["episode"]
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
    _save(fig, os.path.join(results_dir, f"{prefix}_wealth_{constraint_filter}_{timestamp}.png"))


def _plot_social_welfare(aggregated, results_dir, timestamp, n_seeds, constraint_filter,
                         prefix, agent_title):
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ctitle = CONSTRAINT_LABELS.get(constraint_filter, constraint_filter)
    fig.suptitle(
        f"{agent_title} Long-term Social Welfare — {ctitle}  (mean ± std, {n_seeds} seeds, ep 1–{SHORT_TERM})",
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
        for mean, std in [
            (mdf["R_M"], sdf["R_M"]),
            (mdf["R_F"], sdf["R_F"]),
            (R_bar_mean, R_bar_std),
        ]:
            y_min = min(y_min, (mean - std).min())
            y_max = max(y_max, (mean + std).max())
    margin = (y_max - y_min) * 0.05
    ylim = (y_min - margin, y_max + margin)
    for reward, _, mdf, sdf, R_bar_mean, R_bar_std in plot_data:
        color = REWARD_COLORS[reward]
        label = REWARD_LABELS[reward]
        ep = mdf["episode"]
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
    axes[0].set_ylabel(r"$R_M$"); axes[0].set_title("(a) Male Long-term Social Welfare $R_M$")
    axes[1].set_ylabel(r"$R_F$"); axes[1].set_title("(b) Female Long-term Social Welfare $R_F$")
    axes[2].set_ylabel(r"$\bar{R}$"); axes[2].set_title(r"(c) Weighted Average $\bar{R}$")
    _finish_axes(axes)
    plt.tight_layout()
    _save(fig, os.path.join(results_dir, f"{prefix}_social_welfare_{constraint_filter}_{timestamp}.png"))


def _plot_inequality(aggregated, results_dir, timestamp, n_seeds, constraint_filter,
                     prefix, agent_title):
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ctitle = CONSTRAINT_LABELS.get(constraint_filter, constraint_filter)
    fig.suptitle(
        f"{agent_title} Wealth Inequality & Disparity — {ctitle}  (mean ± std, {n_seeds} seeds, ep 1–{SHORT_TERM})",
        fontsize=13, fontweight="bold",
    )
    for reward, _, mdf, sdf in _iter_combos(aggregated, constraint_filter):
        color = REWARD_COLORS[reward]
        label = REWARD_LABELS[reward]
        ep = mdf["episode"]
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
    _save(fig, os.path.join(results_dir, f"{prefix}_inequality_{constraint_filter}_{timestamp}.png"))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for cfg in RESULT_SETS:
        results_dir = cfg["dir"]
        timestamp   = cfg["timestamp"]
        prefix      = cfg["prefix"]
        agent_title = AGENT_TITLES[cfg["agent"]]

        print(f"\n── {agent_title}  ({results_dir}) ──")

        full_agg = load_aggregated(results_dir, timestamp)
        if not full_agg:
            print(f"  No CSVs found — skipping.")
            continue

        n_seeds = 3   # seeds 0, 1, 2
        short_agg = slice_to(full_agg, SHORT_TERM)

        for ct in CONSTRAINTS:
            # Check if any combo matches this constraint
            has_data = any(key.endswith(f"__{ct}") for key in short_agg)
            if not has_data:
                continue

            _plot_comparison(   short_agg, results_dir, timestamp, n_seeds, ct, prefix, agent_title)
            _plot_wealth(       short_agg, results_dir, timestamp, n_seeds, ct, prefix, agent_title)
            _plot_social_welfare(short_agg, results_dir, timestamp, n_seeds, ct, prefix, agent_title)
            _plot_inequality(   short_agg, results_dir, timestamp, n_seeds, ct, prefix, agent_title)

    print("\nDone.")
