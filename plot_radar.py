#!/usr/bin/env python3
"""
Radar plots: spokes = metrics, lines = reward functions.
One plot per (agent × constraint × ep_range).

Output: 24 figures  (3 agents × 4 constraints × 2 ep-ranges)
Naming: final_radar_{tag}_{ep_label}__{constraint}.png

Metrics (spokes):
  - Wealth Gap          (lower = better)
  - Social Welfare R̄   (higher = better)
  - Cumulative Profit   (higher = better)
  - |ρ − 1|            (lower = better, 0 = equal growth)

Normalisation: global-max reference.
  score = v / global_max              (higher-is-better metrics)
  score = 1 − v / global_max          (lower-is-better metrics)
  global_max = max across ALL agents × rewards × constraints × ep_range.
  This preserves magnitudes: 0.5 always means "half of the best observed value".

Episode values: exact episode 100 (short-term) or 1000 (long-term).
"""

import glob
import os
import re
from math import pi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULT_SETS = [
    {"dir": "results_pepg_adapt", "timestamp": "20260325_113301",
     "agent": "PePG Adapt",        "tag": "pepg_adapt"},
    {"dir": "results_pg",         "timestamp": "20260329_162259",
     "agent": "PG (Performative)", "tag": "pg"},
    {"dir": "results_static",     "timestamp": "20260403_151304",
     "agent": "Static",            "tag": "static"},
]

EP_RANGES = [("short_term", 100), ("long_term", 1000)]

REWARDS = [
    "utilitarian_profit",
    "social_welfare",
    "rawlsian_maximin",
    "fairness_lagrangian",
]
CONSTRAINTS = ["predictive", "social", "dm", "two_sided"]

REWARD_LABELS = {
    "utilitarian_profit":  "Util. Profit",
    "social_welfare":      "Social Welfare",
    "rawlsian_maximin":    "Rawlsian Max-Min",
    "fairness_lagrangian": "Fairness Lagr.",
}
REWARD_COLORS = {
    "utilitarian_profit":  "#1f77b4",
    "social_welfare":      "#2ca02c",
    "rawlsian_maximin":    "#ff7f0e",
    "fairness_lagrangian": "#9467bd",
}
CONSTRAINT_LABELS = {
    "predictive": "Predictive",
    "social":     "Social",
    "dm":         "DM",
    "two_sided":  "Two-Sided",
}

# Spokes: (raw column, display label, transform, higher_is_better)
METRICS = [
    ("wealth_gap",        "Wealth\nGap",        lambda x: abs(x),        False),
    ("R_bar",             "Social\nWelfare R̄",  lambda x: abs(x),        True),
    ("cumulative_profit", "Cumulative\nProfit",  lambda x: abs(x),        True),
    ("rho_episode",       "|ρ − 1|\n(equality)", lambda x: abs(x - 1.0), False),
]
METRIC_TRANS = [t   for _, _, t, _   in METRICS]
METRIC_HIB   = [hib for _, _, _, hib in METRICS]

OUT_DIR = "results_radar"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _inject_derived(mdf):
    N_M   = mdf["total_applications_M"]
    N_F   = mdf["total_applications_F"]
    total = (N_M + N_F).replace(0, np.nan)
    mdf["R_bar"] = (N_M * mdf["R_M"] + N_F * mdf["R_F"]) / total
    return mdf


def load_all():
    """Return dict tag → {'{reward}__{constraint}': mean_df}."""
    all_data = {}
    for cfg in RESULT_SETS:
        agg = {}
        for mean_path in sorted(
            glob.glob(os.path.join(cfg["dir"], f"mean_*_{cfg['timestamp']}.csv"))
        ):
            fname = os.path.basename(mean_path)
            m = re.match(r"mean_(.+)__(.+)_\d{8}_\d{6}\.csv$", fname)
            if not m:
                continue
            reward, constraint = m.group(1), m.group(2)
            agg[f"{reward}__{constraint}"] = _inject_derived(pd.read_csv(mean_path))
        all_data[cfg["tag"]] = agg
    return all_data


def episode_value(df, col, n_eps):
    """Value at exactly episode n_eps; falls back to last episode <= n_eps."""
    if col not in df.columns:
        return np.nan
    exact = df[df["episode"] == n_eps]
    if not exact.empty and not exact[col].isna().all():
        return float(exact[col].iloc[-1])
    sub = df[df["episode"] <= n_eps]
    if sub.empty or sub[col].isna().all():
        return np.nan
    return float(sub[col].iloc[-1])

# ---------------------------------------------------------------------------
# Per-agent min-max bounds
# ---------------------------------------------------------------------------

def compute_global_bounds(all_data, n_eps):
    """Min/max of each transformed metric across ALL agents × rewards × constraints."""
    vals = {i: [] for i in range(len(METRICS))}
    for tag, agg in all_data.items():
        for key, mdf in agg.items():
            for i, (col, _, tfn, _) in enumerate(METRICS):
                v = episode_value(mdf, col, n_eps)
                if not np.isnan(v):
                    vals[i].append(tfn(v))
    return {
        i: (min(vs) if vs else 0.0, max(vs) if vs else 1.0)
        for i, vs in vals.items()
    }


def norm_minmax(tv, lo, hi, higher_is_better, floor):
    """
    Min-max normalise into [floor, 1.0].
      floor = 1/N  so the worst run is always visible, never at the centre.
    """
    if np.isnan(tv):
        return floor
    scaled = (tv - lo) / (hi - lo) if hi != lo else 0.5
    if not higher_is_better:
        scaled = 1.0 - scaled
    scaled = np.clip(scaled, 0.0, 1.0)
    return float(floor + (1.0 - floor) * scaled)

# ---------------------------------------------------------------------------
# Radar helpers
# ---------------------------------------------------------------------------

def _setup_axes(ax):
    N      = len(METRICS)
    angles = [2 * pi * i / N for i in range(N)]
    labels = [lbl for _, lbl, _, _ in METRICS]
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
                       fontsize=5.5, color="grey")
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_visible(False)
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.55)


def draw_line(ax, norm_vals, label, color):
    N      = len(norm_vals)
    angles = [2 * pi * i / N for i in range(N)] + [0]
    closed = list(norm_vals) + [norm_vals[0]]
    ax.plot(angles, closed, color=color, linewidth=2.0, label=label)
    ax.fill(angles, closed, color=color, alpha=0.15)

# ---------------------------------------------------------------------------
# One figure per (agent, constraint, ep_range)
# ---------------------------------------------------------------------------

def make_figure(agg, agent_title, constraint, n_eps, ep_label, bounds, out_path):
    # Count available reward functions for this constraint to set floor
    present = [r for r in REWARDS if f"{r}__{constraint}" in agg]
    if not present:
        return False
    floor = 1.0 / len(present)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    fig.suptitle(
        f"{agent_title}  ·  {CONSTRAINT_LABELS[constraint]}  constraint",
        fontsize=13, fontweight="bold", y=1.02,
    )
    _setup_axes(ax)

    for reward in present:
        mdf = agg[f"{reward}__{constraint}"]
        raw_vals  = [episode_value(mdf, col, n_eps) for col, _, _, _ in METRICS]
        tfm_vals  = [METRIC_TRANS[i](raw_vals[i]) if not np.isnan(raw_vals[i]) else np.nan
                     for i in range(len(METRICS))]
        norm_vals = [norm_minmax(tfm_vals[i], *bounds[i], METRIC_HIB[i], floor)
                     for i in range(len(METRICS))]
        draw_line(ax, norm_vals, REWARD_LABELS[reward], REWARD_COLORS[reward])

    handles = [
        mpatches.Patch(color=REWARD_COLORS[r], label=REWARD_LABELS[r])
        for r in present
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.18, 1.12),
        fontsize=10,
        frameon=True,
        title="Reward fn",
        title_fontsize=10,
    )

    plt.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return True

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data …")
    all_data = load_all()
    for tag, agg in all_data.items():
        print(f"  {tag}: {len(agg)} combos")

    print()
    for ep_label, n_eps in EP_RANGES:
        print(f"Generating {ep_label} (ep {n_eps}) …")
        bounds = compute_global_bounds(all_data, n_eps)
        for cfg in RESULT_SETS:
            tag = cfg["tag"]
            agg = all_data[tag]
            for constraint in CONSTRAINTS:
                fname    = f"final_radar_{tag}_{ep_label}__{constraint}.png"
                out_path = os.path.join(OUT_DIR, fname)
                make_figure(agg, cfg["agent"], constraint, n_eps, ep_label, bounds, out_path)

    print("\nDone.")
