#!/usr/bin/env python3
"""
Final-episode radar plots, RL (pg) vs PERL (pepg), ONE figure per agent
(not per agent+constraint): all 6 combos -- RMM/FL/SW under BOTH social
(Outcome) and eo (Opportunity) -- drawn on the SAME axes. Colour carries
the reward function (same colour scheme as every other script, see
REWARD_COLORS), linestyle carries the perspective (solid = social/
Outcome, dashed = eo/Opportunity). Two standalone legends (reward colour,
perspective linestyle), written once and reused.

This replaces an earlier version that drew social and eo as two SEPARATE
files sharing one normalization scale. That made the two plots
technically comparable in principle (the same combo landed at the same
radial position in either file), but a reader looking at one image at a
time has no way to tell that apart from genuine performance -- and in
practice one perspective's own worst-case combo sets the scale, so the
OTHER perspective's merely-mediocre combos can get pushed out near the
rim and look deceptively strong purely because they're better than that
outlier, not because they're actually good. Putting both perspectives on
one chart removes the need to infer anything across files: the
comparison is right there, at a glance.

Axes: wealth gap, profit, inequality ratio, approval rate disparity,
plotted on their raw (untransformed) scale -- no log.

Normalization is a ZERO-ANCHORED RATIO: for higher-is-better (profit),
normalized = value / max; for lower-is-better (wealth_gap, |rho-1|,
approval_rate_disparity), zero-anchored at the genuinely meaningful
"0 = perfect" point, normalized = 1 - value / max. Scale is a fixed MAX
per (agent, metric), computed once across all 6 of that agent's combos
(RMM/FL/SW x {social, eo} -- excluding invisible dm/two_sided so an
unseen combo can't set the scale) -- since all 6 are now on one figure,
this is the honest, non-artifactual use of a shared scale: everything
drawn on this one image genuinely shares one yardstick.

rho_cumulative (the inequality ratio) is NOT a column the aggregate mode
writes -- it's computed here from mu_M_start/mu_F_start (deploy's first
logged episode) and mu_M_end/mu_F_end (deploy's final episode) in the same
mean_*.csv row, matching post_process_rule_policies.py's definition:
(mu_M_end - mu_M_start) / (mu_F_end - mu_F_start).

CAUTION when a combo has high across-seed variance in the underlying
per-episode metrics (see cluster/verify_run.py --diagnose's spread
warning): this plots the mean_*.csv row, i.e. the across-seed MEAN at the
final episode, which can sit in between two genuinely different regimes
for a bimodal combo (e.g. a seed-dependent limit cycle) rather than
describing either one. Check --diagnose's spread warning for the combo
before reading a radar axis as "the" behaviour.
"""

import argparse
import glob
import os
import sys
from math import pi

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import set_paper_style, save_figure, save_legend  # noqa: E402

REWARD_COLORS = {
    "social_welfare":      "#2ca02c",
    "rawlsian_maximin":    "#ff7f0e",
    "fairness_lagrangian": "#9467bd",
}
REWARD_LABELS = {
    "social_welfare": "SW", "rawlsian_maximin": "RMM", "fairness_lagrangian": "FL",
}

AGENTS = [("pepg", "pepg"), ("pg", "pg")]                 # (tag, results subdir)
FAIRNESS_GROUPS = [("social", "social"), ("eo", "eo")]     # (filename label, constraint code)
GROUP_REWARDS = ["rawlsian_maximin", "fairness_lagrangian", "social_welfare"]

METRICS = [
    ("Wealth Gap",               False),
    ("Profit",                   True),
    ("Inequality Ratio",         False),
    ("Approval Rate Disparity",  False),
]
METRIC_LABELS = [m[0] for m in METRICS]
HIGHER_IS_BETTER = [m[1] for m in METRICS]
EPS = 1e-6
FLOOR = 0.04


def load_final_row(root, agent_dir, reward, constraint):
    # sorted()[-1]: timestamps sort lexicographically = chronologically, so
    # this is the MOST RECENT aggregate run. An unsorted matches[0] can pick
    # an older, possibly-ragged run (e.g. one taken while a straggler seed's
    # deploy was still writing its per-seed CSV) instead of the final,
    # complete one -- same convention as plot_reach_rate_run10.py /
    # plot_matthew_run10.py's load_mean_std.
    matches = sorted(glob.glob(os.path.join(root, agent_dir, f"mean_{reward}__{constraint}_*.csv")))
    if not matches:
        raise FileNotFoundError(f"no mean_{reward}__{constraint}_*.csv under {root}/{agent_dir}")
    df = pd.read_csv(matches[-1])
    row = df.iloc[-1].copy()
    first = df.iloc[0]
    d_F = row["mu_F_end"] - first["mu_F_start"]
    d_M = row["mu_M_end"] - first["mu_M_start"]
    row["rho_cumulative"] = d_M / d_F if abs(d_F) > EPS else np.nan
    return row


def raw_metrics(row):
    """Raw (untransformed) [wealth_gap, profit, inequality, approval_rate_
    disparity], in their own direction (not yet flipped for higher_is_
    better -- that happens at normalization time). approval_rate_disparity
    uses the CUMULATIVE approval rates (matching the "cumulative, final
    snapshot" convention already used for wealth_gap and rho_cumulative)."""
    wealth_gap = abs(row["wealth_gap"])
    profit = row["cumulative_profit"]
    inequality = abs(row["rho_cumulative"] - 1.0) if pd.notna(row["rho_cumulative"]) else 0.0
    approval_disparity = abs(row["approval_rate_M_cumulative"] - row["approval_rate_F_cumulative"])
    return [wealth_gap, profit, inequality, approval_disparity]


def compute_agent_scale(root, agent_dir):
    """Max of each raw metric across the 6 relevant combos for this agent
    (RMM/FL/SW under BOTH social and eo). Fixed denominator, not a range
    to stretch -- see module docstring."""
    maxima = [0.0] * len(METRICS)
    for constraint in ("social", "eo"):
        for reward in GROUP_REWARDS:
            row = load_final_row(root, agent_dir, reward, constraint)
            for i, v in enumerate(raw_metrics(row)):
                maxima[i] = max(maxima[i], v)
    return maxima


def normalize(raw_vals, scale):
    """Zero-anchored ratio: value/max (higher-is-better) or 1 - value/max
    (lower-is-better, "0 = perfect" is the genuine anchor). A small clip
    keeps values off the exact center point for legibility without
    stretching the meaningful proportions."""
    normed = []
    for i, v in enumerate(raw_vals):
        m = scale[i]
        ratio = v / m if m > EPS else 0.0
        if not HIGHER_IS_BETTER[i]:
            ratio = 1.0 - ratio
        normed.append(max(ratio, FLOOR))
    return normed


def setup_axes(ax, n_axes):
    angles = [2 * pi * i / n_axes for i in range(n_axes)]
    ax.set_xticks(angles)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color="grey")
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_visible(False)
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.55)


CONSTRAINT_STYLE = {"social": "-", "eo": "--"}
CONSTRAINT_LABELS = {"social": "SP (Outcome)", "eo": "EO (Opportunity)"}


def draw_line(ax, norm_vals, color, linestyle):
    n = len(norm_vals)
    angles = [2 * pi * i / n for i in range(n)] + [0]
    closed = list(norm_vals) + [norm_vals[0]]
    ax.plot(angles, closed, color=color, linestyle=linestyle, linewidth=2.4)
    ax.fill(angles, closed, color=color, alpha=0.10)


def make_figure(root, agent_dir, scale, out_dir, stem):
    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw={"projection": "polar"})
    setup_axes(ax, len(METRIC_LABELS))
    out = {"metric": METRIC_LABELS}
    for constraint, _ in FAIRNESS_GROUPS:
        for reward in GROUP_REWARDS:
            row = load_final_row(root, agent_dir, reward, constraint)
            raw = raw_metrics(row)
            norm_vals = normalize(raw, scale)
            draw_line(ax, norm_vals, REWARD_COLORS[reward], CONSTRAINT_STYLE[constraint])
            out[f"{reward}_{constraint}_raw"] = raw
            out[f"{reward}_{constraint}_norm"] = norm_vals
    save_figure(fig, out_dir, stem, pd.DataFrame(out))
    print(f"  saved -> {os.path.join(out_dir, stem + '.pdf')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="dir containing {pg,pepg}/mean_*.csv and std_*.csv")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "radar_run10"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_paper_style()

    for agent_tag, agent_dir in AGENTS:
        scale = compute_agent_scale(args.root, agent_dir)
        make_figure(args.root, agent_dir, scale, args.out, f"radar_{agent_tag}")

    handles = [Line2D([0], [0], color=REWARD_COLORS[r], lw=2.5) for r in GROUP_REWARDS]
    labels = [REWARD_LABELS[r] for r in GROUP_REWARDS]
    save_legend(handles, labels, args.out, "radar_reward")

    handles = [Line2D([0], [0], color="black", lw=2.5, linestyle=CONSTRAINT_STYLE[c])
               for c, _ in FAIRNESS_GROUPS]
    labels = [CONSTRAINT_LABELS[c] for c, _ in FAIRNESS_GROUPS]
    save_legend(handles, labels, args.out, "radar_perspective")
    print("done.")


if __name__ == "__main__":
    main()
