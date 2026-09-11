#!/usr/bin/env python3
"""
PERL's combined radar (see plot_radar_run10.py) plus ONE extra line: the
frozen-lambda* "balanced algorithm" (fairness_lagrangian / social, lambda
held fixed at lambda-value for the whole run instead of adaptive dual
ascent -- see cluster/lambda_sweep.sub and the session's frozen-lambda
sweep). Drawn in a genuinely distinct style (black, dotted, thicker)
since it shares both reward (FL) and perspective (social/Outcome) with an
existing line and would otherwise be indistinguishable from it.

The scale is RECOMPUTED to include this 7th combo -- it beats every one
of PERL's other six combos on wealth gap, profit, and inequality ratio
simultaneously, so folding it into the same fixed-max normalization the
other six already share (rather than plotting it on its own separate
scale) is what makes the comparison honest: everything on this one chart
uses the same yardstick, this line included.

Reuses plot_radar_run10.py's load_final_row / raw_metrics / REWARD_COLORS
/ setup_axes / draw_line / CONSTRAINT_STYLE directly rather than
re-deriving them, so the two scripts can't quietly drift apart on
definitions.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import set_paper_style, save_figure, save_legend  # noqa: E402
from plot_radar_run10 import (  # noqa: E402
    REWARD_COLORS, REWARD_LABELS, GROUP_REWARDS, FAIRNESS_GROUPS,
    CONSTRAINT_STYLE, CONSTRAINT_LABELS, METRIC_LABELS, HIGHER_IS_BETTER, EPS, FLOOR,
    load_final_row, raw_metrics, setup_axes, draw_line,
)

BALANCED_COLOR = "#000000"
BALANCED_STYLE = ":"
BALANCED_LABEL = r"Balanced ($\lambda^*{=}1.87$, frozen)"


def compute_scale(root, agent_dir, extra_root):
    """Same as plot_radar_run10.compute_agent_scale, but the max also
    includes the extra frozen-lambda combo."""
    maxima = [0.0] * len(METRIC_LABELS)
    for constraint, _ in FAIRNESS_GROUPS:
        for reward in GROUP_REWARDS:
            row = load_final_row(root, agent_dir, reward, constraint)
            for i, v in enumerate(raw_metrics(row)):
                maxima[i] = max(maxima[i], v)
    extra_row = load_final_row(extra_root, agent_dir, "fairness_lagrangian", "social")
    for i, v in enumerate(raw_metrics(extra_row)):
        maxima[i] = max(maxima[i], v)
    return maxima, extra_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="run11-style dir containing pepg/mean_*.csv")
    ap.add_argument("--extra-root", required=True,
                    help="dir containing pepg/mean_fairness_lagrangian__social_*.csv for the "
                         "frozen-lambda campaign (e.g. lambda_sweep_data/lambda_187)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "radar_balanced_run10"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_paper_style()

    agent_dir = "pepg"
    scale, extra_row = compute_scale(args.root, agent_dir, args.extra_root)

    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw={"projection": "polar"})
    setup_axes(ax, len(METRIC_LABELS))

    def normalize(raw_vals):
        out = []
        for i, v in enumerate(raw_vals):
            m = scale[i]
            ratio = v / m if m > EPS else 0.0
            if not HIGHER_IS_BETTER[i]:
                ratio = 1.0 - ratio
            out.append(max(ratio, FLOOR))
        return out

    for constraint, _ in FAIRNESS_GROUPS:
        for reward in GROUP_REWARDS:
            row = load_final_row(args.root, agent_dir, reward, constraint)
            draw_line(ax, normalize(raw_metrics(row)), REWARD_COLORS[reward], CONSTRAINT_STYLE[constraint])

    balanced_norm = normalize(raw_metrics(extra_row))
    ax.plot(*_closed_angles(balanced_norm), color=BALANCED_COLOR, linestyle=BALANCED_STYLE, linewidth=3.2, zorder=20)
    ax.fill(*_closed_angles(balanced_norm), color=BALANCED_COLOR, alpha=0.06, zorder=19)

    save_figure(fig, args.out, "radar_pepg_balanced")
    print(f"  saved -> {os.path.join(args.out, 'radar_pepg_balanced.pdf')}")

    handles = [Line2D([0], [0], color=REWARD_COLORS[r], lw=2.5) for r in GROUP_REWARDS]
    labels = [REWARD_LABELS[r] for r in GROUP_REWARDS]
    save_legend(handles, labels, args.out, "radar_reward")

    handles = [Line2D([0], [0], color="black", lw=2.5, linestyle=CONSTRAINT_STYLE[c])
               for c, _ in FAIRNESS_GROUPS]
    labels = [CONSTRAINT_LABELS[c] for c, _ in FAIRNESS_GROUPS]
    save_legend(handles, labels, args.out, "radar_perspective")

    handles = [Line2D([0], [0], color=BALANCED_COLOR, lw=3.2, linestyle=BALANCED_STYLE)]
    save_legend(handles, [BALANCED_LABEL], args.out, "radar_balanced")
    print("done.")


def _closed_angles(norm_vals):
    from math import pi
    n = len(norm_vals)
    angles = [2 * pi * i / n for i in range(n)] + [0]
    closed = list(norm_vals) + [norm_vals[0]]
    return angles, closed


if __name__ == "__main__":
    main()
