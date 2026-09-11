#!/usr/bin/env python3
"""
Clean 3-line PERL radar: fairness_lagrangian only, under both perspectives,
plus the frozen-lambda* "balanced algorithm" -- dropping RMM/SW entirely
so the chart tells one focused story: FL/Outcome (good fairness, modest
profit) vs. FL/Opportunity (great profit, poor fairness) vs. the frozen-
lambda variant that gets both at once, rather than the full 6-combo
comparison plot_radar_balanced_run10's earlier version drew.

Colour/style, deliberately NOT reusing REWARD_COLORS/CONSTRAINT_STYLE from
plot_radar_run10.py (those encode "one of 3 rewards x one of 2
perspectives" across 6+ lines; with only one reward function on this
chart that encoding is unnecessary and the same purple-solid/dashed pair
undersells how different the frozen variant is):
  FL/Outcome   -- purple, solid       (same purple as FL everywhere else
                                        in the plot suite)
  FL/Opportunity -- purple, dotted    (same reward, weaker perspective --
                                        a lighter touch than a second solid
                                        line would give)
  Balanced (frozen lambda*) -- dark crimson, solid, thick -- a genuinely
                                        different colour, not a variant of
                                        purple, since it isn't just another
                                        point in the FL family, it's a
                                        different algorithm design.

Scale is the zero-anchored ratio from plot_radar_run10.py, recomputed
over just these 3 combos (not all 6) so the chart's own scale matches
what's actually drawn on it.

Reuses plot_radar_run10.py's load_final_row / raw_metrics / setup_axes /
draw_line / METRIC_LABELS / HIGHER_IS_BETTER rather than re-deriving them.
"""

import argparse
import os
import sys
from math import pi

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import set_paper_style, save_figure, save_legend  # noqa: E402
from plot_radar_run10 import (  # noqa: E402
    METRIC_LABELS, HIGHER_IS_BETTER, EPS, FLOOR,
    load_final_row, raw_metrics, setup_axes, draw_line,
)

FL_COLOR = "#9467bd"          # same purple as fairness_lagrangian everywhere else
BALANCED_COLOR = "#8B0000"    # dark crimson -- deliberately not a purple variant

LINES = [
    ("FL / Outcome",    "social", FL_COLOR, "-", 2.4),
    ("FL / Opportunity", "eo",    FL_COLOR, ":", 2.4),
]
BALANCED_LABEL = r"Balanced ($\lambda^*{=}1.87$, frozen)"


def _closed_angles(norm_vals):
    n = len(norm_vals)
    angles = [2 * pi * i / n for i in range(n)] + [0]
    return angles, list(norm_vals) + [norm_vals[0]]


def normalize(raw_vals, scale):
    out = []
    for i, v in enumerate(raw_vals):
        m = scale[i]
        ratio = v / m if m > EPS else 0.0
        if not HIGHER_IS_BETTER[i]:
            ratio = 1.0 - ratio
        out.append(max(ratio, FLOOR))
    return out


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

    rows = {label: load_final_row(args.root, "pepg", "fairness_lagrangian", cons)
            for label, cons, *_ in LINES}
    balanced_row = load_final_row(args.extra_root, "pepg", "fairness_lagrangian", "social")

    scale = [0.0] * len(METRIC_LABELS)
    for row in list(rows.values()) + [balanced_row]:
        for i, v in enumerate(raw_metrics(row)):
            scale[i] = max(scale[i], v)

    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw={"projection": "polar"})
    setup_axes(ax, len(METRIC_LABELS))

    for label, cons, color, style, lw in LINES:
        draw_line(ax, normalize(raw_metrics(rows[label]), scale), color, style)

    bal = normalize(raw_metrics(balanced_row), scale)
    ax.plot(*_closed_angles(bal), color=BALANCED_COLOR, linestyle="-", linewidth=3.2, zorder=20)
    ax.fill(*_closed_angles(bal), color=BALANCED_COLOR, alpha=0.08, zorder=19)

    save_figure(fig, args.out, "radar_pepg_fl_balanced")
    print(f"  saved -> {os.path.join(args.out, 'radar_pepg_fl_balanced.pdf')}")

    handles = [Line2D([0], [0], color=color, lw=2.4, linestyle=style) for _, _, color, style, _ in LINES]
    handles.append(Line2D([0], [0], color=BALANCED_COLOR, lw=3.2, linestyle="-"))
    labels = [label for label, *_ in LINES] + [BALANCED_LABEL]
    save_legend(handles, labels, args.out, "radar_fl_balanced", ncol=1)
    print("done.")


if __name__ == "__main__":
    main()
