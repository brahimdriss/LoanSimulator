#!/usr/bin/env python3
"""
Cleaner MAIN-PAPER version of plot_best_vs_baselines_run10.py's figure:
just 4 lines instead of 8, for the appendix version (all 7 baselines,
full "PERL, Fairness Lagrangian (Equality of Outcome)" label) see that
script instead -- put its output in the appendix, this one in the main
paper.

Lines shown, one per metric (same 5 as plot_best_vs_baselines_run10.py):
  RL    = PG   / fairness_lagrangian / social   (blue)
  PERL  = PePG / fairness_lagrangian / social   (purple, reserved -- same
          colour as plot_radar_run10.py's fairness_lagrangian everywhere
          else in the plot suite)
  Oracle, Pattern Prediction -- the only 2 of the 7 rule-based baselines
          kept, same colours as the appendix version for consistency
          across the two figures.

Series formulas, mean_/std_ CSV lookup, and house style are all imported
from plot_best_vs_baselines_run10.py / plot_radar_run10.py rather than
re-derived, so the two figures can't silently drift apart on definitions.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import set_paper_style, save_figure, save_legend  # noqa: E402
from plot_radar_run10 import REWARD_COLORS  # noqa: E402
from plot_best_vs_baselines_run10 import (  # noqa: E402
    METRICS, load_mean_std, POLICY_COLORS, POLICY_LABELS,
)

RL_AGENT, RL_REWARD, RL_CONSTRAINT = "pg", "fairness_lagrangian", "social"
PERL_AGENT, PERL_REWARD, PERL_CONSTRAINT = "pepg", "fairness_lagrangian", "social"
RL_COLOR = "#0072B2"                       # distinct from PERL's reserved purple
PERL_COLOR = REWARD_COLORS[PERL_REWARD]    # "#9467bd" -- same purple everywhere else

BASELINES_SHOWN = ["oracle", "pattern_prediction"]

LINES = [
    ("RL", RL_AGENT, RL_REWARD, RL_CONSTRAINT, RL_COLOR, 10),
    ("PERL", PERL_AGENT, PERL_REWARD, PERL_CONSTRAINT, PERL_COLOR, 11),
] + [
    (POLICY_LABELS[p], "rule", p, "baseline", POLICY_COLORS[p], 5) for p in BASELINES_SHOWN
]


def plot_one(root, stem_key, ylabel, series_fn, hline, out_dir, n_smooth):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    any_data = False
    out = {}

    for label, agent, reward, constraint, color, zorder in LINES:
        mean_df, std_df = load_mean_std(root, agent, reward, constraint)
        if mean_df is None:
            continue
        any_data = True
        ep = mean_df["episode"]
        mu, sd = series_fn(mean_df, std_df, n_smooth)
        lw = 2.4 if label in ("RL", "PERL") else 1.8
        ax.plot(ep, mu, color=color, lw=lw, zorder=zorder)
        if sd is not None:
            ax.fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.18, lw=0, zorder=zorder - 1)
        if "episode" not in out:
            out["episode"] = ep
        out[f"{label}_mean"] = mu
        if sd is not None:
            out[f"{label}_std"] = sd

    if any_data:
        ax.axhline(hline, color="k", lw=1.0, ls="--", alpha=0.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)

    stem = f"main_{stem_key}"
    save_figure(fig, out_dir, stem, pd.DataFrame(out) if out else None)
    print(f"  saved -> {os.path.join(out_dir, stem + '.pdf')}" + ("" if any_data else "   (no data found)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="dir containing pg/, pepg/ mean_*.csv and rule/ mean_{policy}__baseline_*.csv")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "main_comparison_run10"))
    ap.add_argument("--smooth", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_paper_style()

    for stem_key, ylabel, series_fn, hline in METRICS:
        plot_one(args.root, stem_key, ylabel, series_fn, hline, args.out, args.smooth)

    handles = [Line2D([0], [0], color=color, lw=2.5) for _, _, _, _, color, _ in LINES]
    labels = [label for label, _, _, _, _, _ in LINES]
    save_legend(handles, labels, args.out, "main_comparison")
    print("done.")


if __name__ == "__main__":
    main()
