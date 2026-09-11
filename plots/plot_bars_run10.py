#!/usr/bin/env python3
"""
RL vs PERL bar-chart comparison. Outcome (constraint="social") only -- EO
dropped per earlier request.

FOUR separate PDFs, one per metric (Wealth Gap, Profit, Inequality Ratio,
Approval Rate Disparity) -- each a standalone single-panel grouped bar
chart (RL vs PERL across RMM/FL/SW), no title (house style, see
plots/paper_style.py; the metric is carried by the filename / LaTeX
caption). Arranged into a row in the LaTeX source itself (an
includegraphics grid), not baked into one matplotlib subplot grid, so the
document controls spacing between the four independently. A standalone
RL/PERL legend is written once and reused across all four.

Same final-episode raw values as plot_radar_run10.py's raw_metrics() --
imported directly so the two stay in lockstep.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import set_paper_style, save_figure, save_legend  # noqa: E402
from plot_radar_run10 import load_final_row, raw_metrics, GROUP_REWARDS, METRICS  # noqa: E402

REWARD_LABEL = {"rawlsian_maximin": "RMM", "fairness_lagrangian": "FL", "social_welfare": "SW"}
AGENT_COLORS = {"RL": "#4c72b0", "PERL": "#c44e52"}
CONSTRAINT = "social"   # Equality of Outcome

# (file stem, divisor, suffix, decimals)
METRIC_FILES = [
    ("wealth_gap",          1,    "",  0),
    ("profit",              1e3,  "K", 0),
    ("inequality_ratio",    1,    "",  2),
    ("approval_disparity",  0.01, "%", 2),
]


def fmt_value(v, div, suffix, dec):
    return f"{v / div:.{dec}f}{suffix}"


def make_metric_figure(root, i, stem, div, suffix, dec, out_dir):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    x = range(len(GROUP_REWARDS))
    width = 0.36

    pg_vals = [raw_metrics(load_final_row(root, "pg", r, CONSTRAINT))[i] for r in GROUP_REWARDS]
    pepg_vals = [raw_metrics(load_final_row(root, "pepg", r, CONSTRAINT))[i] for r in GROUP_REWARDS]

    bars_rl = ax.bar([p - width / 2 for p in x], pg_vals, width, color=AGENT_COLORS["RL"])
    bars_perl = ax.bar([p + width / 2 for p in x], pepg_vals, width, color=AGENT_COLORS["PERL"])

    ymax = max(pg_vals + pepg_vals)
    ax.set_ylim(0, ymax * 1.32 if ymax > 0 else 1.0)
    for bars, vals in [(bars_rl, pg_vals), (bars_perl, pepg_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.03,
                     fmt_value(v, div, suffix, dec), ha="center", va="bottom")

    ax.set_xticks(list(x))
    ax.set_xticklabels([REWARD_LABEL[r] for r in GROUP_REWARDS])
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=2)

    out = pd.DataFrame({"reward": [REWARD_LABEL[r] for r in GROUP_REWARDS],
                        "pg": pg_vals, "pepg": pepg_vals})
    save_figure(fig, out_dir, f"bars_{stem}", out)
    print(f"  saved -> {os.path.join(out_dir, 'bars_' + stem + '.pdf')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="dir containing {pg,pepg}/mean_*.csv and std_*.csv")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "radar_run10"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_paper_style()

    for i, ((_, _), (stem, div, suffix, dec)) in enumerate(zip(METRICS, METRIC_FILES)):
        make_metric_figure(args.root, i, stem, div, suffix, dec, args.out)

    handles = [Patch(facecolor=AGENT_COLORS["RL"]), Patch(facecolor=AGENT_COLORS["PERL"])]
    save_legend(handles, ["RL", "PERL"], args.out, "bars_agent")
    print("done.")


if __name__ == "__main__":
    main()
