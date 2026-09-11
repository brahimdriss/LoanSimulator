#!/usr/bin/env python3
"""
Matthew-effect curves, RL (pg) vs PERL (pepg), under social (Equality of
Outcome) and eo (Equality of Opportunity) fairness: mean group wealth
mu_M(t), mu_F(t) over deploy episodes, mean over seeds +/- 1 std, showing
whether/how much each policy accelerates or mitigates the population's own
"rich get richer" wealth divergence (the environment compounds wealth on
every approved, non-defaulted loan; nothing bounds it).

One PDF per (agent, constraint, reward) combo -- no subplot grids, no
titles, no inline legend (house style, see plots/paper_style.py); combine
combos side by side in the LaTeX source instead. A standalone Male/Female
legend is written once and reused across every combo.

Read straight from the aggregated mean_*.csv / std_*.csv (columns
mu_M_end / mu_F_end) -- no re-run needed, same files plot_reach_rate_run10.py
reads. --smooth N applies a centred rolling mean of N episodes.
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import set_paper_style, save_figure, save_legend  # noqa: E402

AGENTS = [("pg", "RL"), ("pepg", "PERL")]
CONSTRAINTS = [("social", "Equality of Outcome"), ("eo", "Equality of Opportunity")]
REWARDS = [("rawlsian_maximin", "RMM"), ("fairness_lagrangian", "FL"), ("social_welfare", "SW")]
GROUPS = [("M", "Male", "#C0392B"), ("F", "Female", "#2980B9")]


def load_mean_std(root, agent, reward, constraint):
    m = sorted(glob.glob(os.path.join(root, agent, f"mean_{reward}__{constraint}_*.csv")))
    s = sorted(glob.glob(os.path.join(root, agent, f"std_{reward}__{constraint}_*.csv")))
    if not m or not s:
        return None, None
    return pd.read_csv(m[-1]), pd.read_csv(s[-1])


def smooth(series, n):
    return series.rolling(n, center=True, min_periods=1).mean() if n > 1 else series


def plot_one(root, agent, constraint, reward, out_dir, n_smooth):
    # 4.4" tall, not 3.6" -- see plot_reach_rate_run10.py's comment on the
    # same fix (tight-bbox undercounting a rotated y-axis label's height).
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    mean_df, std_df = load_mean_std(root, agent, reward, constraint)
    out = None
    if mean_df is not None:
        ep = mean_df["episode"]
        out = {"episode": ep}
        for g, _, color in GROUPS:
            mu = smooth(mean_df[f"mu_{g}_end"], n_smooth)
            sd = smooth(std_df[f"mu_{g}_end"], n_smooth)
            ax.plot(ep, mu, color=color)
            ax.fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.18, lw=0)
            out[f"mu_{g}_mean"] = mu
            out[f"mu_{g}_std"] = sd
        ax.set_xlim(ep.iloc[0], ep.iloc[-1])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean group wealth")

    stem = f"matthew_{agent}_{constraint}_{reward}"
    save_figure(fig, out_dir, stem, pd.DataFrame(out) if out else None)
    print(f"  saved -> {os.path.join(out_dir, stem + '.pdf')}" + ("" if out else "   (no data found)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="dir containing {pg,pepg}/mean_*.csv and std_*.csv")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "matthew_run10"))
    ap.add_argument("--smooth", type=int, default=1,
                    help="centred rolling-mean window in episodes (1 = raw)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_paper_style()

    for agent, _ in AGENTS:
        for constraint, _ in CONSTRAINTS:
            for reward, _ in REWARDS:
                plot_one(args.root, agent, constraint, reward, args.out, args.smooth)

    handles = [Line2D([0], [0], color=color, lw=2.5) for _, _, color in GROUPS]
    labels = [label for _, label, _ in GROUPS]
    save_legend(handles, labels, args.out, "matthew_group")
    print("done.")


if __name__ == "__main__":
    main()
