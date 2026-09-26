#!/usr/bin/env python3
"""
Multiplier trajectories with the three constraints OVERLAID: one standalone
PDF per (agent, perspective), so four figures -- lambda_overlay_{pepg,pg}_
{social,eo} -- each drawing FL, RMM and SW's lambda over the deployment
episodes in the paper's reward colours (plot_radar_run10.REWARD_COLORS),
mean over seeds with a +/-1 std band. One shared legend written once.
House style (plots/paper_style.py): no titles, no inline legend; combine
the four in LaTeX.

Same source as plot_lambda_run10.py: <root>/<agent>/deploy_artifacts/
<agent>_<reward>__<constraint>__seed*_training_trace.csv, column
lambda_wealth. --smooth N applies a centred rolling mean to the mean line
and band (default 1 = raw). --ymax fixes a shared y-limit (the multiplier's
cap is 10 for every cell, so 10.5 is the natural choice).
"""

import argparse, glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import set_paper_style, save_figure, save_legend  # noqa: E402
from plot_radar_run10 import REWARD_COLORS, REWARD_LABELS  # noqa: E402
from plot_lambda_run10 import load_lambda_traces  # noqa: E402

AGENTS = [("pepg", "PERL"), ("pg", "RL")]
CONSTRAINTS = [("social", "Equality of Outcome"), ("eo", "Equality of Opportunity")]
REWARDS = ["fairness_lagrangian", "rawlsian_maximin", "social_welfare"]


def smooth(x, n):
    return pd.Series(x).rolling(n, center=True, min_periods=1).mean().to_numpy() if n > 1 else x


def plot_one(root, agent, constraint, out_dir, n_smooth, ymax):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    out, drawn = {}, 0
    for reward in REWARDS:
        ep, stack = load_lambda_traces(root, agent, reward, constraint)
        if ep is None:
            continue
        mu, sd = smooth(stack.mean(0), n_smooth), smooth(stack.std(0), n_smooth)
        c = REWARD_COLORS[reward]
        ax.plot(ep, mu, color=c, lw=1.8)
        ax.fill_between(ep, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
        out.setdefault("episode", ep)
        out[f"{REWARD_LABELS[reward]}_mean"] = mu
        out[f"{REWARD_LABELS[reward]}_std"] = sd
        out[f"{REWARD_LABELS[reward]}_n_seeds"] = stack.shape[0]
        drawn += 1
    if drawn:
        ax.set_xlim(out["episode"][0], out["episode"][-1])
    # Small negative margin below zero so a curve sitting exactly at the floor
    # (1e-4) is drawn clear of the axis line instead of hidden under it.
    ax.set_ylim(-0.35, ymax) if ymax else ax.set_ylim(bottom=-0.35)
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"$\gamma^t$")
    stem = f"lambda_overlay_{agent}_{constraint}"
    save_figure(fig, out_dir, stem, pd.DataFrame(out) if drawn else None)
    print(f"  saved -> {os.path.join(out_dir, stem + '.pdf')}" + ("" if drawn else "   (no data found)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "run11_data"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "lambda_overlay_run11"))
    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--ymax", type=float, default=10.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_paper_style()
    for agent, _ in AGENTS:
        for constraint, _ in CONSTRAINTS:
            plot_one(args.root, agent, constraint, args.out, args.smooth, args.ymax)
    handles = [Line2D([0], [0], color=REWARD_COLORS[r], lw=2.5) for r in REWARDS]
    save_legend(handles, [REWARD_LABELS[r] for r in REWARDS], args.out, "lambda_overlay_reward")
    print("done.")


if __name__ == "__main__":
    main()
