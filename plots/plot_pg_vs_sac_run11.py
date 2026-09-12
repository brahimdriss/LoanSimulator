#!/usr/bin/env python3
"""
RL (pg) vs SAC, head-to-head on the same 5 per-episode metrics and series
functions as plot_trajectories_run10.py (wealth gap, cumulative profit,
approval-rate disparity, female long-term social welfare R_F, inequality
ratio rho(t)), but with the two agents OVERLAID in one panel instead of
one panel per agent -- that script's series_*/METRICS/load_mean_std are
reused directly, not reimplemented, so the two comparisons can't drift.

Encoding: colour = reward function (RMM/FL/SW, same REWARD_COLORS as
every other figure in this suite), linestyle = agent (solid = RL/pg,
dashed = SAC) -- the same colour+linestyle combination convention as
plot_radar_run10's CONSTRAINT_STYLE, just carrying agent instead of
constraint since constraint is already what splits these into separate
figures (one per (metric, constraint), 5 x 2 = 10 files).

Std bands are drawn for both agents at a lower alpha than a single-agent
panel would use (6 overlapping fills otherwise get visually busy) -- see
plot_trajectories_run10.py's own docstring for the per-metric std-band
caveats (approval disparity in quadrature, inequality ratio has none).

Reads the same aggregated mean_*.csv / std_*.csv as every other plot in
this suite, requires only run11_data/{pg,sac} (or --root pointing wherever
those two live) -- no re-run, no new aggregation.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import set_paper_style, save_figure, save_legend  # noqa: E402
from plot_radar_run10 import REWARD_COLORS, REWARD_LABELS, GROUP_REWARDS  # noqa: E402
from plot_trajectories_run10 import load_mean_std, METRICS, CONSTRAINTS  # noqa: E402

AGENTS = [("pg", "RL"), ("sac", "SAC")]
AGENT_STYLE = {"pg": "-", "sac": "--"}


def plot_one(root, constraint, stem_key, ylabel, series_fn, hline, out_dir, n_smooth):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    any_data = False
    out = {}
    xmin = xmax = None
    for agent, _ in AGENTS:
        ls = AGENT_STYLE[agent]
        for reward in GROUP_REWARDS:
            mean_df, std_df = load_mean_std(root, agent, reward, constraint)
            if mean_df is None:
                continue
            any_data = True
            color = REWARD_COLORS[reward]
            ep = mean_df["episode"]
            mu, sd = series_fn(mean_df, std_df, n_smooth)
            ax.plot(ep, mu, color=color, linestyle=ls)
            if sd is not None:
                ax.fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.12, lw=0)
            out["episode"] = ep
            out[f"{agent}_{reward}_mean"] = mu
            if sd is not None:
                out[f"{agent}_{reward}_std"] = sd
            xmin = ep.iloc[0] if xmin is None else min(xmin, ep.iloc[0])
            xmax = ep.iloc[-1] if xmax is None else max(xmax, ep.iloc[-1])
    if any_data:
        ax.axhline(hline, color="k", lw=1.0, ls=":", alpha=0.6)
        ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)

    stem = f"pgvsac_{stem_key}_{constraint}"
    import pandas as pd
    save_figure(fig, out_dir, stem, pd.DataFrame(out) if out else None)
    print(f"  saved -> {os.path.join(out_dir, stem + '.pdf')}" + ("" if any_data else "   (no data found)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run11_data"),
        help="dir containing {pg,sac}/mean_*.csv and std_*.csv")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "pg_vs_sac_run11"))
    ap.add_argument("--smooth", type=int, default=1,
                    help="centred rolling-mean window in episodes (1 = raw)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_paper_style()

    for stem_key, ylabel, series_fn, hline in METRICS:
        for constraint, _ in CONSTRAINTS:
            plot_one(args.root, constraint, stem_key, ylabel, series_fn,
                     hline, args.out, args.smooth)

    reward_handles = [Line2D([0], [0], color=REWARD_COLORS[r], lw=2.5) for r in GROUP_REWARDS]
    reward_labels = [REWARD_LABELS[r] for r in GROUP_REWARDS]
    save_legend(reward_handles, reward_labels, args.out, "pgvsac_reward")

    agent_handles = [Line2D([0], [0], color="black", lw=2.5, linestyle=AGENT_STYLE[a]) for a, _ in AGENTS]
    agent_labels = [label for _, label in AGENTS]
    save_legend(agent_handles, agent_labels, args.out, "pgvsac_agent")
    print("done.")


if __name__ == "__main__":
    main()
