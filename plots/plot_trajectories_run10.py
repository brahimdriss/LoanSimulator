#!/usr/bin/env python3
"""
Trajectory (line) plots, one PDF per (agent, constraint, metric): wealth
gap, cumulative profit, approval-rate disparity, long-term social welfare
R_bar, and inequality ratio rho(t), all OVER DEPLOY EPISODES (mean +/- 1
std over seeds where that's well-defined -- see caveats below). Within
each figure the three reward functions (RMM, FL, SW) are OVERLAID with a
fixed colour each -- same axis/overlay structure as pg_adapt.py /
pepg_adapt.py's own aggregate-mode plots (_plot_inequality,
_plot_social_welfare: one panel per metric, all reward functions overlaid
within it, split by constraint), just redone per metric as its own
standalone file in the paper house style (see plots/paper_style.py): no
titles/subplot grids, larger legible fonts, one shared standalone
reward-function legend reused across every figure -- 2 agents x 2
constraints x 5 metrics = 20 files.

Reward colours match plot_radar_run10.py's REWARD_COLORS (imported
directly, not duplicated) so a reward function is the same colour in
every figure across the whole plot suite.

Reads the same aggregated mean_*.csv / std_*.csv as reach_rate / matthew /
radar / bars -- no re-run needed.

Caveats, inherited from the codebase's existing precedent for these
derived quantities:
  * approval-rate disparity's std band combines approval_rate_M/F's stds
    in quadrature (sqrt(std_M^2 + std_F^2)) -- an approximation (treats
    the two groups' seed-to-seed noise as independent), not an exact
    propagation.
  * long-term social welfare R_bar(t) = (N_M(t)*R_M(t) + N_F(t)*R_F(t)) /
    (N_M(t)+N_F(t)), N_g = total_applications_g -- and its std band --
    are computed with the EXACT formula pepg_adapt.py's own
    _plot_social_welfare already uses (verbatim, not a new derivation),
    including that formula's own std approximation
    (sqrt((N_M*std_M)^2+(N_F*std_F)^2)/total).
  * inequality ratio rho_cumulative(t) = (mu_M_end(t) - mu_M_start(ep 1))
    / (mu_F_end(t) - mu_F_start(ep 1)) is a RATIO of two already-averaged
    quantities, not a proper per-seed ratio then averaged, and has NO std
    band -- a correct one needs the raw per-seed episodes.csv files, not
    just the aggregated mean_/std_ CSVs. Mean line only, exactly as
    post_process_rule_policies.py already does for the same quantity.
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import set_paper_style, save_figure, save_legend  # noqa: E402
from plot_radar_run10 import REWARD_COLORS, REWARD_LABELS, GROUP_REWARDS  # noqa: E402

AGENTS = [("pg", "RL"), ("pepg", "PERL")]
CONSTRAINTS = [("social", "Equality of Outcome"), ("eo", "Equality of Opportunity")]


def load_mean_std(root, agent, reward, constraint):
    m = sorted(glob.glob(os.path.join(root, agent, f"mean_{reward}__{constraint}_*.csv")))
    s = sorted(glob.glob(os.path.join(root, agent, f"std_{reward}__{constraint}_*.csv")))
    if not m or not s:
        return None, None
    return pd.read_csv(m[-1]), pd.read_csv(s[-1])


def smooth(series, n):
    return series.rolling(n, center=True, min_periods=1).mean() if n > 1 else series


def series_wealth_gap(mdf, sdf, n):
    return smooth(mdf["wealth_gap"], n), smooth(sdf["wealth_gap"], n)


def series_profit(mdf, sdf, n):
    return smooth(mdf["cumulative_profit"], n), smooth(sdf["cumulative_profit"], n)


def series_approval_disparity(mdf, sdf, n):
    mean = mdf["approval_rate_M_cumulative"] - mdf["approval_rate_F_cumulative"]
    std = np.sqrt(sdf["approval_rate_M_cumulative"] ** 2 + sdf["approval_rate_F_cumulative"] ** 2)
    return smooth(mean, n), smooth(std, n)


def series_social_welfare(mdf, sdf, n):
    """R_bar(t), verbatim formula from pepg_adapt.py's _plot_social_welfare."""
    N_M, N_F = mdf["total_applications_M"], mdf["total_applications_F"]
    total = (N_M + N_F).replace(0, np.nan)
    r_bar = (N_M * mdf["R_M"] + N_F * mdf["R_F"]) / total
    r_bar_std = np.sqrt((N_M * sdf["R_M"]) ** 2 + (N_F * sdf["R_F"]) ** 2) / total
    return smooth(r_bar, n), smooth(r_bar_std, n)


def series_inequality_ratio(mdf, sdf, n):
    d_M = mdf["mu_M_end"] - mdf["mu_M_start"].iloc[0]
    d_F = mdf["mu_F_end"] - mdf["mu_F_start"].iloc[0]
    rho = d_M / d_F.replace(0, np.nan)
    return smooth(rho, n), None   # no std band -- see module docstring


METRICS = [
    ("wealth_gap", "Wealth Gap", series_wealth_gap, 0.0),
    ("profit", "Cumulative Profit", series_profit, 0.0),
    ("approval_disparity", "Approval Rate Disparity", series_approval_disparity, 0.0),
    ("social_welfare", r"Long-Term Social Welfare $\bar{R}$", series_social_welfare, 0.0),
    ("inequality_ratio", r"Inequality Ratio $\rho(t)$", series_inequality_ratio, 1.0),
]


def plot_one(root, agent, constraint, stem_key, ylabel, series_fn, hline, out_dir, n_smooth):
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    any_data = False
    out = {}
    for reward in GROUP_REWARDS:
        mean_df, std_df = load_mean_std(root, agent, reward, constraint)
        if mean_df is None:
            continue
        any_data = True
        color = REWARD_COLORS[reward]
        ep = mean_df["episode"]
        mu, sd = series_fn(mean_df, std_df, n_smooth)
        ax.plot(ep, mu, color=color)
        if sd is not None:
            ax.fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.18, lw=0)
        out["episode"] = ep
        out[f"{reward}_mean"] = mu
        if sd is not None:
            out[f"{reward}_std"] = sd
    if any_data:
        ax.axhline(hline, color="k", lw=1.0, ls="--", alpha=0.6)
        ax.set_xlim(mean_df["episode"].iloc[0], mean_df["episode"].iloc[-1])
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)

    stem = f"traj_{stem_key}_{agent}_{constraint}"
    save_figure(fig, out_dir, stem, pd.DataFrame(out) if out else None)
    print(f"  saved -> {os.path.join(out_dir, stem + '.pdf')}" + ("" if any_data else "   (no data found)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="dir containing {pg,pepg}/mean_*.csv and std_*.csv")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "trajectories_run10"))
    ap.add_argument("--smooth", type=int, default=1,
                    help="centred rolling-mean window in episodes (1 = raw)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_paper_style()

    for stem_key, ylabel, series_fn, hline in METRICS:
        for agent, _ in AGENTS:
            for constraint, _ in CONSTRAINTS:
                plot_one(args.root, agent, constraint, stem_key, ylabel, series_fn,
                         hline, args.out, args.smooth)

    handles = [Line2D([0], [0], color=REWARD_COLORS[r], lw=2.5) for r in GROUP_REWARDS]
    labels = [REWARD_LABELS[r] for r in GROUP_REWARDS]
    save_legend(handles, labels, args.out, "trajectories_reward")
    print("done.")


if __name__ == "__main__":
    main()
