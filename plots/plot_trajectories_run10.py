#!/usr/bin/env python3
"""
Trajectory (line) plots, RL (pg) vs PERL (pepg), one PDF per (agent,
reward, metric): wealth gap, cumulative profit, inequality ratio, and
approval-rate disparity, all OVER DEPLOY EPISODES (mean +/- 1 std over
seeds where that's well-defined -- see caveats below). Equality of
Outcome (social) and Equality of Opportunity (eo) are OVERLAID on the
same axes so the two constraint types are directly comparable for a
given agent/reward/metric, rather than split across separate files.

These are the full-trajectory counterparts of plot_radar_run10.py /
plot_bars_run10.py's four final-episode snapshot metrics -- read from the
identical aggregated mean_*.csv / std_*.csv (columns wealth_gap,
cumulative_profit, approval_rate_{M,F}_cumulative, mu_{M,F}_start/end), no
re-run needed. This is the same {wealth_gap, approval_disparity,
rho_cumulative} trio pg_adapt.py / pepg_adapt.py's own aggregate mode has
always sketched (see their _plot_inequality), redone in the paper house
style (see plots/paper_style.py): one figure per (agent, reward, metric),
no titles/subplot grids, larger legible fonts, standalone shared legend.

Caveats, both inherited from the codebase's existing precedent for these
derived quantities (post_process_rule_policies.py):
  * approval-rate disparity's std band combines approval_rate_M/F's stds
    in quadrature (sqrt(std_M^2 + std_F^2)), the same convention used for
    R_bar's std elsewhere in this codebase -- an approximation (treats the
    two groups' seed-to-seed noise as independent), not an exact
    propagation.
  * inequality ratio rho_cumulative(t) = (mu_M_end(t) - mu_M_start(ep 1))
    / (mu_F_end(t) - mu_F_start(ep 1)) is a RATIO of two already-averaged
    quantities, not a proper per-seed ratio then averaged, and has NO std
    band -- computing one correctly needs the raw per-seed episodes.csv
    files, not just the aggregated mean_/std_ CSVs. Drawn as a mean line
    only, exactly as post_process_rule_policies.py already does for the
    same quantity.
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

AGENTS = [("pg", "RL"), ("pepg", "PERL")]
CONSTRAINTS = [("social", "Equality of Outcome"), ("eo", "Equality of Opportunity")]
REWARDS = [("rawlsian_maximin", "RMM"), ("fairness_lagrangian", "FL"), ("social_welfare", "SW")]
CONSTRAINT_COLORS = {"social": "#0072B2", "eo": "#D55E00"}   # Okabe-Ito blue / vermillion


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


def series_inequality_ratio(mdf, sdf, n):
    d_M = mdf["mu_M_end"] - mdf["mu_M_start"].iloc[0]
    d_F = mdf["mu_F_end"] - mdf["mu_F_start"].iloc[0]
    rho = (d_M / d_F.replace(0, np.nan))
    return smooth(rho, n), None   # no std band -- see module docstring


METRICS = [
    ("wealth_gap", "Wealth Gap", series_wealth_gap, 0.0),
    ("profit", "Cumulative Profit", series_profit, 0.0),
    ("approval_disparity", "Approval Rate Disparity", series_approval_disparity, 0.0),
    ("inequality_ratio", r"Inequality Ratio $\rho(t)$", series_inequality_ratio, 1.0),
]


def plot_one(root, agent, reward, stem_key, ylabel, series_fn, hline, out_dir, n_smooth):
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    any_data = False
    out = {}
    for constraint, _ in CONSTRAINTS:
        mean_df, std_df = load_mean_std(root, agent, reward, constraint)
        if mean_df is None:
            continue
        any_data = True
        color = CONSTRAINT_COLORS[constraint]
        ep = mean_df["episode"]
        mu, sd = series_fn(mean_df, std_df, n_smooth)
        ax.plot(ep, mu, color=color)
        if sd is not None:
            ax.fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.18, lw=0)
        out["episode"] = ep
        out[f"{constraint}_mean"] = mu
        if sd is not None:
            out[f"{constraint}_std"] = sd
    if any_data:
        ax.axhline(hline, color="k", lw=1.0, ls="--", alpha=0.6)
        ax.set_xlim(mean_df["episode"].iloc[0], mean_df["episode"].iloc[-1])
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)

    stem = f"traj_{stem_key}_{agent}_{reward}"
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
            for reward, _ in REWARDS:
                plot_one(args.root, agent, reward, stem_key, ylabel, series_fn,
                         hline, args.out, args.smooth)

    handles = [Line2D([0], [0], color=CONSTRAINT_COLORS[c], lw=2.5) for c, _ in CONSTRAINTS]
    labels = [label for _, label in CONSTRAINTS]
    save_legend(handles, labels, args.out, "trajectories_constraint")
    print("done.")


if __name__ == "__main__":
    main()
