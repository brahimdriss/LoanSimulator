#!/usr/bin/env python3
"""
Trajectory (line) plots, one PDF per (agent, constraint, metric): wealth
gap, cumulative profit, approval-rate disparity, FEMALE long-term social
welfare R_F, and inequality ratio rho(t), all OVER DEPLOY EPISODES (mean
+/- 1 std over seeds where that's well-defined -- see caveats below).
Deliberately R_F alone, not the N-weighted R_bar average pepg_adapt.py's
own plots use: R_bar lets an overtly male-preferring policy's high R_M
mask a poor R_F behind a decent-looking weighted mean. Within
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
  * female long-term social welfare is R_F(t) read straight off the
    aggregated mean_/std_ columns, no derived formula involved.
  * inequality ratio rho_cumulative(t) = (mu_M_end(t) - mu_M_start(ep 1))
    / (mu_F_end(t) - mu_F_start(ep 1)) is a RATIO of two already-averaged
    quantities, not a per-seed ratio then averaged. Its band is a
    bootstrap over seeds of that same ratio of seed-means, computed from
    the per-seed checkpoints (bootstrap_rho_band), omitted for the first
    50 episodes where the denominator is near zero; --no-rho-band restores
    the old mean-line-only figure.
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


def bootstrap_rho_band(root, agent, reward, constraint, n_boot=2000, start=50, seed=0):
    """Per-episode bootstrap std of rho(t) = mean_s dM_s(t) / mean_s dF_s(t),
    resampling SEEDS with replacement -- the same ratio-of-seed-means the
    plotted line uses, and the same convention as the tables' rho std.
    dM_s(t) = mu_M_end(t) - mu_M_start(ep 1) for seed s (likewise dF).
    Returned as a float array over episodes, NaN for the first `start`
    episodes, where dF is near zero and the ratio is not meaningful.
    Reads the per-seed checkpoints (<root>/<agent>/checkpoints/
    seed*_<reward>__<constraint>.csv); returns (None, None) if absent."""
    files = sorted(glob.glob(os.path.join(
        root, agent, "checkpoints", f"seed*_{reward}__{constraint}.csv")))
    if not files:
        return None, None
    dfs = [pd.read_csv(f) for f in files]
    T = min(len(d) for d in dfs)
    dM = np.stack([d["mu_M_end"].values[:T] - d["mu_M_start"].values[0] for d in dfs])
    dF = np.stack([d["mu_F_end"].values[:T] - d["mu_F_start"].values[0] for d in dfs])
    S = dM.shape[0]
    rng = np.random.default_rng(seed)
    acc, acc2, done = np.zeros(T), np.zeros(T), 0
    while done < n_boot:
        b = min(200, n_boot - done)
        idx = rng.integers(0, S, size=(b, S))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = dM[idx].mean(axis=1) / dF[idx].mean(axis=1)
        acc += r.sum(axis=0); acc2 += (r ** 2).sum(axis=0); done += b
    mean_b = acc / n_boot
    sd = np.sqrt(np.maximum(acc2 / n_boot - mean_b ** 2, 0.0))
    sd[:start] = np.nan
    point = dM.mean(axis=0) / dF.mean(axis=0)
    return point, sd


def series_wealth_gap(mdf, sdf, n):
    """|wealth_gap| on the mean, same convention plot_bars_run10.py and
    post_process_rule_policies.py already use (abs on the mean only, std
    left as the raw column -- not recomputed for the abs transform)."""
    return smooth(mdf["wealth_gap"].abs(), n), smooth(sdf["wealth_gap"], n)


def series_profit(mdf, sdf, n):
    return smooth(mdf["cumulative_profit"], n), smooth(sdf["cumulative_profit"], n)


def series_approval_disparity(mdf, sdf, n):
    mean = mdf["approval_rate_M_cumulative"] - mdf["approval_rate_F_cumulative"]
    std = np.sqrt(sdf["approval_rate_M_cumulative"] ** 2 + sdf["approval_rate_F_cumulative"] ** 2)
    return smooth(mean, n), smooth(std, n)


def series_social_welfare(mdf, sdf, n):
    """R_F(t), the FEMALE group's own long-term social welfare -- not the
    N-weighted R_bar average (dropped deliberately: R_bar lets an overtly
    male-preferring policy's high R_M mask a poor R_F behind a decent-
    looking weighted mean, exactly the failure mode this whole project is
    about; R_F alone can't be hidden that way)."""
    return smooth(mdf["R_F"], n), smooth(sdf["R_F"], n)


def series_inequality_ratio(mdf, sdf, n):
    d_M = mdf["mu_M_end"] - mdf["mu_M_start"].iloc[0]
    d_F = mdf["mu_F_end"] - mdf["mu_F_start"].iloc[0]
    rho = d_M / d_F.replace(0, np.nan)
    return smooth(rho, n), None   # no std band -- see module docstring


METRICS = [
    ("wealth_gap", "|Wealth Gap|", series_wealth_gap, 0.0),
    ("profit", "Cumulative Profit", series_profit, 0.0),
    ("approval_disparity", "Approval Rate Disparity", series_approval_disparity, 0.0),
    ("social_welfare", r"Female Welfare $R_F$", series_social_welfare, 0.0),
    ("inequality_ratio", r"Inequality Ratio $\rho(t)$", series_inequality_ratio, 1.0),
]


def plot_one(root, agent, constraint, stem_key, ylabel, series_fn, hline, out_dir, n_smooth,
             rho_band=True):
    # 4.4" tall, not 3.6" -- see plot_reach_rate_run10.py's comment on the
    # same fix (tight-bbox undercounting a rotated y-axis label's height,
    # margin here is thin even for the shorter labels since some of these
    # metrics also carry LaTeX like $\bar{R}$/$\rho(t)$).
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
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
        if stem_key == "inequality_ratio" and rho_band:
            point, bsd = bootstrap_rho_band(root, agent, reward, constraint)
            if bsd is not None:
                # the per-seed point estimate must be the plotted line
                assert np.allclose(point[len(point) // 2:], mu.values[len(point) // 2:len(point)],
                                   rtol=1e-6, equal_nan=True), f"rho mismatch {agent} {reward} {constraint}"
                sd = smooth(pd.Series(bsd), n_smooth)
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
    ap.add_argument("--metrics", default="",
                    help="comma-separated subset of metric stems to (re)draw, e.g. inequality_ratio")
    ap.add_argument("--no-rho-band", action="store_true",
                    help="draw rho(t) as a mean line only (the old behaviour)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_paper_style()

    wanted = {m for m in args.metrics.split(",") if m}
    for stem_key, ylabel, series_fn, hline in METRICS:
        if wanted and stem_key not in wanted:
            continue
        for agent, _ in AGENTS:
            for constraint, _ in CONSTRAINTS:
                plot_one(args.root, agent, constraint, stem_key, ylabel, series_fn,
                         hline, args.out, args.smooth, rho_band=not args.no_rho_band)

    handles = [Line2D([0], [0], color=REWARD_COLORS[r], lw=2.5) for r in GROUP_REWARDS]
    labels = [REWARD_LABELS[r] for r in GROUP_REWARDS]
    save_legend(handles, labels, args.out, "trajectories_reward")
    print("done.")


if __name__ == "__main__":
    main()
