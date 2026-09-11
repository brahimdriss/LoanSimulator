#!/usr/bin/env python3
"""
Best learned combo vs. the 7 non-learning rule-based baselines, overlaid
on one figure per metric: wealth gap, cumulative profit, approval-rate
disparity, long-term social welfare R_bar, and inequality ratio rho(t),
all over deploy episodes.

The "best combo" is fixed as PePG / fairness_lagrangian / social
(Equality of Outcome) -- picked by inspecting the final-episode radar/bars
numbers: it has the smallest wealth gap and highest approval-rate
"fairness" of any combo while ALSO posting the second-highest profit in
the whole campaign, a combination no other reward/constraint pairing
matches. Change BEST_AGENT/BEST_REWARD/BEST_CONSTRAINT below to compare a
different combo instead.

Colour: the best combo is drawn in the SAME purple as fairness_lagrangian
in plot_radar_run10.py's REWARD_COLORS (imported, not duplicated) --
purple is reserved for it alone. The 7 baselines use a colourblind-safe
Okabe-Ito-derived palette with the reddish-purple entry swapped for a
neutral grey specifically so nothing collides with that reserved purple.

Data: the best combo's mean_*.csv/std_*.csv under <root>/<agent>/, and
the baselines' mean_{policy}__baseline_*.csv/std_..._*.csv under
<root>/rule/ (written by test_rule_based_policies.py --aggregate; run
that first if <root>/rule has no mean_*.csv yet). House style throughout
(plots/paper_style.py): no titles/subplot grids, one shared standalone
legend, larger legible fonts.

Caveats carried over from plot_trajectories_run10.py (same formulas):
approval disparity's std band combines the two groups' stds in
quadrature (an approximation); rho(t) has no std band (needs raw
per-seed data to compute correctly, not just aggregated mean/std).
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
from plot_radar_run10 import REWARD_COLORS  # noqa: E402

BEST_AGENT, BEST_REWARD, BEST_CONSTRAINT = "pepg", "fairness_lagrangian", "social"
BEST_COLOR = REWARD_COLORS[BEST_REWARD]   # "#9467bd" -- reserved for this combo only
BEST_LABEL = "PERL, Fairness Lagrangian (Equality of Outcome)"

POLICY_LABELS = {
    "always_approve":             "Always Approve",
    "always_reject":              "Always Reject",
    "uniform_acceptance":         "Uniform Acceptance",
    "oracle":                     "Oracle",
    "pattern_prediction":         "Pattern Prediction",
    "rich_becomes_richer":        "Rich Becomes Richer",
    "reverse_rich_becomes_richer": "Reverse Rich->Richer",
}
# Okabe-Ito colourblind-safe set, reddish-purple (#CC79A7) swapped for a
# neutral grey so nothing competes with BEST_COLOR's purple.
POLICY_COLORS = {
    "always_approve":              "#0072B2",  # blue
    "always_reject":               "#D55E00",  # vermillion
    "uniform_acceptance":          "#E69F00",  # orange
    "oracle":                      "#009E73",  # bluish green
    "pattern_prediction":          "#B8860B",  # muted gold
    "rich_becomes_richer":         "#56B4E9",  # sky blue
    "reverse_rich_becomes_richer": "#7F7F7F",  # grey
}
POLICY_ORDER = list(POLICY_LABELS)


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
    N_M, N_F = mdf["total_applications_M"], mdf["total_applications_F"]
    total = (N_M + N_F).replace(0, np.nan)
    r_bar = (N_M * mdf["R_M"] + N_F * mdf["R_F"]) / total
    r_bar_std = np.sqrt((N_M * sdf["R_M"]) ** 2 + (N_F * sdf["R_F"]) ** 2) / total
    return smooth(r_bar, n), smooth(r_bar_std, n)


def series_inequality_ratio(mdf, sdf, n):
    d_M = mdf["mu_M_end"] - mdf["mu_M_start"].iloc[0]
    d_F = mdf["mu_F_end"] - mdf["mu_F_start"].iloc[0]
    rho = d_M / d_F.replace(0, np.nan)
    return smooth(rho, n), None


METRICS = [
    ("wealth_gap", "Wealth Gap", series_wealth_gap, 0.0),
    ("profit", "Cumulative Profit", series_profit, 0.0),
    ("approval_disparity", "Approval Rate Disparity", series_approval_disparity, 0.0),
    ("social_welfare", r"Long-Term Social Welfare $\bar{R}$", series_social_welfare, 0.0),
    ("inequality_ratio", r"Inequality Ratio $\rho(t)$", series_inequality_ratio, 1.0),
]


def plot_one(root, stem_key, ylabel, series_fn, hline, out_dir, n_smooth):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    any_data = False
    out = {}

    mean_df, std_df = load_mean_std(root, BEST_AGENT, BEST_REWARD, BEST_CONSTRAINT)
    if mean_df is not None:
        any_data = True
        ep = mean_df["episode"]
        mu, sd = series_fn(mean_df, std_df, n_smooth)
        ax.plot(ep, mu, color=BEST_COLOR, zorder=10)
        if sd is not None:
            ax.fill_between(ep, mu - sd, mu + sd, color=BEST_COLOR, alpha=0.20, lw=0, zorder=9)
        out["episode"] = ep
        out["best_mean"] = mu
        if sd is not None:
            out["best_std"] = sd

    for policy in POLICY_ORDER:
        mean_df, std_df = load_mean_std(root, "rule", policy, "baseline")
        if mean_df is None:
            continue
        any_data = True
        color = POLICY_COLORS[policy]
        ep = mean_df["episode"]
        mu, sd = series_fn(mean_df, std_df, n_smooth)
        ax.plot(ep, mu, color=color, lw=1.6, alpha=0.9)
        if sd is not None:
            ax.fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.08, lw=0)
        if "episode" not in out:
            out["episode"] = ep
        out[f"{policy}_mean"] = mu
        if sd is not None:
            out[f"{policy}_std"] = sd

    if any_data:
        ax.axhline(hline, color="k", lw=1.0, ls="--", alpha=0.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)

    stem = f"best_vs_baselines_{stem_key}"
    save_figure(fig, out_dir, stem, pd.DataFrame(out) if out else None)
    print(f"  saved -> {os.path.join(out_dir, stem + '.pdf')}" + ("" if any_data else "   (no data found)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="dir containing pepg/mean_*.csv (or whichever BEST_AGENT) and "
                         "rule/mean_{policy}__baseline_*.csv (run test_rule_based_policies.py "
                         "--aggregate first if the latter don't exist yet)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "best_vs_baselines_run10"))
    ap.add_argument("--smooth", type=int, default=1,
                    help="centred rolling-mean window in episodes (1 = raw)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_paper_style()

    for stem_key, ylabel, series_fn, hline in METRICS:
        plot_one(args.root, stem_key, ylabel, series_fn, hline, args.out, args.smooth)

    handles = [Line2D([0], [0], color=BEST_COLOR, lw=3.0)]
    labels = [BEST_LABEL]
    for policy in POLICY_ORDER:
        handles.append(Line2D([0], [0], color=POLICY_COLORS[policy], lw=2.0))
        labels.append(POLICY_LABELS[policy])
    save_legend(handles, labels, args.out, "best_vs_baselines", ncol=2)
    print("done.")


if __name__ == "__main__":
    main()
