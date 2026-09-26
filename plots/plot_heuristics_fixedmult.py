#!/usr/bin/env python3
"""
Heuristics figure of the appendix (rule-based policies vs PeRL), with the
PeRL overlay switched to the FIXED-multiplier FL / Equality-of-Outcome run
(lambda_187_snap: gamma* = 1.87, 10 seeds) so the figure agrees with the
PeRL / Outcome / FL row of the main-text table. The earlier version of this
figure overlaid the adaptive-multiplier run11 cell instead.

The seven baselines are NOT recomputed: their plotted series are read back
from the per-metric CSVs that plot_best_vs_baselines_run10.py saved with
the previous figure (plots/best_vs_baselines_run11/*.csv), since the raw
rule-policy runs live on the cluster. The PeRL series is computed from the
fixed-multiplier campaign's aggregated mean_/std_ CSVs with exactly the same
series functions, so both come out under one convention.

Output file names match the paper's iclr_2027_plots/full_heuristic_plots/:
wealth_gap, inequality_ratio, cumulative_profit, social_welfare (+ legend).
"""
import argparse, glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import set_paper_style, save_figure, save_legend, figure_set_bbox  # noqa: E402
from plot_best_vs_baselines_run10 import (  # noqa: E402
    POLICY_LABELS, POLICY_COLORS, POLICY_ORDER, BEST_COLOR,
    series_wealth_gap, series_profit, series_social_welfare, series_inequality_ratio,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BEST_LABEL = "PeRL"
# (paper file stem, baseline-CSV stem, y label, series fn, reference line)
PANELS = [
    ("wealth_gap",        "wealth_gap",       "|Wealth Gap|",              series_wealth_gap,       0.0),
    ("inequality_ratio",  "inequality_ratio", r"Inequality Ratio $\rho(t)$", series_inequality_ratio, 1.0),
    ("cumulative_profit", "profit",           "Cumulative Profit",         series_profit,           0.0),
    ("social_welfare",    "social_welfare",   r"Female Welfare $R_F$",     series_social_welfare,   0.0),
]


def load_fixed_multiplier(root):
    """Latest aggregate of the fixed-multiplier campaign, checked against the
    per-seed checkpoints (an earlier partial aggregate also sits in the dir)."""
    d = os.path.join(root, "pepg")
    m = sorted(glob.glob(os.path.join(d, "mean_fairness_lagrangian__social_*.csv")))[-1]
    s = m.replace("/mean_", "/std_")
    mdf, sdf = pd.read_csv(m), pd.read_csv(s)
    seeds = sorted(glob.glob(os.path.join(d, "checkpoints", "seed*_fairness_lagrangian__social.csv")))
    per_seed = np.mean([pd.read_csv(f).cumulative_profit.iloc[-1] for f in seeds])
    assert len(seeds) == 10 and np.isclose(mdf.cumulative_profit.iloc[-1], per_seed, rtol=1e-6), \
        f"aggregate {os.path.basename(m)} does not match the {len(seeds)} per-seed runs"
    print(f"  PeRL overlay: {os.path.basename(m)} ({len(seeds)} seeds, final profit {per_seed/1e6:.3f}M)")
    return mdf, sdf


def build(panel, baseline_dir, mdf, sdf):
    stem, bstem, ylabel, fn, hline = panel
    base = pd.read_csv(os.path.join(baseline_dir, f"best_vs_baselines_{bstem}.csv"))
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    out = {"episode": base["episode"]}
    mu, sd = fn(mdf, sdf, 1)
    ep = mdf["episode"]
    ax.plot(ep, mu, color=BEST_COLOR, zorder=10)
    if sd is not None:
        ax.fill_between(ep, mu - sd, mu + sd, color=BEST_COLOR, alpha=0.20, lw=0, zorder=9)
    out["perl_fixed_mean"] = mu.values
    if sd is not None:
        out["perl_fixed_std"] = sd.values
    for policy in POLICY_ORDER:
        bm = base[f"{policy}_mean"]
        ax.plot(base["episode"], bm, color=POLICY_COLORS[policy], lw=1.6, alpha=0.9)
        if f"{policy}_std" in base:
            bs = base[f"{policy}_std"]
            ax.fill_between(base["episode"], bm - bs, bm + bs, color=POLICY_COLORS[policy], alpha=0.08, lw=0)
        out[f"{policy}_mean"] = bm.values
    ax.axhline(hline, color="k", lw=1.0, ls="--", alpha=0.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    return fig, stem, pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-root", default=os.path.join(ROOT, "lambda_sweep_data", "lambda_187_snap"))
    ap.add_argument("--baselines", default=os.path.join(HERE, "best_vs_baselines_run11"))
    ap.add_argument("--out", default=os.path.join(HERE, "full_heuristic_plots_fixedmult"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_paper_style()
    mdf, sdf = load_fixed_multiplier(args.fixed_root)
    built = [build(p, args.baselines, mdf, sdf) for p in PANELS]
    bbox = figure_set_bbox([f for f, _, _ in built])
    for fig, stem, data in built:
        save_figure(fig, args.out, stem, data, bbox=bbox)
        print(f"  saved -> {os.path.join(args.out, stem + '.pdf')}")
    handles = [Line2D([0], [0], color=BEST_COLOR, lw=3.0)] + \
              [Line2D([0], [0], color=POLICY_COLORS[p], lw=2.0) for p in POLICY_ORDER]
    labels = [BEST_LABEL] + [POLICY_LABELS[p] for p in POLICY_ORDER]
    save_legend(handles, labels, args.out, "heuristics", ncol=len(labels))
    print("done.")


if __name__ == "__main__":
    main()
