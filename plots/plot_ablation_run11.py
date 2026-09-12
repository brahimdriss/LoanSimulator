#!/usr/bin/env python3
"""
Robustness-check plots for the three ablations (loan_simulator/ablation.py,
cluster/ablation.sub): performative strength, initial wealth-gap scale,
and PERL's replay buffer capacity. FL/social only (the one combo with a
genuinely live lambda), 4 seeds per ablated setting; the 1x / buffer=50
point at every axis is run11's FL/social result (10 seeds) and is treated
as the BASELINE these bars are measured against, not re-run.

Encoding, chosen over a 3-point dot-and-whisker line after that version
looked misleading -- three discrete multiplicative settings connected by a
line implies a continuous trend that was never tested, and each metric's
own huge absolute scale (wealth gap in the thousands, profit in the
hundreds of thousands) meant the baseline's seed-to-seed noise dwarfed any
real shift, making every panel just look like three overlapping error
bars:

  * y-axis is PERCENT CHANGE FROM THE 1x/buffer=50 BASELINE, not the
    metric's absolute value. This is the actual question a robustness
    check asks ("did perturbing this knob move the result much?") and it
    strips out each metric's absolute scale so the y-range only ever
    spans the size of the shift.
  * the baseline point itself is DROPPED (it is 0% by construction) --
    only the two perturbed settings appear, as separate bars, not a line.
    Bars read as independent measurements; a connecting line would still
    imply an interpolation between three points that was never measured.
  * std is propagated through the ratio (mv/mb - 1) via the standard
    first-order formula for two independent means,
    SE(pct) = |mv/mb| * sqrt((sv/mv)^2 + (sb/mb)^2), sv/sb/mv/mb all from
    the same per-seed final-episode values as before.

Metrics (final-episode, per seed, then averaged -- NOT recomputed from the
aggregated mean_/std_ CSVs, since seed-level values are what a std band
across seeds means): |wealth_gap|, approval-rate disparity
|appM_cumulative - appF_cumulative|, cumulative_profit, and the inequality
ratio rho = (mu_M_end - mu_M_start[ep1]) / (mu_F_end - mu_F_start[ep1]) --
same definitions as plots/plot_pg_vs_sac_run11.py and the run11 tables.
NOTE on rho specifically: it is itself a ratio centred near 1, so "percent
change in rho" is a ratio-of-a-ratio -- read it as "how much did the
already-somewhat-arbitrary rho number move," not as a physically
meaningful percentage the way profit's is.

Reads loan_simulator/ablation.py's per-seed checkpoint CSVs directly
(ablation_data/<campaign>/<agent>/checkpoints/seed*_fairness_lagrangian__
social.csv and run11_data/<agent>/checkpoints/... for the baseline) -- not
the aggregated mean_/std_ files, which only carry the cross-seed mean/std
and can't be re-aggregated onto a different axis without the raw seeds.
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import set_paper_style, save_figure, save_legend  # noqa: E402

RL_COLOR = "#0072B2"
PERL_COLOR = "#9467bd"
AGENT_COLOR = {"pg": RL_COLOR, "pepg": PERL_COLOR}
AGENT_LABEL = {"pg": "RL", "pepg": "PERL"}

METRICS = [
    ("wealth_gap", "|Wealth Gap|"),
    ("disparity", "Approval Rate Disparity"),
    ("profit", "Cumulative Profit"),
    ("rho", r"Inequality Ratio $\rho$"),
]


def seed_finals(root, agent):
    files = sorted(glob.glob(os.path.join(
        root, agent, "checkpoints", "seed*_fairness_lagrangian__social.csv")))
    rows = []
    for f in files:
        df = pd.read_csv(f)
        first, last = df.iloc[0], df.iloc[-1]
        d_M = last["mu_M_end"] - first["mu_M_start"]
        d_F = last["mu_F_end"] - first["mu_F_start"]
        rows.append(dict(
            wealth_gap=abs(last["wealth_gap"]),
            disparity=abs(last["approval_rate_M_cumulative"] - last["approval_rate_F_cumulative"]),
            profit=last["cumulative_profit"],
            rho=d_M / d_F if d_F != 0 else np.nan,
        ))
    if not rows:
        return None
    return pd.DataFrame(rows)


def mean_std(root, agent):
    df = seed_finals(root, agent)
    if df is None:
        return None
    return {m: (df[m].mean(), df[m].std()) for m, _ in METRICS}


def pct_change(root_val, root_base, agent, metric_key):
    """(mean_val/mean_base - 1) * 100, with std propagated through the
    ratio of two independent means. NaN if either side is missing or the
    baseline mean is ~0 (would blow up the ratio; doesn't occur for any
    metric here -- see the module's baseline sanity check)."""
    v, b = mean_std(root_val, agent), mean_std(root_base, agent)
    if v is None or b is None:
        return np.nan, np.nan
    mv, sv = v[metric_key]
    mb, sb = b[metric_key]
    if mb == 0 or np.isclose(mb, 0):
        return np.nan, np.nan
    pct = (mv / mb - 1) * 100
    rel_err = np.sqrt((sv / mv) ** 2 + (sb / mb) ** 2) if mv != 0 else np.nan
    pct_std = abs(mv / mb) * rel_err * 100
    return pct, pct_std


# Each axis: knob value -> data root. The baseline knob (None) always
# resolves to run11_data and is the denominator every bar is measured
# against; it does not get its own bar.
AXES = {
    "performative": {
        "baseline": 1.0,
        "knobs": {0.5: "ablation_perf05", 2.0: "ablation_perf2"},
        "agents": ["pg", "pepg"],
        "xlabel": "Performative Scale",
        "tick_fmt": lambda k: f"{k:g}x",
    },
    "wealthgap": {
        "baseline": 1.0,
        "knobs": {0.5: "ablation_gap05", 2.0: "ablation_gap2"},
        "agents": ["pg", "pepg"],
        "xlabel": "Initial Wealth-Gap Scale",
        "tick_fmt": lambda k: f"{k:g}x",
    },
    "buffer": {
        "baseline": 50,
        "knobs": {10: "ablation_buf10", 200: "ablation_buf200"},
        "agents": ["pepg"],
        "xlabel": "PERL Buffer Capacity (episodes)",
        "tick_fmt": lambda k: f"{int(k)}",
    },
}


def plot_axis_metric(axis_key, axis, metric_key, ylabel, ablation_root, run11_root, out_dir):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    knobs = list(axis["knobs"].keys())
    agents = axis["agents"]
    width = 0.35 if len(agents) == 2 else 0.5
    x = np.arange(len(knobs))
    out = {"knob": knobs}
    any_data = False

    for i, agent in enumerate(agents):
        pcts, errs = [], []
        for k in knobs:
            root = os.path.join(ablation_root, axis["knobs"][k])
            pct, err = pct_change(root, run11_root, agent, metric_key)
            pcts.append(pct); errs.append(err)
        pcts, errs = np.array(pcts), np.array(errs)
        if np.all(np.isnan(pcts)):
            continue
        any_data = True
        offset = (i - (len(agents) - 1) / 2) * width
        ax.bar(x + offset, pcts, width, yerr=errs, color=AGENT_COLOR[agent],
                capsize=4, error_kw=dict(lw=1.5, ecolor="black", alpha=0.7))
        out[f"{agent}_pct"] = pcts
        out[f"{agent}_pct_std"] = errs

    ax.axhline(0, color="k", lw=1.2, ls="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([axis["tick_fmt"](k) for k in knobs])
    ax.set_xlabel(axis["xlabel"])
    ax.set_ylabel(f"% Change in {ylabel}\nfrom Baseline")
    ax.margins(x=0.3)

    stem = f"ablation_{axis_key}_{metric_key}_pct"
    save_figure(fig, out_dir, stem, pd.DataFrame(out) if any_data else None)
    print(f"  saved -> {os.path.join(out_dir, stem + '.pdf')}" + ("" if any_data else "   (no data found)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-root", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ablation_data"))
    ap.add_argument("--run11-root", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run11_data"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "ablation_run11"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_paper_style()

    for axis_key, axis in AXES.items():
        for metric_key, ylabel in METRICS:
            plot_axis_metric(axis_key, axis, metric_key, ylabel,
                              args.ablation_root, args.run11_root, args.out)

    handles = [Patch(facecolor=RL_COLOR), Patch(facecolor=PERL_COLOR)]
    labels = ["RL", "PERL"]
    save_legend(handles, labels, args.out, "ablation_agent")
    print("done.")


if __name__ == "__main__":
    main()
