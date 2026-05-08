#!/usr/bin/env python3

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Policy metadata
# ---------------------------------------------------------------------------

POLICY_NAMES = [
    "always_approve",
    "always_reject",
    "uniform_acceptance",
    "oracle",
    "pattern_prediction",
    "rich_becomes_richer",
    "reverse_rich_becomes_richer",
]

POLICY_COLORS = {
    "always_approve":              "#1f77b4",
    "always_reject":               "#d62728",
    "uniform_acceptance":          "#ff7f0e",
    "oracle":                      "#17becf",
    "pattern_prediction":          "#9467bd",
    "rich_becomes_richer":         "#8c564b",
    "reverse_rich_becomes_richer": "#e377c2",
}

POLICY_LABELS = {
    "always_approve":              "Always Approve",
    "always_reject":               "Always Reject",
    "uniform_acceptance":          "Uniform Acceptance",
    "oracle":                      "Oracle",
    "pattern_prediction":          "Pattern Prediction",
    "rich_becomes_richer":         "Rich Becomes Richer",
    "reverse_rich_becomes_richer": "Reverse Rich->Richer",
}

MAX_EPISODES = 500


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

def setup_plot_style():
    plt.rcParams.update({
        "font.family":       ["STIXGeneral"],
        "font.size":         16,
        "axes.labelsize":    18,
        "axes.titlesize":    19,
        "legend.fontsize":   14,
        "legend.framealpha": 0.9,
        "xtick.labelsize":   15,
        "ytick.labelsize":   15,
        "figure.dpi":        200,
        "savefig.dpi":       200,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "axes.axisbelow":    True,
    })


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def load_and_aggregate(checkpoint_dir, seeds, policy_name):
    """Load per-seed CSVs and return (mean_df, std_df, n_loaded)."""
    dfs = []
    for seed in seeds:
        path = os.path.join(checkpoint_dir, f"policy_{policy_name}_seed{seed}.csv")
        if not os.path.exists(path):
            print(f"    WARNING: missing {path}")
            continue
        try:
            dfs.append(pd.read_csv(path))
        except Exception as exc:
            print(f"    WARNING: could not read {path}: {exc}")

    if not dfs:
        return None, None, 0

    min_len = min(len(df) for df in dfs)
    dfs = [df.iloc[:min_len].reset_index(drop=True) for df in dfs]

    numeric_cols = dfs[0].select_dtypes(include=[np.number]).columns
    stacked = np.stack([df[numeric_cols].values for df in dfs])

    mean_df = dfs[0].copy()
    std_df  = dfs[0].copy()
    mean_df[numeric_cols] = stacked.mean(axis=0)
    std_df[numeric_cols]  = stacked.std(axis=0)

    # Cut off at MAX_EPISODES
    mask    = mean_df["episode"] <= MAX_EPISODES
    mean_df = mean_df[mask].reset_index(drop=True)
    std_df  = std_df[mask].reset_index(drop=True)

    return mean_df, std_df, len(dfs)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _band(ax, x, mean, std, color, lw=2.0, alpha=0.12):
    ax.plot(x, mean, color=color, lw=lw)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha)


def _single(aggregated, col_fn, ylabel, out_path, hline=None,
            ylim_bottom=None, ylim_top=None):
    """Render one single-panel PDF with all policies as mean +- std bands."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, (mdf, sdf) in aggregated.items():
        mean, std = col_fn(mdf, sdf)
        _band(ax, mdf["episode"], mean, std, POLICY_COLORS.get(name))

    if hline is not None:
        ax.axhline(hline, color="k", lw=0.9, ls="-")

    ax.set_xlabel("Episode (T)")
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    if ylim_top is not None:
        ax.set_ylim(top=ylim_top)
    plt.tight_layout()

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def generate_plots(aggregated, results_dir, timestamp):
    # Pre-compute R_bar for each policy
    agg_with_rbar = {}
    for name, (mdf, sdf) in aggregated.items():
        N_M     = mdf["total_applications_M"]
        N_F     = mdf["total_applications_F"]
        total   = (N_M + N_F).replace(0, np.nan)
        mdf     = mdf.copy()
        sdf     = sdf.copy()
        mdf["R_bar"] = (N_M * mdf["R_M"] + N_F * mdf["R_F"]) / total
        sdf["R_bar"] = np.sqrt((N_M * sdf["R_M"])**2 + (N_F * sdf["R_F"])**2) / total
        # rho_cumulative
        mu_M_0 = mdf["mu_M_start"].iloc[0]
        mu_F_0 = mdf["mu_F_start"].iloc[0]
        mdf["rho_cumulative"] = (mdf["mu_M_end"] - mu_M_0) / \
                                (mdf["mu_F_end"] - mu_F_0).replace(0, np.nan)
        sdf["rho_cumulative"] = sdf["rho_episode"] * 0  # zero std placeholder
        agg_with_rbar[name] = (mdf, sdf)

    p = lambda col: (lambda m, s: (m[col], s[col]))

    _single(agg_with_rbar,
            lambda m, s: (m["wealth_gap"].abs(), s["wealth_gap"]),
            "|Wealth-Gap|",
            os.path.join(results_dir, f"rule_wealth_gap_{timestamp}.pdf"),
            hline=0, ylim_bottom=0)

    _single(agg_with_rbar,
            p("cumulative_profit"),
            "Cumulative Profit",
            os.path.join(results_dir, f"rule_cumulative_profit_{timestamp}.pdf"),
            hline=0, ylim_bottom=-50000)

    _single(agg_with_rbar,
            p("rho_cumulative"),
            r"Inequality Ratio $\rho(t)$",
            os.path.join(results_dir, f"rule_inequality_ratio_{timestamp}.pdf"),
            hline=1, ylim_bottom=0.9)

    _single(agg_with_rbar,
            p("R_bar"),
            "Avg Long-term Social Welfare",
            os.path.join(results_dir, f"rule_social_welfare_{timestamp}.pdf"),
            hline=0, ylim_bottom=-0.001)



def main():
    parser = argparse.ArgumentParser(
        description="Aggregate rule-based policy results across seeds and generate plots."
    )
    parser.add_argument("--results-dir",    type=str, default="./rule_policy_results")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--seeds",          type=int, default=10)
    parser.add_argument("--seed-list",      type=int, nargs="+", default=None)
    parser.add_argument("--no-plots",       action="store_true")
    parser.add_argument("--log-scale",      action="store_true",
                        help="Apply symlog y-scale to all plots")
    args = parser.parse_args()

    seeds          = args.seed_list if args.seed_list is not None else list(range(args.seeds))
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.results_dir, "checkpoints")
    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Checkpoint dir : {checkpoint_dir}")
    print(f"Seeds          : {seeds}")
    print(f"Results dir    : {args.results_dir}")
    print(f"Max episodes   : {MAX_EPISODES}")
    print()

    aggregated = {}
    for name in POLICY_NAMES:
        mdf, sdf, n = load_and_aggregate(checkpoint_dir, seeds, name)
        if mdf is None:
            print(f"  SKIP {name} -- no data found")
            continue
        aggregated[name] = (mdf, sdf)
        mdf.to_csv(os.path.join(args.results_dir, f"mean_policy_{name}_{timestamp}.csv"), index=False)
        sdf.to_csv(os.path.join(args.results_dir, f"std_policy_{name}_{timestamp}.csv"),  index=False)
        print(f"  {name}: {len(mdf)} episodes, {n} seeds loaded")

    if not aggregated:
        print("No data found. Exiting.")
        return

    # Summary table
    print(f"\n{'=' * 82}")
    print(f"{'Policy':<32} {'R_M mean+-std':>18} {'R_F mean+-std':>18} {'profit ($k)':>12}")
    print(f"{'-' * 82}")
    for name, (mdf, sdf) in aggregated.items():
        R_M_m = mdf["R_M"].iloc[-1];               R_M_s = sdf["R_M"].iloc[-1]
        R_F_m = mdf["R_F"].iloc[-1];               R_F_s = sdf["R_F"].iloc[-1]
        pr_m  = mdf["cumulative_profit"].iloc[-1];  pr_s  = sdf["cumulative_profit"].iloc[-1]
        print(
            f"  {POLICY_LABELS.get(name, name):<30}  "
            f"{R_M_m:+.4f} +- {R_M_s:.4f}  "
            f"{R_F_m:+.4f} +- {R_F_s:.4f}  "
            f"{pr_m:>8.1f} +- {pr_s:.1f}"
        )
    print(f"{'=' * 82}")

    if not args.no_plots:
        print("\nGenerating plots...")
        generate_plots(aggregated, args.results_dir, timestamp)

    print("\nDone.")


if __name__ == "__main__":
    main()