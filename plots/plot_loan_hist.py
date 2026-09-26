#!/usr/bin/env python3
"""
Cumulative-loan distribution over individuals at the end of deployment, in
the style of the ICLR draft's Figures 15/16 ("Cumulative loan distributions"):
x = individual index after sorting, binned into --bins consecutive bins of
N/--bins individuals; y = total loans received by the individuals in that
bin, averaged over seeds. One PDF per (agent, constraint, reward, group),
red = male, blue = female, black dashed line = the uniform reference (what
every bin would receive if the group's loans were spread evenly). House
style from plots/paper_style.py: no titles, no inline legends, a standalone
group legend written once.

Sorting (--sort):
  loans   individuals sorted by their own cumulative loan count, descending
          (default). A rank plot: the steeper the fall from the left, the
          more the group's loans are concentrated on a few recipients; bars
          at zero on the right are individuals who never received a loan.
  wealth  individuals sorted by wealth at the first snapshot episode
          (--wealth-episode, default 100), descending, so the left of the
          axis is the group's richest members. Shows whether loans follow
          wealth. The run11 population.npz index itself is NOT wealth
          ordered (checked: corr(index, wealth) = 0.00), so an unsorted
          plot would be flat noise; some explicit sort is required.

Data: per-seed deploy population.npz (loan_counts_{M,F}_ep{N}, X_{male,
female}_ep{N}); --episode picks the snapshot (default 3000). A CSV with the
bin totals and a stats CSV (mean / median / p10 / p90 / zero share per
group) are written next to the figures. Y-limits are shared across every
figure drawn in one invocation so panels are comparable; pass --ymax to
force the value across invocations (e.g. run11 vs lambda_187_snap).
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import set_paper_style, save_figure, save_legend  # noqa: E402

AGENTS = ["pg", "pepg"]
CONSTRAINTS = ["social", "eo"]
REWARDS = ["rawlsian_maximin", "fairness_lagrangian", "social_welfare"]
GROUPS = [("M", "male", "Male", "#C0392B"), ("F", "female", "Female", "#2980B9")]


def load_seeds(root, agent, reward, constraint, episode, wealth_episode):
    """list over seeds of {group: (counts, wealth_at_wealth_episode)}."""
    paths = sorted(glob.glob(os.path.join(
        root, agent, "deploy_artifacts", f"{agent}_{reward}__{constraint}__seed*_population.npz")))
    out = []
    for p in paths:
        d = np.load(p)
        if f"loan_counts_M_ep{episode}" not in d.files:
            continue
        seed = {}
        for g, gname, _, _ in GROUPS:
            c = d[f"loan_counts_{g}_ep{episode}"].astype(np.float64)
            wk = f"X_{gname}_ep{wealth_episode}"
            w = d[wk].astype(np.float64) if wk in d.files else None
            seed[g] = (c, w)
        out.append(seed)
    return out


def binned_totals(counts, order, n_bins):
    """Sum of counts per consecutive bin of the sorted individuals."""
    s = counts[order]
    N = len(s)
    edges = np.linspace(0, N, n_bins + 1).astype(int)
    totals = np.array([s[edges[i]:edges[i + 1]].sum() for i in range(n_bins)])
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges) * 0.85
    return centers, totals, widths, edges


def combo_bins(seeds, g, n_bins, sort):
    """Per-seed binned totals (each seed sorted on its own), stacked."""
    rows, centers, widths, edges = [], None, None, None
    for s in seeds:
        c, w = s[g]
        if sort == "loans":
            order = np.argsort(-c, kind="stable")
        else:
            if w is None:
                raise SystemExit("--sort wealth needs X_{group}_ep{wealth_episode} in the npz")
            order = np.argsort(-w, kind="stable")
        centers, totals, widths, edges = binned_totals(c, order, n_bins)
        rows.append(totals)
    return centers, np.stack(rows), widths, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episode", type=int, default=3000)
    ap.add_argument("--wealth-episode", type=int, default=100)
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--sort", choices=["loans", "wealth"], default="loans")
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--agents", default=",".join(AGENTS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_paper_style()

    combos = {}
    for agent in args.agents.split(","):
        for constraint in CONSTRAINTS:
            for reward in REWARDS:
                seeds = load_seeds(args.root, agent, reward, constraint, args.episode, args.wealth_episode)
                if seeds:
                    combos[(agent, constraint, reward)] = seeds
    if not combos:
        print("no data found"); return

    # First pass: bins for every (combo, group), to fix a shared y-limit.
    prepared = {}
    for key, seeds in combos.items():
        for g, _, _, _ in GROUPS:
            prepared[key + (g,)] = combo_bins(seeds, g, args.bins, args.sort)
    ymax = args.ymax or 1.05 * max(v[1].mean(axis=0).max() for v in prepared.values())
    print(f"  sort={args.sort}, episode {args.episode}, {args.bins} bins, shared ymax {ymax:.0f}")

    stat_rows = []
    for (agent, constraint, reward, g), (centers, stack, widths, edges) in prepared.items():
        gname, g_label, color = next((n, l, c) for gg, n, l, c in GROUPS if gg == g)
        mean_tot = stack.mean(axis=0)
        N = edges[-1]
        uniform = mean_tot.sum() / args.bins

        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        ax.bar(centers, mean_tot, width=widths, color=color, alpha=0.8, edgecolor="none")
        ax.axhline(uniform, color="k", ls="--", lw=1.2)
        ax.set_xlim(0, N)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Individual index" + (" (by loans received)" if args.sort == "loans"
                                            else f" (by wealth, episode {args.wealth_episode})"))
        ax.set_ylabel("Total loans received")

        stem = f"loandist_{agent}_{constraint}_{reward}_{gname}"
        out = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:],
                            "loans_mean": mean_tot, "loans_std": stack.std(axis=0)})
        save_figure(fig, args.out, stem, out)
        print(f"  saved -> {os.path.join(args.out, stem + '.pdf')}")

        seeds = combos[(agent, constraint, reward)]
        allc = np.concatenate([s[g][0] for s in seeds])
        stat_rows.append({
            "agent": agent, "constraint": constraint, "reward": reward, "group": g_label,
            "n_seeds": len(seeds), "mean": allc.mean(), "median": np.median(allc),
            "p10": np.percentile(allc, 10), "p90": np.percentile(allc, 90),
            "zero_share": float(np.mean([np.mean(s[g][0] == 0) for s in seeds])),
            "top10pct_share": float(np.mean([np.sort(s[g][0])[::-1][:len(s[g][0]) // 10].sum() / max(s[g][0].sum(), 1)
                                             for s in seeds])),
        })

    handles = [Patch(facecolor=c, alpha=0.8) for _, _, _, c in GROUPS] + [Line2D([0], [0], color="k", ls="--", lw=1.5)]
    labels = [l for _, _, l, _ in GROUPS] + ["Uniform"]
    save_legend(handles, labels, args.out, "loandist_group")

    sdf = pd.DataFrame(stat_rows)
    sdf.to_csv(os.path.join(args.out, "loandist_stats.csv"), index=False)
    print(sdf.round(3).to_string(index=False))
    print("done.")


if __name__ == "__main__":
    main()
