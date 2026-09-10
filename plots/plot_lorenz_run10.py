#!/usr/bin/env python3
"""
Lorenz curves of cumulative loan concentration, RL (pg) vs PERL (pepg),
under social (Equality of Outcome) and eo (Equality of Opportunity)
fairness, at multiple deploy-episode checkpoints. Paper figure, in the
style of Eutopia_2026.pdf Fig. 4 / Fig. 17: one figure per (agent,
constraint), one panel per reward function (RMM, FL, SW), red = male,
blue = female, curves light -> dark with episode, dashed diagonal =
perfect equality.

Data source: the per-seed deploy population.npz files written by
pg_adapt.py / pepg_adapt.py. With --population-snapshot-episodes set at
deploy time (cluster/lorenz.sub, campaign run10_lorenz) each file carries
loan_counts_{M,F}_ep{N} for every requested episode. Without it (the
original run10 files) only the final-episode loan_counts_{M,F} exist, and
the script falls back to a single-checkpoint curve labelled "final".

Lorenz curve per (seed, group, episode): sort individuals by cumulative
loan count ascending, cumulative population fraction vs cumulative loan
fraction (same construction as individual_loan_frequency.compute_lorenz).
Averaged across seeds on a shared population-fraction grid. A Gini table
(1 - 2 * area under the Lorenz curve) is written alongside the figures.
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"

AGENTS = [("pg", "RL"), ("pepg", "PERL")]
CONSTRAINTS = [("social", "Equality of Outcome"), ("eo", "Equality of Opportunity")]
REWARDS = [("rawlsian_maximin", "RMM"), ("fairness_lagrangian", "FL"), ("social_welfare", "SW")]
GROUPS = [("M", "Male", cm.Reds), ("F", "Female", cm.Blues)]
N_GRID = 200


def lorenz(counts, n_points=N_GRID):
    """(cum_pop_fraction, cum_loan_fraction), each length n_points+1 with a
    (0,0) origin. Zero-total counts give the flat zero curve."""
    s = np.sort(np.asarray(counts, dtype=np.float64))
    cum = np.cumsum(s)
    n = len(s)
    idx = np.linspace(0, n - 1, n_points, dtype=int)
    pop = np.concatenate([[0.0], (idx + 1) / n])
    loan = np.concatenate([[0.0], cum[idx] / cum[-1]]) if cum[-1] > 0 else np.zeros(n_points + 1)
    return pop, loan


def gini(pop, loan):
    trap = getattr(np, "trapezoid", None) or np.trapz
    return 1.0 - 2.0 * trap(loan, pop)


def snapshot_episodes(npz):
    """Episodes with loan_counts_M_ep{N} keys, ascending. Empty if the file
    predates snapshotting (run10 originals)."""
    eps = [int(m.group(1)) for k in npz.files
           for m in [re.match(r"loan_counts_M_ep(\d+)$", k)] if m]
    return sorted(eps)


def load_curves(root, agent, reward, constraint):
    """dict episode -> {group: (pop, mean_loan_curve, n_seeds)}."""
    paths = sorted(glob.glob(os.path.join(
        root, agent, "deploy_artifacts", f"{agent}_{reward}__{constraint}__seed*_population.npz")))
    if not paths:
        return {}
    per_ep = {}
    for p in paths:
        d = np.load(p)
        eps = snapshot_episodes(d)
        keyed = {ep: (f"loan_counts_M_ep{ep}", f"loan_counts_F_ep{ep}") for ep in eps}
        if not keyed:
            keyed = {"final": ("loan_counts_M", "loan_counts_F")}
        for ep, (kM, kF) in keyed.items():
            for g, k in (("M", kM), ("F", kF)):
                pop, loan = lorenz(d[k])
                per_ep.setdefault(ep, {}).setdefault(g, []).append((pop, loan))
    out = {}
    for ep, groups in per_ep.items():
        out[ep] = {}
        for g, curves in groups.items():
            pop = curves[0][0]
            loans = np.stack([c[1] for c in curves])
            out[ep][g] = (pop, loans.mean(axis=0), len(curves))
    return out


def episode_sort_key(ep):
    return (1, 0) if ep == "final" else (0, ep)


def plot_agent_constraint(root, agent, agent_label, constraint, constraint_label,
                          out_dir, gini_rows):
    fig, axes = plt.subplots(1, len(REWARDS), figsize=(3.2 * len(REWARDS), 3.2),
                             sharex=True, sharey=True)
    any_data = False
    for ax, (reward, reward_label) in zip(axes, REWARDS):
        curves = load_curves(root, agent, reward, constraint)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(reward_label, fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, lw=0.4, color="gray", alpha=0.35)
        if not curves:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=9, color="gray")
            continue
        any_data = True
        eps = sorted(curves, key=episode_sort_key)
        shades = np.linspace(0.35, 0.95, len(eps))
        for g, g_label, cmap in GROUPS:
            for shade, ep in zip(shades, eps):
                if g not in curves[ep]:
                    continue
                pop, loan, n = curves[ep][g]
                ax.plot(pop, loan, color=cmap(shade), lw=1.4)
                gini_rows.append({
                    "agent": agent_label, "constraint": constraint, "reward": reward_label,
                    "group": g_label, "episode": ep, "gini": gini(pop, loan), "n_seeds": n,
                })
    axes[0].set_ylabel("Cumulative loan fraction", fontsize=10)
    for ax in axes:
        ax.set_xlabel("Cumulative population fraction", fontsize=10)
    fig.suptitle(f"{agent_label}, {constraint_label}", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(out_dir, f"lorenz_{agent}_{constraint}.pdf")
    fig.savefig(out, format="pdf", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  saved -> {out}" + ("" if any_data else "   (no data found)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/run10_lorenz"),
                    help="dir containing {pg,pepg}/deploy_artifacts/*_population.npz")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "lorenz_run10"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gini_rows = []
    for agent, agent_label in AGENTS:
        for constraint, constraint_label in CONSTRAINTS:
            plot_agent_constraint(args.root, agent, agent_label, constraint, constraint_label,
                                  args.out, gini_rows)
    if gini_rows:
        gdf = pd.DataFrame(gini_rows)
        gpath = os.path.join(args.out, "gini_table.csv")
        gdf.to_csv(gpath, index=False)
        print(f"  saved -> {gpath}")
        print(gdf.pivot_table(index=["agent", "constraint", "reward", "group"],
                              columns="episode", values="gini").round(3).to_string())
    print("done.")


if __name__ == "__main__":
    main()
