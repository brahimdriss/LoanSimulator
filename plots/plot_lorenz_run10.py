#!/usr/bin/env python3
"""
Lorenz curves of cumulative loan concentration, RL (pg) vs PERL (pepg),
under social (Equality of Outcome) and eo (Equality of Opportunity)
fairness, at multiple deploy-episode checkpoints. Paper figures, in the
style of Eutopia_2026.pdf Fig. 4 / Fig. 17: red = male, blue = female,
curves light -> dark with episode, dashed diagonal = perfect equality.

One PDF per (agent, constraint, reward) combo -- no subplot grids, no
titles (house style, see plots/paper_style.py); combine combos side by
side in the LaTeX source instead. Two standalone legends are written once
and reused across every combo: which colour is which group, and what
episode each shade corresponds to.

Data source: the per-seed deploy population.npz files written by
pg_adapt.py / pepg_adapt.py. With --population-snapshot-episodes set at
deploy time (cluster/lorenz.sub / full_fix.sub) each file carries
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
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import set_paper_style, save_figure, save_legend  # noqa: E402

AGENTS = [("pg", "RL"), ("pepg", "PERL")]
CONSTRAINTS = [("social", "Equality of Outcome"), ("eo", "Equality of Opportunity")]
REWARDS = [("rawlsian_maximin", "RMM"), ("fairness_lagrangian", "FL"), ("social_welfare", "SW")]
GROUPS = [("M", "Male", cm.Reds), ("F", "Female", cm.Blues)]
GROUP_SOLID = {"M": "#C0392B", "F": "#2980B9"}  # fixed swatch colour for the legend
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


def plot_one(root, agent, constraint, reward, reward_label, out_dir, gini_rows):
    """One combo -> one standalone Lorenz-curve figure. Returns the sorted
    episode list actually drawn (for the shared shade legend)."""
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    curves = load_curves(root, agent, reward, constraint)
    eps = sorted(curves, key=episode_sort_key)
    out = {}
    if curves:
        shades = np.linspace(0.35, 0.95, len(eps))
        for g, g_label, cmap in GROUPS:
            for shade, ep in zip(shades, eps):
                if g not in curves[ep]:
                    continue
                pop, loan, n = curves[ep][g]
                ax.plot(pop, loan, color=cmap(shade), lw=2.0)
                out[f"pop"] = pop
                out[f"{g}_loan_ep{ep}"] = loan
                gini_rows.append({
                    "agent": agent, "constraint": constraint, "reward": reward,
                    "group": g_label, "episode": ep, "gini": gini(pop, loan), "n_seeds": n,
                })

    ax.set_xlabel("Cumulative population fraction")
    ax.set_ylabel("Cumulative loan fraction")
    stem = f"lorenz_{agent}_{constraint}_{reward}"
    save_figure(fig, out_dir, stem, pd.DataFrame(out) if out else None)
    print(f"  saved -> {os.path.join(out_dir, stem + '.pdf')}" + ("" if curves else "   (no data found)"))
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/run10_lorenz"),
                    help="dir containing {pg,pepg}/deploy_artifacts/*_population.npz")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "lorenz_run10"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_paper_style()

    gini_rows = []
    all_eps = set()
    for agent, _ in AGENTS:
        for constraint, _ in CONSTRAINTS:
            for reward, reward_label in REWARDS:
                eps = plot_one(args.root, agent, constraint, reward, reward_label, args.out, gini_rows)
                all_eps.update(e for e in eps if e != "final")

    # Shared standalone legends, written once.
    group_handles = [Line2D([0], [0], color=GROUP_SOLID[g], lw=2.5) for g, _, _ in GROUPS]
    group_labels = [lbl for _, lbl, _ in GROUPS]
    save_legend(group_handles, group_labels, args.out, "lorenz_group")

    if all_eps:
        eps_sorted = sorted(all_eps)
        shades = np.linspace(0.35, 0.95, len(eps_sorted))
        ep_handles = [Line2D([0], [0], color=cm.Greys(s), lw=2.5) for s in shades]
        ep_labels = [f"ep {e}" for e in eps_sorted]
        save_legend(ep_handles, ep_labels, args.out, "lorenz_episode_shade",
                    ncol=min(len(ep_labels), 7))

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
