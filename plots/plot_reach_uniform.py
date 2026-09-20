#!/usr/bin/env python3
"""
Reach rate of a uniformly random lender (Uniform Acceptance, a_i ~ U(0,1),
so ~50% of applicants approved regardless of group, wealth or score) over
the agents' 3,000-episode deployment horizon, one line per group, mean over
seeds with a +/-1 std band. Companion panel: applications per episode, the
quantity that actually moves.

The point of the figure: the policy is group-blind and constant, so any
separation between the two reach curves is produced entirely by the
population's response to the lending, not by the lender. Reach rate is
approval rate x (applications / N_g), the approval rate is fixed at ~0.5,
and applications follow the Hawkes base intensity
    f(mu_g) = (N_g/N_ref) * max(0.5, 2(1 + C (mu_g/mubar - 1))),  C = 2,
which rises with a group's relative wealth and clips at a floor once the
group falls far enough behind. The dashed guides mark the two structural
limits this imposes: mu_g/mubar <= 2 caps the richer group's intensity at
24 (reach ~0.10 at 50% approval), and the max(0.5, .) clip floors the
poorer group's at 2 (reach ~0.008).

Reads the per-seed deploy CSVs written by test_rule_based_policies.py
(rule_{policy}__baseline__seed{S}_episodes.csv), not the aggregated
mean_/std_ files, so it works straight after the runs finish.
"""

import argparse, glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import set_paper_style, save_figure, save_legend  # noqa: E402

GROUPS = [("M", "Male", "#C0392B"), ("F", "Female", "#2980B9")]
N_G = 12000
N_REF = 3000
C_MATTHEW = 2.0


def f_base(ratio):
    return (N_G / N_REF) * max(0.5, 2 * (1 + C_MATTHEW * (ratio - 1)))


def load(root, policy):
    fs = sorted(glob.glob(os.path.join(
        root, "rule", "deploy_artifacts", f"rule_{policy}__baseline__seed*_episodes.csv")))
    if not fs:
        raise SystemExit(f"no per-seed episodes CSVs under {root}/rule/deploy_artifacts")
    return [pd.read_csv(f) for f in fs], fs


def band(ax, ep, mat, color, smooth):
    m = pd.Series(mat.mean(0)).rolling(smooth, center=True, min_periods=1).mean()
    s = pd.Series(mat.std(0)).rolling(smooth, center=True, min_periods=1).mean()
    ax.plot(ep, m, color=color, lw=1.8)
    ax.fill_between(ep, m - s, m + s, color=color, alpha=0.18, lw=0)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "ua_data"))
    ap.add_argument("--policy", default="uniform_acceptance")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "reach_uniform"))
    ap.add_argument("--smooth", type=int, default=25)
    ap.add_argument("--guides", action="store_true", help="draw the structural ceiling/floor")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_paper_style()

    dfs, fs = load(args.root, args.policy)
    n = min(len(d) for d in dfs)
    ep = dfs[0]["episode"].values[:n]
    print(f"  {len(dfs)} seeds, {n} episodes: {', '.join(os.path.basename(f).split('seed')[1][0] for f in fs)}")

    out = {"episode": ep}
    for key, ylabel, stem, cols in [
        ("reach", "Reach rate (unique recipients / N)", "reach_uniform",
         {"M": "reach_rate_M", "F": "reach_rate_F"}),
        ("apps", "Applications per episode", "applications_uniform",
         {"M": "applications_M_episode", "F": "applications_F_episode"}),
    ]:
        fig, ax = plt.subplots(figsize=(4.8, 4.4))
        for g, _, color in GROUPS:
            mat = np.stack([d[cols[g]].values[:n] for d in dfs])
            m = band(ax, ep, mat, color, args.smooth)
            out[f"{key}_{g}_mean"] = m.values
            out[f"{key}_{g}_std"] = mat.std(0)
        if args.guides and key == "reach":
            # Only the FLOOR is a clean structural prediction: once a group's
            # relative wealth drops below 0.625 the max(0.5, .) clip pins its
            # base intensity at 2 per unit time, i.e. ~200 applications and
            # ~0.008 reach at 50% approval, whatever happens afterwards. The
            # corresponding base-rate ceiling for the richer group (mu/mubar
            # -> 2, intensity 24, reach ~0.10) is NOT a bound on the observed
            # curve, because the Hawkes excitation term adds intensity on top
            # of the base rate and the red group ends at 0.127; so it is not
            # drawn.
            ax.axhline(0.5 * f_base(0.0) * 100 / N_G, color="0.45", ls=":", lw=1.1)
        ax.set_xlim(ep[0], ep[-1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        save_figure(fig, args.out, stem, None)
        print(f"  saved -> {os.path.join(args.out, stem + '.pdf')}")

    pd.DataFrame(out).to_csv(os.path.join(args.out, "reach_uniform.csv"), index=False)
    handles = [Line2D([0], [0], color=c, lw=2.5) for _, _, c in GROUPS]
    save_legend(handles, [l for _, l, _ in GROUPS], args.out, "reach_uniform_group")

    # end-state summary
    print("\n  last 100 episodes (mean over seeds):")
    for g, lbl, _ in GROUPS:
        r = np.mean([d[f"reach_rate_{g}"].values[n-100:n].mean() for d in dfs])
        a = np.mean([d[f"applications_{g}_episode"].values[n-100:n].mean() for d in dfs])
        print(f"    {lbl:7s} reach {r:.4f}   applications/ep {a:7.0f}")
    print("  first 100 episodes:")
    for g, lbl, _ in GROUPS:
        r = np.mean([d[f"reach_rate_{g}"].values[:100].mean() for d in dfs])
        a = np.mean([d[f"applications_{g}_episode"].values[:100].mean() for d in dfs])
        print(f"    {lbl:7s} reach {r:.4f}   applications/ep {a:7.0f}")


if __name__ == "__main__":
    main()
