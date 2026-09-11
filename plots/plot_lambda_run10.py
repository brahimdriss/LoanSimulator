#!/usr/bin/env python3
"""
Deploy-phase lambda_wealth trajectory, one PDF per (reward, constraint)
combo, mean +/- std over seeds. Read straight from the per-seed
*_training_trace.csv files under <root>/<agent>/deploy_artifacts/ (written
by pg_adapt.py / pepg_adapt.py -- columns episode, episode_reward,
lambda_wealth, lambda_approval), not the aggregated mean_/std_ CSVs, since
those don't carry the training trace.

lambda_approval is NOT plotted: every combo in this campaign uses
constraint_type in {"social", "eo"}, which only ever updates
lambda_wealth (see loan_simulator/agent.py / pepg/agent.py
_update_lambdas) -- lambda_approval sits frozen at its unused init value
the whole run.

House style (plots/paper_style.py): no titles/subplot grids, larger
legible fonts, all combos for one agent saved at one shared bbox (see
paper_style.figure_set_bbox) so the files come out at identical page
dimensions if placed side by side in LaTeX.
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import set_paper_style, save_figure, figure_set_bbox  # noqa: E402

CONSTRAINTS = [("social", "Equality of Outcome"), ("eo", "Equality of Opportunity")]
REWARDS = [("rawlsian_maximin", "RMM"), ("fairness_lagrangian", "FL"), ("social_welfare", "SW")]
LINE_COLOR = "#4c4c4c"


def load_lambda_traces(root, agent, reward, constraint):
    """[n_seeds, n_episodes] array of lambda_wealth, seeds truncated to the
    shortest trace so they stack cleanly."""
    paths = sorted(glob.glob(os.path.join(
        root, agent, "deploy_artifacts", f"{agent}_{reward}__{constraint}__seed*_training_trace.csv")))
    if not paths:
        return None, None
    dfs = [pd.read_csv(p) for p in paths]
    n = min(len(d) for d in dfs)
    ep = dfs[0]["episode"].iloc[:n].to_numpy()
    stack = np.stack([d["lambda_wealth"].iloc[:n].to_numpy() for d in dfs])
    return ep, stack


def plot_one(root, agent, reward, reward_label, constraint):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ep, stack = load_lambda_traces(root, agent, reward, constraint)
    any_data = ep is not None
    out = None
    if any_data:
        mu, sd = stack.mean(axis=0), stack.std(axis=0)
        ax.plot(ep, mu, color=LINE_COLOR)
        ax.fill_between(ep, mu - sd, mu + sd, color=LINE_COLOR, alpha=0.18, lw=0)
        ax.set_xlim(ep[0], ep[-1])
        out = {"episode": ep, "lambda_wealth_mean": mu, "lambda_wealth_std": sd,
               "n_seeds": stack.shape[0]}
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"$\lambda$")

    stem = f"lambda_{agent}_{reward}_{constraint}"
    return fig, stem, (pd.DataFrame(out) if out else None), any_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="dir containing <agent>/deploy_artifacts/*_training_trace.csv")
    ap.add_argument("--agent", default="pepg", choices=["pepg", "pg"])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "lambda_run10"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_paper_style()

    built = []
    for constraint, _ in CONSTRAINTS:
        for reward, reward_label in REWARDS:
            built.append(plot_one(args.root, args.agent, reward, reward_label, constraint))

    shared_bbox = figure_set_bbox([fig for fig, _, _, _ in built])
    for fig, stem, data, any_data in built:
        save_figure(fig, args.out, stem, data, bbox=shared_bbox)
        print(f"  saved -> {os.path.join(args.out, stem + '.pdf')}" + ("" if any_data else "   (no data found)"))
    print("done.")


if __name__ == "__main__":
    main()
