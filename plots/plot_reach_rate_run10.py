#!/usr/bin/env python3
"""
Reach-rate curves, RL (pg) vs PERL (pepg), under social (Equality of
Outcome) and eo (Equality of Opportunity) fairness. Paper figure, in the
style of Eutopia_2026.pdf Fig. 5: one figure per (agent, constraint), one
panel per reward function (RMM, FL, SW), red = male, blue = female, line =
mean over seeds, band = +/- 1 std over seeds.

reach_rate_g(t) = unique individuals in group g who received a loan in
episode t / N_g. Read straight from run10's aggregated mean_*.csv and
std_*.csv (columns reach_rate_M / reach_rate_F), so no re-run is needed --
unlike the Lorenz curves, this is per-episode data run10 already logged.

--smooth N applies a centred rolling mean of N episodes to the mean line
and band (default 1 = raw per-episode values).
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"

AGENTS = [("pg", "RL"), ("pepg", "PERL")]
CONSTRAINTS = [("social", "Equality of Outcome"), ("eo", "Equality of Opportunity")]
REWARDS = [("rawlsian_maximin", "RMM"), ("fairness_lagrangian", "FL"), ("social_welfare", "SW")]
GROUPS = [("M", "Male", "#C0392B"), ("F", "Female", "#2980B9")]


def load_mean_std(root, agent, reward, constraint):
    m = sorted(glob.glob(os.path.join(root, agent, f"mean_{reward}__{constraint}_*.csv")))
    s = sorted(glob.glob(os.path.join(root, agent, f"std_{reward}__{constraint}_*.csv")))
    if not m or not s:
        return None, None
    return pd.read_csv(m[-1]), pd.read_csv(s[-1])


def smooth(series, n):
    return series.rolling(n, center=True, min_periods=1).mean() if n > 1 else series


def plot_agent_constraint(root, agent, agent_label, constraint, constraint_label,
                          out_dir, n_smooth):
    fig, axes = plt.subplots(1, len(REWARDS), figsize=(3.4 * len(REWARDS), 3.0), sharey=True)
    for ax, (reward, reward_label) in zip(axes, REWARDS):
        mean_df, std_df = load_mean_std(root, agent, reward, constraint)
        ax.set_title(reward_label, fontsize=11)
        ax.set_xlabel("Episode", fontsize=10)
        ax.grid(True, lw=0.4, color="gray", alpha=0.35)
        if mean_df is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")
            continue
        ep = mean_df["episode"]
        for g, g_label, color in GROUPS:
            mu = smooth(mean_df[f"reach_rate_{g}"], n_smooth)
            sd = smooth(std_df[f"reach_rate_{g}"], n_smooth)
            ax.plot(ep, mu, color=color, lw=1.4, label=g_label)
            ax.fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.18, lw=0)
        ax.set_xlim(ep.iloc[0], ep.iloc[-1])
    axes[0].set_ylabel("Reach rate (unique recipients / N)", fontsize=10)
    axes[0].set_ylim(bottom=0)
    axes[-1].legend(fontsize=9, frameon=False, loc="best")
    fig.suptitle(f"{agent_label}, {constraint_label}", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(out_dir, f"reach_rate_{agent}_{constraint}.pdf")
    fig.savefig(out, format="pdf", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/run10_full_eo_results"),
                    help="dir containing {pg,pepg}/mean_*.csv and std_*.csv")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "reach_rate_run10"))
    ap.add_argument("--smooth", type=int, default=1,
                    help="centred rolling-mean window in episodes (1 = raw)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for agent, agent_label in AGENTS:
        for constraint, constraint_label in CONSTRAINTS:
            plot_agent_constraint(args.root, agent, agent_label, constraint, constraint_label,
                                  args.out, args.smooth)
    print("done.")


if __name__ == "__main__":
    main()
