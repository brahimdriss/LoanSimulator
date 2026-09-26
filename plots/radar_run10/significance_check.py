#!/usr/bin/env python3
"""
Paired significance check for the RL-vs-PERL winner tables: for each
(reward, constraint, metric), pull the FINAL-EPISODE raw value from every
individual seed checkpoint (not the pre-aggregated mean/std CSVs -- this
lets the derived metric, approval_rate_disparity, be computed per-seed and
tested directly, rather than approximating its variance from the two
marginal approval-rate stds), then run a paired Wilcoxon signed-rank test
(pepg vs pg, paired by seed 0-9) per metric.
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DATA_ROOT = os.path.join(os.path.expanduser("~"), "run10_full_eo_results")

GROUP_REWARDS = ["rawlsian_maximin", "fairness_lagrangian", "social_welfare"]
REWARD_LABEL = {"rawlsian_maximin": "RMM", "fairness_lagrangian": "FL", "social_welfare": "SW"}
CONSTRAINTS = [("eo", "EO"), ("social", "Outcome")]

METRICS = [
    ("Wealth Gap",              False),
    ("Profit",                  True),
    ("Inequality Ratio",        False),
    ("Approval Rate Disparity", False),
]


def raw_metrics_row(row):
    wealth_gap = abs(row["wealth_gap"])
    profit = row["cumulative_profit"]
    inequality = abs(row["rho_cumulative"] - 1.0)
    approval_disparity = abs(
        row["approval_rate_M_cumulative"] - row["approval_rate_F_cumulative"]
    )
    return [wealth_gap, profit, inequality, approval_disparity]


def per_seed_values(agent_dir, reward, constraint):
    """Return array (n_seeds, 4) of final-episode raw metrics, one row per
    seed, sorted by seed number so pepg/pg arrays are paired correctly."""
    paths = sorted(
        glob.glob(os.path.join(DATA_ROOT, agent_dir, "checkpoints",
                                f"seed*_{reward}__{constraint}.csv")),
        key=lambda p: int(os.path.basename(p).split("_")[0].replace("seed", "")),
    )
    vals = []
    for p in paths:
        df = pd.read_csv(p)
        vals.append(raw_metrics_row(df.iloc[-1]))
    return np.array(vals)


if __name__ == "__main__":
    for constraint, label in CONSTRAINTS:
        print(f"\n=== {label} ({constraint}) ===")
        for reward in GROUP_REWARDS:
            pepg_vals = per_seed_values("pepg", reward, constraint)
            pg_vals = per_seed_values("pg", reward, constraint)
            n_pepg, n_pg = len(pepg_vals), len(pg_vals)
            print(f" {REWARD_LABEL[reward]}  (n_pepg={n_pepg}, n_pg={n_pg})")
            for i, (mlabel, hib) in enumerate(METRICS):
                pv, gv = pepg_vals[:, i], pg_vals[:, i]
                n = min(len(pv), len(gv))
                pv, gv = pv[:n], gv[:n]
                diff = pv - gv
                mean_pepg, mean_pg = pv.mean(), gv.mean()
                if hib:
                    winner = "PERL" if mean_pepg > mean_pg else "RL"
                else:
                    winner = "PERL" if mean_pepg < mean_pg else "RL"
                try:
                    if np.allclose(diff, 0):
                        pval = 1.0
                    else:
                        _, pval = wilcoxon(pv, gv)
                except ValueError:
                    pval = float("nan")
                sig = "***" if pval < 0.01 else ("*" if pval < 0.05 else "ns")
                print(f"    {mlabel:26s} winner={winner:5s} "
                      f"PERL={mean_pepg:.4g} PG={mean_pg:.4g}  "
                      f"p={pval:.4f} [{sig}]")
