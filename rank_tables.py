#!/usr/bin/env python3
"""
Ranking tables from per-seed checkpoint CSVs.

For each agent (pepg_adapt, static) and each episode range (short=100, long=1000):
  Table A — rows=constraints,     cols=metrics: best-ranked reward function + mean±std + KW p-value
  Table B — rows=reward functions, cols=metrics: best-ranked constraint     + mean±std + KW p-value

Metrics:
  cumulative_profit  → higher is better
  wealth_gap         → lower  is better
  R_bar              → higher is better  (computed: (N_M*R_M + N_F*R_F)/(N_M+N_F))
  rho_episode        → lower  is better

Ranking: pandas.rank() on mean-across-seeds values.
P-value: scipy.stats.kruskal() across groups (one value per seed = 3 obs/group).
"""

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import kruskal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AGENTS = {
    "pepg_adapt": "results_pepg_adapt/checkpoints",
    "static":     "results_static/checkpoints",
}

SEEDS      = [0, 1, 2]
REWARDS    = ["utilitarian_profit", "social_welfare", "rawlsian_maximin", "fairness_lagrangian"]
CONSTRAINTS = ["predictive", "social", "dm", "two_sided"]

REWARD_LABELS = {
    "utilitarian_profit":  "Util. Profit",
    "social_welfare":      "Social Welfare",
    "rawlsian_maximin":    "Rawlsian Max-Min",
    "fairness_lagrangian": "Fairness Lagr.",
}
CONSTRAINT_LABELS = {
    "predictive": "Predictive",
    "social":     "Social",
    "dm":         "DM",
    "two_sided":  "Two-Sided",
}

EP_RANGES = [("short_term", 100), ("long_term", 1000)]

# (column, display name, higher_is_better)
METRICS = [
    ("cumulative_profit", "Cumulative Profit", True),
    ("wealth_gap",        "Wealth Gap",        False),
    ("R_bar",             "Social Welfare R̄",  True),
    ("rho_episode",       "Inequality ρ",       False),
]

OUT_DIR = "results_ranking"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _inject_R_bar(df):
    N_M   = df["total_applications_M"]
    N_F   = df["total_applications_F"]
    total = (N_M + N_F).replace(0, np.nan)
    df["R_bar"] = (N_M * df["R_M"] + N_F * df["R_F"]) / total
    return df


def load_checkpoints(ckpt_dir):
    """
    Returns dict (reward, constraint, seed) → DataFrame.
    """
    data = {}
    for path in glob.glob(os.path.join(ckpt_dir, "seed*.csv")):
        fname = os.path.basename(path)
        m = re.match(r"seed(\d+)_(.+)__(.+)\.csv$", fname)
        if not m:
            continue
        seed, reward, constraint = int(m.group(1)), m.group(2), m.group(3)
        data[(reward, constraint, seed)] = _inject_R_bar(pd.read_csv(path))
    return data

# ---------------------------------------------------------------------------
# Get the final-episode value for a metric within an ep range
# ---------------------------------------------------------------------------

def final_val(df, col, n_eps):
    sub = df[df["episode"] <= n_eps]
    if sub.empty or col not in sub.columns:
        return np.nan
    return sub[col].iloc[-1]

# ---------------------------------------------------------------------------
# Build seed-level value matrix for a fixed (grouping_key, metric)
#
# group_vals: dict  group_key → list of per-seed values
# ---------------------------------------------------------------------------

def get_group_vals(data, fixed_key, varying_keys, fixed_is_constraint, col, n_eps):
    """
    fixed_is_constraint=True  → fixed_key is a constraint, varying_keys are rewards
    fixed_is_constraint=False → fixed_key is a reward,     varying_keys are constraints
    """
    group_vals = {}
    for vk in varying_keys:
        reward     = vk         if fixed_is_constraint else fixed_key
        constraint = fixed_key  if fixed_is_constraint else vk
        vals = []
        for seed in SEEDS:
            key = (reward, constraint, seed)
            if key not in data:
                continue
            v = final_val(data[key], col, n_eps)
            if not np.isnan(v):
                vals.append(v)
        if vals:
            group_vals[vk] = vals
    return group_vals   # dict: varying_key → [seed values]

# ---------------------------------------------------------------------------
# Rank + KW for one cell
# ---------------------------------------------------------------------------

def rank_and_test(group_vals, higher_is_better):
    """
    group_vals: dict key → list of floats (seed values)
    Returns:
      winner        : key with rank 1
      winner_mean   : float
      winner_std    : float
      kw_pvalue     : float (nan if < 2 groups with > 1 obs)
      rank_series   : pd.Series key → rank (1 = best)
    """
    if not group_vals:
        return None, np.nan, np.nan, np.nan, pd.Series(dtype=float)

    means = pd.Series({k: np.mean(v) for k, v in group_vals.items()})

    # rank: ascending=True means smallest gets rank 1
    # for higher_is_better we want largest → rank 1, so ascending=False
    ranks = means.rank(ascending=not higher_is_better, method="min")

    winner     = ranks.idxmin()
    w_vals     = group_vals[winner]
    w_mean     = np.mean(w_vals)
    w_std      = np.std(w_vals, ddof=1) if len(w_vals) > 1 else 0.0

    # KW: need ≥ 2 groups each with ≥ 1 observation
    groups_for_kw = [v for v in group_vals.values() if len(v) >= 1]
    if len(groups_for_kw) >= 2:
        try:
            _, pval = kruskal(*groups_for_kw)
        except ValueError:
            pval = np.nan
    else:
        pval = np.nan

    return winner, w_mean, w_std, pval, ranks

# ---------------------------------------------------------------------------
# Build one table
# ---------------------------------------------------------------------------

def build_table(data, row_keys, row_labels, row_is_constraint,
                varying_keys, varying_labels, n_eps):
    """
    Returns a DataFrame with MultiIndex columns:
      metric → (winner, mean±std, p-value)
    """
    records = []
    for rk in row_keys:
        row = {"Row": row_labels[rk]}
        for col, metric_name, hib in METRICS:
            varying = varying_keys
            group_vals = get_group_vals(
                data, rk, varying, row_is_constraint, col, n_eps
            )
            winner, w_mean, w_std, pval, _ = rank_and_test(group_vals, hib)

            winner_label = varying_labels.get(winner, winner) if winner else "—"
            mean_std_str = f"{w_mean:.3f} ± {w_std:.3f}" if winner else "—"
            pval_str     = f"{pval:.3f}" if not np.isnan(pval) else "—"

            row[f"{metric_name} | Best"]     = winner_label
            row[f"{metric_name} | Mean±Std"] = mean_std_str
            row[f"{metric_name} | KW p"]     = pval_str
        records.append(row)

    df = pd.DataFrame(records).set_index("Row")
    return df

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for agent_tag, ckpt_dir in AGENTS.items():
        print(f"\n{'='*70}")
        print(f"Agent: {agent_tag}  ({ckpt_dir})")
        print(f"{'='*70}")

        data = load_checkpoints(ckpt_dir)
        print(f"  Loaded {len(data)} (reward, constraint, seed) entries.")

        for ep_label, n_eps in EP_RANGES:
            print(f"\n  ── {ep_label}  (ep 1–{n_eps}) ──")

            # Table A: rows = constraints, best reward function per cell
            tA = build_table(
                data,
                row_keys      = CONSTRAINTS,
                row_labels    = CONSTRAINT_LABELS,
                row_is_constraint = True,
                varying_keys  = REWARDS,
                varying_labels= REWARD_LABELS,
                n_eps         = n_eps,
            )
            print(f"\n  Table A — Best reward function per constraint")
            print(tA.to_string())
            fname = f"table_A_{agent_tag}_{ep_label}.csv"
            tA.to_csv(os.path.join(OUT_DIR, fname))
            print(f"  Saved → {os.path.join(OUT_DIR, fname)}")

            # Table B: rows = rewards, best constraint per cell
            tB = build_table(
                data,
                row_keys      = REWARDS,
                row_labels    = REWARD_LABELS,
                row_is_constraint = False,
                varying_keys  = CONSTRAINTS,
                varying_labels= CONSTRAINT_LABELS,
                n_eps         = n_eps,
            )
            print(f"\n  Table B — Best constraint per reward function")
            print(tB.to_string())
            fname = f"table_B_{agent_tag}_{ep_label}.csv"
            tB.to_csv(os.path.join(OUT_DIR, fname))
            print(f"  Saved → {os.path.join(OUT_DIR, fname)}")

    print(f"\nAll tables saved to '{OUT_DIR}/'")
