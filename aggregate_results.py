#!/usr/bin/env python3

import argparse
import os
import glob
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from run_multi_seed import add_derived_columns, aggregate_across_seeds
from pg_run import (
    VALID_COMBOS as PG_COMBOS,
    _combo_key,
    plot_comparison_agg,
    plot_wealth_agg,
    plot_social_welfare_agg,
    plot_inequality_agg,
)
from pepg_adapt import (
    PEPG_COMBOS,
    _plot_comparison,
    _plot_wealth,
    _plot_social_welfare,
    _plot_inequality,
    print_summary_table,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_seed_key_from_filename(fname):
    """
    Extract (seed, key) from filenames like:
      seed0_utilitarian_profit__two_sided_20260410_123456.csv
      seed0_utilitarian_profit__two_sided.csv   (no timestamp)
    """
    base = os.path.basename(fname)
    base = re.sub(r"\.csv$", "", base)
    # Strip trailing timestamp _YYYYMMDD_HHMMSS
    base = re.sub(r"_\d{8}_\d{6}$", "", base)
    m = re.match(r"^seed(\d+)_(.+)$", base)
    if m:
        return int(m.group(1)), m.group(2)
    return None, None


def load_per_seed_csvs(results_dirs, prefix=""):
    """
    Load per-seed CSVs from one or more results directories.
    Returns dict: seed -> {combo_key -> DataFrame}
    """
    seed_to_results = defaultdict(dict)

    for d in results_dirs:
        pattern = os.path.join(d, f"{prefix}seed*.csv")
        files = glob.glob(pattern)
        for f in files:
            seed, key = _parse_seed_key_from_filename(os.path.basename(f).replace(prefix, "", 1))
            if seed is None:
                continue
            try:
                df = add_derived_columns(pd.read_csv(f))
                # If duplicate (same seed+key from multiple dirs), prefer the longer one
                if key in seed_to_results[seed]:
                    if len(df) > len(seed_to_results[seed][key]):
                        seed_to_results[seed][key] = df
                else:
                    seed_to_results[seed][key] = df
                print(f"  Loaded: seed={seed} key={key} ({len(df)} episodes) ← {f}")
            except Exception as e:
                print(f"  SKIP {f}: {e}")

    return seed_to_results


def filter_complete_seeds(seed_to_results, expected_keys):
    """Return only seeds that have all expected_keys."""
    complete = [
        seed_to_results[s]
        for s in sorted(seed_to_results.keys())
        if expected_keys.issubset(seed_to_results[s].keys())
    ]
    missing = [
        s for s in sorted(seed_to_results.keys())
        if not expected_keys.issubset(seed_to_results[s].keys())
    ]
    if missing:
        print(f"  Seeds with incomplete combos (excluded): {missing}")
    return complete


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aggregate and plot pepg_adapt / pg_adapt results")
    parser.add_argument("--results-dirs", nargs="+", required=True,
                        help="One or more results directories to merge")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to save plots/CSVs (defaults to first results-dir)")
    parser.add_argument("--mode", choices=["pepg", "pg"], default="pepg",
                        help="Which agent type produced the results")
    parser.add_argument("--reward", type=str, default="all")
    parser.add_argument("--constraint", type=str, default="all")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dirs[0]
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Select combos
    all_combos = PEPG_COMBOS if args.mode == "pepg" else PG_COMBOS
    combos = all_combos
    if args.reward != "all":
        combos = [(r, c) for r, c in combos if r == args.reward]
    if args.constraint != "all":
        combos = [(r, c) for r, c in combos if c == args.constraint]
    expected_keys = {_combo_key(r, c) for r, c in combos}
    print(f"\nExpecting combos: {sorted(expected_keys)}")

    # ── Deploy results ────────────────────────────────────────────────────────
    print(f"\nLoading deploy (test) results from: {args.results_dirs}")
    seed_to_results = load_per_seed_csvs(args.results_dirs, prefix="")
    seed_results = filter_complete_seeds(seed_to_results, expected_keys)

    if not seed_results:
        print("\nNo complete seeds found for deploy results. Check your --results-dirs.")
    else:
        n = len(seed_results)
        print(f"\nAggregating {n} complete seeds…")
        aggregated = aggregate_across_seeds(seed_results)

        for key, (mdf, sdf) in aggregated.items():
            mdf.to_csv(os.path.join(output_dir, f"mean_{key}_{timestamp}.csv"), index=False)
            sdf.to_csv(os.path.join(output_dir, f"std_{key}_{timestamp}.csv"),  index=False)
        print(f"  Aggregated CSVs saved to {output_dir}/")

        print_summary_table(aggregated, n)

        if not args.no_plots:
            print("  Generating plots…")
            for ct in ["predictive", "social", "dm", "two_sided"]:
                _plot_comparison(aggregated,     output_dir, timestamp, n, ct)
                _plot_wealth(aggregated,         output_dir, timestamp, n, ct)
                _plot_social_welfare(aggregated, output_dir, timestamp, n, ct)
                _plot_inequality(aggregated,     output_dir, timestamp, n, ct)

    # ── Train results ─────────────────────────────────────────────────────────
    print(f"\nLoading train results…")
    seed_to_train = load_per_seed_csvs(args.results_dirs, prefix="train_metrics_")
    # Also try without prefix (some runs save without it)
    train_results = filter_complete_seeds(seed_to_train, expected_keys)

    if train_results:
        n = len(train_results)
        print(f"  Aggregating {n} complete seeds (train)…")
        train_agg = aggregate_across_seeds(train_results)
        for key, (mdf, sdf) in train_agg.items():
            mdf.to_csv(os.path.join(output_dir, f"train_mean_{key}_{timestamp}.csv"), index=False)
            sdf.to_csv(os.path.join(output_dir, f"train_std_{key}_{timestamp}.csv"),  index=False)
        if not args.no_plots:
            for ct in ["predictive", "social", "dm", "two_sided"]:
                plot_comparison_agg(train_agg, output_dir, timestamp, n, ct, prefix="train_")
                plot_wealth_agg(train_agg, output_dir, timestamp, n, ct, prefix="train_")
                plot_social_welfare_agg(train_agg, output_dir, timestamp, n, ct, prefix="train_")
                plot_inequality_agg(train_agg, output_dir, timestamp, n, ct, prefix="train_")
    else:
        print("  No complete train results found — skipping train plots.")

    print("\nDone.")


if __name__ == "__main__":
    main()
