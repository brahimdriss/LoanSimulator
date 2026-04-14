#!/usr/bin/env python3
"""
Regenerate all PePG adapt plots from existing mean_*.csv / std_*.csv files.
Picks the latest file for each combo key (by timestamp in filename).

Usage
-----
  python replot_pepg.py
  python replot_pepg.py --results-dir results_pepg_adapt --output-dir results_pepg_adapt/plots
"""

import argparse
import glob
import os
import re
from datetime import datetime

import pandas as pd

from pepg_adapt import (
    _plot_comparison,
    _plot_wealth,
    _plot_social_welfare,
    _plot_inequality,
    print_summary_table,
    PEPG_COMBOS,
)
from pg_run import _combo_key


def _load_latest(results_dir, prefix):
    """
    Load mean_*.csv / std_*.csv pairs, keeping the latest timestamp per combo key.
    Returns dict: combo_key -> (mean_df, std_df)
    """
    pattern = os.path.join(results_dir, f"{prefix}*.csv")
    files = glob.glob(pattern)

    # Group by combo key, track latest timestamp
    latest = {}  # key -> (timestamp_str, filepath)
    for f in files:
        base = os.path.basename(f)
        # Strip prefix and .csv
        inner = re.sub(r"^" + re.escape(prefix), "", base)
        inner = re.sub(r"\.csv$", "", inner)
        # Extract trailing timestamp _YYYYMMDD_HHMMSS
        m = re.search(r"^(.+?)_(\d{8}_\d{6})$", inner)
        if m:
            key, ts = m.group(1), m.group(2)
        else:
            key, ts = inner, "00000000_000000"
        if key not in latest or ts > latest[key][0]:
            latest[key] = (ts, f)

    result = {}
    for key, (ts, f) in sorted(latest.items()):
        result[key] = pd.read_csv(f)
        print(f"  {prefix}{key}  ← {os.path.basename(f)}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results_pepg_adapt")
    parser.add_argument("--output-dir",  default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nLoading mean CSVs from {args.results_dir}/")
    mean_files = _load_latest(args.results_dir, "mean_")
    print(f"\nLoading std CSVs from {args.results_dir}/")
    std_files  = _load_latest(args.results_dir, "std_")

    # Build aggregated dict
    common_keys = set(mean_files.keys()) & set(std_files.keys())
    if not common_keys:
        print("No matching mean/std pairs found.")
        return

    aggregated = {k: (mean_files[k], std_files[k]) for k in common_keys}
    print(f"\nFound {len(aggregated)} combo(s): {sorted(aggregated.keys())}")

    n_seeds = 3  # for plot titles — adjust if needed
    print_summary_table(aggregated, n_seeds)

    print("\nGenerating plots…")
    for ct in ["predictive", "social", "dm", "two_sided"]:
        _plot_comparison(aggregated,     output_dir, timestamp, n_seeds, ct)
        _plot_wealth(aggregated,         output_dir, timestamp, n_seeds, ct)
        _plot_social_welfare(aggregated, output_dir, timestamp, n_seeds, ct)
        _plot_inequality(aggregated,     output_dir, timestamp, n_seeds, ct)

    print(f"\nDone. Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
