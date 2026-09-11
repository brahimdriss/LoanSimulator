#!/usr/bin/env python3
"""
Is a probe_policies.py FLAT flag genuine input-blindness, or a symptom of
the probe's fixed synthetic range (X in [20,600], mu in [40,1000], both in
the same 'k' units as the environment) no longer covering where the
population actually sits after a long deploy?

For each deployed policy this compares:
  (a) the actual final mu_M/mu_F and X range from its population.npz
      (or, lacking a snapshot, mu_M_end/mu_F_end from its episodes.csv), to
  (b) probe_policies.py's synthetic OBS_RANGES ceiling for those columns,
and re-runs the FLAT check with the observation range widened to cover the
actual final wealth, holding every other column at probe_policies.py's
existing OBS_RANGES.

Usage (same --root layout as verify_run.py / probe_policies.py):
  python3 cluster/check_flat_range.py --root ~/eutopia_runs/run11 \
      --agent pepg --combo rawlsian_maximin__social
  (drop --combo to check every FLAT-flagged cell probe_policies.py found;
   this re-imports probe_policies.py's own OBS_RANGES/FLAT_STD/N_ROWS so the
   two scripts can't silently drift apart.)
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_policies import (  # noqa: E402
    OBS_RANGES, FLAT_STD, N_ROWS, BATCH_SEED, probe, load_net, synthetic_batch,
)


def actual_wealth_range(deploy_dir, agent, combo, seed):
    """(X_lo, X_hi, mu_lo, mu_hi) actually reached by episode's end, from the
    population.npz if present (covers X directly), else from the episodes
    CSV's mu_M_end/mu_F_end only (X range then falls back to OBS_RANGES)."""
    stem = f"{agent}_{combo}__seed{seed}"
    npz_path = os.path.join(deploy_dir, f"{stem}_population.npz")
    x_lo = x_hi = mu_lo = mu_hi = None
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        X_all = np.concatenate([d["X_male"], d["X_female"]])
        x_lo, x_hi = float(X_all.min()), float(X_all.max())
        mu_lo = min(float(d["X_male"].mean()), float(d["X_female"].mean()))
        mu_hi = max(float(d["X_male"].mean()), float(d["X_female"].mean()))
    csv_path = os.path.join(deploy_dir, f"{stem}_episodes.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        row = pd.read_csv(csv_path).iloc[-1]
        mu_lo = min(mu_lo or row["mu_M_end"], row["mu_M_end"], row["mu_F_end"])
        mu_hi = max(mu_hi or row["mu_F_end"], row["mu_M_end"], row["mu_F_end"])
    return x_lo, x_hi, mu_lo, mu_hi


def widened_batch(x_hi, mu_hi, n=N_ROWS, seed=BATCH_SEED):
    rng = np.random.default_rng(seed)
    cols = []
    for name, lo, hi in OBS_RANGES:
        if name == "S":
            cols.append(rng.integers(0, 2, size=n).astype(np.float32))
        elif name == "X/100" and x_hi is not None:
            cols.append(rng.uniform(lo, max(hi, x_hi / 100.0 * 1.1), size=n).astype(np.float32))
        elif name in ("mu_R/100", "mu_B/100") and mu_hi is not None:
            cols.append(rng.uniform(lo, max(hi, mu_hi / 100.0 * 1.1), size=n).astype(np.float32))
        else:
            cols.append(rng.uniform(lo, hi, size=n).astype(np.float32))
    return torch.from_numpy(np.stack(cols, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--agent", required=True, choices=["pepg", "pg"])
    ap.add_argument("--combo", default=None, help="e.g. rawlsian_maximin__social; default: all")
    args = ap.parse_args()

    deploy_dir = os.path.join(args.root, args.agent, "deploy_artifacts")
    pat = re.compile(rf"^{re.escape(args.agent)}_(.+)__seed(\d+)_deployed\.pt$")
    rows = []
    for f in sorted(glob.glob(os.path.join(deploy_dir, "*.pt"))):
        m = pat.match(os.path.basename(f))
        if not m:
            continue
        combo, seed = m.group(1), int(m.group(2))
        if args.combo and combo != args.combo:
            continue
        rows.append((f, combo, seed))

    print(f"{'combo':30s} {'seed':>4s}  {'probe hi (mu,X)':>16s}  {'actual hi (mu,X)':>17s}  "
          f"{'old std':>8s} {'old flag':>8s}  {'widened std':>11s} {'widened flag':>13s}")
    for f, combo, seed in rows:
        net, legacy = load_net(f)

        old_batch = synthetic_batch()
        old_stats, old_flags = probe(net, old_batch)

        x_lo, x_hi, mu_lo, mu_hi = actual_wealth_range(deploy_dir, args.agent, combo, seed)
        wide_batch = widened_batch(x_hi, mu_hi)
        wide_stats, wide_flags = probe(net, wide_batch)

        old_flag = "FLAT" if "FLAT" in old_flags else "-"
        wide_flag = "FLAT" if "FLAT" in wide_flags else "-"
        mu_hi_disp = f"{mu_hi:.0f}" if mu_hi is not None else "?"
        x_hi_disp = f"{x_hi:.0f}" if x_hi is not None else "?"
        print(f"{combo:30s} {seed:4d}  {'100/1000':>16s}  {mu_hi_disp+'/'+x_hi_disp:>17s}  "
              f"{old_stats['p_std']:8.4f} {old_flag:>8s}  {wide_stats['p_std']:11.4f} {wide_flag:>13s}")


if __name__ == "__main__":
    main()
