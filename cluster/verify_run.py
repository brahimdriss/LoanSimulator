#!/usr/bin/env python3
"""
Pre-aggregation verifier: did the sharded campaign actually finish, completely
and cleanly?

Aggregation silently tolerates missing seeds -- it averages whatever it finds --
so a shard that died would quietly reduce the sample size rather than error.
This checks that up front.

Usage:
    python3 cluster/verify_run.py                       # ~/eutopia_runs/main
    python3 cluster/verify_run.py --root ~/eutopia_runs/main --seeds 20
    python3 cluster/verify_run.py --logs ~/eutopia_logs  # also scan condor .err
    python3 cluster/verify_run.py --seeds 10 --diagnose   # + per-seed basin table

Exit status is 0 only if every check passes, so it can gate a script:
    python3 cluster/verify_run.py && run_job.sh pepg aggregate
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

COMBOS = [
    ("utilitarian_profit", "dm"), ("utilitarian_profit", "two_sided"),
    ("social_welfare", "social"), ("social_welfare", "two_sided"),
    ("social_welfare", "eo"),
    ("rawlsian_maximin", "social"), ("rawlsian_maximin", "dm"),
    ("rawlsian_maximin", "two_sided"), ("rawlsian_maximin", "eo"),
    ("fairness_lagrangian", "social"),
    ("fairness_lagrangian", "dm"), ("fairness_lagrangian", "two_sided"),
    ("fairness_lagrangian", "eo"),
]

# Metrics that must be finite and, where noted, non-degenerate.
KEY_METRICS = ["wealth_gap", "R_M", "R_F", "cumulative_profit",
               "reach_rate_M", "reach_rate_F", "mu_M_end", "mu_F_end"]

OK, BAD, WARN = "PASS", "FAIL", "WARN"
_fail = []
_warn = []


def warn(label, clean, detail=""):
    """Report but do NOT fail the run. For things that are suspicious rather
    than definitely wrong -- e.g. a policy that lends to nobody may be the
    genuine optimum for some objectives."""
    print(f"  [{OK if clean else WARN}] {label}" + (f"  -- {detail}" if detail else ""))
    if not clean:
        _warn.append(label)


def check(label, ok, detail=""):
    print(f"  [{OK if ok else BAD}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        _fail.append(label)
    return ok


def verify_agent(root, agent, seeds):
    print(f"\n=== {agent} ===")
    adir = os.path.join(root, agent)
    if not os.path.isdir(adir):
        check(f"{agent}: results dir exists", False, adir)
        return

    # --- 1. every (seed, combo) checkpoint present -------------------------
    ckdir = os.path.join(adir, "checkpoints")
    found, missing = set(), []
    for f in glob.glob(os.path.join(ckdir, "seed*_*.csv")):
        m = re.match(r"seed(\d+)_(.+)\.csv$", os.path.basename(f))
        if m:
            found.add((int(m.group(1)), m.group(2)))
    for s in range(seeds):
        for r, c in COMBOS:
            if (s, f"{r}__{c}") not in found:
                missing.append(f"seed{s}/{r}__{c}")
    expected = seeds * len(COMBOS)
    check(f"checkpoints complete ({len(found)}/{expected})", not missing,
          "" if not missing else f"{len(missing)} missing, e.g. {missing[:3]}")

    # --- 2. deploy artefacts (4 files per seed/combo) ---------------------
    dep = os.path.join(adir, "deploy_artifacts")
    n_pt = len(glob.glob(os.path.join(dep, "*_deployed.pt")))
    n_ep = len(glob.glob(os.path.join(dep, "*_episodes.csv")))
    n_tr = len(glob.glob(os.path.join(dep, "*_training_trace.csv")))
    n_np = len(glob.glob(os.path.join(dep, "*_population.npz")))
    check(f"deploy artefacts ({n_pt} weights, {n_ep} episodes, "
          f"{n_tr} traces, {n_np} populations)",
          n_pt == n_ep == n_tr == n_np == expected,
          f"expected {expected} of each")

    # --- 3. data sanity: finite, right length, non-degenerate -------------
    bad_finite, bad_len, degenerate, ep_counts = [], [], [], []
    for f in sorted(glob.glob(os.path.join(ckdir, "seed*_*.csv"))):
        try:
            df = pd.read_csv(f)
        except Exception as e:
            bad_finite.append(f"{os.path.basename(f)}: unreadable ({e})")
            continue
        ep_counts.append(len(df))
        for col in KEY_METRICS:
            if col not in df.columns:
                bad_len.append(f"{os.path.basename(f)}: missing column {col}")
                continue
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(v).all():
                bad_finite.append(f"{os.path.basename(f)}:{col} "
                                  f"({int((~np.isfinite(v)).sum())} non-finite)")
        # a policy that never lent at all across a whole run is suspicious
        if "total_loans_M" in df.columns and df["total_loans_M"].iloc[-1] == 0:
            degenerate.append(f"{os.path.basename(f)}: zero loans to group M")

    check("all metrics finite (no NaN/Inf)", not bad_finite,
          "" if not bad_finite else f"{len(bad_finite)} issues, e.g. {bad_finite[:2]}")
    check("expected columns present", not bad_len,
          "" if not bad_len else str(bad_len[:2]))
    warn("no zero-lending runs", not degenerate,
         "" if not degenerate else f"{len(degenerate)}, e.g. {degenerate[:2]}")

    if ep_counts:
        lo, hi = min(ep_counts), max(ep_counts)
        check(f"episode counts consistent ({lo}..{hi} rows)", lo == hi,
              "" if lo == hi else "shards ran different episode counts")

    # --- 4. per-shard console logs reached the end ------------------------
    logs = glob.glob(os.path.join(adir, "logs", "console_seed*.log"))
    unfinished = [os.path.basename(l) for l in logs
                  if "Done." not in open(l, errors="ignore").read()[-4000:]]
    check(f"shard logs ended cleanly ({len(logs)} logs)",
          logs and not unfinished,
          "" if logs else "no shard logs found"
          if not logs else f"unfinished: {unfinished[:3]}")


# --- degenerate-basin diagnostics (--diagnose) -------------------------------
# The run10 post-mortem found policies collapsing to three degenerate basins
# that the aggregate means hid: a uniform coin flip (Beta(1,1): cumulative
# approval exactly 0.500/0.500 for both groups), approve-all (>0.95) and
# reject-all (<0.05). This table shows every (combo, seed) cell so a collapsed
# seed can never again disappear into a mean.
COINFLIP_TOL = 0.005      # |rate - 0.5| for both groups  -> 'x'
REJECT_ALL = 0.05         # max(M, F) below this          -> '0'
APPROVE_ALL = 0.95        # min(M, F) above this          -> '1'
SPREAD_WARN = 0.15        # per-combo max-min of approval_rate_M across seeds
APPROVAL_COLS = ("approval_rate_M_cumulative", "approval_rate_F_cumulative")


def _basin_flag(m, f):
    if abs(m - 0.5) <= COINFLIP_TOL and abs(f - 0.5) <= COINFLIP_TOL:
        return "x"
    if max(m, f) < REJECT_ALL:
        return "0"
    if min(m, f) > APPROVE_ALL:
        return "1"
    return "."


def diagnose_agent(root, agent, seeds):
    """Per-agent table of final-episode cumulative approval (M/F) per
    (combo, seed), flagged by basin. Warns (does not fail) on any coin-flip
    cell or on a combo whose seeds disagree on approval_rate_M by more than
    SPREAD_WARN."""
    print(f"\n=== {agent}: basin diagnostics (final-row "
          f"approval_rate_M_cumulative/approval_rate_F_cumulative) ===")
    ckdir = os.path.join(root, agent, "checkpoints")
    cells = {}                                  # (combo, seed) -> (M, F) or None
    combos_on_disk = set()
    for f in glob.glob(os.path.join(ckdir, "seed*_*.csv")):
        m = re.match(r"seed(\d+)_(.+)\.csv$", os.path.basename(f))
        if not m:
            continue
        seed, combo = int(m.group(1)), m.group(2)
        combos_on_disk.add(combo)
        if seed >= seeds:
            continue
        try:
            df = pd.read_csv(f, usecols=list(APPROVAL_COLS))
            last = df.dropna().iloc[-1]
            cells[(combo, seed)] = (float(last[APPROVAL_COLS[0]]),
                                    float(last[APPROVAL_COLS[1]]))
        except Exception as e:
            print(f"  ! {os.path.basename(f)}: {e}")
            cells[(combo, seed)] = None
    if not combos_on_disk:
        warn(f"{agent}: diagnose found checkpoints", False, ckdir)
        return

    # known combos in canonical order first, then anything unexpected on disk
    order = [f"{r}__{c}" for r, c in COMBOS if f"{r}__{c}" in combos_on_disk]
    order += sorted(combos_on_disk - set(order))
    width = max(len(c) for c in order)
    seed_cols = list(range(seeds))
    print("  " + "combo".ljust(width) + "  " + "  ".join(f"s{s}".center(10) for s in seed_cols))

    counts = {"x": 0, "0": 0, "1": 0, ".": 0}
    total = 0
    x_combos, spread_combos = [], []
    for combo in order:
        row = [combo.ljust(width)]
        m_vals = []
        for s in seed_cols:
            cell = cells.get((combo, s))
            if cell is None:
                row.append("   --     ")
                continue
            m, f = cell
            flag = _basin_flag(m, f)
            counts[flag] += 1
            total += 1
            m_vals.append(m)
            row.append(f"{m:.2f}/{f:.2f}{flag}".center(10))
            if flag == "x" and combo not in x_combos:
                x_combos.append(combo)
        if len(m_vals) >= 2 and (max(m_vals) - min(m_vals)) > SPREAD_WARN:
            spread_combos.append(f"{combo} (spread {max(m_vals) - min(m_vals):.2f})")
        print("  " + "  ".join(row))

    print(f"  flags: x=coin-flip(0.50/0.50)  0=reject-all(<{REJECT_ALL})  "
          f"1=approve-all(>{APPROVE_ALL})  .=other")
    print(f"  {agent} summary: x={counts['x']}  0={counts['0']}  1={counts['1']}  "
          f"of {total} cells")
    problems = []
    if counts["x"]:
        problems.append(f"{counts['x']} coin-flip cell(s) in {x_combos}")
    if spread_combos:
        problems.append(f"approval_rate_M spread > {SPREAD_WARN} across seeds in "
                        f"{spread_combos}")
    if problems:
        print(f"  WARNING: {agent}: " + "; ".join(problems))
        _warn.append(f"{agent}: degenerate basins -- " + "; ".join(problems))
    else:
        print(f"  [{OK}] {agent}: no coin-flip cells, per-combo spread <= {SPREAD_WARN}")


def scan_condor_logs(logdir):
    print(f"\n=== condor stderr ({logdir}) ===")
    errs = glob.glob(os.path.join(logdir, "*.err"))
    if not errs:
        check("condor .err files present", False, "none found")
        return
    # tqdm writes progress bars to stderr, so filter to genuine failures.
    pat = re.compile(r"Traceback|Error:|ERROR|Killed|MemoryError|"
                     r"CUDA|Segmentation fault|blas_thread_init", re.I)
    hits = []
    for e in errs:
        for line in open(e, errors="ignore"):
            if pat.search(line) and "0 errors" not in line:
                hits.append(f"{os.path.basename(e)}: {line.strip()[:110]}")
                break
    check(f"no errors in {len(errs)} condor .err files", not hits,
          "" if not hits else f"{len(hits)} files, e.g. {hits[:2]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/eutopia_runs/main"))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--agents", nargs="+", default=["pepg", "pg"])
    ap.add_argument("--logs", default=None, help="condor log dir to scan for errors")
    ap.add_argument("--diagnose", action="store_true",
                    help="after the checks, print a per-(combo, seed) table of final "
                         "cumulative approval rates flagged for degenerate basins "
                         "(coin-flip / reject-all / approve-all)")
    a = ap.parse_args()

    print(f"Verifying campaign at {a.root}  (expecting {a.seeds} seeds "
          f"x {len(COMBOS)} combos = {a.seeds * len(COMBOS)} runs per agent)")
    for agent in a.agents:
        verify_agent(a.root, agent, a.seeds)
    if a.diagnose:
        for agent in a.agents:
            diagnose_agent(a.root, agent, a.seeds)
    if a.logs:
        scan_condor_logs(os.path.expanduser(a.logs))

    print("\n" + "=" * 62)
    if _warn:
        print(f"{WARN}: {len(_warn)} warning(s) -- worth a look, not blocking:")
        for w in _warn:
            print(f"   - {w}")
        print()
    if _fail:
        print(f"{BAD}: {len(_fail)} check(s) failed -- do NOT aggregate yet:")
        for f in _fail:
            print(f"   - {f}")
        print("\nRe-submitting the array is safe: shards skip work already"
              "\ncheckpointed, so only the missing pieces are recomputed.")
        sys.exit(1)
    print(f"{OK}: all checks passed -- safe to aggregate.")
    sys.exit(0)


if __name__ == "__main__":
    main()
