#!/bin/bash
# Wrapper executed by HTCondor for the Eutopia experiments.
#
#   run_job.sh <pepg|pg>                       -> all seeds in one job, then aggregate
#   run_job.sh <pepg|pg> <seed>                -> ONE seed, all 13 combos, no plots
#   run_job.sh <pepg|pg> <seed> <r> <c>        -> ONE (seed, combo) pair only
#   run_job.sh <pepg|pg> aggregate             -> aggregate + plot from existing checkpoints
#   run_job.sh shard <index>                   -> array shard; maps a flat 0..2N-1
#                                                  index onto (agent, seed) -- 13 combos/job
#   run_job.sh combo <index>                   -> array shard; maps a flat 0..2*13N-1
#                                                  index onto (agent, seed, reward, constraint)
#                                                  -- ONE combo/job, no internal worker pool
#   run_job.sh pair <0|1> <mode>               -> 2-job form; 0=pepg, 1=pg
#
# The shard/aggregate split exists because the parallelism in this workload is
# across independent (seed, combo) runs, not inside any one of them: the policy
# net is ~18k parameters on batches of ~20, so a single job pinned to one node
# leaves the cluster idle.
#
# `shard` packs all 13 combos of one seed into one job (8-worker internal pool)
# -- fewer Condor jobs, but the 8 workers contend with each other for the same
# host's cache/memory bandwidth once all are CPU-active (measured: episode time
# for a fixed combo drifted 46.6s -> 57.5s purely from sibling-worker load, and
# combos starting later in the same shard ran ~1.8x slower than the first ones).
# `combo` avoids that entirely by giving every (seed, combo) pair its own Condor
# job on its own host -- more jobs, but each one runs in isolation.
#
# Sharding is safe because the pipeline checkpoints every (seed, combo) pair to
# <results>/checkpoints/ and skips pairs it already finds. Shards therefore
# write disjoint filenames into a SHARED results dir, and the aggregate pass
# simply re-reads all of them.
set -euo pipefail

# --- flat array index -> (agent, seed) ------------------------------------
# HTCondor's macro language cannot nest $() calls: $CHOICE($INT($(Process)/20))
# fails to parse ("$INT($(Process is invalid index"). Only `queue N` with a
# bare $(Process) is portable, so the submit file passes the flat index and the
# arithmetic happens here, in bash, where it is unambiguous.
#   index 0 .. N_SEEDS-1        -> pepg, seed = index
#   index N_SEEDS .. 2*N_SEEDS-1 -> pg,  seed = index - N_SEEDS
if [ "${1:-}" = "shard" ]; then
  _IDX="${2:?usage: run_job.sh shard <array-index>}"
  _NS="${N_SEEDS:-20}"
  if [ "$_IDX" -lt "$_NS" ]; then
    set -- pepg "$_IDX"
  else
    set -- pg "$(( _IDX - _NS ))"
  fi
fi

# --- flat array index -> (agent, seed, reward, constraint) -----------------
# Same $CHOICE/$INT nesting problem as `shard` -- the arithmetic happens here.
# Order matches VALID_COMBOS in pg_run.py / PEPG_COMBOS in pepg_adapt.py /
# COMBOS in verify_run.py (the `predictive` column is on hold; utilitarian_
# profit/eo is undefined -- same reason it has no /social entry -- so 13
# combos, not 14).
_COMBOS_R=(utilitarian_profit utilitarian_profit \
           social_welfare social_welfare social_welfare \
           rawlsian_maximin rawlsian_maximin rawlsian_maximin rawlsian_maximin \
           fairness_lagrangian fairness_lagrangian fairness_lagrangian fairness_lagrangian)
_COMBOS_C=(dm two_sided \
           social two_sided eo \
           social dm two_sided eo \
           social dm two_sided eo)

if [ "${1:-}" = "combo" ]; then
  _IDX="${2:?usage: run_job.sh combo <array-index>}"
  _NS="${N_SEEDS:-3}"
  _NC=${#_COMBOS_R[@]}
  _PER_AGENT=$(( _NS * _NC ))
  if [ "$_IDX" -lt "$_PER_AGENT" ]; then
    _AGENT=pepg; _LOCAL=$_IDX
  else
    _AGENT=pg; _LOCAL=$(( _IDX - _PER_AGENT ))
  fi
  _SEED=$(( _LOCAL / _NC ))
  _CI=$(( _LOCAL % _NC ))
  set -- "$_AGENT" "$_SEED" "${_COMBOS_R[$_CI]}" "${_COMBOS_C[$_CI]}"
fi

# Two-job forms (all-seeds, aggregate) map index 0 -> pepg, 1 -> pg. Same
# reason as `shard`: $CHOICE($(Process), pepg, pg) is the very nesting pattern
# HTCondor rejects, so no submit file uses $CHOICE at all.
if [ "${1:-}" = "pair" ]; then
  _IDX="${2:?usage: run_job.sh pair <0|1> <mode>}"
  _MODE="${3:?usage: run_job.sh pair <0|1> <mode>}"
  if [ "$_IDX" -eq 0 ]; then set -- pepg "$_MODE"; else set -- pg "$_MODE"; fi
fi

AGENT="${1:?usage: run_job.sh <pepg|pg> [seed|aggregate]  |  shard <index>  |  combo <index>  |  pair <0|1> <mode>}"
MODE="${2:-all}"
REWARD="${3:-}"
CONSTRAINT="${4:-}"

# Single-combo jobs run alone in their own process -- no internal pool needed,
# so default to 1 worker instead of 8. (Still overridable via WORKERS=.)
if [ -n "$REWARD" ] && [ -n "$CONSTRAINT" ]; then
  WORKERS_DEFAULT=1
else
  WORKERS_DEFAULT=8
fi

# --- paths ----------------------------------------------------------------
PROJ="${PROJ:-$HOME/LoanSimulator}"
OUT="${OUT:-$HOME/eutopia_runs}"
cd "$PROJ"

# Login-node /tmp is only 1GB; keep temp work in the cluster home.
mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

# Thread control. The entry scripts also set these before importing numpy/torch
# (see the BLAS thread guard at the top of pepg_adapt.py), so this is belt and
# braces -- but Condor hard-enforces CPU limits, and OpenBLAS defaulting to one
# thread per visible core would oversubscribe the slot.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# The cluster provides `python3`, not `python`. getenv=True carries the
# submitting shell's PATH into the job -- if that shell didn't have the venv
# activated (e.g. a fresh terminal), `command -v python3` silently resolves
# to the system interpreter instead, which has none of the project's deps.
# That's a real failure mode, not hypothetical: a 200-job array once failed
# 100% uniformly (ModuleNotFoundError: pandas) this exact way. Prefer the
# known venv location explicitly so a forgotten `source activate` doesn't
# silently break every job in a submission. Override with PYTHON=... for a
# different venv/conda env.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ] && [ -x "$HOME/eutopia-venv/bin/python3" ]; then
  PYTHON="$HOME/eutopia-venv/bin/python3"
fi
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [ -z "$PYTHON" ]; then
  echo "ERROR: no python3/python found on PATH" >&2
  exit 127
fi

# --- output layout --------------------------------------------------------
# FIXED (not timestamped) so every shard of a campaign writes to one place and
# the aggregate pass can find them. Set CAMPAIGN to keep separate runs apart.
CAMPAIGN="${CAMPAIGN:-main}"
RESULTS="$OUT/$CAMPAIGN/$AGENT"
WEIGHTS="$RESULTS/weights"
mkdir -p "$RESULTS" "$WEIGHTS"

# --- shared experiment configuration --------------------------------------
# Identical for both agents; a flag differing between them would invalidate the
# comparison, so it lives in exactly one place. Override TRAIN_EPISODES/
# DEPLOY_EPISODES via env for a different scale of run (e.g. a bigger
# confirmatory run after a pilot looks promising) -- default 500/1000 matches
# the original pilot scale.
COMMON=(
  --data           "$PROJ/adult.csv"
  --train-episodes "${TRAIN_EPISODES:-500}"
  --deploy-episodes "${DEPLOY_EPISODES:-1000}"
  --N-male         12000
  --N-female       12000
  --T              100
  --dt             0.5
  --workers        "${WORKERS:-$WORKERS_DEFAULT}"
  --weights-dir    "$WEIGHTS"
  --results-dir    "$RESULTS"
)
N_SEEDS="${N_SEEDS:-20}"

case "$AGENT" in
  pepg) SCRIPT=pepg_adapt.py ;;
  pg)   SCRIPT=pg_adapt.py ;;
  *)    echo "unknown agent '$AGENT' (expected pepg|pg)" >&2; exit 2 ;;
esac

# --- mode-specific args ---------------------------------------------------
if [ "$MODE" = "aggregate" ]; then
  # Every checkpoint already exists; this pass loads them, aggregates across
  # seeds and writes the CSVs and figures.
  MODE_ARGS=( --seeds "$N_SEEDS" )
  LOGNAME="aggregate"
elif [ "$MODE" = "all" ]; then
  MODE_ARGS=( --seeds "$N_SEEDS" )
  LOGNAME="all"
elif [ -n "$REWARD" ] && [ -n "$CONSTRAINT" ]; then
  # One (seed, combo) pair -- the isolated `combo` sharding mode.
  MODE_ARGS=( --seed-list "$MODE" --reward "$REWARD" --constraint "$CONSTRAINT" --no-plots )
  LOGNAME="seed${MODE}_${REWARD}__${CONSTRAINT}"
else
  # One seed, all 13 combos. --no-plots because N shards each rendering the
  # same figures from a single seed would be waste; the aggregate pass draws
  # them once.
  MODE_ARGS=( --seed-list "$MODE" --no-plots )
  LOGNAME="seed$MODE"
fi

echo "=== Eutopia ==================================================="
echo "  agent    : $AGENT"
echo "  mode     : $MODE"
echo "  campaign : $CAMPAIGN"
echo "  results  : $RESULTS"
echo "  host     : $(hostname)"
echo "  started  : $(date)"
echo "  python   : $PYTHON ($($PYTHON --version 2>&1))"
echo "  args     : ${COMMON[*]} ${MODE_ARGS[*]}"
echo "==============================================================="

mkdir -p "$RESULTS/logs"
"$PYTHON" -u "$SCRIPT" "${COMMON[@]}" "${MODE_ARGS[@]}" 2>&1 \
  | tee "$RESULTS/logs/console_${LOGNAME}.log"

echo "=== finished: $(date) ==="
echo "checkpoints so far: $(find "$RESULTS/checkpoints" -name '*.csv' 2>/dev/null | wc -l)"
du -sh "$RESULTS" 2>/dev/null || true
