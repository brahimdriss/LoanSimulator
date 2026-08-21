#!/bin/bash
# Wrapper executed by HTCondor for the Eutopia experiments.
#
#   run_job.sh <pepg|pg>                 -> all seeds in one job, then aggregate
#   run_job.sh <pepg|pg> <seed>          -> ONE seed only, no plots (array shard)
#   run_job.sh <pepg|pg> aggregate       -> aggregate + plot from existing checkpoints
#   run_job.sh shard <index>             -> array shard; maps a flat 0..2N-1
#                                           index onto (agent, seed)
#   run_job.sh pair <0|1> <mode>         -> 2-job form; 0=pepg, 1=pg
#
# The shard/aggregate split exists because the parallelism in this workload is
# across independent (seed, combo) runs, not inside any one of them: the policy
# net is ~18k parameters on batches of ~20, so a single job pinned to one node
# leaves the cluster idle. Sharding by seed turns ~25h of wall clock into ~2-3h.
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

# Two-job forms (all-seeds, aggregate) map index 0 -> pepg, 1 -> pg. Same
# reason as `shard`: $CHOICE($(Process), pepg, pg) is the very nesting pattern
# HTCondor rejects, so no submit file uses $CHOICE at all.
if [ "${1:-}" = "pair" ]; then
  _IDX="${2:?usage: run_job.sh pair <0|1> <mode>}"
  _MODE="${3:?usage: run_job.sh pair <0|1> <mode>}"
  if [ "$_IDX" -eq 0 ]; then set -- pepg "$_MODE"; else set -- pg "$_MODE"; fi
fi

AGENT="${1:?usage: run_job.sh <pepg|pg> [seed|aggregate]  |  shard <index>  |  pair <0|1> <mode>}"
MODE="${2:-all}"

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

# The cluster provides `python3`, not `python`. Override with PYTHON=... for a
# venv/conda env.
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
# comparison, so it lives in exactly one place.
COMMON=(
  --data           "$PROJ/adult.csv"
  --train-episodes 500
  --deploy-episodes 1000
  --N-male         12000
  --N-female       12000
  --T              100
  --dt             0.5
  --workers        "${WORKERS:-8}"
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
else
  # One seed. --no-plots because 20 shards each rendering the same figures from
  # a single seed would be waste; the aggregate pass draws them once.
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
