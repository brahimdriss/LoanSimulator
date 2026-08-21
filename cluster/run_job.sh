#!/bin/bash
# Wrapper executed by HTCondor for one (agent) run of the Eutopia experiments.
#
# $1 = agent: "pepg" or "pg"
# Everything else is fixed here so the two agents are guaranteed to run under
# identical settings -- an accidental flag difference between them would
# invalidate the comparison.
set -euo pipefail

AGENT="${1:?usage: run_job.sh <pepg|pg>}"

# --- paths (EDIT: point PROJ at wherever you rsync'd the repo) -------------
PROJ="${PROJ:-$HOME/LoanSimulator}"
OUT="${OUT:-$HOME/eutopia_runs}"
cd "$PROJ"

# Login-node /tmp is only 1GB; keep temp/pip work in the cluster home.
mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

# Thread control: each Condor slot gets request_cpus cores and CPU limits are
# HARD-enforced. Letting BLAS/torch spawn one thread per physical core would
# oversubscribe the slot and slow the job down rather than speed it up.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# The cluster nodes provide `python3`, not `python` (login2 has no `python`
# on PATH at all), so resolve it explicitly rather than assuming. Override by
# exporting PYTHON=... if you are using a venv/conda env.
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
if [ -z "$PYTHON" ]; then
  echo "ERROR: no python3/python found on PATH" >&2
  exit 127
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS="$OUT/${AGENT}_${STAMP}"
WEIGHTS="$RESULTS/weights"
mkdir -p "$RESULTS" "$WEIGHTS"

# --- shared experiment configuration --------------------------------------
# Identical for both agents. Combos are already matched (10 each, `predictive`
# on hold). See the repo notes for why each value is what it is.
COMMON=(
  --data           "$PROJ/adult.csv"
  --train-episodes 500
  --deploy-episodes 1000
  --seeds          20
  --N-male         12000
  --N-female       12000
  --T              100
  --dt             0.5
  --workers        "${WORKERS:-8}"
  --weights-dir    "$WEIGHTS"
  --results-dir    "$RESULTS"
)

echo "=== Eutopia run ==============================================="
echo "  agent    : $AGENT"
echo "  results  : $RESULTS"
echo "  host     : $(hostname)"
echo "  started  : $(date)"
echo "  python   : $PYTHON ($($PYTHON --version 2>&1))"
echo "  config   : ${COMMON[*]}"
echo "==============================================================="

case "$AGENT" in
  pepg) SCRIPT=pepg_adapt.py ;;
  pg)   SCRIPT=pg_adapt.py ;;
  *)    echo "unknown agent '$AGENT' (expected pepg|pg)" >&2; exit 2 ;;
esac

# Tee everything so the full console log is preserved alongside the CSVs,
# not just in Condor's .out file.
"$PYTHON" -u "$SCRIPT" "${COMMON[@]}" 2>&1 | tee "$RESULTS/console.log"

echo "=== finished: $(date) ==="
echo "artefacts under: $RESULTS"
du -sh "$RESULTS" 2>/dev/null || true
