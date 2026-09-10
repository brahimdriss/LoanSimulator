#!/usr/bin/env bash
# HTCondor executable for ONE rule-based baseline policy x ONE seed, deployed
# for the agents' Phase-2 horizon on the same performative environment.
#   usage: run_rule.sh <policy> <seed>
# Env (same names / defaults as run_job.sh): CAMPAIGN, DEPLOY_EPISODES,
# SNAPSHOT_EPISODES, PROJ, OUT, PYTHON.  Writes
#   $OUT/$CAMPAIGN/rule/deploy_artifacts/rule_<policy>__baseline__seed<S>_{episodes.csv,population.npz}
set -euo pipefail
PROJ="${PROJ:-$HOME/LoanSimulator}"
OUT="${OUT:-$HOME/eutopia_runs}"
CAMPAIGN="${CAMPAIGN:?export CAMPAIGN=<name> (e.g. run11) before submitting}"

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export MPLBACKEND=Agg

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ] && [ -x "$HOME/eutopia-venv/bin/python3" ]; then
  PYTHON="$HOME/eutopia-venv/bin/python3"
fi
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
[ -n "$PYTHON" ] || { echo "ERROR: no python3 found" >&2; exit 1; }

POLICY="${1:?policy name}"
SEED="${2:?seed}"
ART="$OUT/$CAMPAIGN/rule/deploy_artifacts"
mkdir -p "$ART"
cd "$PROJ"

ARGS=(
  --data "$PROJ/adult.csv"
  --policy "$POLICY"
  --seed "$SEED"
  --episodes "${DEPLOY_EPISODES:-3000}"
  --N-male 12000
  --N-female 12000
  --T 100
  --dt 0.5
  --deploy-artifacts-dir "$ART"
  --no-plots
)
if [ -n "${SNAPSHOT_EPISODES:-}" ]; then
  ARGS+=( --population-snapshot-episodes "$SNAPSHOT_EPISODES" )
fi

echo "=== Eutopia rule-based baseline ==============================="
echo "  policy   : $POLICY   seed: $SEED"
echo "  campaign : $CAMPAIGN"
echo "  artefacts: $ART"
echo "  host     : $(hostname)   started: $(date)"
echo "  python   : $PYTHON ($($PYTHON --version 2>&1))"
echo "  args     : ${ARGS[*]}"
echo "==============================================================="
"$PYTHON" test_rule_based_policies.py "${ARGS[@]}"
echo "=== finished: $(date) ==="
