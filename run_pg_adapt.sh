#!/bin/bash
#SBATCH --job-name=pg_adapt
#SBATCH --account=def-zhijing
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/pg_adapt/slurm_%j.out
#SBATCH --error=logs/pg_adapt/slurm_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11
module load gcc
source $SCRATCH/envs/circuit-tracing/bin/activate

cd ~/projects/def-zhijing/vpalit/LoanSimulator

mkdir -p logs/pg_adapt
mkdir -p $SCRATCH/loansim/weights_pg_adapt
mkdir -p $SCRATCH/loansim/results_pg_adapt

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Run ───────────────────────────────────────────────────────────────────────
python pg_adapt.py \
    --data              $SCRATCH/loansim/adult_income.csv \
    --N-male            3000 \
    --N-female          3000 \
    --train-episodes    1000 \
    --deploy-episodes   1000 \
    --warmup            20 \
    --seeds             5 \
    --workers           12 \
    --weights-dir       $SCRATCH/loansim/weights_pg_adapt \
    --results-dir       $SCRATCH/loansim/results_pg_adapt \
    --save-per-seed-csv
