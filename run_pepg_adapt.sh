#!/bin/bash
#SBATCH --job-name=pepg_adapt
#SBATCH --account=def-zhijing
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/pepg_adapt/slurm_%j.out
#SBATCH --error=logs/pepg_adapt/slurm_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11
module load gcc
source $SCRATCH/envs/circuit-tracing/bin/activate

cd ~/projects/def-zhijing/vpalit/LoanSimulator

mkdir -p logs/pepg_adapt
mkdir -p weights_pepg_adapt
mkdir -p results_pepg_adapt

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Run ───────────────────────────────────────────────────────────────────────
python pepg_adapt.py \
    --data              adult.csv \
    --N-male            3000 \
    --N-female          3000 \
    --train-episodes    1000 \
    --deploy-episodes   1000 \
    --warmup            20 \
    --buffer-capacity   50 \
    --seeds             10 \
    --workers           12 \
    --weights-dir       weights_pepg_adapt \
    --results-dir       results_pepg_adapt \
    --save-per-seed-csv
