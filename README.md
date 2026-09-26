# Eutopia

Performative lending simulator built on the UCI Adult Income dataset (`adult.csv`), with the two agents compared in the paper:

- `pepg_adapt.py`: **PeRL**, performative policy gradient, which differentiates through the population's response to the policy.
- `pg_adapt.py`: **RL**, vanilla policy gradient on the same Beta policy.

Both scripts pre-train (Phase 1), then deploy on the performative environment with continued updates (Phase 2). `sac_adapt.py` runs the same pipeline with Soft Actor-Critic, and `test_rule_based_policies.py` deploys the rule-based baselines.

## Setup

```bash
uv sync
```

Requires Python 3.12 or later; the dependencies are listed in `pyproject.toml`.

## Running

Settings used in the paper: 1,000 pre-training episodes, 3,000 deployment episodes, 10 seeds.

```bash
COMMON="--data adult.csv --train-episodes 1000 --deploy-episodes 3000 \
        --N-male 12000 --N-female 12000 --T 100 --dt 0.5 --seeds 10"

python pepg_adapt.py $COMMON --reward fairness_lagrangian --constraint social \
    --weights-dir results/pepg/weights --results-dir results/pepg
python pg_adapt.py   $COMMON --reward fairness_lagrangian --constraint social \
    --weights-dir results/pg/weights   --results-dir results/pg
```

- `--reward`: `fairness_lagrangian` (FL), `rawlsian_maximin` (RMM), `social_welfare` (SW), `utilitarian_profit` (MaxUtil), or `all`.
- `--constraint`: `social` (Equality of Outcome), `eo` (Equality of Opportunity), `dm` (no constraint, used with `utilitarian_profit`), or `all`.
- `--seeds 10` runs seeds 0 to 9; `--seed-list 0 3 7` runs specific seeds instead. `--workers N` sets the number of parallel worker processes.
- `--population-snapshot-episodes 100,1000,2000,3000` saves per-individual loan counts at those deployment episodes (used by the Lorenz and loan-distribution plots).

Every (seed, combination) result is checkpointed under `--results-dir/checkpoints/` and skipped on restart; per-seed episode logs and population files are written to `--results-dir/deploy_artifacts/`.

### Other experiments

```bash
# PeRL with a fixed multiplier gamma* = 1.87 (FL, Equality of Outcome)
python pepg_adapt.py $COMMON --reward fairness_lagrangian --constraint social \
    --lambda-wealth-override 1.87 --freeze-lambda \
    --weights-dir results/pepg_fixed/weights --results-dir results/pepg_fixed

# Robustness: performative strength or initial wealth gap scaled by 0.5 or 2
python pepg_adapt.py $COMMON --reward fairness_lagrangian --constraint social \
    --performative-scale 2 --weights-dir results/perf2/weights --results-dir results/perf2
python pepg_adapt.py $COMMON --reward fairness_lagrangian --constraint social \
    --wealth-gap-scale 0.5 --weights-dir results/gap05/weights --results-dir results/gap05

# SAC baseline (same flags as pg_adapt.py)
python sac_adapt.py $COMMON --reward fairness_lagrangian --constraint social \
    --weights-dir results/sac/weights --results-dir results/sac

# Rule-based baselines, one policy and one seed per call
python test_rule_based_policies.py --data adult.csv --policy uniform_acceptance --seed 0 \
    --episodes 3000 --N-male 12000 --N-female 12000 --T 100 --dt 0.5 \
    --deploy-artifacts-dir results/rule/deploy_artifacts --no-plots
```

Rule-based policies: `always_approve`, `always_reject`, `oracle`, `pattern_prediction`, `rich_becomes_richer`, `reverse_rich_becomes_richer`, `uniform_acceptance`.

## Figures

The scripts in `plots/` draw the paper's figures from the results directories above; most take `--root <results dir>` and `--out <output dir>` (see each script's `--help`).
