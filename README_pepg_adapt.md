# `pepg_adapt.py` — PePG Performative Pre-train → Performative Deploy

Trains a **Performative Policy Gradient (PePG)** agent on a static lending environment, then deploys it with continued performative updates in a persistent testing environment.

## Two-Phase Pipeline

| Phase | Environment | What happens |
|-------|-------------|--------------|
| **1 — Train** | `IncomeEnvironment` (resets each episode) | PePGAgentV2 trained with full performative gradients (Hawkes + wealth terms) |
| **2 — Deploy** | `TestingIncomeEnvironment` (state preserved) | Weights loaded, agent continues updating in the true performative setting |

After both phases, results are aggregated across seeds and plots are saved.

## Combos

12 combinations: 4 reward functions × 3 constraint types.

| Reward | Constraint |
|--------|------------|
| `utilitarian_profit` | `wealth`, `approval_rate`, `both` |
| `social_welfare` | `wealth`, `approval_rate`, `both` |
| `rawlsian_maximin` | `wealth`, `approval_rate`, `both` |
| `fairness_lagrangian` | `wealth`, `approval_rate`, `both` |

## Checkpointing

- **Phase 1**: If a weight file already exists for a combo, training is skipped automatically on restart.
- **Phase 2**: Each (seed, combo) result is saved to `results-dir/checkpoints/` immediately upon completion. On restart, completed combos are loaded and skipped.

## Usage

### Full run (train + deploy)

```bash
python pepg_adapt.py \
  --N-male 3000 --N-female 3000 \
  --train-episodes 500 \
  --deploy-episodes 500 \
  --warmup 20 \
  --buffer-capacity 50 \
  --weights-dir /content/drive/MyDrive/Long-term-Fairness-NeurIPS/weights_pepg_adapt \
  --results-dir /content/drive/MyDrive/Long-term-Fairness-NeurIPS/results_pepg_adapt \
  --seeds 3
```

### Skip training (weights already saved)

If Phase 1 completed and you only need to run or resume the deploy phase:

```bash
python pepg_adapt.py \
  --skip-train \
  --N-male 3000 --N-female 3000 \
  --deploy-episodes 500 \
  --warmup 20 \
  --buffer-capacity 50 \
  --weights-dir /content/drive/MyDrive/Long-term-Fairness-NeurIPS/weights_pepg_adapt \
  --results-dir /content/drive/MyDrive/Long-term-Fairness-NeurIPS/results_pepg_adapt \
  --seeds 3
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--seeds` | `3` | Number of random seeds |
| `--train-episodes` | `500` | Training episodes per combo (Phase 1) |
| `--deploy-episodes` | `500` | Deploy episodes per combo (Phase 2) |
| `--warmup` | `20` | Episodes using standard PG before performative gradients activate |
| `--buffer-capacity` | `50` | Replay buffer size (episodes) |
| `--N-male` / `--N-female` | `3000` | Population size per group |
| `--T` | `100` | Episode length |
| `--dt` | `0.5` | Timestep size |
| `--workers` | 80% of CPUs | Parallel workers |
| `--skip-train` | — | Skip Phase 1, load existing weights |
| `--reward` | `all` | Filter to a single reward function |
| `--constraint` | `all` | Filter to a single constraint type |
| `--no-plots` | — | Skip plot generation |

## Outputs

All outputs saved to `--results-dir`:

- `checkpoints/seed{N}_{reward}__{constraint}.csv` — per-combo deploy results
- `mean_{combo}_{timestamp}.csv` / `std_{combo}_{timestamp}.csv` — aggregated CSVs
- `pepg_adapt_comparison_{constraint}_{timestamp}.png` — wealth gap, approval disparity, profit, R_g
- `pepg_adapt_wealth_{constraint}_{timestamp}.png` — μ_M, μ_F, cumulative profit, ρ
- `pepg_adapt_social_welfare_{constraint}_{timestamp}.png` — R_M, R_F, R̄
- `pepg_adapt_inequality_{constraint}_{timestamp}.png` — wealth gap, approval disparity, ρ(t)
