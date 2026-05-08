# Eutopia Simulator

Performative loan simulator built on the UCI Adult Income dataset. Two training scripts are provided.

---

## `pepg_adapt.py` — PePG: Performative Pre-train → Performative Deploy

Trains a PePG agent on the performative environment, then deploys it with continued updates.

```bash
python pepg_adapt.py \
  --data path/to/adult.csv \
  --train-episodes 500 \
  --deploy-episodes 500 \
  --seeds 3 \
  --weights-dir ./weights_pepg_adapt \
  --results-dir ./results_pepg_adapt
```

To skip training and use existing weights:

```bash
python pepg_adapt.py \
  --skip-train \
  --data path/to/adult.csv \
  --deploy-episodes 500 \
  --seeds 3 \
  --weights-dir ./weights_pepg_adapt \
  --results-dir ./results_pepg_adapt
```

---

## `pg_adapt.py` — PG: Static Pre-train → Performative Deploy

Trains a standard policy gradient agent on a static environment, then fine-tunes it on the performative environment.

```bash
python pg_adapt.py \
  --data path/to/adult.csv \
  --train-episodes 500 \
  --deploy-episodes 500 \
  --seeds 3 \
  --weights-dir ./weights_pg \
  --results-dir ./results_pg
```

To skip training and use existing weights:

```bash
python pg_adapt.py \
  --skip-train \
  --data path/to/adult.csv \
  --deploy-episodes 500 \
  --seeds 3 \
  --weights-dir ./weights_pg \
  --results-dir ./results_pg
```

---

Results (CSVs and plots) are saved to `--results-dir`. Completed seed/combo checkpoints are saved incrementally and skipped on restart.
