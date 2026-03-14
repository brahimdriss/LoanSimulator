#!/usr/bin/env python3
"""
Test-only script for PG agents.

Skips training entirely — discovers existing weights in --weights-dir and
runs Phase 2 (testing) + Phase 3 (aggregate + plot + save CSVs).

Weight files must match the naming convention written by pg_run.py:
    {reward}__{constraint}__seed{seed}.pt

Usage
-----
  python pg_test_only.py \
      --weights-dir /content/drive/MyDrive/Long-term-Fairness-NeurIPS/weights \
      --results-dir /content/drive/MyDrive/Long-term-Fairness-NeurIPS/results \
      --seeds 5 --test-episodes 500 --N-male 3000 --N-female 3000
"""

import argparse
import multiprocessing as mp
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

from loan_simulator.testing.data_loader import TestingAdultIncomeDataLoader
from loan_simulator.testing.environment import TestingIncomeEnvironment
from loan_simulator.transition_learner import TransitionParameterLearner
from run_multi_seed import add_derived_columns, aggregate_across_seeds
from pg_run import (
    VALID_COMBOS,
    CONSTRAINT_LABELS,
    _PolicyNet,
    _get_action,
    _combo_key,
    print_summary_table,
    plot_comparison_agg,
    plot_wealth_agg,
    plot_social_welfare_agg,
    plot_inequality_agg,
)
from test_rule_based_policies import setup_plot_style  # noqa: F401


# ---------------------------------------------------------------------------
# Test worker (identical logic to pg_run._test_worker)
# ---------------------------------------------------------------------------

def _test_worker(cfg):
    seed         = cfg["seed"]
    reward       = cfg["reward_function"]
    constraint   = cfg["constraint_type"]
    weights_path = cfg["weights_path"]
    run_id       = cfg.get("run_id", 0)
    total        = cfg.get("total_runs", 1)

    try:
        np.random.seed(seed)

        theta     = cfg["theta"]
        male_X    = cfg["male_X"]
        female_X  = cfg["female_X"]
        gt_male   = cfg["gt_male"]
        gt_female = cfg["gt_female"]

        env = TestingIncomeEnvironment(
            theta_params=theta,
            initial_wealth_male=male_X,
            initial_wealth_female=female_X,
            ground_truth_male=gt_male,
            ground_truth_female=gt_female,
            N_male=cfg["N_male"],
            N_female=cfg["N_female"],
            T=cfg["T"],
            dt=cfg["dt"],
            seed=seed,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_net = _PolicyNet(
            input_dim=12, hidden_dim=cfg.get("hidden_dim", 128)
        ).to(device)
        checkpoint = torch.load(
            weights_path, map_location=device, weights_only=False
        )
        policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        policy_net.eval()

        test_eps = cfg["test_episodes"]
        for ep in range(test_eps):
            obs, _ = env.reset()
            done = False
            while not done:
                action = _get_action(policy_net, device, obs)
                obs, _, term, trunc, _ = env.step(np.array([action]))
                done = term or trunc

            if (ep + 1) % max(1, test_eps // 5) == 0 or ep + 1 == test_eps:
                app_M = env.episode_loans_M / max(env.episode_applications_M, 1)
                app_F = env.episode_loans_F / max(env.episode_applications_F, 1)
                print(
                    f"  [{run_id:3d}/{total}] TEST  "
                    f"seed={seed}  {reward[:4]}/{constraint[:4]}  "
                    f"ep {ep+1}/{test_eps}  "
                    f"μ_M={env.mu_M:.1f}  μ_F={env.mu_F:.1f}  "
                    f"appM={app_M:.3f} appF={app_F:.3f}"
                )

        env.finalize_episode_metrics()
        df = env.get_episode_metrics_dataframe()
        key = _combo_key(reward, constraint)
        print(f"  [{run_id:3d}/{total}] TEST OK  seed={seed}  {reward}/{constraint}")
        return seed, key, df

    except Exception as exc:
        import traceback
        print(
            f"  [{run_id}/{total}] TEST FAIL  "
            f"seed={seed}  {reward}/{constraint}: {exc}"
        )
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test-only: load PG weights and run on TestingIncomeEnvironment"
    )

    # Seeds / combos
    parser.add_argument("--seeds",       type=int, default=5,
                        help="Number of seeds (0..seeds-1) (default: 5)")
    parser.add_argument("--seed-list",   type=int, nargs="+", default=None,
                        help="Explicit seed list (overrides --seeds)")
    parser.add_argument("--reward",      type=str, default="all",
                        choices=["all", "utilitarian_profit", "social_welfare",
                                 "rawlsian_maximin", "fairness_lagrangian"])
    parser.add_argument("--constraint",  type=str, default="all",
                        choices=["all", "predictive", "social", "dm", "two_sided"])

    # Environment
    parser.add_argument("--test-episodes", type=int,   default=500)
    parser.add_argument("--N-male",        type=int,   default=3000)
    parser.add_argument("--N-female",      type=int,   default=3000)
    parser.add_argument("--T",             type=int,   default=100)
    parser.add_argument("--dt",            type=float, default=0.5)
    parser.add_argument("--hidden-dim",    type=int,   default=128)
    parser.add_argument("--credit-threshold", type=float, default=0.5)

    # I/O
    parser.add_argument("--data",         type=str, default=None,
                        help="Path to adult.csv (None = auto-download)")
    parser.add_argument("--weights-dir",  type=str, default="./weights_pg")
    parser.add_argument("--results-dir",  type=str, default="./pg_results")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory for per-combo checkpoints (default: results-dir/checkpoints). "
                             "Existing checkpoints are loaded and those combos skipped.")
    parser.add_argument("--workers",      type=int, default=None)
    parser.add_argument("--no-plots",     action="store_true")
    parser.add_argument("--save-per-seed-csv", action="store_true")

    args = parser.parse_args()

    seeds  = args.seed_list if args.seed_list is not None else list(range(args.seeds))
    n_seeds = len(seeds)

    combos = VALID_COMBOS
    if args.reward != "all":
        combos = [(r, c) for r, c in combos if r == args.reward]
    if args.constraint != "all":
        combos = [(r, c) for r, c in combos if c == args.constraint]

    if not combos:
        print("No valid combos match the requested filter. Exiting.")
        return

    if args.workers is None:
        args.workers = min(int(mp.cpu_count() * 0.8), 20)

    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("PG TEST-ONLY (resume from saved weights)")
    print("=" * 70)
    print(f"  Seeds          : {seeds}")
    print(f"  Combos         : {len(combos)}")
    print(f"  Test episodes  : {args.test_episodes}")
    print(f"  Workers        : {args.workers}")
    print(f"  Population     : {args.N_male}M + {args.N_female}F")
    print(f"  Weights dir    : {args.weights_dir}")
    print(f"  Results dir    : {args.results_dir}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Discover weights
    # ------------------------------------------------------------------
    found = []
    missing = []
    for seed in seeds:
        for reward, constraint in combos:
            weights_path = os.path.join(
                args.weights_dir,
                f"{reward}__{constraint}__seed{seed}.pt",
            )
            if os.path.exists(weights_path):
                found.append({"seed": seed, "reward": reward,
                               "constraint": constraint, "weights_path": weights_path})
            else:
                missing.append(f"  MISSING: {weights_path}")

    if missing:
        print(f"\nWarning — {len(missing)} weight file(s) not found:")
        for m in missing:
            print(m)

    if not found:
        print("No weight files found. Exiting.")
        return

    print(f"\nFound {len(found)} weight file(s). Proceeding to test…")

    # ------------------------------------------------------------------
    # Load test data once
    # ------------------------------------------------------------------
    print("\n  Loading test data (once)…")
    test_loader = TestingAdultIncomeDataLoader(
        filepath=args.data,
        sample_size=20000,
        credit_threshold=args.credit_threshold,
    )
    test_loader.load_data()
    test_loader.preprocess()
    test_theta = TransitionParameterLearner(
        default_rate_min=0.14, default_rate_max=0.16
    )
    test_theta.fit(test_loader.data)
    _male_X   = test_loader.male_data["X"].values
    _female_X = test_loader.female_data["X"].values
    _gt_male  = test_loader.male_data["ground_truth_approval"].values
    _gt_female = test_loader.female_data["ground_truth_approval"].values

    # ------------------------------------------------------------------
    # Load existing checkpoints — skip already-done combos
    # ------------------------------------------------------------------
    import pandas as pd
    seed_to_results = defaultdict(dict)
    n_loaded = 0
    for entry in found:
        seed = entry["seed"]
        key  = _combo_key(entry["reward"], entry["constraint"])
        ckpt = os.path.join(checkpoint_dir, f"seed{seed}_{key}.csv")
        if os.path.exists(ckpt):
            try:
                df = add_derived_columns(pd.read_csv(ckpt))
                seed_to_results[seed][key] = df
                n_loaded += 1
            except Exception:
                pass  # corrupt checkpoint — re-run this combo

    if n_loaded:
        print(f"\n  Loaded {n_loaded} checkpoint(s) — skipping those combos.")

    # ------------------------------------------------------------------
    # Build test configs (skip already checkpointed)
    # ------------------------------------------------------------------
    test_configs = []
    for entry in found:
        seed = entry["seed"]
        key  = _combo_key(entry["reward"], entry["constraint"])
        if key in seed_to_results.get(seed, {}):
            continue  # already done
        cfg = {
            "seed":            seed,
            "reward_function": entry["reward"],
            "constraint_type": entry["constraint"],
            "weights_path":    entry["weights_path"],
            "test_episodes":   args.test_episodes,
            "N_male":          args.N_male,
            "N_female":        args.N_female,
            "T":               args.T,
            "dt":              args.dt,
            "hidden_dim":      args.hidden_dim,
            "theta":           test_theta,
            "male_X":          _male_X,
            "female_X":        _female_X,
            "gt_male":         _gt_male,
            "gt_female":       _gt_female,
            "run_id":          len(test_configs) + 1,
            "total_runs":      len(found) - n_loaded,
        }
        test_configs.append(cfg)

    if not test_configs:
        print("\n  All combos already checkpointed — skipping to aggregation.")
    else:
        print(f"\n  Testing {len(test_configs)} agents ({n_loaded} already done)…")
        with mp.Pool(processes=args.workers) as pool:
            for raw in pool.imap_unordered(_test_worker, test_configs):
                if raw is None:
                    continue
                seed, key, df = raw
                df = add_derived_columns(df)
                seed_to_results[seed][key] = df

                # Save checkpoint immediately so progress survives interruption
                ckpt = os.path.join(checkpoint_dir, f"seed{seed}_{key}.csv")
                df.to_csv(ckpt, index=False)
                print(f"  Checkpointed: seed{seed}_{key}")

                if args.save_per_seed_csv:
                    path = os.path.join(
                        args.results_dir, f"seed{seed}_{key}_{timestamp}.csv"
                    )
                    df.to_csv(path, index=False)

    # Keep only seeds that have all expected combos
    expected_keys = {_combo_key(r, c) for r, c in combos}
    seed_results = [
        seed_to_results[s]
        for s in seeds
        if expected_keys.issubset(seed_to_results[s].keys())
    ]

    if not seed_results:
        print("No complete seed results to aggregate. Exiting.")
        return

    n_complete = len(seed_results)
    print(f"\n  {n_complete}/{n_seeds} seeds complete with all combos")

    # ------------------------------------------------------------------
    # Aggregate + Plot
    # ------------------------------------------------------------------
    print(f"\n  Aggregating mean ± std across {n_complete} seeds…")
    aggregated = aggregate_across_seeds(seed_results)

    for key, (mdf, sdf) in aggregated.items():
        mdf.to_csv(
            os.path.join(args.results_dir, f"mean_{key}_{timestamp}.csv"),
            index=False,
        )
        sdf.to_csv(
            os.path.join(args.results_dir, f"std_{key}_{timestamp}.csv"),
            index=False,
        )
    print(f"  Aggregated CSVs saved to {args.results_dir}/")

    print_summary_table(aggregated, n_complete)

    if not args.no_plots:
        print("  Generating plots…")
        for ct in ["predictive", "social", "dm", "two_sided"]:
            plot_comparison_agg(aggregated, args.results_dir, timestamp, n_complete, ct)
            plot_wealth_agg(aggregated, args.results_dir, timestamp, n_complete, ct)
            plot_social_welfare_agg(aggregated, args.results_dir, timestamp, n_complete, ct)
            plot_inequality_agg(aggregated, args.results_dir, timestamp, n_complete, ct)

    print("\nDone.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
