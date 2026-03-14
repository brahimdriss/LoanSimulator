"""
Command-line entry point for PePG V2 (Performative Policy Gradient) experiments.

Single-seed usage:
    python pepg_main.py --episodes 100 --reward social_welfare --constraint wealth

Multi-seed parallel usage:
    python pepg_main.py --seeds 5 --seed 1 --episodes 100 --reward all --workers 8

Load a pre-trained agent:
    python pepg_main.py --load ./weights_pepg_v2/pepg_social_welfare_wealth_seed0.pt
"""

import multiprocessing as mp
import os
import pickle
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from loan_simulator.pepg import (
    load_pepg_v2_agent,
    pepg_train_worker,
    run_pepg_v2_experiment,
)
from loan_simulator.utils import get_gpu_count


def main():
    import argparse

    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Run PePG V2 Experiment with Multi-Processing Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Experiment configuration
    parser.add_argument(
        "--data", type=str, default=None, help="Path to adult.csv data file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of training episodes (default: 100)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of random seeds to run in parallel (default: 1 for single run)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Starting seed or single seed if --seeds=1 (default: 1)",
    )
    parser.add_argument(
        "--constraint",
        type=str,
        default="wealth",
        choices=["approval_rate", "wealth", "both"],
        help="Constraint type (default: wealth)",
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="all",
        choices=[
            "all",
            "social_welfare",
            "rawlsian_maximin",
            "fairness_lagrangian",
            "utilitarian_profit",
        ],
        help="Reward function to use (default: all for multi-seed, social_welfare for single)",
    )

    # PePG specific parameters
    parser.add_argument("--buffer-capacity", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--hawkes-weight", type=float, default=1.0)
    parser.add_argument("--wealth-weight", type=float, default=1.0)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--reward-weight", type=float, default=1.0)

    parser.add_argument("--lambda-wealth", type=float, default=2.0)
    parser.add_argument("--lambda-approval", type=float, default=2.0)
    parser.add_argument("--lambda-lr", type=float, default=1e-2)

    # Output directories
    parser.add_argument("--save-weights", action="store_false")
    parser.add_argument("--weights-dir", type=str, default="./weights_pepg_v2")
    parser.add_argument("--save-lambda", action="store_true")
    parser.add_argument("--plot-lambda", action="store_true")
    parser.add_argument(
        "--lambda-dir", type=str, default="./lambda_trajectories_pepg_v2"
    )
    parser.add_argument("--save-episode-metrics", action="store_true")
    parser.add_argument("--plot-episode-metrics", action="store_true")
    parser.add_argument(
        "--episode-metrics-dir", type=str, default="./episode_metrics_pepg_v2"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Plot gradient decomposition results"
    )
    parser.add_argument(
        "--plot-loss", action="store_true", help="Plot loss propagation during training"
    )
    parser.add_argument(
        "--loss-dir",
        type=str,
        default="./loss_plots_pepg_v2",
        help="Directory to save loss plots",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./pepg_v2_results",
        help="Directory to save aggregated results (default: ./pepg_v2_results)",
    )

    # Multiprocessing parameters
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: 80%% of CPU count, max 20)",
    )

    parser.add_argument("--load", type=str, default=None)

    args = parser.parse_args()

    if args.seeds == 1:
        # ------------------------------------------------------------------ #
        # Single-seed run
        # ------------------------------------------------------------------ #
        if args.reward == "all":
            args.reward = "social_welfare"

        if args.load:
            agent, env, theta, loader = load_pepg_v2_agent(
                weights_path=args.load, data_filepath=args.data, seed=args.seed
            )
        else:
            agent, env, theta, loader = run_pepg_v2_experiment(
                data_filepath=args.data,
                num_episodes=args.episodes,
                seed=args.seed,
                reward_function=args.reward,
                constraint_type=args.constraint,
                lambda_wealth=args.lambda_wealth,
                lambda_approval=args.lambda_approval,
                lambda_lr=args.lambda_lr,
                buffer_capacity=args.buffer_capacity,
                warmup_episodes=args.warmup,
                hawkes_weight=args.hawkes_weight,
                wealth_weight=args.wealth_weight,
                transition_weight=args.transition_weight,
                reward_weight=args.reward_weight,
                save_weights=args.save_weights,
                weights_dir=args.weights_dir,
                save_lambda=args.save_lambda,
                lambda_dir=args.lambda_dir,
                plot_lambda=args.plot_lambda,
                save_episode_metrics_flag=args.save_episode_metrics,
                episode_metrics_dir=args.episode_metrics_dir,
                plot_episode_metrics_flag=args.plot_episode_metrics,
                plot_results=args.plot,
                plot_loss=args.plot_loss,
                loss_dir=args.loss_dir,
            )
    else:
        # ------------------------------------------------------------------ #
        # Multi-seed parallel run
        # ------------------------------------------------------------------ #
        start_time = time.time()

        if args.workers is None:
            available_cpus = cpu_count()
            args.workers = min(int(available_cpus * 0.8), 20)

        if args.reward == "all":
            reward_functions = [
                "social_welfare",
                "rawlsian_maximin",
                "fairness_lagrangian",
                "utilitarian_profit",
            ]
        else:
            reward_functions = [args.reward]

        print(f"\n{'=' * 80}")
        print(f"PEPG V2 MULTI-SEED EXPERIMENT")
        print(f"{'=' * 80}")
        print(f"  Seeds: {args.seeds} (starting from seed {args.seed})")
        print(f"  Reward functions: {reward_functions}")
        print(f"  Constraint: {args.constraint}")
        print(f"  Episodes: {args.episodes}")
        print(f"  Workers: {args.workers}")
        print(f"  GPUs: {get_gpu_count()}")
        print(f"{'=' * 80}\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = os.path.join(
            args.results_dir, f"pepg_v2_experiment_{timestamp}"
        )
        os.makedirs(base_output_dir, exist_ok=True)
        print(f"Base output directory: {base_output_dir}")

        configs = []
        gpu_count = get_gpu_count()

        for seed in range(args.seed, args.seed + args.seeds):
            for reward_func in reward_functions:
                config = {
                    "seed": seed,
                    "reward_function": reward_func,
                    "constraint_type": args.constraint,
                    "data_filepath": args.data,
                    "num_episodes": args.episodes,
                    "base_output_dir": base_output_dir,
                    "gpu_id": (len(configs) % gpu_count) if gpu_count > 0 else -1,
                    "run_id": len(configs) + 1,
                    "lambda_wealth": args.lambda_wealth,
                    "lambda_approval": args.lambda_approval,
                    "lambda_lr": args.lambda_lr,
                    "buffer_capacity": args.buffer_capacity,
                    "warmup_episodes": args.warmup,
                    "hawkes_weight": args.hawkes_weight,
                    "wealth_weight": args.wealth_weight,
                    "transition_weight": args.transition_weight,
                    "reward_weight": args.reward_weight,
                }
                configs.append(config)

        total_runs = len(configs)
        for config in configs:
            config["total_runs"] = total_runs

        print(f"Total runs: {total_runs}")

        aggregated_dir = os.path.join(base_output_dir, "aggregated_results")
        os.makedirs(aggregated_dir, exist_ok=True)

        print("\n[Phase 1/2] Training with parallel workers...")

        with Pool(processes=args.workers) as pool:
            train_results = pool.map(pepg_train_worker, configs)

        train_results_path = os.path.join(aggregated_dir, "train_results.pkl")
        with open(train_results_path, "wb") as f:
            pickle.dump(train_results, f)

        successful_trains = sum(1 for r in train_results if r["success"])
        print(f"\nTraining complete: {successful_trains}/{len(configs)} successful")

        print("\n[Phase 2/2] Aggregating results...")

        summary_records = []
        for result in train_results:
            if result["success"]:
                summary_records.append(result["metrics"])

        if summary_records:
            summary_df = pd.DataFrame(summary_records)
            summary_path = os.path.join(aggregated_dir, "training_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"Training summary saved to: {summary_path}")

            print(f"\n{'=' * 60}")
            print("AGGREGATED RESULTS BY REWARD FUNCTION")
            print("=" * 60)

            for reward_func in reward_functions:
                subset = summary_df[summary_df["reward_function"] == reward_func]
                if len(subset) > 0:
                    print(f"\n{reward_func.upper()}:")
                    if "train_rho_mean" in subset.columns:
                        rho_values = subset["train_rho_mean"].dropna()
                        if len(rho_values) > 0:
                            print(
                                f"  ρ (mean±std): {rho_values.mean():.4f} ± {rho_values.std():.4f}"
                            )
                    if "final_wealth_gap" in subset.columns:
                        gap_values = subset["final_wealth_gap"].dropna()
                        if len(gap_values) > 0:
                            print(
                                f"  Wealth gap: ${gap_values.mean():.3f}k ± ${gap_values.std():.3f}k"
                            )
                    if "total_profit" in subset.columns:
                        profit_values = subset["total_profit"].dropna()
                        if len(profit_values) > 0:
                            print(
                                f"  Total profit: ${profit_values.mean():.2f}k ± ${profit_values.std():.2f}k"
                            )

        elapsed_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT COMPLETE")
        print(
            f"  Total time: {elapsed_time:.1f} seconds ({elapsed_time / 60:.1f} minutes)"
        )
        print(f"  Results: {base_output_dir}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
