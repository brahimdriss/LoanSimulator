import argparse

import numpy as np

from loan_simulator import (
    load_trained_agent,
    run_single_reward_function,
    run_experiment,
)
from loan_simulator.plotting import plot_lambda_trajectory, plot_episode_metrics


def main():
    parser = argparse.ArgumentParser(description="Run Adult Income RL Experiment")

    # Basic arguments
    parser.add_argument("--data", type=str, default=None, help="Path to adult.csv")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--constraint",
        type=str,
        default="wealth",
        choices=["approval_rate", "wealth", "both"],
        help="Constraint type: approval_rate, wealth, or both",
    )

    # Save options
    parser.add_argument("--save", action="store_true", help="Save results (pickle)")
    parser.add_argument(
        "--save-weights", action="store_false", help="Save model weights"
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="./weights",
        help="Directory for saved weights",
    )

    # Lambda trajectory options
    parser.add_argument(
        "--save-lambda",
        action="store_true",
        help="Save lambda trajectory arrays (csv and npy)",
    )
    parser.add_argument(
        "--plot-lambda", action="store_true", help="Plot lambda trajectories"
    )
    parser.add_argument(
        "--lambda-dir",
        type=str,
        default="./lambda_trajectories",
        help="Directory for lambda data",
    )

    # Episode metrics options
    parser.add_argument(
        "--save-episode-metrics",
        action="store_false",
        help="Save episode-level metrics (ρ, R_M, R_F)",
    )
    parser.add_argument(
        "--plot-episode-metrics", action="store_true", help="Plot episode-level metrics"
    )
    parser.add_argument(
        "--episode-metrics-dir",
        type=str,
        default="./episode_metrics",
        help="Directory for episode metrics",
    )

    # Single reward function mode
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run single reward function instead of all",
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="social_welfare",
        choices=[
            "social_welfare",
            "rawlsian_maximin",
            "fairness_lagrangian",
            "utilitarian_profit",
        ],
        help="Reward function (used with --single)",
    )

    # Lambda parameters (for single mode)
    parser.add_argument(
        "--lambda-wealth", type=float, default=2.0, help="Initial lambda for wealth"
    )
    parser.add_argument(
        "--lambda-approval",
        type=float,
        default=2.0,
        help="Initial lambda for approval rate",
    )
    parser.add_argument(
        "--lambda-lr",
        type=float,
        default=1e-2,
        help="Learning rate for lambda optimization",
    )

    # Load model
    parser.add_argument(
        "--load", type=str, default=None, help="Path to load saved model weights"
    )

    args = parser.parse_args()

    if args.load:
        agent, env, theta, loader = load_trained_agent(
            weights_path=args.load, data_filepath=args.data, seed=args.seed
        )
        print("\nModel loaded successfully. Ready for inference.")

        if args.plot_lambda and agent.lambda_history.get("wealth"):
            plot_lambda_trajectory(
                agent.lambda_history,
                agent.reward_func_name,
                agent.constraint_type,
                args.seed,
                save_path=None,
                show=True,
            )

        if args.plot_episode_metrics and agent.episode_metrics.get("episode"):
            plot_episode_metrics(
                agent.episode_metrics,
                agent.reward_func_name,
                agent.constraint_type,
                args.seed,
                save_path=None,
                show=True,
            )

    elif args.single:
        agent, env, theta, loader = run_single_reward_function(
            data_filepath=args.data,
            num_episodes=args.episodes,
            seed=args.seed,
            reward_function=args.reward,
            constraint_type=args.constraint,
            lambda_wealth=args.lambda_wealth,
            lambda_approval=args.lambda_approval,
            lambda_lr=args.lambda_lr,
            save_weights=args.save_weights,
            weights_dir=args.weights_dir,
            save_lambda=args.save_lambda,
            lambda_dir=args.lambda_dir,
            plot_lambda=args.plot_lambda,
            save_episode_metrics_flag=args.save_episode_metrics,
            episode_metrics_dir=args.episode_metrics_dir,
            plot_episode_metrics_flag=args.plot_episode_metrics,
        )
    else:
        results, loader, theta = run_experiment(
            data_filepath=args.data,
            num_episodes=args.episodes,
            seed=args.seed,
            constraint_type=args.constraint,
            save_results_flag=args.save,
            save_weights=args.save_weights,
            weights_dir=args.weights_dir,
            save_lambda=args.save_lambda,
            lambda_dir=args.lambda_dir,
            plot_lambda=args.plot_lambda,
        )


if __name__ == "__main__":
    main()
