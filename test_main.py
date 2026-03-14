import argparse
import glob
import os

import numpy as np

from loan_simulator.testing import run_testing, list_checkpoints


def main():
    parser = argparse.ArgumentParser(
        description="Test Trained Agent on Income Environment (Episode-wise Metrics)"
    )

    # Required (but not for --list-checkpoints)
    parser.add_argument(
        "--weights", type=str, default=None, help="Path to trained model weights (.pt)"
    )

    # Data
    parser.add_argument("--data", type=str, default=None, help="Path to adult.csv")

    # Testing parameters
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes (continuous)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--N-male", type=int, default=3000, help="Number of male individuals"
    )
    parser.add_argument(
        "--N-female", type=int, default=3000, help="Number of female individuals"
    )
    parser.add_argument("--T", type=int, default=100, help="Time horizon per episode")
    parser.add_argument("--dt", type=float, default=0.5, help="Time step")
    parser.add_argument(
        "--credit-threshold",
        type=float,
        default=0.5,
        help="Creditworthiness threshold for ground truth approval",
    )

    # Options
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic policy"
    )
    parser.add_argument("--save", action="store_false", help="Save results")
    parser.add_argument(
        "--results-dir", type=str, default="./test_results", help="Results directory"
    )

    # Checkpoint options
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for checkpoint files",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save checkpoint every N episodes",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore existing checkpoints",
    )
    parser.add_argument(
        "--clear-checkpoints",
        action="store_true",
        help="Clear existing checkpoints before starting",
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List available checkpoints and exit",
    )

    args = parser.parse_args()

    if args.list_checkpoints:
        list_checkpoints(args.checkpoint_dir)
        return

    if args.weights is None:
        parser.error(
            "--weights is required for testing "
            "(use --list-checkpoints to see available checkpoints)"
        )

    if args.clear_checkpoints:
        model_name = os.path.basename(args.weights).replace(".pt", "")
        pattern = f"{args.checkpoint_dir}/checkpoint_{model_name}_seed{args.seed}.pkl"
        for f in glob.glob(pattern):
            os.remove(f)
            print(f"Removed checkpoint: {f}")

    run_testing(
        weights_path=args.weights,
        data_filepath=args.data,
        num_episodes=args.episodes,
        seed=args.seed,
        N_male=args.N_male,
        N_female=args.N_female,
        T=args.T,
        dt=args.dt,
        deterministic=args.deterministic,
        save_results=args.save,
        results_dir=args.results_dir,
        credit_threshold=args.credit_threshold,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
