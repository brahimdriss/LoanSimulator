import os
import pickle
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch


def save_results(results, reward_func, constraint_type, seed, save_dir="./results"):
    """Save results for a specific reward function, constraint type, and seed."""
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/{reward_func}_{constraint_type}_seed{seed}_{timestamp}.pkl"

    save_data = {
        "reward_function": reward_func,
        "constraint_type": constraint_type,
        "seed": seed,
        "history": results["history"],
        "episode_rewards": results["episode_rewards"],
        "lambda_history": results.get("lambda_history", {}),
        "timestamp": timestamp,
    }

    with open(filename, "wb") as f:
        pickle.dump(save_data, f)

    print(f"  Saved results to {filename}")
    return filename


def save_lambda_history(
    lambda_history,
    reward_func,
    constraint_type,
    seed,
    save_dir="./lambda_trajectories",
    format="both",
):
    """Save lambda trajectory arrays separately for easy analysis.

    Args:
        lambda_history: dict with 'wealth' and 'approval' lists
        reward_func: Name of reward function
        constraint_type: 'wealth', 'approval_rate', or 'both'
        seed: Random seed
        save_dir: Directory to save
        format: 'csv', 'npy', or 'both'
    """
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{save_dir}/{reward_func}_{constraint_type}_seed{seed}_{timestamp}"

    df = pd.DataFrame(
        {
            "episode": range(len(lambda_history.get("wealth", []))),
            "lambda_wealth": lambda_history.get("wealth", []),
            "lambda_approval": lambda_history.get("approval", []),
        }
    )

    saved_files = []

    if format in ["csv", "both"]:
        csv_file = f"{base_filename}_lambda.csv"
        df.to_csv(csv_file, index=False)
        saved_files.append(csv_file)
        print(f"  Lambda history saved to {csv_file}")

    if format in ["npy", "both"]:
        npy_file = f"{base_filename}_lambda.npy"
        np.save(
            npy_file,
            {
                "wealth": np.array(lambda_history.get("wealth", [])),
                "approval": np.array(lambda_history.get("approval", [])),
            },
        )
        saved_files.append(npy_file)
        print(f"  Lambda history saved to {npy_file}")

    return saved_files


def save_episode_metrics(
    episode_metrics,
    reward_func,
    constraint_type,
    seed,
    save_dir="./episode_metrics",
    format="both",
):
    """Save episode-level metrics separately for analysis.

    Args:
        episode_metrics: dict with episode-level metrics
        reward_func: Name of reward function
        constraint_type: 'wealth', 'approval_rate', or 'both'
        seed: Random seed
        save_dir: Directory to save
        format: 'csv', 'npy', or 'both'
    """
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{save_dir}/{reward_func}_{constraint_type}_seed{seed}_{timestamp}"

    df = pd.DataFrame(episode_metrics)

    saved_files = []

    if format in ["csv", "both"]:
        csv_file = f"{base_filename}_episode_metrics.csv"
        df.to_csv(csv_file, index=False)
        saved_files.append(csv_file)
        print(f"  Episode metrics saved to {csv_file}")

    if format in ["npy", "both"]:
        npy_file = f"{base_filename}_episode_metrics.npy"
        np.save(npy_file, episode_metrics)
        saved_files.append(npy_file)
        print(f"  Episode metrics saved to {npy_file}")

    return saved_files


def load_trained_agent(weights_path, data_filepath=None, seed=0):
    """Load a trained agent from saved weights.

    Args:
        weights_path: Path to saved weights file (.pt)
        data_filepath: Path to adult.csv (needed to recreate environment)
        seed: Random seed

    Returns:
        agent: Loaded PolicyGradientAgent
        env: Environment
        theta_learner: Transition parameters
        loader: Data loader
    """
    from .data_loader import AdultIncomeDataLoader
    from .transition_learner import TransitionParameterLearner
    from .environment import IncomeEnvironment
    from .agent import PolicyGradientAgent

    print(f"Loading trained agent from {weights_path}...")

    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location="cpu")

    reward_function = checkpoint.get("reward_function", "social_welfare")
    constraint_type = checkpoint.get("constraint_type", "wealth")

    print(f"  Reward function: {reward_function}")
    print(f"  Constraint type: {constraint_type}")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    loader = AdultIncomeDataLoader(filepath=data_filepath, sample_size=20000)
    loader.load_data()
    loader.preprocess()

    theta_learner = TransitionParameterLearner(
        default_rate_min=0.02, default_rate_max=0.15
    )
    theta_learner.fit(loader.data)

    env = IncomeEnvironment(
        theta_params=theta_learner,
        initial_wealth_male=loader.male_data["X"].values,
        initial_wealth_female=loader.female_data["X"].values,
        N_male=3000,
        N_female=3000,
        T=100,
        dt=0.5,
        seed=seed,
    )

    lambda_wealth = checkpoint.get("final_lambda_wealth", 2.0)
    lambda_approval = checkpoint.get("final_lambda_approval", 2.0)

    agent = PolicyGradientAgent(
        env,
        reward_function=reward_function,
        constraint_type=constraint_type,
        lambda_wealth=lambda_wealth,
        lambda_approval=lambda_approval,
    )

    agent.load_model(weights_path)

    return agent, env, theta_learner, loader
