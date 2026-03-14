import os
import random
from datetime import datetime

import numpy as np
import torch

from .data_loader import AdultIncomeDataLoader
from .transition_learner import TransitionParameterLearner
from .environment import IncomeEnvironment
from .agent import PolicyGradientAgent
from .io_utils import save_results, save_lambda_history, save_episode_metrics
from .plotting import plot_lambda_trajectory, plot_episode_metrics


def run_single_reward_function(
    data_filepath=None,
    num_episodes=100,
    seed=0,
    reward_function="social_welfare",
    constraint_type="wealth",
    lambda_wealth=2.0,
    lambda_approval=2.0,
    lambda_lr=1e-2,
    save_weights=True,
    weights_dir="./weights",
    save_lambda=True,
    lambda_dir="./lambda_trajectories",
    plot_lambda=True,
    save_episode_metrics_flag=True,
    episode_metrics_dir="./episode_metrics",
    plot_episode_metrics_flag=True,
):
    """Run experiment with a SINGLE reward function and optionally save weights.

    Args:
        data_filepath: Path to adult.csv (None for auto-download)
        num_episodes: Number of training episodes
        seed: Random seed
        reward_function: One of 'social_welfare', 'rawlsian_maximin',
                         'fairness_lagrangian', 'utilitarian_profit'
        constraint_type: 'approval_rate', 'wealth', or 'both'
        lambda_wealth: Initial lambda for wealth constraint
        lambda_approval: Initial lambda for approval rate constraint
        lambda_lr: Learning rate for lambda optimization
        save_weights: Whether to save model weights
        weights_dir: Directory to save weights
        save_lambda: Whether to save lambda trajectory arrays
        lambda_dir: Directory to save lambda trajectories
        plot_lambda: Whether to plot lambda trajectories
        save_episode_metrics_flag: Whether to save episode-level metrics
        episode_metrics_dir: Directory to save episode metrics
        plot_episode_metrics_flag: Whether to plot episode-level metrics

    Returns:
        agent, env, theta_learner, loader
    """
    os.makedirs(weights_dir, exist_ok=True)

    print("=" * 80)
    print(f"SINGLE REWARD FUNCTION EXPERIMENT")
    print(f"  Reward: {reward_function.upper()}")
    print(f"  Constraint: {constraint_type.upper()}")
    print(f"  Seed: {seed}")
    print("=" * 80)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"\nCUDA Available: {torch.cuda.get_device_name(0)}")
    else:
        print("\nUsing CPU")

    print(f"\n[STEP 1] Loading Adult Income Data...")
    loader = AdultIncomeDataLoader(filepath=data_filepath, sample_size=20000)
    loader.load_data()
    loader.preprocess()

    print(f"\n[STEP 2] Learning θ Parameters...")
    theta_learner = TransitionParameterLearner(
        default_rate_min=0.02, default_rate_max=0.15
    )
    theta_learner.fit(loader.data)

    print(f"\n[STEP 3] Creating Environment...")
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

    print(f"\n[STEP 4] Training Agent...")
    agent = PolicyGradientAgent(
        env,
        reward_function=reward_function,
        constraint_type=constraint_type,
        lambda_wealth=lambda_wealth,
        lambda_approval=lambda_approval,
        lambda_lr=lambda_lr,
    )

    agent.train(num_episodes=num_episodes)

    print(f"\n[STEP 5] Running Test Episode...")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = agent.get_action(obs)
        obs, _, terminated, truncated, _ = env.step(np.array([action]))
        done = terminated or truncated

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE: {reward_function} ({constraint_type})")
    print("=" * 60)
    print(f"  Initial wealth gap: ${env.history['wealth_gap'][0]:.3f}k")
    print(f"  Final wealth gap: ${env.history['wealth_gap'][-1]:.3f}k")
    print(f"  Approval disparity: {env.history['approval_disparity'][-1]:.4f}")
    print(f"  Total profit: ${sum(env.history['profit']):.3f}k")

    if reward_function != "utilitarian_profit":
        final_lambda_w, final_lambda_a = agent._get_current_lambdas()
        if constraint_type == "wealth":
            print(f"  Final λ_wealth: {final_lambda_w:.4f}")
        elif constraint_type == "approval_rate":
            print(f"  Final λ_approval: {final_lambda_a:.4f}")
        else:
            print(f"  Final λ_wealth: {final_lambda_w:.4f}")
            print(f"  Final λ_approval: {final_lambda_a:.4f}")

    if agent.episode_metrics["rho_episode"]:
        rho_values = np.array(agent.episode_metrics["rho_episode"])
        R_M_values = np.array(agent.episode_metrics["R_M"])
        R_F_values = np.array(agent.episode_metrics["R_F"])
        print(f"\n  --- Episode-Level Metrics (Training) ---")
        print(
            f"  ρ (inequality ratio): mean={np.mean(rho_values):.4f}, std={np.std(rho_values):.4f}"
        )
        print(f"  R_M (male welfare):   final={R_M_values[-1]:.6f}")
        print(f"  R_F (female welfare): final={R_F_values[-1]:.6f}")
        print(
            f"  R_M / R_F:            {R_M_values[-1] / R_F_values[-1]:.4f}"
            if abs(R_F_values[-1]) > 1e-8
            else "  R_M / R_F: N/A"
        )

    if save_weights:
        print(f"\n[STEP 6] Saving Model Weights...")
        weights_filename = (
            f"{weights_dir}/{reward_function}_{constraint_type}_seed{seed}.pt"
        )
        agent.save_model(weights_filename)
        print(f"  Saved to: {weights_filename}")

    if save_lambda and reward_function != "utilitarian_profit":
        print(f"\n[STEP 7] Saving Lambda Trajectories...")
        save_lambda_history(
            agent.lambda_history,
            reward_function,
            constraint_type,
            seed,
            save_dir=lambda_dir,
            format="both",
        )

    if plot_lambda and reward_function != "utilitarian_profit":
        print(f"\n[STEP 8] Plotting Lambda Trajectories...")
        plot_path = f"{lambda_dir}/{reward_function}_{constraint_type}_seed{seed}_lambda_plot.png"
        os.makedirs(lambda_dir, exist_ok=True)
        plot_lambda_trajectory(
            agent.lambda_history,
            reward_function,
            constraint_type,
            seed,
            save_path=plot_path,
            show=True,
        )

    if save_episode_metrics_flag:
        print(f"\n[STEP 9] Saving Episode Metrics...")
        os.makedirs(episode_metrics_dir, exist_ok=True)
        save_episode_metrics(
            agent.episode_metrics,
            reward_function,
            constraint_type,
            seed,
            save_dir=episode_metrics_dir,
            format="both",
        )

    if plot_episode_metrics_flag:
        print(f"\n[STEP 10] Plotting Episode Metrics...")
        os.makedirs(episode_metrics_dir, exist_ok=True)
        plot_path = f"{episode_metrics_dir}/{reward_function}_{constraint_type}_seed{seed}_episode_metrics.png"
        plot_episode_metrics(
            agent.episode_metrics,
            reward_function,
            constraint_type,
            seed,
            save_path=plot_path,
            show=True,
        )

    return agent, env, theta_learner, loader


def run_experiment(
    data_filepath=None,
    num_episodes=100,
    seed=0,
    constraint_type="wealth",
    save_results_flag=False,
    save_weights=True,
    weights_dir="./weights",
    save_lambda=False,
    lambda_dir="./lambda_trajectories",
    plot_lambda=False,
):
    """Run Adult Income experiment with all reward functions.

    Args:
        constraint_type: 'approval_rate', 'wealth', or 'both'
        save_weights: Whether to save model weights for each reward function
        weights_dir: Directory to save weights
        save_lambda: Whether to save lambda trajectory arrays
        lambda_dir: Directory to save lambda trajectories
        plot_lambda: Whether to plot lambda trajectories
    """
    if save_weights:
        os.makedirs(weights_dir, exist_ok=True)
    if save_lambda:
        os.makedirs(lambda_dir, exist_ok=True)

    print("=" * 80)
    print(
        f"ADULT INCOME EXPERIMENT - SEED {seed} - CONSTRAINT: {constraint_type.upper()}"
    )
    print("=" * 80)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"\nCUDA Available: {torch.cuda.get_device_name(0)}")
    else:
        print("\nUsing CPU")

    loader = AdultIncomeDataLoader(filepath=data_filepath, sample_size=20000)
    loader.load_data()
    loader.preprocess()

    theta_learner = TransitionParameterLearner(
        default_rate_min=0.02, default_rate_max=0.15
    )
    theta_learner.fit(loader.data)

    reward_functions = [
        "social_welfare",
        "rawlsian_maximin",
        "fairness_lagrangian",
        "utilitarian_profit",
    ]

    results = {}
    saved_files = []

    for reward_func in reward_functions:
        print(f"\n{'=' * 60}")
        print(f"Training: {reward_func.upper().replace('_', ' ')} ({constraint_type})")
        print("=" * 60)

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

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

        if reward_func == "utilitarian_profit":
            lambda_wealth, lambda_approval = 0.0, 0.0
        elif reward_func == "social_welfare":
            lambda_wealth, lambda_approval = 2.0, 2.0
        elif reward_func == "rawlsian_maximin":
            lambda_wealth, lambda_approval = 5.0, 5.0
        elif reward_func == "fairness_lagrangian":
            lambda_wealth, lambda_approval = 10.0, 10.0

        agent = PolicyGradientAgent(
            env,
            reward_function=reward_func,
            constraint_type=constraint_type,
            lambda_wealth=lambda_wealth,
            lambda_approval=lambda_approval,
            lambda_lr=1e-2,
        )
        agent.train(num_episodes=num_episodes)

        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = agent.get_action(obs)
            obs, _, terminated, truncated, _ = env.step(np.array([action]))
            done = terminated or truncated

        results[reward_func] = {
            "agent": agent,
            "history": env.history,
            "episode_rewards": agent.episode_rewards,
            "lambda_history": agent.lambda_history,
            "episode_metrics": agent.episode_metrics,
        }

        if save_results_flag:
            saved_file = save_results(
                results[reward_func], reward_func, constraint_type, seed
            )
            saved_files.append(saved_file)

        if save_weights:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            weights_filename = (
                f"{weights_dir}/{reward_func}_{constraint_type}_seed{seed}_{timestamp}.pt"
            )
            agent.save_model(weights_filename)

        print(f"\nSummary for {reward_func}:")
        print(f"  Initial wealth gap: ${env.history['wealth_gap'][0]:.3f}k")
        print(f"  Final wealth gap: ${env.history['wealth_gap'][-1]:.3f}k")
        print(f"  Approval disparity: {env.history['approval_disparity'][-1]:.4f}")
        print(f"  Total profit: ${sum(env.history['profit']):.3f}k")
        if reward_func != "utilitarian_profit":
            final_lambda_w, final_lambda_a = agent._get_current_lambdas()
            if constraint_type == "wealth":
                print(f"  Final λ_wealth: {final_lambda_w:.4f}")
            elif constraint_type == "approval_rate":
                print(f"  Final λ_approval: {final_lambda_a:.4f}")
            else:
                print(f"  Final λ_wealth: {final_lambda_w:.4f}")
                print(f"  Final λ_approval: {final_lambda_a:.4f}")

        if agent.episode_metrics["rho_episode"]:
            rho_values = np.array(agent.episode_metrics["rho_episode"])
            R_M_values = np.array(agent.episode_metrics["R_M"])
            R_F_values = np.array(agent.episode_metrics["R_F"])
            print(f"  --- Episode Metrics ---")
            print(
                f"  ρ mean: {np.mean(rho_values):.4f}, R_M: {R_M_values[-1]:.6f}, R_F: {R_F_values[-1]:.6f}"
            )

    return results, loader, theta_learner
