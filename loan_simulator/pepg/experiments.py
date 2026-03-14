import os
import random

import numpy as np
import torch

from ..data_loader import AdultIncomeDataLoader
from ..environment import IncomeEnvironment
from ..io_utils import save_episode_metrics, save_lambda_history
from ..plotting import plot_episode_metrics, plot_lambda_trajectory
from ..transition_learner import TransitionParameterLearner
from .agent import PePGAgentV2
from .plotting import _plot_pepg_v2_results, plot_loss_propagation, plot_loss_summary


def run_pepg_v2_experiment(
    data_filepath: str = None,
    num_episodes: int = 100,
    seed: int = 0,
    reward_function: str = "social_welfare",
    constraint_type: str = "wealth",
    lambda_wealth: float = 2.0,
    lambda_approval: float = 2.0,
    lambda_lr: float = 1e-2,
    buffer_capacity: int = 50,
    warmup_episodes: int = 0,
    hawkes_weight: float = 1.0,
    wealth_weight: float = 1.0,
    transition_weight: float = 1.0,
    reward_weight: float = 1.0,
    save_weights: bool = True,
    weights_dir: str = "./weights_pepg_v2",
    save_lambda: bool = True,
    lambda_dir: str = "./lambda_trajectories_pepg_v2",
    plot_lambda: bool = True,
    save_episode_metrics_flag: bool = True,
    episode_metrics_dir: str = "./episode_metrics_pepg_v2",
    plot_episode_metrics_flag: bool = True,
    plot_results: bool = True,
    plot_loss: bool = True,
    loss_dir: str = "./loss_plots_pepg_v2",
):
    """Run PePG V2 experiment."""
    os.makedirs(weights_dir, exist_ok=True)

    print("=" * 80)
    print(f"PERFORMATIVE POLICY GRADIENT V2 (Explicit Hawkes) EXPERIMENT")
    print(f"  Reward: {reward_function.upper()}")
    print(f"  Constraint: {constraint_type.upper()}")
    print(f"  Seed: {seed}")
    print(f"  Warmup: {warmup_episodes} episodes")
    print("=" * 80)

    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        print(f"\nCUDA Available: {torch.cuda.get_device_name(0)}")
    else:
        print("\nUsing CPU")

    # Load data
    print(f"\n[STEP 1] Loading Adult Income Data...")
    loader = AdultIncomeDataLoader(filepath=data_filepath, sample_size=20000)
    loader.load_data()
    loader.preprocess()

    # Learn theta parameters
    print(f"\n[STEP 2] Learning θ Parameters...")
    theta_learner = TransitionParameterLearner(
        default_rate_min=0.02, default_rate_max=0.15
    )
    theta_learner.fit(loader.data)

    # Create environment
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

    # Create PePG V2 agent
    print(f"\n[STEP 4] Creating PePG V2 Agent...")
    agent = PePGAgentV2(
        env,
        reward_function=reward_function,
        constraint_type=constraint_type,
        lambda_wealth=lambda_wealth,
        lambda_approval=lambda_approval,
        lambda_lr=lambda_lr,
        buffer_capacity=buffer_capacity,
        warmup_episodes=warmup_episodes,
        alpha_R=env.alpha_R,
        alpha_B=env.alpha_B,
        beta_R=env.beta_R,
        beta_B=env.beta_B,
        hawkes_weight=hawkes_weight,
        wealth_weight=wealth_weight,
        transition_weight=transition_weight,
        reward_weight=reward_weight,
    )

    # Train
    print(f"\n[STEP 5] Training PePG V2 Agent...")
    agent.train(num_episodes=num_episodes, use_performative=True)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Final wealth gap: ${env.history['wealth_gap'][-1]:.3f}k")
    print(f"  Approval disparity: {env.history['approval_disparity'][-1]:.4f}")
    print(f"  Total profit: ${sum(env.history['profit']):.3f}k")

    if agent.episode_metrics["rho_episode"]:
        rho_values = np.array(agent.episode_metrics["rho_episode"])
        print(f"  Mean ρ: {np.mean(rho_values):.4f}")

    # Save weights
    if save_weights:
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = (
            f"{weights_dir}/pepg_{reward_function}_{constraint_type}_seed{seed}.pt"
        )
        agent.save_model(weights_path)

    # Save lambda trajectories
    if save_lambda and reward_function != "utilitarian_profit":
        os.makedirs(lambda_dir, exist_ok=True)
        save_lambda_history(
            agent.lambda_history,
            f"pepg_v2_{reward_function}",
            constraint_type,
            seed,
            save_dir=lambda_dir,
            format="both",
        )

    # Plot lambda trajectories
    if plot_lambda and reward_function != "utilitarian_profit":
        os.makedirs(lambda_dir, exist_ok=True)
        lambda_plot_path = (
            f"{lambda_dir}/pepg_v2_{reward_function}_{constraint_type}_seed{seed}_lambda.png"
        )
        plot_lambda_trajectory(
            agent.lambda_history,
            f"PePG_v2 {reward_function}",
            constraint_type,
            seed,
            save_path=lambda_plot_path,
            show=True,
        )

    # Save episode metrics
    if save_episode_metrics_flag:
        os.makedirs(episode_metrics_dir, exist_ok=True)
        save_episode_metrics(
            agent.episode_metrics,
            f"pepg_v2_{reward_function}",
            constraint_type,
            seed,
            save_dir=episode_metrics_dir,
            format="both",
        )

    # Plot episode metrics
    if plot_episode_metrics_flag:
        os.makedirs(episode_metrics_dir, exist_ok=True)
        episode_plot_path = (
            f"{episode_metrics_dir}/pepg_v2_{reward_function}_{constraint_type}_seed{seed}_metrics.png"
        )
        plot_episode_metrics(
            agent.episode_metrics,
            f"PePG_v2 {reward_function}",
            constraint_type,
            seed,
            save_path=episode_plot_path,
            show=True,
        )

    # Plot gradient decomposition results
    if plot_results:
        _plot_pepg_v2_results(
            agent, reward_function, constraint_type, seed, weights_dir
        )

    # Plot loss propagation
    if plot_loss:
        os.makedirs(loss_dir, exist_ok=True)
        loss_plot_path = (
            f"{loss_dir}/pepg_v2_{reward_function}_{constraint_type}_seed{seed}_loss.png"
        )
        plot_loss_propagation(
            agent.loss_history,
            reward_function=reward_function,
            constraint_type=constraint_type,
            seed=seed,
            save_path=loss_plot_path,
            show=True,
        )

        loss_summary_path = (
            f"{loss_dir}/pepg_v2_{reward_function}_{constraint_type}_seed{seed}_loss_summary.png"
        )
        plot_loss_summary(
            agent.loss_history,
            save_path=loss_summary_path,
            show=True,
        )

    return agent, env, theta_learner, loader


def load_pepg_v2_agent(
    weights_path: str, data_filepath: str = None, seed: int = 0
):
    """Load a trained PePG V2 agent."""
    print(f"Loading PePG V2 agent from {weights_path}...")

    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location="cpu")

    reward_function = checkpoint.get("reward_function", "social_welfare")
    constraint_type = checkpoint.get("constraint_type", "wealth")

    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Load data
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

    agent = PePGAgentV2(
        env,
        reward_function=reward_function,
        constraint_type=constraint_type,
        lambda_wealth=checkpoint.get("final_lambda_wealth", 2.0),
        lambda_approval=checkpoint.get("final_lambda_approval", 2.0),
        buffer_capacity=checkpoint.get("buffer_capacity", 50),
        warmup_episodes=checkpoint.get("warmup_episodes", 0),
        alpha_R=checkpoint.get("alpha_R", 0.3),
        alpha_B=checkpoint.get("alpha_B", 0.3),
        beta_R=checkpoint.get("beta_R", 2.0),
        beta_B=checkpoint.get("beta_B", 2.0),
        hawkes_weight=checkpoint.get("hawkes_weight", 1.0),
        wealth_weight=checkpoint.get("wealth_weight", 1.0),
    )

    agent.load_model(weights_path)

    return agent, env, theta_learner, loader
