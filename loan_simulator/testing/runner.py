import glob
import os
import pickle
from datetime import datetime

import numpy as np
import torch

from ..transition_learner import TransitionParameterLearner
from .data_loader import TestingAdultIncomeDataLoader
from .environment import TestingIncomeEnvironment
from .agent import TestingAgent


def list_checkpoints(checkpoint_dir: str = "./checkpoints"):
    """List all available checkpoints and their status."""
    if not os.path.exists(checkpoint_dir):
        print(f"No checkpoint directory found at {checkpoint_dir}")
        return []

    checkpoints = glob.glob(f"{checkpoint_dir}/checkpoint_*.pkl")

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return []

    print(f"\nAvailable checkpoints in {checkpoint_dir}:")
    print("-" * 70)

    checkpoint_info = []
    for cp_path in sorted(checkpoints):
        try:
            with open(cp_path, "rb") as f:
                cp = pickle.load(f)
            info = {
                "path": cp_path,
                "episodes": cp["total_episodes"],
                "time": cp["current_time"],
                "mu_M": cp["mu_M"],
                "mu_F": cp["mu_F"],
            }
            checkpoint_info.append(info)
            print(f"  {os.path.basename(cp_path)}")
            print(f"    Episodes: {info['episodes']}, Time: {info['time']:.1f}")
            print(f"    μ_M: ${info['mu_M']:.2f}k, μ_F: ${info['mu_F']:.2f}k")
        except Exception as e:
            print(f"  {os.path.basename(cp_path)} - ERROR: {e}")

    print("-" * 70)
    return checkpoint_info


def run_testing(
    weights_path: str,
    data_filepath: str = None,
    num_episodes: int = 10,
    seed: int = 42,
    N_male: int = 1000,
    N_female: int = 1000,
    T: int = 500,
    dt: float = 0.5,
    deterministic: bool = False,
    save_results: bool = True,
    results_dir: str = "./test_results",
    credit_threshold: float = 0.5,
    checkpoint_dir: str = "./checkpoints",
    checkpoint_every: int = 1,
    resume: bool = True,
):
    """
    Run testing with a trained agent (no training, no wealth reset between episodes).
    All metrics are tracked per-episode.

    Args:
        weights_path: Path to saved model weights (.pt)
        data_filepath: Path to adult.csv (None for auto-download)
        num_episodes: Number of continuous episodes to run
        seed: Random seed
        N_male, N_female: Number of individuals per group
        T, dt: Time horizon and step size per episode
        deterministic: Use mode of Beta distribution instead of sampling
        save_results: Save episode metrics CSV and summary
        results_dir: Directory for output files
        credit_threshold: Creditworthiness threshold for ground truth approval
        checkpoint_dir: Directory for checkpoint files
        checkpoint_every: Save checkpoint every N episodes
        resume: If True, search for an existing checkpoint and resume

    Returns:
        episode_metrics_df, env, agent
    """
    print("=" * 80)
    print("TESTING TRAINED AGENT (EPISODE-WISE METRICS)")
    print("=" * 80)
    print(f"  Model: {weights_path}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Continuous play: YES (no wealth reset between episodes)")
    print(f"  Credit threshold for ground truth: {credit_threshold}")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print(f"  Resume from checkpoint: {resume}")
    print("=" * 80)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if save_results:
        os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_name = os.path.basename(weights_path).replace(".pt", "")
    checkpoint_path = f"{checkpoint_dir}/checkpoint_{model_name}_seed{seed}.pkl"

    # Load data
    print("\n[1] Loading Data...")
    loader = TestingAdultIncomeDataLoader(
        filepath=data_filepath,
        sample_size=20000,
        credit_threshold=credit_threshold,
    )
    loader.load_data()
    loader.preprocess()

    # Learn theta parameters (uses Bernoulli default distribution)
    theta_learner = TransitionParameterLearner(
        default_rate_min=0.02, default_rate_max=0.15
    )
    theta_learner.fit(loader.data)

    # Create testing environment
    env = TestingIncomeEnvironment(
        theta_params=theta_learner,
        initial_wealth_male=loader.male_data["X"].values,
        initial_wealth_female=loader.female_data["X"].values,
        ground_truth_male=loader.male_data["ground_truth_approval"].values,
        ground_truth_female=loader.female_data["ground_truth_approval"].values,
        N_male=N_male,
        N_female=N_female,
        T=T,
        dt=dt,
        seed=seed,
    )

    # Load trained agent
    agent = TestingAgent()
    agent.load_model(weights_path)

    # Resume from checkpoint if available
    start_episode = 0
    if resume and os.path.exists(checkpoint_path):
        print(f"\n[4.5] Found existing checkpoint, resuming...")
        env.load_checkpoint(checkpoint_path)
        start_episode = env.total_episodes
        print(f"  Resuming from episode {start_episode + 1}")

    if start_episode >= num_episodes:
        print(f"\n  All {num_episodes} episodes already completed!")
    else:
        remaining = num_episodes - start_episode
        print(
            f"\n[5] Running {remaining} Episodes "
            f"(Episodes {start_episode + 1} to {num_episodes})..."
        )

        for episode in range(start_episode, num_episodes):
            obs, _ = env.reset()
            done = False

            while not done:
                action = agent.get_action(obs, deterministic=deterministic)
                obs, _, terminated, truncated, _ = env.step(np.array([action]))
                done = terminated or truncated

            if (episode + 1) % 10 == 0 or episode + 1 == num_episodes:
                print(f"\n  Episode {episode + 1}/{num_episodes} Complete:")
                print(f"    Timesteps this episode: {env.episode_timesteps}")
                print(f"    Total time elapsed: {env.current_time:.1f}")
                print(f"    μ_M: ${env.mu_M:.2f}k, μ_F: ${env.mu_F:.2f}k")
                print(f"    Wealth Gap: ${env.mu_M - env.mu_F:.2f}k")
                print(
                    f"    Episode Approval Rate M: "
                    f"{env.episode_loans_M / max(env.episode_applications_M, 1):.4f}"
                )
                print(
                    f"    Episode Approval Rate F: "
                    f"{env.episode_loans_F / max(env.episode_applications_F, 1):.4f}"
                )
                print(f"    Episode Profit: ${env.episode_profit:.2f}k")
                print(f"    Cumulative Profit: ${env.cumulative_profit:.2f}k")

            if (episode + 1) % checkpoint_every == 0:
                env.save_checkpoint(checkpoint_path)

    env.finalize_episode_metrics()
    episode_metrics_df = env.get_episode_metrics_dataframe()

    # ------------------------------------------------------------------ #
    # Final summary
    # ------------------------------------------------------------------ #
    print("\n[6] Computing Final Metrics...")

    total_M = env.tp_M + env.fp_M + env.tn_M + env.fn_M
    total_F = env.tp_F + env.fp_F + env.tn_F + env.fn_F
    accuracy_M = (env.tp_M + env.tn_M) / max(total_M, 1)
    accuracy_F = (env.tp_F + env.tn_F) / max(total_F, 1)
    precision_M = env.tp_M / max(env.tp_M + env.fp_M, 1)
    precision_F = env.tp_F / max(env.tp_F + env.fp_F, 1)
    recall_M = env.tp_M / max(env.tp_M + env.fn_M, 1)
    recall_F = env.tp_F / max(env.tp_F + env.fn_F, 1)

    print("\n" + "=" * 80)
    print("FINAL TESTING RESULTS")
    print("=" * 80)
    print(f"\n  Model: {agent.reward_func_name} ({agent.constraint_type})")
    print(f"  Episodes: {num_episodes}")
    print(f"  Total Time: {env.current_time:.1f}")
    print(f"  Hawkes Events (M/F): {len(env.event_times_R)}/{len(env.event_times_B)}")

    print(f"\n  --- Wealth Metrics ---")
    print(f"  Initial μ_M,0: ${env.mu_M_0:.2f}k")
    print(f"  Initial μ_F,0: ${env.mu_F_0:.2f}k")
    print(f"  Final μ_M: ${env.mu_M:.2f}k")
    print(f"  Final μ_F: ${env.mu_F:.2f}k")
    print(f"  Final Wealth Gap: ${env.mu_M - env.mu_F:.2f}k")

    print(f"\n  --- Long-term Social Welfare R_g(t) = (μ_g,t - μ_g,0) / t ---")
    if env.current_time > 1e-8:
        R_M_final = (env.mu_M - env.mu_M_0) / env.current_time
        R_F_final = (env.mu_F - env.mu_F_0) / env.current_time
        R_total_final = (R_M_final + R_F_final) / 2
    else:
        R_M_final = R_F_final = R_total_final = 0.0
    print(f"  R_M (Male):   {R_M_final:.6f} ($1000s per time unit)")
    print(f"  R_F (Female): {R_F_final:.6f} ($1000s per time unit)")
    print(f"  R_total:      {R_total_final:.6f} ($1000s per time unit)")
    print(
        f"  R_M / R_F:    {R_M_final / R_F_final:.4f}"
        if abs(R_F_final) > 1e-8
        else "  R_M / R_F:    N/A (R_F ≈ 0)"
    )

    if len(episode_metrics_df) > 0:
        rho_values = episode_metrics_df["rho_episode"].values
        print(f"\n  --- Inequality Ratio ρ = Δμ_M / Δμ_F (per episode) ---")
        print(f"  Mean ρ:   {np.mean(rho_values):.4f}")
        print(f"  Std ρ:    {np.std(rho_values):.4f}")
        print(f"  Min ρ:    {np.min(rho_values):.4f}")
        print(f"  Max ρ:    {np.max(rho_values):.4f}")
        print(f"  Last 5 episodes: {[f'{r:.3f}' for r in rho_values[-5:]]}")

    print(f"\n  --- Approval Metrics ---")
    print(f"  Total Applications M: {env.total_applications_M}")
    print(f"  Total Applications F: {env.total_applications_F}")
    print(f"  Total Loans Approved M: {env.total_loans_M}")
    print(f"  Total Loans Approved F: {env.total_loans_F}")
    print(
        f"  Final Approval Rate M: "
        f"{env.total_loans_M / max(env.total_applications_M, 1):.4f}"
    )
    print(
        f"  Final Approval Rate F: "
        f"{env.total_loans_F / max(env.total_applications_F, 1):.4f}"
    )

    print(f"\n  --- Success Probabilities ---")
    print(f"  Total Defaults M: {env.total_defaults_M}")
    print(f"  Total Defaults F: {env.total_defaults_F}")
    print(
        f"  Final Success Prob M: "
        f"{1 - env.total_defaults_M / max(env.total_loans_M, 1):.4f}"
    )
    print(
        f"  Final Success Prob F: "
        f"{1 - env.total_defaults_F / max(env.total_loans_F, 1):.4f}"
    )

    print(f"\n  --- Ground Truth Comparison (Males) ---")
    print(f"  True Positives:  {env.tp_M}")
    print(f"  False Positives: {env.fp_M}")
    print(f"  True Negatives:  {env.tn_M}")
    print(f"  False Negatives: {env.fn_M}")
    print(f"  Accuracy:  {accuracy_M:.4f}")
    print(f"  Precision: {precision_M:.4f}")
    print(f"  Recall:    {recall_M:.4f}")

    print(f"\n  --- Ground Truth Comparison (Females) ---")
    print(f"  True Positives:  {env.tp_F}")
    print(f"  False Positives: {env.fp_F}")
    print(f"  True Negatives:  {env.tn_F}")
    print(f"  False Negatives: {env.fn_F}")
    print(f"  Accuracy:  {accuracy_F:.4f}")
    print(f"  Precision: {precision_F:.4f}")
    print(f"  Recall:    {recall_F:.4f}")

    print(f"\n  --- Profit ---")
    print(f"  Cumulative Profit: ${env.cumulative_profit:.2f}k")

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    if save_results:
        print("\n[7] Saving Results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_path = (
            f"{results_dir}/test_episode_metrics_"
            f"{agent.reward_func_name}_{agent.constraint_type}_seed{seed}_{timestamp}.csv"
        )
        episode_metrics_df.to_csv(csv_path, index=False)
        print(f"  Saved episode metrics to {csv_path}")

        # Remove checkpoint after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    return episode_metrics_df, env, agent
