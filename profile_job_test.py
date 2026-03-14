#!/usr/bin/env python3
"""
Profiling script for testing runs to diagnose performance and memory issues.

Tracks per-episode:
- Number of decisions
- Number of applicants (M/F)
- Number of approvals (M/F)
- Time per episode
- GPU memory usage
- Approval rates
- Ground truth accuracy metrics

Can detect runaway conditions and stop early.
"""

import argparse
import gc
import os
import time
from datetime import datetime

import numpy as np
import torch

# Thresholds for detecting runaway
DECISION_GROWTH_THRESHOLD = 5.0  # If decisions grow 5x from baseline, warn
MEMORY_THRESHOLD_GB = 15.0  # Stop if GPU memory exceeds this
MAX_DECISIONS_PER_EPISODE = 50000  # Stop if decisions exceed this


def get_gpu_memory_gb():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


def get_gpu_memory_reserved_gb():
    """Get reserved GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / (1024**3)
    return 0.0


def profile_test_job(
    weights_path: str,
    seed: int,
    episodes: int,
    N_male: int = 1000,
    N_female: int = 1000,
    T: int = 500,
    dt: float = 0.5,
    deterministic: bool = False,
    stop_on_runaway: bool = True,
    verbose: bool = True,
):
    """
    Profile a single testing job with detailed metrics.

    Returns dict with profiling results.
    """
    # Import here to avoid issues
    from adult_income_testing import (
        AdultIncomeDataLoader,
        TransitionParameterLearner,
        TestingIncomeEnvironment,
        TestingAgent,
    )

    print("=" * 70)
    print(f"PROFILING TEST: {weights_path}")
    print(f"Seed: {seed}, Episodes: {episodes}, Stop on runaway: {stop_on_runaway}")
    print("=" * 70)

    # Set seeds
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Initial GPU memory: {get_gpu_memory_gb():.2f} GB")

    # Load data
    print("\n[1/4] Loading data...")
    loader = AdultIncomeDataLoader(filepath=None, sample_size=20000)
    loader.load_data()
    loader.preprocess()

    # Learn parameters
    print("[2/4] Learning transition parameters...")
    theta_learner = TransitionParameterLearner(
        default_rate_min=0.02, default_rate_max=0.15
    )
    theta_learner.fit(loader.data)

    # Create testing environment
    print("[3/4] Creating testing environment...")
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
    print("[4/4] Loading trained agent...")
    agent = TestingAgent()
    agent.load_model(weights_path)

    # Profiling data
    profile_data = {
        'episode': [],
        'num_decisions': [],
        'num_applicants_M': [],
        'num_applicants_F': [],
        'num_approvals_M': [],
        'num_approvals_F': [],
        'episode_time_sec': [],
        'gpu_memory_gb': [],
        'gpu_reserved_gb': [],
        'approval_rate_M': [],
        'approval_rate_F': [],
        'mu_M': [],
        'mu_F': [],
        'wealth_gap': [],
        'cumulative_profit': [],
        # Ground truth metrics
        'accuracy_M': [],
        'accuracy_F': [],
        'precision_M': [],
        'precision_F': [],
        'recall_M': [],
        'recall_F': [],
    }

    baseline_decisions = None
    stopped_early = False
    stop_reason = None

    print("\n" + "=" * 70)
    print("TESTING WITH PROFILING")
    print("=" * 70)
    print(f"{'Ep':<5} {'Decisions':<10} {'Applicants':<14} {'Approvals':<12} {'Time':<8} {'GPU MB':<10} {'μ_M':<10} {'μ_F':<10} {'Status'}")
    print("-" * 105)

    for ep in range(episodes):
        ep_start = time.time()

        # Clear cache before episode
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Reset for new episode (soft reset - no wealth reset)
        obs, _ = env.reset()
        done = False
        step_count = 0

        # Run episode
        while not done:
            action = agent.get_action(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(np.array([action]))
            done = terminated or truncated
            step_count += 1

        # Count metrics from this episode
        num_decisions = step_count
        num_applicants_M = env.episode_applications_M
        num_applicants_F = env.episode_applications_F
        num_approvals_M = env.episode_loans_M
        num_approvals_F = env.episode_loans_F

        # Set baseline
        if baseline_decisions is None:
            baseline_decisions = max(num_decisions, 100)

        # Check for runaway
        gpu_mem = get_gpu_memory_gb()
        gpu_reserved = get_gpu_memory_reserved_gb()

        status = "OK"
        if stop_on_runaway:
            if num_decisions > MAX_DECISIONS_PER_EPISODE:
                status = "RUNAWAY:DECISIONS"
                stopped_early = True
                stop_reason = f"Decisions ({num_decisions}) exceeded {MAX_DECISIONS_PER_EPISODE}"
            elif num_decisions > baseline_decisions * DECISION_GROWTH_THRESHOLD:
                status = "RUNAWAY:GROWTH"
                stopped_early = True
                stop_reason = f"Decisions grew {num_decisions/baseline_decisions:.1f}x from baseline {baseline_decisions}"
            elif gpu_mem > MEMORY_THRESHOLD_GB:
                status = "RUNAWAY:MEMORY"
                stopped_early = True
                stop_reason = f"GPU memory ({gpu_mem:.1f}GB) exceeded {MEMORY_THRESHOLD_GB}GB"

        if stopped_early:
            ep_time = time.time() - ep_start
            print(f"{ep:<5} {num_decisions:<10} {num_applicants_M}M/{num_applicants_F}F    {num_approvals_M}M/{num_approvals_F}F    {ep_time:<8.1f} {gpu_mem*1024:<10.0f} {env.mu_M:<10.2f} {env.mu_F:<10.2f} {status}")
            print(f"\n*** STOPPING EARLY: {stop_reason}")
            break

        ep_time = time.time() - ep_start

        # Get approval rates
        approval_rate_M = num_approvals_M / max(num_applicants_M, 1)
        approval_rate_F = num_approvals_F / max(num_applicants_F, 1)

        # Compute accuracy metrics
        total_decisions_M = env.tp_M + env.fp_M + env.tn_M + env.fn_M
        total_decisions_F = env.tp_F + env.fp_F + env.tn_F + env.fn_F

        accuracy_M = (env.tp_M + env.tn_M) / max(total_decisions_M, 1)
        accuracy_F = (env.tp_F + env.tn_F) / max(total_decisions_F, 1)

        precision_M = env.tp_M / max(env.tp_M + env.fp_M, 1)
        precision_F = env.tp_F / max(env.tp_F + env.fp_F, 1)

        recall_M = env.tp_M / max(env.tp_M + env.fn_M, 1)
        recall_F = env.tp_F / max(env.tp_F + env.fn_F, 1)

        # Store profiling data
        profile_data['episode'].append(ep)
        profile_data['num_decisions'].append(num_decisions)
        profile_data['num_applicants_M'].append(num_applicants_M)
        profile_data['num_applicants_F'].append(num_applicants_F)
        profile_data['num_approvals_M'].append(num_approvals_M)
        profile_data['num_approvals_F'].append(num_approvals_F)
        profile_data['episode_time_sec'].append(ep_time)
        profile_data['gpu_memory_gb'].append(gpu_mem)
        profile_data['gpu_reserved_gb'].append(gpu_reserved)
        profile_data['approval_rate_M'].append(approval_rate_M)
        profile_data['approval_rate_F'].append(approval_rate_F)
        profile_data['mu_M'].append(env.mu_M)
        profile_data['mu_F'].append(env.mu_F)
        profile_data['wealth_gap'].append(env.mu_M - env.mu_F)
        profile_data['cumulative_profit'].append(env.cumulative_profit)
        profile_data['accuracy_M'].append(accuracy_M)
        profile_data['accuracy_F'].append(accuracy_F)
        profile_data['precision_M'].append(precision_M)
        profile_data['precision_F'].append(precision_F)
        profile_data['recall_M'].append(recall_M)
        profile_data['recall_F'].append(recall_F)

        # Print progress
        if ep % 1 == 0 or ep < 10:  # Print every episode for profiling
            print(f"{ep:<5} {num_decisions:<10} {num_applicants_M}M/{num_applicants_F}F    {num_approvals_M}M/{num_approvals_F}F    {ep_time:<8.1f} {gpu_mem*1024:<10.0f} {env.mu_M:<10.2f} {env.mu_F:<10.2f} {status}")

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("PROFILING SUMMARY")
    print("=" * 70)

    if profile_data['episode']:
        print(f"Episodes completed: {len(profile_data['episode'])}")
        print(f"Stopped early: {stopped_early} ({stop_reason if stop_reason else 'N/A'})")
        print(f"\nDecisions per episode:")
        print(f"  First: {profile_data['num_decisions'][0]}")
        print(f"  Last:  {profile_data['num_decisions'][-1]}")
        print(f"  Max:   {max(profile_data['num_decisions'])}")
        print(f"  Growth: {profile_data['num_decisions'][-1] / max(profile_data['num_decisions'][0], 1):.1f}x")

        print(f"\nApplicants per episode:")
        print(f"  Male   - First: {profile_data['num_applicants_M'][0]}, Last: {profile_data['num_applicants_M'][-1]}, Max: {max(profile_data['num_applicants_M'])}")
        print(f"  Female - First: {profile_data['num_applicants_F'][0]}, Last: {profile_data['num_applicants_F'][-1]}, Max: {max(profile_data['num_applicants_F'])}")

        print(f"\nTime per episode:")
        print(f"  First: {profile_data['episode_time_sec'][0]:.1f}s")
        print(f"  Last:  {profile_data['episode_time_sec'][-1]:.1f}s")
        print(f"  Total: {sum(profile_data['episode_time_sec']):.1f}s")

        print(f"\nGPU Memory:")
        print(f"  First: {profile_data['gpu_memory_gb'][0]*1024:.0f} MB")
        print(f"  Last:  {profile_data['gpu_memory_gb'][-1]*1024:.0f} MB")
        print(f"  Max:   {max(profile_data['gpu_memory_gb'])*1024:.0f} MB")

        print(f"\nWealth (final):")
        print(f"  μ_M: ${profile_data['mu_M'][-1]:.2f}k")
        print(f"  μ_F: ${profile_data['mu_F'][-1]:.2f}k")
        print(f"  Gap: ${profile_data['wealth_gap'][-1]:.2f}k")

        print(f"\nApproval rates (final episode):")
        print(f"  Male:   {profile_data['approval_rate_M'][-1]:.4f}")
        print(f"  Female: {profile_data['approval_rate_F'][-1]:.4f}")

        print(f"\nAccuracy vs ground truth (cumulative):")
        print(f"  Male   - Accuracy: {profile_data['accuracy_M'][-1]:.4f}, Precision: {profile_data['precision_M'][-1]:.4f}, Recall: {profile_data['recall_M'][-1]:.4f}")
        print(f"  Female - Accuracy: {profile_data['accuracy_F'][-1]:.4f}, Precision: {profile_data['precision_F'][-1]:.4f}, Recall: {profile_data['recall_F'][-1]:.4f}")

        print(f"\nProfit:")
        print(f"  Cumulative: ${profile_data['cumulative_profit'][-1]:.2f}k")

    return {
        'profile_data': profile_data,
        'stopped_early': stopped_early,
        'stop_reason': stop_reason,
        'agent': agent,
        'env': env,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Profile a testing job to diagnose performance and memory issues',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile testing with a trained model
  uv run python profile_job_test.py --weights ./models/social_welfare_approval_rate_seed1.pt --seed 1 --episodes 50

  # Profile with different population sizes
  uv run python profile_job_test.py --weights ./models/model.pt --seed 1 --episodes 50 --N_male 2000 --N_female 2000

  # Profile without stopping on runaway (may OOM)
  uv run python profile_job_test.py --weights ./models/model.pt --seed 1 --episodes 100 --no-stop

  # Profile with deterministic policy
  uv run python profile_job_test.py --weights ./models/model.pt --seed 1 --episodes 50 --deterministic
        """
    )

    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model weights (.pt file)')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Number of episodes to profile (default: 50)')
    parser.add_argument('--N_male', type=int, default=3000,
                        help='Number of male individuals (default: 1000)')
    parser.add_argument('--N_female', type=int, default=3000,
                        help='Number of female individuals (default: 1000)')
    parser.add_argument('--T', type=int, default=100,
                        help='Time horizon (default: 500)')
    parser.add_argument('--dt', type=float, default=0.5,
                        help='Time step (default: 0.5)')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic policy (mode instead of sampling)')
    parser.add_argument('--no-stop', action='store_true',
                        help='Do not stop on runaway detection (may OOM)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save profiling data to this file (numpy .npz)')

    args = parser.parse_args()

    results = profile_test_job(
        weights_path=args.weights,
        seed=args.seed,
        episodes=args.episodes,
        N_male=args.N_male,
        N_female=args.N_female,
        T=args.T,
        dt=args.dt,
        deterministic=args.deterministic,
        stop_on_runaway=not args.no_stop,
    )

    if args.save:
        np.savez(args.save, **results['profile_data'])
        print(f"\nSaved profiling data to {args.save}")


if __name__ == '__main__':
    main()
