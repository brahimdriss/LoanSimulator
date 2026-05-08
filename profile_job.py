#!/usr/bin/env python3

import argparse
import gc
import os
import sys
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


def profile_job(reward: str, constraint: str, seed: int, episodes: int,
                stop_on_runaway: bool = True, verbose: bool = True):
    """
    Profile a single training job with detailed metrics.

    Returns dict with profiling results.
    """
    # Import here to avoid issues
    from par_pepg import (
        PePGAgentV2,
        run_pepg_v2_experiment,
    )
    from adult_income_training import (
        AdultIncomeDataLoader,
        TransitionParameterLearner,
        IncomeEnvironment,
    )

    print("=" * 70)
    print(f"PROFILING: {reward} + {constraint} + seed{seed}")
    print(f"Episodes: {episodes}, Stop on runaway: {stop_on_runaway}")
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

    # Create environment
    print("[3/4] Creating environment...")
    env = IncomeEnvironment(
        theta_params=theta_learner,
        initial_wealth_male=loader.male_data['X'].values,
        initial_wealth_female=loader.female_data['X'].values,
        N_male=3000,
        N_female=3000,
        T=100, # TODO
        dt=0.5,
        seed=seed
    )

    # Create agent
    print("[4/4] Creating agent...")
    agent = PePGAgentV2(
        env,
        reward_function=reward,
        constraint_type=constraint,
        lambda_wealth=2.0,
        lambda_approval=2.0,
        lambda_lr=1e-2,
        buffer_capacity=50,
        warmup_episodes=0,
        alpha_R=env.alpha_R,
        alpha_B=env.alpha_B,
        beta_R=env.beta_R,
        beta_B=env.beta_B,
    )

    # Profiling data
    profile_data = {
        'episode': [],
        'num_decisions': [],
        'num_applicants_M': [],
        'num_applicants_F': [],
        'num_approvals_M': [],
        'num_approvals_F': [],
        'num_hawkes_events_M': [],
        'num_hawkes_events_F': [],
        'episode_time_sec': [],
        'gpu_memory_gb': [],
        'gpu_reserved_gb': [],
        'reward': [],
        'grad_H_norm': [],
        'grad_W_norm': [],
        'approval_rate_M': [],
        'approval_rate_F': [],
    }

    baseline_decisions = None
    stopped_early = False
    stop_reason = None

    print("\n" + "=" * 70)
    print("TRAINING WITH PROFILING")
    print("=" * 70)
    print(f"{'Ep':<5} {'Decisions':<10} {'Applicants':<14} {'Approvals':<12} {'Hawkes':<12} {'Time':<8} {'GPU MB':<10} {'Reward':<12} {'Status'}")
    print("-" * 105)

    for ep in range(episodes):
        ep_start = time.time()

        # Clear cache before episode
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Collect episode with tracking
        episode_data = agent._collect_episode(use_soft_reset=True)
        decisions = episode_data['decisions']

        # Count metrics
        num_decisions = len(decisions)
        num_applicants_M = sum(1 for d in decisions if d['group'] == 'male')
        num_applicants_F = sum(1 for d in decisions if d['group'] == 'female')
        num_approvals_M = sum(1 for d in decisions if d['approved'] and d['group'] == 'male')
        num_approvals_F = sum(1 for d in decisions if d['approved'] and d['group'] == 'female')
        num_hawkes_M = len(agent.decision_tracker.hawkes_events_R)
        num_hawkes_F = len(agent.decision_tracker.hawkes_events_B)

        # Set baseline
        if baseline_decisions is None:
            baseline_decisions = max(num_decisions, 100)

        # Check for runaway BEFORE gradient computation
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
            print(f"{ep:<5} {num_decisions:<10} {num_applicants_M}M/{num_applicants_F}F    {num_approvals_M}M/{num_approvals_F}F    {num_hawkes_M}M/{num_hawkes_F}F    {ep_time:<8.1f} {gpu_mem*1024:<10.0f} {'N/A':<12} {status}")
            print(f"\n*** STOPPING EARLY: {stop_reason}")
            break

        # Compute gradients (this is where OOM usually happens)
        try:
            agent.optimizer.zero_grad()
            total_grad, grad_info = agent._compute_full_pepg_gradient(decisions)

            # Apply gradient
            if total_grad.numel() > 0:
                idx = 0
                for p in agent.policy_net.parameters():
                    numel = p.numel()
                    if idx + numel <= total_grad.numel():
                        p.grad = total_grad[idx:idx+numel].view(p.shape)
                    idx += numel
                torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 1.0)
                agent.optimizer.step()

            # Update lambdas
            if agent.learnable_lambdas is not None:
                agent._update_lambdas()

            grad_H = grad_info['hawkes_norm']
            grad_W = grad_info['wealth_norm']

        except torch.cuda.OutOfMemoryError as e:
            status = "OOM"
            stopped_early = True
            stop_reason = f"CUDA OOM at episode {ep}: {e}"
            grad_H = -1
            grad_W = -1
            print(f"\n*** OOM ERROR: {stop_reason}")
            break

        # Record episode reward
        episode_reward = sum(episode_data['rewards'])
        agent.episode_rewards.append(episode_reward)
        agent._update_episode_metrics(episode_data)

        ep_time = time.time() - ep_start

        # Get approval rates
        approval_rate_M = env.total_loans_R / max(env.total_applications_R, 1)
        approval_rate_F = env.total_loans_B / max(env.total_applications_B, 1)

        # Store profiling data
        profile_data['episode'].append(ep)
        profile_data['num_decisions'].append(num_decisions)
        profile_data['num_applicants_M'].append(num_applicants_M)
        profile_data['num_applicants_F'].append(num_applicants_F)
        profile_data['num_approvals_M'].append(num_approvals_M)
        profile_data['num_approvals_F'].append(num_approvals_F)
        profile_data['num_hawkes_events_M'].append(num_hawkes_M)
        profile_data['num_hawkes_events_F'].append(num_hawkes_F)
        profile_data['episode_time_sec'].append(ep_time)
        profile_data['gpu_memory_gb'].append(gpu_mem)
        profile_data['gpu_reserved_gb'].append(gpu_reserved)
        profile_data['reward'].append(episode_reward)
        profile_data['grad_H_norm'].append(grad_H)
        profile_data['grad_W_norm'].append(grad_W)
        profile_data['approval_rate_M'].append(approval_rate_M)
        profile_data['approval_rate_F'].append(approval_rate_F)

        # Print progress
        if ep % 1 == 0 or ep < 10:  # Print every episode for profiling
            print(f"{ep:<5} {num_decisions:<10} {num_applicants_M}M/{num_applicants_F}F    {num_approvals_M}M/{num_approvals_F}F    {num_hawkes_M}M/{num_hawkes_F}F    {ep_time:<8.1f} {gpu_mem*1024:<10.0f} {episode_reward:<12.0f} {status}")

        # Cleanup
        del episode_data
        del decisions
        del total_grad
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
        print(f"  Growth: {profile_data['num_decisions'][-1] / profile_data['num_decisions'][0]:.1f}x")

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

        print(f"\nApproval rates (final):")
        print(f"  Male:   {profile_data['approval_rate_M'][-1]:.4f}")
        print(f"  Female: {profile_data['approval_rate_F'][-1]:.4f}")

    return {
        'profile_data': profile_data,
        'stopped_early': stopped_early,
        'stop_reason': stop_reason,
        'agent': agent,
        'env': env,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Profile a PePG training job to diagnose OOM issues',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile a known failing job (rawlsian_maximin)
  uv run python profile_job.py --reward rawlsian_maximin --constraint approval_rate --seed 1 --episodes 50

  # Profile social_welfare seed5 which fails
  uv run python profile_job.py --reward social_welfare --constraint wealth --seed 5 --episodes 50

  # Profile without stopping on runaway (dangerous - may OOM)
  uv run python profile_job.py --reward rawlsian_maximin --constraint approval_rate --seed 1 --episodes 100 --no-stop

  # Profile a successful job for comparison
  uv run python profile_job.py --reward social_welfare --constraint approval_rate --seed 1 --episodes 50
        """
    )

    parser.add_argument('--reward', type=str, required=True,
                        choices=['social_welfare', 'rawlsian_maximin', 'fairness_lagrangian', 'utilitarian_profit'],
                        help='Reward function')
    parser.add_argument('--constraint', type=str, required=True,
                        choices=['approval_rate', 'wealth', 'both'],
                        help='Constraint type')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of episodes to profile (default: 50)')
    parser.add_argument('--no-stop', action='store_true',
                        help='Do not stop on runaway detection (may OOM)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save profiling data to this file (numpy .npz)')

    args = parser.parse_args()

    results = profile_job(
        reward=args.reward,
        constraint=args.constraint,
        seed=args.seed,
        episodes=args.episodes,
        stop_on_runaway=not args.no_stop,
    )

    if args.save:
        import numpy as np
        np.savez(args.save, **results['profile_data'])
        print(f"\nSaved profiling data to {args.save}")


if __name__ == '__main__':
    main()