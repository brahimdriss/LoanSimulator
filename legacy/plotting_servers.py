#!/usr/bin/env python3

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
from typing import Dict, List, Optional


# Configuration
JOB_MATRIX = {
    "rewards": [
        "social_welfare",
        "rawlsian_maximin",
        "fairness_lagrangian",
        "utilitarian_profit",
    ],
    "constraints": ["approval_rate", "wealth", "both"],
    "seeds": range(1, 2),
}

REWARD_LABELS = {
    "social_welfare": "Social Welfare",
    "rawlsian_maximin": "Rawlsian Maximin",
    "fairness_lagrangian": "Fairness Lagrangian",
    "utilitarian_profit": "Utilitarian Profit",
}

REWARD_SHORT_LABELS = {
    "social_welfare": "SW",
    "rawlsian_maximin": "RM",
    "fairness_lagrangian": "FL",
    "utilitarian_profit": "UP",
}

COLORS = {
    "social_welfare": "#1f77b4",
    "rawlsian_maximin": "#ff7f0e",
    "fairness_lagrangian": "#2ca02c",
    "utilitarian_profit": "#d62728",
}

LINESTYLES = {
    "approval_rate": "-",
    "wealth": "--",
    "both": ":",
}

CONSTRAINT_LABELS = {
    "approval_rate": "Approval Rate",
    "wealth": "Wealth",
    "both": "Both",
}


def setup_plot_style():
    """Setup matplotlib style with Liberation Serif font."""
    plt.rcParams.update({
        # 'font.family': 'serif',
        'font.family': ['Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 8,
        'legend.framealpha': 0.9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 200,
        'savefig.dpi': 200,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.axisbelow': True,
    })


def parse_filename(filename: str) -> Optional[Dict]:
    """
    Parse a result filename to extract reward, constraint, and seed.

    Expected format: pg_test_v2_{reward}_{constraint}_seed{N}_{timestamp}_episode_metrics.csv
    """
    name = os.path.basename(filename)
    if not name.endswith('.csv'):
        return None
    name = name[:-4]  # Remove .csv

    if not name.startswith('pg_test_v2_'):
        return None
    name = name[len('pg_test_v2_'):]

    # Remove _episode_metrics suffix
    if name.endswith('_episode_metrics'):
        name = name[:-len('_episode_metrics')]

    # Try to match each reward type
    for reward in JOB_MATRIX["rewards"]:
        if name.startswith(reward + "_"):
            remaining = name[len(reward) + 1:]

            # Try to match each constraint
            for constraint in JOB_MATRIX["constraints"]:
                if remaining.startswith(constraint + "_"):
                    seed_part = remaining[len(constraint) + 1:]

                    # Extract seed number
                    seed_match = re.match(r'seed(\d+)', seed_part)
                    if seed_match:
                        seed = int(seed_match.group(1))
                        return {
                            'reward': reward,
                            'constraint': constraint,
                            'seed': seed
                        }
    return None


def load_results(directory_paths: List[str]) -> Dict:
    """
    Load all result files from multiple server directories.

    Returns a nested dict: {reward: {constraint: {seed: DataFrame}}}
    """
    results = defaultdict(lambda: defaultdict(dict))

    for directory_path in directory_paths:
        server_name = os.path.basename(os.path.normpath(directory_path))
        pattern = os.path.join(directory_path, "pg_test_v2_*.csv")
        files = glob.glob(pattern)

        print(f"Found {len(files)} CSV files in {server_name}")

        for filepath in files:
            parsed = parse_filename(filepath)
            if parsed:
                seed = parsed['seed']
                reward = parsed['reward']
                constraint = parsed['constraint']

                if seed in results[reward][constraint]:
                    print(f"  Warning: duplicate seed {seed} for {reward}/{constraint} "
                          f"(from {server_name}), keeping first occurrence")
                    continue

                try:
                    df = pd.read_csv(filepath)
                    results[reward][constraint][seed] = df
                    print(f"  Loaded: {reward} / {constraint} / seed{seed} ({server_name})")
                except Exception as e:
                    print(f"  Error loading {filepath}: {e}")

    return results


def aggregate_across_seeds(results: Dict) -> Dict:
    """
    Aggregate results across seeds for each (reward, constraint) pair.
    Returns dict with mean and std DataFrames.
    """
    aggregated = defaultdict(dict)

    for reward in results:
        for constraint in results[reward]:
            dfs = list(results[reward][constraint].values())
            seeds = list(results[reward][constraint].keys())

            if len(dfs) > 0:
                # Find minimum length across seeds (in case of different episode counts)
                min_len = min(len(df) for df in dfs)
                dfs_trimmed = [df.iloc[:min_len] for df in dfs]

                # Stack and compute statistics
                stacked = np.stack([df.select_dtypes(include=[np.number]).values
                                   for df in dfs_trimmed], axis=0)

                mean_values = np.mean(stacked, axis=0)
                std_values = np.std(stacked, axis=0)

                # Create mean DataFrame
                mean_df = pd.DataFrame(mean_values,
                                       columns=dfs_trimmed[0].select_dtypes(include=[np.number]).columns)
                std_df = pd.DataFrame(std_values,
                                      columns=dfs_trimmed[0].select_dtypes(include=[np.number]).columns)

                aggregated[reward][constraint] = {
                    'mean': mean_df,
                    'std': std_df,
                    'n_seeds': len(dfs),
                    'seeds': seeds,
                    'n_episodes': min_len
                }

                print(f"  {REWARD_LABELS[reward]} / {constraint}: "
                      f"{len(dfs)} seed(s), {min_len} episodes")

    return aggregated


def get_reward_aggregated_data(aggregated: Dict, metric: str, final_only: bool = False):
    """
    Get data aggregated across constraints for each reward.

    Returns:
        If final_only: dict {reward: (mean, std)}
        Else: dict {reward: (mean_trajectory, std_trajectory)}
    """
    data = {}

    for reward in JOB_MATRIX["rewards"]:
        if reward not in aggregated:
            continue

        values = []
        for constraint in aggregated[reward]:
            df_mean = aggregated[reward][constraint]['mean']
            if final_only:
                values.append(df_mean[metric].iloc[-1])
            else:
                values.append(df_mean[metric].values)

        if values:
            if final_only:
                data[reward] = (np.mean(values), np.std(values))
            else:
                # Align trajectories to minimum length
                min_len = min(len(v) for v in values)
                values_aligned = [v[:min_len] for v in values]
                mean_traj = np.mean(values_aligned, axis=0)
                std_traj = np.std(values_aligned, axis=0)
                data[reward] = (mean_traj, std_traj)

    return data


def create_plot_1(aggregated: Dict, constraint: str, output_dir: str, use_log_scale: bool = False, zoom: bool = False):
    """
    Create first plot for a specific constraint with 3 subplots:
    1. Approval Rate Disparity (M-F) - Bar graph (final)
    2. Approval Rate Disparity (M-F) - Trajectory
    3. Wealth Gap Disparity (M-F) - Bar graph (final)

    All values are aggregated across seeds.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    rewards = JOB_MATRIX["rewards"]
    x = np.arange(len(rewards))
    bar_width = 0.6

    # ========== Subplot 1: Approval Rate Disparity Bar Graph ==========
    ax = axes[0]

    disparities = []
    disparities_std = []

    for reward in rewards:
        if reward in aggregated and constraint in aggregated[reward]:
            df_mean = aggregated[reward][constraint]['mean']
            df_std = aggregated[reward][constraint]['std']

            disparity = df_mean['approval_rate_M_episode'].iloc[-1] - df_mean['approval_rate_F_episode'].iloc[-1]
            disp_std = np.sqrt(df_std['approval_rate_M_episode'].iloc[-1]**2 +
                              df_std['approval_rate_F_episode'].iloc[-1]**2)
            disparities.append(disparity)
            disparities_std.append(disp_std)
        else:
            disparities.append(0)
            disparities_std.append(0)

    bars = ax.bar(x, disparities, bar_width,
                  color=[COLORS[r] for r in rewards],
                  edgecolor='black', linewidth=0.5,
                  yerr=disparities_std, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([REWARD_LABELS[r] for r in rewards], rotation=30, ha='right')
    ax.set_ylabel('Approval Rate Disparity (M-F)')
    ax.set_title('(a) Final Approval Rate Disparity')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    if use_log_scale:
        ax.set_yscale('symlog', linthresh=1e-4, linscale=0.5)
    elif zoom and disparities:
        # Zoom to data range with 20% padding
        valid_vals = [d for d in disparities if d != 0]
        if valid_vals:
            data_min = min(valid_vals) - max(disparities_std)
            data_max = max(valid_vals) + max(disparities_std)
            padding = 0.2 * (data_max - data_min) if data_max != data_min else 0.1
            ax.set_ylim(data_min - padding, data_max + padding)

    # ========== Subplot 2: Approval Rate Disparity Trajectory ==========
    ax = axes[1]

    for reward in rewards:
        if reward not in aggregated or constraint not in aggregated[reward]:
            continue

        df_mean = aggregated[reward][constraint]['mean']
        df_std = aggregated[reward][constraint]['std']

        disparity_mean = df_mean['approval_rate_M_episode'] - df_mean['approval_rate_F_episode']
        disparity_std = np.sqrt(df_std['approval_rate_M_episode']**2 + df_std['approval_rate_F_episode']**2)

        episodes = np.arange(len(disparity_mean))

        line, = ax.plot(episodes, disparity_mean,
                       label=f"{REWARD_LABELS[reward]}",
                       color=COLORS[reward],
                       linewidth=1.5)

        ax.fill_between(episodes,
                       disparity_mean - disparity_std,
                       disparity_mean + disparity_std,
                       color=COLORS[reward], alpha=0.15)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Approval Rate Disparity (M-F)')
    ax.set_title('(b) Approval Rate Disparity Trajectory')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(loc='best', fontsize=7)

    if use_log_scale:
        ax.set_yscale('symlog', linthresh=1e-4, linscale=0.5)

    # ========== Subplot 3: Wealth Gap Bar Graph ==========
    ax = axes[2]

    wealth_gaps = []
    wealth_gaps_std = []

    for reward in rewards:
        if reward in aggregated and constraint in aggregated[reward]:
            df_mean = aggregated[reward][constraint]['mean']
            df_std = aggregated[reward][constraint]['std']
            wealth_gaps.append(df_mean['wealth_gap'].iloc[-1])
            wealth_gaps_std.append(df_std['wealth_gap'].iloc[-1])
        else:
            wealth_gaps.append(0)
            wealth_gaps_std.append(0)

    bars = ax.bar(x, wealth_gaps, bar_width,
                  color=[COLORS[r] for r in rewards],
                  edgecolor='black', linewidth=0.5,
                  yerr=wealth_gaps_std, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([REWARD_LABELS[r] for r in rewards], rotation=30, ha='right')
    ax.set_ylabel('Wealth Gap (M-F)')
    ax.set_title('(c) Final Wealth Gap Disparity')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'plot_1_{constraint}_approval_wealth_disparity.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_plot_2(aggregated: Dict, constraint: str, output_dir: str, use_log_scale: bool = False, zoom: bool = False):
    """
    Create second plot for a specific constraint with 3 subplots:
    1. Wealth Gap Disparity (M-F) - Trajectory
    2. Inequality Ratio (rho) - Bar graph (final)
       Formula: rho = (mu_M_end[last] - mu_M_start[first]) / (mu_F_end[last] - mu_F_start[first])
    3. Inequality Ratio (rho_t) - Trajectory
       Formula: rho(t) = (mu_M_end[t] - mu_M_start[first]) / (mu_F_end[t] - mu_F_start[first])

    All values are aggregated across seeds.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    rewards = JOB_MATRIX["rewards"]
    x = np.arange(len(rewards))
    bar_width = 0.6

    # ========== Subplot 1: Wealth Gap Trajectory ==========
    ax = axes[0]

    for reward in rewards:
        if reward not in aggregated or constraint not in aggregated[reward]:
            continue

        df_mean = aggregated[reward][constraint]['mean']
        df_std = aggregated[reward][constraint]['std']

        wealth_gap_mean = df_mean['wealth_gap']
        wealth_gap_std = df_std['wealth_gap']

        episodes = np.arange(len(wealth_gap_mean))

        ax.plot(episodes, wealth_gap_mean,
               label=f"{REWARD_LABELS[reward]}",
               color=COLORS[reward],
               linewidth=1.5)

        ax.fill_between(episodes,
                       wealth_gap_mean - wealth_gap_std,
                       wealth_gap_mean + wealth_gap_std,
                       color=COLORS[reward], alpha=0.15)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Wealth Gap (M-F)')
    ax.set_title('(a) Wealth Gap Trajectory')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(loc='best', fontsize=7)

    # ========== Subplot 2: Inequality Ratio Bar Graph ==========
    ax = axes[1]

    rho_values = []
    rho_std = []

    print(f"\n  --- Inequality Ratio Calculation (Constraint: {constraint}) ---")

    for reward in rewards:
        if reward in aggregated and constraint in aggregated[reward]:
            df_mean = aggregated[reward][constraint]['mean']
            df_std = aggregated[reward][constraint]['std']

            # Compute rho = (mu_M_end[last] - mu_M_start[first]) / (mu_F_end[last] - mu_F_start[first])
            mu_M_start_first = df_mean['mu_M_start'].iloc[0]
            mu_M_end_last = df_mean['mu_M_end'].iloc[-1]
            mu_F_start_first = df_mean['mu_F_start'].iloc[0]
            mu_F_end_last = df_mean['mu_F_end'].iloc[-1]

            delta_M = mu_M_end_last - mu_M_start_first
            delta_F = mu_F_end_last - mu_F_start_first

            if delta_F != 0:
                rho = delta_M / delta_F
            else:
                rho = np.nan

            rho_values.append(rho if not np.isnan(rho) else 0)

            # Approximate std using error propagation
            # rho = delta_M / delta_F
            # std(rho) ≈ |rho| * sqrt((std_delta_M/delta_M)^2 + (std_delta_F/delta_F)^2)
            mu_M_start_std_first = df_std['mu_M_start'].iloc[0]
            mu_M_end_std_last = df_std['mu_M_end'].iloc[-1]
            mu_F_start_std_first = df_std['mu_F_start'].iloc[0]
            mu_F_end_std_last = df_std['mu_F_end'].iloc[-1]

            std_delta_M = np.sqrt(mu_M_start_std_first**2 + mu_M_end_std_last**2)
            std_delta_F = np.sqrt(mu_F_start_std_first**2 + mu_F_end_std_last**2)

            if delta_M != 0 and delta_F != 0 and not np.isnan(rho):
                rho_std_val = abs(rho) * np.sqrt((std_delta_M/abs(delta_M))**2 + (std_delta_F/abs(delta_F))**2)
            else:
                rho_std_val = 0

            rho_std.append(rho_std_val)

            print(f"  {REWARD_LABELS[reward]}:")
            print(f"    delta_M = mu_M_end[last]({mu_M_end_last:.4f}) - mu_M_start[first]({mu_M_start_first:.4f}) = {delta_M:.4f}")
            print(f"    delta_F = mu_F_end[last]({mu_F_end_last:.4f}) - mu_F_start[first]({mu_F_start_first:.4f}) = {delta_F:.4f}")
            print(f"    rho = {rho:.4f}")
        else:
            rho_values.append(0)
            rho_std.append(0)

    bars = ax.bar(x, rho_values, bar_width,
                  color=[COLORS[r] for r in rewards],
                  edgecolor='black', linewidth=0.5,
                  yerr=rho_std, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([REWARD_LABELS[r] for r in rewards], rotation=30, ha='right')
    ax.set_ylabel(r'Inequality Ratio ($\rho$)')
    ax.set_title(r'(b) Final Inequality Ratio ($\rho$)')
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Equal growth (ρ=1)')
    ax.legend(loc='best', fontsize=7)

    if use_log_scale:
        ax.set_yscale('symlog', linthresh=1e-2, linscale=0.5)
    elif zoom and rho_values:
        valid_vals = [v for v in rho_values if v != 0]
        if valid_vals:
            data_min = min(valid_vals) - max(rho_std)
            data_max = max(valid_vals) + max(rho_std)
            padding = 0.2 * (data_max - data_min) if data_max != data_min else 0.1
            ax.set_ylim(data_min - padding, data_max + padding)

    # ========== Subplot 3: Inequality Ratio Trajectory ==========
    ax = axes[2]

    for reward in rewards:
        if reward not in aggregated or constraint not in aggregated[reward]:
            continue

        df_mean = aggregated[reward][constraint]['mean']
        df_std = aggregated[reward][constraint]['std']
        n_episodes = len(df_mean)

        # Fixed starting point (first row) - use mu_start
        mu_M_start_first = df_mean['mu_M_start'].iloc[0]
        mu_F_start_first = df_mean['mu_F_start'].iloc[0]

        # Compute rho trajectory: rho(t) = (mu_M_end[t] - mu_M_start[0]) / (mu_F_end[t] - mu_F_start[0])
        rho_traj = []
        rho_std_traj = []

        for t in range(n_episodes):
            mu_M_end_t = df_mean['mu_M_end'].iloc[t]
            mu_F_end_t = df_mean['mu_F_end'].iloc[t]

            delta_M_t = mu_M_end_t - mu_M_start_first
            delta_F_t = mu_F_end_t - mu_F_start_first

            if delta_F_t != 0:
                rho_t = delta_M_t / delta_F_t
            else:
                rho_t = np.nan

            rho_traj.append(rho_t)

            # Std propagation
            mu_M_end_std_t = df_std['mu_M_end'].iloc[t]
            mu_M_start_std_first = df_std['mu_M_start'].iloc[0]
            mu_F_end_std_t = df_std['mu_F_end'].iloc[t]
            mu_F_start_std_first = df_std['mu_F_start'].iloc[0]

            std_delta_M_t = np.sqrt(mu_M_end_std_t**2 + mu_M_start_std_first**2)
            std_delta_F_t = np.sqrt(mu_F_end_std_t**2 + mu_F_start_std_first**2)

            if delta_M_t != 0 and delta_F_t != 0 and not np.isnan(rho_t):
                rho_std_t = abs(rho_t) * np.sqrt((std_delta_M_t/abs(delta_M_t))**2 + (std_delta_F_t/abs(delta_F_t))**2)
            else:
                rho_std_t = 0

            rho_std_traj.append(rho_std_t)

        rho_traj = np.array(rho_traj)
        rho_std_traj = np.array(rho_std_traj)
        episodes = np.arange(n_episodes)

        ax.plot(episodes, rho_traj,
               label=f"{REWARD_LABELS[reward]}",
               color=COLORS[reward],
               linewidth=1.5)

        ax.fill_between(episodes,
                       rho_traj - rho_std_traj,
                       rho_traj + rho_std_traj,
                       color=COLORS[reward], alpha=0.15)

    ax.set_xlabel('Episode')
    ax.set_ylabel(r'Inequality Ratio ($\rho_t$)')
    ax.set_title(r'(c) Inequality Ratio ($\rho_t$) Trajectory')
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.legend(loc='best', fontsize=7)

    if use_log_scale:
        ax.set_yscale('symlog', linthresh=1e-2, linscale=0.5)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'plot_2_{constraint}_wealth_inequality_ratio.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_plot_3(aggregated: Dict, constraint: str, output_dir: str, use_log_scale: bool = False, zoom: bool = False):
    """
    Create third plot for a specific constraint with 3 subplots:
    1. Cumulative Bank Profit - Trajectory
    2. Accuracy - Bar graph (final, avg M/F)
    3. Precision - Bar graph (final, avg M/F)

    Bank Profit formula:
        successful_M = loans_M - defaults_M
        successful_F = loans_F - defaults_F
        bank_profit_episode = (successful_M + successful_F) * loan_amount * interest_rate - defaults_total * loan_amount
        bank_profit_cumulative = cumsum(bank_profit_episode)

    All values are aggregated across seeds.
    """
    # Bank profit parameters
    loan_amount = 30.0
    interest_rate = 0.15

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    rewards = JOB_MATRIX["rewards"]
    x = np.arange(len(rewards))
    bar_width = 0.6

    # ========== Subplot 1: Cumulative Bank Profit Trajectory ==========
    ax = axes[0]

    for reward in rewards:
        if reward not in aggregated or constraint not in aggregated[reward]:
            continue

        df_mean = aggregated[reward][constraint]['mean']
        df_std = aggregated[reward][constraint]['std']

        # Check if required columns exist
        required_cols = ['loans_M_episode', 'loans_F_episode', 'defaults_M_episode', 'defaults_F_episode']
        missing_cols = [col for col in required_cols if col not in df_mean.columns]
        if missing_cols:
            print(f"  {REWARD_LABELS[reward]}: Missing columns {missing_cols} for bank profit, skipping")
            continue

        # Calculate bank profit per episode
        successful_M = df_mean['loans_M_episode'] - df_mean['defaults_M_episode']
        successful_F = df_mean['loans_F_episode'] - df_mean['defaults_F_episode']
        defaults_total = df_mean['defaults_M_episode'] + df_mean['defaults_F_episode']

        bank_profit_episode = (successful_M + successful_F) * loan_amount * interest_rate - defaults_total * loan_amount
        bank_profit_cumulative = bank_profit_episode.cumsum()

        # Std propagation for bank profit (approximate)
        loans_M_std = df_std['loans_M_episode'] if 'loans_M_episode' in df_std.columns else 0
        loans_F_std = df_std['loans_F_episode'] if 'loans_F_episode' in df_std.columns else 0
        defaults_M_std = df_std['defaults_M_episode'] if 'defaults_M_episode' in df_std.columns else 0
        defaults_F_std = df_std['defaults_F_episode'] if 'defaults_F_episode' in df_std.columns else 0

        # Approximate std for cumulative (grows with sqrt of episodes for independent errors)
        bank_profit_episode_std = np.sqrt(
            (loans_M_std * loan_amount * interest_rate)**2 +
            (loans_F_std * loan_amount * interest_rate)**2 +
            (defaults_M_std * loan_amount * (1 + interest_rate))**2 +
            (defaults_F_std * loan_amount * (1 + interest_rate))**2
        )
        # Cumulative std (approximate)
        bank_profit_cumulative_std = np.sqrt(np.cumsum(bank_profit_episode_std**2))

        episodes = np.arange(len(bank_profit_cumulative))

        ax.plot(episodes, bank_profit_cumulative,
               label=f"{REWARD_LABELS[reward]}",
               color=COLORS[reward],
               linewidth=1.5)

        ax.fill_between(episodes,
                       bank_profit_cumulative - bank_profit_cumulative_std,
                       bank_profit_cumulative + bank_profit_cumulative_std,
                       color=COLORS[reward], alpha=0.15)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Bank Profit')
    ax.set_title('(a) Cumulative Bank Profit Trajectory')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(loc='best', fontsize=7)

    # ========== Subplot 2: Accuracy Bar Graph ==========
    ax = axes[1]

    accuracy_values = []
    accuracy_std = []

    for reward in rewards:
        if reward in aggregated and constraint in aggregated[reward]:
            df_mean = aggregated[reward][constraint]['mean']
            df_std = aggregated[reward][constraint]['std']

            acc_m = df_mean['accuracy_M'].iloc[-1]
            acc_f = df_mean['accuracy_F'].iloc[-1]
            acc_avg = (acc_m + acc_f) / 2

            acc_m_std = df_std['accuracy_M'].iloc[-1]
            acc_f_std = df_std['accuracy_F'].iloc[-1]
            acc_std = np.sqrt(acc_m_std**2 + acc_f_std**2) / 2

            accuracy_values.append(acc_avg)
            accuracy_std.append(acc_std)
        else:
            accuracy_values.append(0)
            accuracy_std.append(0)

    bars = ax.bar(x, accuracy_values, bar_width,
                  color=[COLORS[r] for r in rewards],
                  edgecolor='black', linewidth=0.5,
                  yerr=accuracy_std, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([REWARD_LABELS[r] for r in rewards], rotation=30, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('(b) Final Accuracy (Avg M/F)')
    ax.set_ylim(0, 1.05)

    # ========== Subplot 3: Precision Bar Graph ==========
    ax = axes[2]

    precision_values = []
    precision_std = []

    for reward in rewards:
        if reward in aggregated and constraint in aggregated[reward]:
            df_mean = aggregated[reward][constraint]['mean']
            df_std = aggregated[reward][constraint]['std']

            prec_m = df_mean['precision_M'].iloc[-1]
            prec_f = df_mean['precision_F'].iloc[-1]
            prec_avg = (prec_m + prec_f) / 2

            prec_m_std = df_std['precision_M'].iloc[-1]
            prec_f_std = df_std['precision_F'].iloc[-1]
            prec_std = np.sqrt(prec_m_std**2 + prec_f_std**2) / 2

            precision_values.append(prec_avg)
            precision_std.append(prec_std)
        else:
            precision_values.append(0)
            precision_std.append(0)

    bars = ax.bar(x, precision_values, bar_width,
                  color=[COLORS[r] for r in rewards],
                  edgecolor='black', linewidth=0.5,
                  yerr=precision_std, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([REWARD_LABELS[r] for r in rewards], rotation=30, ha='right')
    ax.set_ylabel('Precision')
    ax.set_title('(c) Final Precision (Avg M/F)')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'plot_3_{constraint}_profit_accuracy_precision.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_plot_4(aggregated: Dict, constraint: str, output_dir: str, use_log_scale: bool = False, zoom: bool = False):
    """
    Create fourth plot for a specific constraint with 3 subplots showing trajectories:
    1. Red Group (M) Long-term Social Welfare trajectory
    2. Blue Group (F) Long-term Social Welfare trajectory
    3. Weighted Average Long-term Social Welfare trajectory

    Formula at each episode t (0-indexed, so t=0 is episode 1):
        R_M(t) = (mu_M_end[t] - mu_M_start[0]) / (t+1)
        R_F(t) = (mu_F_end[t] - mu_F_start[0]) / (t+1)
        R_bar(t) = (N_M[t] * R_M(t) + N_F[t] * R_F(t)) / (N_M[t] + N_F[t])

    All values are aggregated across seeds.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    rewards = JOB_MATRIX["rewards"]

    print(f"\n  --- Long-term Social Welfare Trajectory (Constraint: {constraint}) ---")

    # Store all trajectories for potential y-axis adjustment
    all_welfare_M = []
    all_welfare_F = []
    all_welfare_avg = []

    # ========== Compute and plot for each reward ==========
    for reward in rewards:
        if reward not in aggregated or constraint not in aggregated[reward]:
            print(f"  {REWARD_LABELS[reward]}: No data found, skipping")
            continue

        df_mean = aggregated[reward][constraint]['mean']
        df_std = aggregated[reward][constraint]['std']
        n_episodes = len(df_mean)

        # Check if required columns exist
        required_cols = ['mu_M_start', 'mu_M_end', 'mu_F_start', 'mu_F_end',
                        'total_applications_M', 'total_applications_F']
        missing_cols = [col for col in required_cols if col not in df_mean.columns]
        if missing_cols:
            print(f"  {REWARD_LABELS[reward]}: Missing columns {missing_cols}, skipping")
            continue

        # Fixed starting point (first row)
        mu_M_start_first = df_mean['mu_M_start'].iloc[0]
        mu_F_start_first = df_mean['mu_F_start'].iloc[0]

        print(f"\n  {REWARD_LABELS[reward]}:")
        print(f"    n_episodes: {n_episodes}")
        print(f"    mu_M_start[0]: {mu_M_start_first:.4f}, mu_F_start[0]: {mu_F_start_first:.4f}")
        print(f"    mu_M_end[-1]: {df_mean['mu_M_end'].iloc[-1]:.4f}, mu_F_end[-1]: {df_mean['mu_F_end'].iloc[-1]:.4f}")
        print(f"    N_M[-1]: {df_mean['total_applications_M'].iloc[-1]:.0f}, N_F[-1]: {df_mean['total_applications_F'].iloc[-1]:.0f}")

        # Compute welfare trajectories
        welfare_M_traj = []
        welfare_M_std_traj = []
        welfare_F_traj = []
        welfare_F_std_traj = []
        welfare_avg_traj = []
        welfare_avg_std_traj = []

        for t in range(n_episodes):
            mu_M_end_t = df_mean['mu_M_end'].iloc[t]
            mu_F_end_t = df_mean['mu_F_end'].iloc[t]
            N_M_t = df_mean['total_applications_M'].iloc[t]
            N_F_t = df_mean['total_applications_F'].iloc[t]

            # Calculate R_M(t) = (mu_M_end[t] - mu_M_start[0]) / (t+1)
            R_M_t = (mu_M_end_t - mu_M_start_first) / (t + 1)

            # Calculate R_F(t) = (mu_F_end[t] - mu_F_start[0]) / (t+1)
            R_F_t = (mu_F_end_t - mu_F_start_first) / (t + 1)

            welfare_M_traj.append(R_M_t)
            welfare_F_traj.append(R_F_t)

            # Weighted average: R_bar(t) = (N_M[t] * R_M(t) + N_F[t] * R_F(t)) / (N_M[t] + N_F[t])
            total_N_t = N_M_t + N_F_t
            if total_N_t > 0:
                R_bar_t = (N_M_t * R_M_t + N_F_t * R_F_t) / total_N_t
            else:
                R_bar_t = np.nan
            welfare_avg_traj.append(R_bar_t)

            # Std propagation for R_M
            mu_M_end_std_t = df_std['mu_M_end'].iloc[t]
            mu_M_start_std = df_std['mu_M_start'].iloc[0]
            welfare_M_std_t = np.sqrt(mu_M_end_std_t**2 + mu_M_start_std**2) / (t + 1)
            welfare_M_std_traj.append(welfare_M_std_t)

            # Std propagation for R_F
            mu_F_end_std_t = df_std['mu_F_end'].iloc[t]
            mu_F_start_std = df_std['mu_F_start'].iloc[0]
            welfare_F_std_t = np.sqrt(mu_F_end_std_t**2 + mu_F_start_std**2) / (t + 1)
            welfare_F_std_traj.append(welfare_F_std_t)

            # Std for weighted average (approximate)
            if total_N_t > 0:
                w_M = N_M_t / total_N_t
                w_F = N_F_t / total_N_t
                welfare_avg_std_t = np.sqrt((w_M * welfare_M_std_t)**2 + (w_F * welfare_F_std_t)**2)
            else:
                welfare_avg_std_t = 0
            welfare_avg_std_traj.append(welfare_avg_std_t)

        welfare_M_traj = np.array(welfare_M_traj)
        welfare_F_traj = np.array(welfare_F_traj)
        welfare_avg_traj = np.array(welfare_avg_traj)
        welfare_M_std_traj = np.array(welfare_M_std_traj)
        welfare_F_std_traj = np.array(welfare_F_std_traj)
        welfare_avg_std_traj = np.array(welfare_avg_std_traj)
        episodes = np.arange(n_episodes)

        # Print diagnostics
        valid_M = ~np.isnan(welfare_M_traj)
        valid_F = ~np.isnan(welfare_F_traj)
        print(f"    Valid M points: {np.sum(valid_M)}/{n_episodes}, range: [{np.nanmin(welfare_M_traj):.6f}, {np.nanmax(welfare_M_traj):.6f}]")
        print(f"    Valid F points: {np.sum(valid_F)}/{n_episodes}, range: [{np.nanmin(welfare_F_traj):.6f}, {np.nanmax(welfare_F_traj):.6f}]")

        # Store for axis limits
        all_welfare_M.extend(welfare_M_traj[valid_M])
        all_welfare_F.extend(welfare_F_traj[valid_F])
        all_welfare_avg.extend(welfare_avg_traj[~np.isnan(welfare_avg_traj)])

        # Plot M trajectory
        ax = axes[0]
        ax.plot(episodes, welfare_M_traj,
               label=f"{REWARD_LABELS[reward]}",
               color=COLORS[reward],
               linewidth=1.5)
        ax.fill_between(episodes,
                       welfare_M_traj - welfare_M_std_traj,
                       welfare_M_traj + welfare_M_std_traj,
                       color=COLORS[reward], alpha=0.15)

        # Plot F trajectory
        ax = axes[1]
        ax.plot(episodes, welfare_F_traj,
               label=f"{REWARD_LABELS[reward]}",
               color=COLORS[reward],
               linewidth=1.5)
        ax.fill_between(episodes,
                       welfare_F_traj - welfare_F_std_traj,
                       welfare_F_traj + welfare_F_std_traj,
                       color=COLORS[reward], alpha=0.15)

        # Plot Average trajectory
        ax = axes[2]
        ax.plot(episodes, welfare_avg_traj,
               label=f"{REWARD_LABELS[reward]}",
               color=COLORS[reward],
               linewidth=1.5)
        ax.fill_between(episodes,
                       welfare_avg_traj - welfare_avg_std_traj,
                       welfare_avg_traj + welfare_avg_std_traj,
                       color=COLORS[reward], alpha=0.15)

    # Configure subplot 1 (M)
    ax = axes[0]
    ax.set_xlabel('Episode')
    ax.set_ylabel(r'$R_M$ (Long-term Social Welfare)')
    ax.set_title('(a) Red Group (M) Long-term Social Welfare')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(loc='best', fontsize=7)
    if use_log_scale:
        ax.set_yscale('symlog', linthresh=1e-6, linscale=0.5)

    # Configure subplot 2 (F)
    ax = axes[1]
    ax.set_xlabel('Episode')
    ax.set_ylabel(r'$R_F$ (Long-term Social Welfare)')
    ax.set_title('(b) Blue Group (F) Long-term Social Welfare')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(loc='best', fontsize=7)
    if use_log_scale:
        ax.set_yscale('symlog', linthresh=1e-6, linscale=0.5)

    # Configure subplot 3 (Weighted Average)
    ax = axes[2]
    ax.set_xlabel('Episode')
    ax.set_ylabel(r'$\bar{R}$ (Weighted Long-term Social Welfare)')
    ax.set_title('(c) Weighted Average Long-term Social Welfare')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(loc='best', fontsize=7)
    if use_log_scale:
        ax.set_yscale('symlog', linthresh=1e-6, linscale=0.5)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'plot_4_{constraint}_longterm_social_welfare.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def print_summary_statistics(aggregated: Dict):
    """Print summary statistics for all metrics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (Final Episode Values - Aggregated Over Seeds)")
    print("="*80)

    for reward in JOB_MATRIX["rewards"]:
        if reward not in aggregated:
            continue
        print(f"\n{REWARD_LABELS[reward]}:")
        print("-" * 50)

        for constraint in aggregated[reward]:
            df_mean = aggregated[reward][constraint]['mean']
            df_std = aggregated[reward][constraint]['std']
            n_seeds = aggregated[reward][constraint]['n_seeds']

            print(f"  Constraint: {constraint} (n_seeds={n_seeds})")

            # Approval rate disparity
            apr_m = df_mean['approval_rate_M_episode'].iloc[-1]
            apr_f = df_mean['approval_rate_F_episode'].iloc[-1]
            apr_disp = apr_m - apr_f
            apr_m_std = df_std['approval_rate_M_episode'].iloc[-1]
            apr_f_std = df_std['approval_rate_F_episode'].iloc[-1]
            print(f"    Approval Rate M: {apr_m:.4f} ± {apr_m_std:.4f}")
            print(f"    Approval Rate F: {apr_f:.4f} ± {apr_f_std:.4f}")
            print(f"    Approval Rate Disparity (M-F): {apr_disp:.4f}")

            # Wealth gap
            wealth_gap = df_mean['wealth_gap'].iloc[-1]
            wealth_gap_std = df_std['wealth_gap'].iloc[-1]
            print(f"    Wealth Gap: {wealth_gap:.4f} ± {wealth_gap_std:.4f}")

            # Inequality ratio: rho = (mu_M_end[last] - mu_M_start[first]) / (mu_F_end[last] - mu_F_start[first])
            mu_M_start_first = df_mean['mu_M_start'].iloc[0]
            mu_M_end_last = df_mean['mu_M_end'].iloc[-1]
            mu_F_start_first = df_mean['mu_F_start'].iloc[0]
            mu_F_end_last = df_mean['mu_F_end'].iloc[-1]

            delta_M = mu_M_end_last - mu_M_start_first
            delta_F = mu_F_end_last - mu_F_start_first

            if delta_F != 0:
                rho = delta_M / delta_F
            else:
                rho = np.nan

            # Approximate std using error propagation
            mu_M_start_std_first = df_std['mu_M_start'].iloc[0]
            mu_M_end_std_last = df_std['mu_M_end'].iloc[-1]
            mu_F_start_std_first = df_std['mu_F_start'].iloc[0]
            mu_F_end_std_last = df_std['mu_F_end'].iloc[-1]

            std_delta_M = np.sqrt(mu_M_start_std_first**2 + mu_M_end_std_last**2)
            std_delta_F = np.sqrt(mu_F_start_std_first**2 + mu_F_end_std_last**2)

            if delta_M != 0 and delta_F != 0 and not np.isnan(rho):
                rho_std = abs(rho) * np.sqrt((std_delta_M/abs(delta_M))**2 + (std_delta_F/abs(delta_F))**2)
            else:
                rho_std = 0

            print(f"    Inequality Ratio (ρ): {rho:.4f} ± {rho_std:.4f}")

            # Long-term social welfare (final value of trajectory)
            # R_M = (mu_M_end[last] - mu_M_start[0]) / T
            # R_F = (mu_F_end[last] - mu_F_start[0]) / T
            # R_bar = (N_M * R_M + N_F * R_F) / (N_M + N_F)
            n_episodes = len(df_mean)
            mu_M_end_last = df_mean['mu_M_end'].iloc[-1]
            mu_M_start_first = df_mean['mu_M_start'].iloc[0]
            N_M_last = df_mean['total_applications_M'].iloc[-1]
            mu_F_end_last = df_mean['mu_F_end'].iloc[-1]
            mu_F_start_first = df_mean['mu_F_start'].iloc[0]
            N_F_last = df_mean['total_applications_F'].iloc[-1]

            welfare_M = (mu_M_end_last - mu_M_start_first) / n_episodes
            welfare_F = (mu_F_end_last - mu_F_start_first) / n_episodes

            total_N = N_M_last + N_F_last
            if total_N > 0:
                welfare_avg = (N_M_last * welfare_M + N_F_last * welfare_F) / total_N
            else:
                welfare_avg = 0

            print(f"    Long-term Social Welfare R_M: {welfare_M:.6f}")
            print(f"    Long-term Social Welfare R_F: {welfare_F:.6f}")
            print(f"    Long-term Social Welfare R_bar (weighted): {welfare_avg:.6f}")


def print_aggregated_by_reward(aggregated: Dict):
    """Print results aggregated by reward function for each constraint separately."""
    print("\n" + "="*80)
    print("AGGREGATED RESULTS BY REWARD FUNCTION (Per Constraint)")
    print("="*80)

    all_table_data = {}

    for constraint in JOB_MATRIX["constraints"]:
        # Check if any data exists for this constraint
        has_data = any(constraint in aggregated.get(r, {}) for r in JOB_MATRIX["rewards"])
        if not has_data:
            continue

        print(f"\n--- Constraint: {CONSTRAINT_LABELS[constraint]} ---")

        # Collect data for table
        table_data = []

        for reward in JOB_MATRIX["rewards"]:
            if reward not in aggregated or constraint not in aggregated[reward]:
                continue

            df_mean = aggregated[reward][constraint]['mean']
            df_std = aggregated[reward][constraint]['std']
            n_seeds = aggregated[reward][constraint]['n_seeds']
            n_episodes = len(df_mean)

            apr_disp = df_mean['approval_rate_M_episode'].iloc[-1] - df_mean['approval_rate_F_episode'].iloc[-1]
            apr_disp_std = np.sqrt(df_std['approval_rate_M_episode'].iloc[-1]**2 +
                                   df_std['approval_rate_F_episode'].iloc[-1]**2)

            wealth_gap = df_mean['wealth_gap'].iloc[-1]
            wealth_gap_std = df_std['wealth_gap'].iloc[-1]

            # Inequality ratio: rho = (mu_M_end[last] - mu_M_start[first]) / (mu_F_end[last] - mu_F_start[first])
            mu_M_start_first = df_mean['mu_M_start'].iloc[0]
            mu_M_end_last = df_mean['mu_M_end'].iloc[-1]
            mu_F_start_first = df_mean['mu_F_start'].iloc[0]
            mu_F_end_last = df_mean['mu_F_end'].iloc[-1]

            delta_M = mu_M_end_last - mu_M_start_first
            delta_F = mu_F_end_last - mu_F_start_first

            if delta_F != 0:
                rho = delta_M / delta_F
            else:
                rho = np.nan

            # Approximate std using error propagation
            mu_M_start_std_first = df_std['mu_M_start'].iloc[0]
            mu_M_end_std_last = df_std['mu_M_end'].iloc[-1]
            mu_F_start_std_first = df_std['mu_F_start'].iloc[0]
            mu_F_end_std_last = df_std['mu_F_end'].iloc[-1]

            std_delta_M = np.sqrt(mu_M_start_std_first**2 + mu_M_end_std_last**2)
            std_delta_F = np.sqrt(mu_F_start_std_first**2 + mu_F_end_std_last**2)

            if delta_M != 0 and delta_F != 0 and not np.isnan(rho):
                rho_std = abs(rho) * np.sqrt((std_delta_M/abs(delta_M))**2 + (std_delta_F/abs(delta_F))**2)
            else:
                rho_std = 0

            # Long-term social welfare
            # R_M = (mu_M_end[last] - mu_M_start[0]) / T
            # R_F = (mu_F_end[last] - mu_F_start[0]) / T
            # R_bar = (N_M * R_M + N_F * R_F) / (N_M + N_F)
            mu_M_end_last = df_mean['mu_M_end'].iloc[-1]
            mu_M_start_first = df_mean['mu_M_start'].iloc[0]
            N_M_last = df_mean['total_applications_M'].iloc[-1]
            mu_F_end_last = df_mean['mu_F_end'].iloc[-1]
            mu_F_start_first = df_mean['mu_F_start'].iloc[0]
            N_F_last = df_mean['total_applications_F'].iloc[-1]

            welfare_M = (mu_M_end_last - mu_M_start_first) / n_episodes
            welfare_F = (mu_F_end_last - mu_F_start_first) / n_episodes

            total_N = N_M_last + N_F_last
            if total_N > 0:
                welfare_avg = (N_M_last * welfare_M + N_F_last * welfare_F) / total_N
            else:
                welfare_avg = 0

            row = {
                'constraint': constraint,
                'reward': reward,
                'label': REWARD_LABELS[reward],
                'apr_disp_mean': apr_disp,
                'apr_disp_std': apr_disp_std,
                'wealth_gap_mean': wealth_gap,
                'wealth_gap_std': wealth_gap_std,
                'rho_mean': rho if not np.isnan(rho) else 0,
                'rho_std': rho_std if not np.isnan(rho_std) else 0,
                'welfare_M': welfare_M,
                'welfare_F': welfare_F,
                'welfare_avg': welfare_avg,
                'n_seeds': n_seeds,
                'n_episodes': n_episodes
            }
            table_data.append(row)

        if table_data:
            # Print table header
            print(f"\n{'Reward Function':<22} | {'Apr. Rate Disp.':<18} | {'Wealth Gap':<18} | {'Inequality (ρ)':<18} | {'Welfare (R_bar)':<14} | {'Seeds'}")
            print("-" * 115)

            for row in table_data:
                print(f"{row['label']:<22} | "
                      f"{row['apr_disp_mean']:>7.4f} ± {row['apr_disp_std']:<7.4f} | "
                      f"{row['wealth_gap_mean']:>7.4f} ± {row['wealth_gap_std']:<7.4f} | "
                      f"{row['rho_mean']:>7.4f} ± {row['rho_std']:<7.4f} | "
                      f"{row['welfare_avg']:>12.6f} | "
                      f"{row['n_seeds']}")

            print("-" * 115)

            all_table_data[constraint] = table_data

    return all_table_data


def main(directory_paths: List[str], output_dir: str, use_log_scale: bool = False, zoom: bool = False, test: bool = False):
    """Main function to generate all plots from multiple server directories."""
    setup_plot_style()

    # Validate directories
    valid_dirs = []
    for d in directory_paths:
        if os.path.isdir(d):
            valid_dirs.append(d)
        else:
            print(f"Warning: Directory not found, skipping: {d}")

    if not valid_dirs:
        print("Error: No valid directories found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from {len(valid_dirs)} server directories:")
    for d in valid_dirs:
        print(f"  - {d}")
    print(f"Output directory: {output_dir}")
    print(f"Log scale: {'Enabled' if use_log_scale else 'Disabled'}")
    print(f"Zoom mode: {'Enabled' if zoom else 'Disabled'}")
    print(f"Test mode: {'Enabled (last 100 episodes)' if test else 'Disabled'}")
    print("-" * 50)

    # Load results from all directories
    results = load_results(valid_dirs)

    if not results:
        print("No results found! Check if files match the expected pattern:")
        print("  pg_test_v2_{reward}_{constraint}_seed{N}_{timestamp}_episode_metrics.csv")
        return

    # Trim to last 100 episodes if --test is set
    if test:
        n_tail = 100
        print(f"\n[Test mode] Trimming all data to last {n_tail} episodes")
        for reward in results:
            for constraint in results[reward]:
                for seed in results[reward][constraint]:
                    df = results[reward][constraint][seed]
                    if len(df) > n_tail:
                        results[reward][constraint][seed] = df.tail(n_tail).reset_index(drop=True)

    # Aggregate across seeds
    print("\n" + "="*80)
    print("AGGREGATING RESULTS ACROSS SEEDS")
    print("="*80)
    aggregated = aggregate_across_seeds(results)

    # Print detailed summary statistics
    print_summary_statistics(aggregated)

    # Print aggregated results by reward function (per constraint)
    all_table_data = print_aggregated_by_reward(aggregated)

    # Save aggregated results to CSV (one per constraint)
    for constraint, table_data in all_table_data.items():
        results_df = pd.DataFrame(table_data)
        results_csv_path = os.path.join(output_dir, f'aggregated_results_{constraint}.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"Aggregated results saved to: {results_csv_path}")

    # Create plots for each constraint
    print("\n" + "="*80)
    print("GENERATING PLOTS (4 plots × 3 constraints = 12 plots)")
    print("="*80)

    generated_plots = []
    for constraint in JOB_MATRIX["constraints"]:
        has_data = any(constraint in aggregated[r] for r in aggregated)
        if has_data:
            print(f"\n--- Constraint: {CONSTRAINT_LABELS[constraint]} ---")
            create_plot_1(aggregated, constraint, output_dir, use_log_scale, zoom)
            create_plot_2(aggregated, constraint, output_dir, use_log_scale, zoom)
            create_plot_3(aggregated, constraint, output_dir, use_log_scale, zoom)
            create_plot_4(aggregated, constraint, output_dir, use_log_scale, zoom)
            generated_plots.append(f"plot_1_{constraint}_approval_wealth_disparity.png")
            generated_plots.append(f"plot_2_{constraint}_wealth_inequality_ratio.png")
            generated_plots.append(f"plot_3_{constraint}_profit_accuracy_precision.png")
            generated_plots.append(f"plot_4_{constraint}_longterm_social_welfare.png")
        else:
            print(f"\n--- Constraint: {CONSTRAINT_LABELS[constraint]} --- (no data, skipping)")

    print(f"\nAll outputs saved to: {output_dir}")
    print("Generated plots:")
    for plot in generated_plots:
        print(f"  - {plot}")
    print("Generated CSV files:")
    for constraint in all_table_data.keys():
        print(f"  - aggregated_results_{constraint}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate fairness metric plots from experiment results split across multiple servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python plotting_servers.py episode_metrics_pg_test_bob episode_metrics_pg_test_flanders episode_metrics_pg_test_desktop
    python plotting_servers.py episode_metrics_pg_test_* --log-scale
    python plotting_servers.py episode_metrics_pg_test_* --zoom
    python plotting_servers.py episode_metrics_pg_test_* --test
    python plotting_servers.py episode_metrics_pg_test_bob episode_metrics_pg_test_flanders episode_metrics_pg_test_desktop -o plots_output

Expected file format in each server directory:
    pg_test_v2_{reward}_{constraint}_seed{N}_{timestamp}_episode_metrics.csv

Reward functions: social_welfare, rawlsian_maximin, fairness_lagrangian, utilitarian_profit
Constraints: approval_rate, wealth, both

Visualization options:
    --log-scale : Use symmetric log scale (good for data spanning orders of magnitude)
    --zoom      : Auto-zoom y-axis to data range (good for small differences)
    --test      : Only plot the last 100 episodes (quick sanity check)

Output (12 plots total - 4 per constraint):
    For each constraint:
      - plot_1_{constraint}_approval_wealth_disparity.png
        (a) Final Approval Rate Disparity bar graph
        (b) Approval Rate Disparity trajectory
        (c) Final Wealth Gap bar graph
      - plot_2_{constraint}_wealth_inequality_ratio.png
        (a) Wealth Gap trajectory
        (b) Final Inequality Ratio bar graph
        (c) Inequality Ratio trajectory
      - plot_3_{constraint}_profit_accuracy_precision.png
        (a) Cumulative Bank Profit trajectory
            Formula: profit
        (b) Final Accuracy bar graph (avg M/F)
        (c) Final Precision bar graph (avg M/F)
      - plot_4_{constraint}_longterm_social_welfare.png
        (a) Red Group (M) Long-term Social Welfare trajectory
        (b) Blue Group (F) Long-term Social Welfare trajectory
        (c) Weighted Average Long-term Social Welfare trajectory
        Formula: R_M(t) = (mu_M_end[t] - mu_M_start[0]) / (t+1)
                 R_F(t) = (mu_F_end[t] - mu_F_start[0]) / (t+1)
                 R_bar(t) = (N_M[t]*R_M(t) + N_F[t]*R_F(t)) / (N_M[t] + N_F[t])
        """
    )

    parser.add_argument(
        "directories",
        type=str,
        nargs='+',
        help="Paths to server result directories (e.g. episode_metrics_pg_test_bob episode_metrics_pg_test_flanders episode_metrics_pg_test_desktop)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="plots_servers",
        help="Output directory for plots (default: plots_servers)"
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        dest="log_scale",
        help="Use symmetric log scale for approval rate disparity and inequality ratio plots"
    )
    parser.add_argument(
        "--zoom",
        action="store_true",
        dest="zoom",
        help="Zoom y-axis to data range for better visualization of small differences"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        dest="test",
        help="Only plot the last 100 episodes (quick sanity check)"
    )

    args = parser.parse_args()
    main(args.directories, args.output_dir, args.log_scale, args.zoom, args.test)
