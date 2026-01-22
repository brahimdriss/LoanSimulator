#!/usr/bin/env python3
"""
Plotting script for fairness experiment results.
Generates visualizations for approval rate disparity, wealth gap, and inequality ratio.

Usage:
    python plot_fairness_results.py /path/to/static_sequential_results --log-scale
    python plot_fairness_results.py /path/to/pepg_sequential_results
"""

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
from typing import Dict, Optional


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
    
    Expected format: test_episode_metrics_{reward}_{constraint}_seed{N}_{timestamp}.csv
    """
    # Remove extension and prefix
    name = os.path.basename(filename)
    if not name.endswith('.csv'):
        return None
    name = name[:-4]  # Remove .csv
    
    if not name.startswith('test_episode_metrics_'):
        return None
    name = name[len('test_episode_metrics_'):]
    
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


def load_results(directory_path: str) -> Dict:
    """
    Load all result files from the directory.
    
    Returns a nested dict: {reward: {constraint: {seed: DataFrame}}}
    """
    results = defaultdict(lambda: defaultdict(dict))
    
    # Find all CSV files
    pattern = os.path.join(directory_path, "test_episode_metrics_*.csv")
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} result files")
    
    for filepath in files:
        parsed = parse_filename(filepath)
        if parsed:
            try:
                df = pd.read_csv(filepath)
                results[parsed['reward']][parsed['constraint']][parsed['seed']] = df
                print(f"  Loaded: {parsed['reward']} / {parsed['constraint']} / seed{parsed['seed']}")
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


def create_plot_1(aggregated: Dict, constraint: str, output_dir: str, use_log_scale: bool = False):
    """
    Create first plot for a specific constraint with 3 subplots:
    1. Approval Rate Disparity (M-F) - Bar graph (final)
    2. Approval Rate Disparity (M-F) - Trajectory
    3. Wealth Gap Disparity (M-F) - Bar graph (final)
    
    All values are aggregated across seeds.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    # fig.suptitle(f'Constraint: {CONSTRAINT_LABELS[constraint]}', fontsize=14, fontweight='bold', y=1.02)
    
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
            
            # Disparity from seed-aggregated mean
            disparity = df_mean['approval_rate_M_episode'].iloc[-1] - df_mean['approval_rate_F_episode'].iloc[-1]
            # Propagate std: std(A-B) ≈ sqrt(std_A^2 + std_B^2)
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
        ax.set_yscale('symlog', linthresh=0.1)
    # ========== Subplot 2: Approval Rate Disparity Trajectory ==========
    ax = axes[1]
    
    for reward in rewards:
        if reward not in aggregated or constraint not in aggregated[reward]:
            continue
            
        df_mean = aggregated[reward][constraint]['mean']
        df_std = aggregated[reward][constraint]['std']
        
        disparity_mean = df_mean['approval_rate_M_episode'] - df_mean['approval_rate_F_episode']
        # Approximate std for difference
        disparity_std = np.sqrt(df_std['approval_rate_M_episode']**2 + df_std['approval_rate_F_episode']**2)
        
        episodes = np.arange(len(disparity_mean))
        
        line, = ax.plot(episodes, disparity_mean, 
                       label=f"{REWARD_LABELS[reward]}",
                       color=COLORS[reward],
                       linewidth=1.5)
        
        # Add shaded region for std
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
        ax.set_yscale('symlog', linthresh=0.1)
    
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


def create_plot_2(aggregated: Dict, constraint: str, output_dir: str, use_log_scale: bool = False):
    """
    Create second plot for a specific constraint with 3 subplots:
    1. Wealth Gap Disparity (M-F) - Trajectory
    2. Inequality Ratio (rho_t) - Bar graph (final)
    3. Inequality Ratio (rho_t) - Trajectory
    
    All values are aggregated across seeds.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    #fig.suptitle(f'Constraint: {CONSTRAINT_LABELS[constraint]}', fontsize=14, fontweight='bold', y=1.02)
    
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
    
    for reward in rewards:
        if reward in aggregated and constraint in aggregated[reward]:
            df_mean = aggregated[reward][constraint]['mean']
            df_std = aggregated[reward][constraint]['std']
            rho_values.append(df_mean['rho_episode'].iloc[-1])
            rho_std.append(df_std['rho_episode'].iloc[-1])
        else:
            rho_values.append(0)
            rho_std.append(0)
    
    bars = ax.bar(x, rho_values, bar_width,
                  color=[COLORS[r] for r in rewards],
                  edgecolor='black', linewidth=0.5,
                  yerr=rho_std, capsize=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels([REWARD_LABELS[r] for r in rewards], rotation=30, ha='right')
    ax.set_ylabel(r'Inequality Ratio ($\rho_t$)')
    ax.set_title(r'(b) Final Inequality Ratio ($\rho_t$)')
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Equal growth (ρ=1)')
    ax.legend(loc='best', fontsize=7)
    
    if use_log_scale:
        ax.set_yscale('symlog', linthresh=0.1)
    
    # ========== Subplot 3: Inequality Ratio Trajectory ==========
    ax = axes[2]
    
    for reward in rewards:
        if reward not in aggregated or constraint not in aggregated[reward]:
            continue
            
        df_mean = aggregated[reward][constraint]['mean']
        df_std = aggregated[reward][constraint]['std']
        
        rho_mean = df_mean['rho_episode']
        rho_std_vals = df_std['rho_episode']
        
        episodes = np.arange(len(rho_mean))
        
        ax.plot(episodes, rho_mean,
               label=f"{REWARD_LABELS[reward]}",
               color=COLORS[reward],
               linewidth=1.5)
        
        ax.fill_between(episodes,
                       rho_mean - rho_std_vals,
                       rho_mean + rho_std_vals,
                       color=COLORS[reward], alpha=0.15)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel(r'Inequality Ratio ($\rho_t$)')
    ax.set_title(r'(c) Inequality Ratio ($\rho_t$) Trajectory')
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.legend(loc='best', fontsize=7)
    
    if use_log_scale:
        ax.set_yscale('symlog', linthresh=0.1)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'plot_2_{constraint}_wealth_inequality_ratio.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_plot_3(aggregated: Dict, constraint: str, output_dir: str, use_log_scale: bool = False):
    """
    Create third plot for a specific constraint with 3 subplots:
    1. Cumulative Profit - Trajectory
    2. Accuracy - Bar graph (final, avg M/F)
    3. Precision - Bar graph (final, avg M/F)
    
    All values are aggregated across seeds.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    rewards = JOB_MATRIX["rewards"]
    x = np.arange(len(rewards))
    bar_width = 0.6
    
    # ========== Subplot 1: Cumulative Profit Trajectory ==========
    ax = axes[0]
    
    for reward in rewards:
        if reward not in aggregated or constraint not in aggregated[reward]:
            continue
            
        df_mean = aggregated[reward][constraint]['mean']
        df_std = aggregated[reward][constraint]['std']
        
        profit_mean = df_mean['cumulative_profit']
        profit_std = df_std['cumulative_profit']
        
        episodes = np.arange(len(profit_mean))
        
        ax.plot(episodes, profit_mean,
               label=f"{REWARD_LABELS[reward]}",
               color=COLORS[reward],
               linewidth=1.5)
        
        ax.fill_between(episodes,
                       profit_mean - profit_std,
                       profit_mean + profit_std,
                       color=COLORS[reward], alpha=0.15)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Profit')
    ax.set_title('(a) Cumulative Profit Trajectory')
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
            
            # Average accuracy across M and F
            acc_m = df_mean['accuracy_M'].iloc[-1]
            acc_f = df_mean['accuracy_F'].iloc[-1]
            acc_avg = (acc_m + acc_f) / 2
            
            # Propagate std
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
    
    if use_log_scale:
        ax.set_yscale('symlog', linthresh=0.1)
    # ========== Subplot 3: Precision Bar Graph ==========
    ax = axes[2]
    
    precision_values = []
    precision_std = []
    
    for reward in rewards:
        if reward in aggregated and constraint in aggregated[reward]:
            df_mean = aggregated[reward][constraint]['mean']
            df_std = aggregated[reward][constraint]['std']
            
            # Average precision across M and F
            prec_m = df_mean['precision_M'].iloc[-1]
            prec_f = df_mean['precision_F'].iloc[-1]
            prec_avg = (prec_m + prec_f) / 2
            
            # Propagate std
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

    if use_log_scale:
        ax.set_yscale('symlog', linthresh=0.1)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'plot_3_{constraint}_profit_accuracy_precision.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_plot_4(aggregated: Dict, constraint: str, output_dir: str, use_log_scale: bool = False):
    """
    Create fourth plot for a specific constraint with 3 subplots:
    1. Red Group (M) Long-term Social Welfare - Bar graph
       Formula: (mu_M_end[last] - mu_M_start[first]) / n_episodes / total_applications_M[last]
    2. Blue Group (F) Long-term Social Welfare - Bar graph
       Formula: (mu_F_end[last] - mu_F_start[first]) / n_episodes / total_applications_F[last]
    3. Average Long-term Social Welfare - Bar graph
       Formula: Average of M and F values
    
    All values are aggregated across seeds.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    rewards = JOB_MATRIX["rewards"]
    x = np.arange(len(rewards))
    bar_width = 0.6
    
    # Compute long-term social welfare for each reward
    welfare_M_values = []
    welfare_M_std = []
    welfare_F_values = []
    welfare_F_std = []
    welfare_avg_values = []
    welfare_avg_std = []
    
    print(f"\n  --- Long-term Social Welfare Calculation (Constraint: {constraint}) ---")
    
    for reward in rewards:
        if reward in aggregated and constraint in aggregated[reward]:
            df_mean = aggregated[reward][constraint]['mean']
            df_std = aggregated[reward][constraint]['std']
            n_episodes = len(df_mean)
            
            # Red Group (M) Long-term Social Welfare
            # (mu_M_end[last] - mu_M_start[first]) / n_episodes / total_applications_M[last]
            mu_M_end_last = df_mean['mu_M_end'].iloc[-1]
            mu_M_start_first = df_mean['mu_M_start'].iloc[0]
            total_apps_M_last = df_mean['total_applications_M'].iloc[-1]
            
            # Blue Group (F) Long-term Social Welfare
            mu_F_end_last = df_mean['mu_F_end'].iloc[-1]
            mu_F_start_first = df_mean['mu_F_start'].iloc[0]
            total_apps_F_last = df_mean['total_applications_F'].iloc[-1]
            
            # Print diagnostic info
            print(f"\n  {REWARD_LABELS[reward]}:")
            print(f"    n_episodes: {n_episodes}")
            print(f"    M: mu_end[last]={mu_M_end_last:.4f}, mu_start[first]={mu_M_start_first:.4f}, "
                  f"delta={mu_M_end_last - mu_M_start_first:.4f}, total_apps={total_apps_M_last:.0f}")
            print(f"    F: mu_end[last]={mu_F_end_last:.4f}, mu_start[first]={mu_F_start_first:.4f}, "
                  f"delta={mu_F_end_last - mu_F_start_first:.4f}, total_apps={total_apps_F_last:.0f}")
            
            if total_apps_M_last > 0:
                welfare_M = (mu_M_end_last - mu_M_start_first) / n_episodes / total_apps_M_last
            else:
                welfare_M = 0
            
            if total_apps_F_last > 0:
                welfare_F = (mu_F_end_last - mu_F_start_first) / n_episodes / total_apps_F_last
            else:
                welfare_F = 0
            
            print(f"    Welfare M: {welfare_M:.8f}")
            print(f"    Welfare F: {welfare_F:.8f}")
            
            # Approximate std for M (using error propagation)
            mu_M_end_std = df_std['mu_M_end'].iloc[-1]
            mu_M_start_std = df_std['mu_M_start'].iloc[0]
            
            if total_apps_M_last > 0:
                welfare_M_std_val = np.sqrt(mu_M_end_std**2 + mu_M_start_std**2) / n_episodes / total_apps_M_last
            else:
                welfare_M_std_val = 0
            
            welfare_M_values.append(welfare_M)
            welfare_M_std.append(welfare_M_std_val)
            
            # Approximate std for F
            mu_F_end_std = df_std['mu_F_end'].iloc[-1]
            mu_F_start_std = df_std['mu_F_start'].iloc[0]
            
            if total_apps_F_last > 0:
                welfare_F_std_val = np.sqrt(mu_F_end_std**2 + mu_F_start_std**2) / n_episodes / total_apps_F_last
            else:
                welfare_F_std_val = 0
            
            welfare_F_values.append(welfare_F)
            welfare_F_std.append(welfare_F_std_val)
            
            # Average Long-term Social Welfare
            welfare_avg = (welfare_M + welfare_F) / 2
            welfare_avg_std_val = np.sqrt(welfare_M_std_val**2 + welfare_F_std_val**2) / 2
            
            welfare_avg_values.append(welfare_avg)
            welfare_avg_std.append(welfare_avg_std_val)
        else:
            welfare_M_values.append(0)
            welfare_M_std.append(0)
            welfare_F_values.append(0)
            welfare_F_std.append(0)
            welfare_avg_values.append(0)
            welfare_avg_std.append(0)
    
    
    # ========== Subplot 1: Red Group (M) Long-term Social Welfare ==========
    ax = axes[0]
    
    bars = ax.bar(x, welfare_M_values, bar_width,
                  color=[COLORS[r] for r in rewards],  # Use reward-specific colors
                  edgecolor='black', linewidth=0.5,
                  yerr=welfare_M_std, capsize=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels([REWARD_LABELS[r] for r in rewards], rotation=30, ha='right')
    ax.set_ylabel(r'$\bar{R}_M$ (Long-term Social Welfare)')
    ax.set_title('(a) Red Group (M) Long-term Social Welfare')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    if use_log_scale:
        ax.set_yscale('symlog', linthresh=0.1)
    # ========== Subplot 2: Blue Group (F) Long-term Social Welfare ==========
    ax = axes[1]
    
    bars = ax.bar(x, welfare_F_values, bar_width,
                  color=[COLORS[r] for r in rewards],  # Use reward-specific colors
                  edgecolor='black', linewidth=0.5,
                  yerr=welfare_F_std, capsize=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels([REWARD_LABELS[r] for r in rewards], rotation=30, ha='right')
    ax.set_ylabel(r'$\bar{R}_F$ (Long-term Social Welfare)')
    ax.set_title('(b) Blue Group (F) Long-term Social Welfare')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    if use_log_scale:
        ax.set_yscale('symlog', linthresh=0.1)
    # ========== Subplot 3: Average Long-term Social Welfare ==========
    ax = axes[2]
    
    bars = ax.bar(x, welfare_avg_values, bar_width,
                  color=[COLORS[r] for r in rewards],
                  edgecolor='black', linewidth=0.5,
                  yerr=welfare_avg_std, capsize=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels([REWARD_LABELS[r] for r in rewards], rotation=30, ha='right')
    ax.set_ylabel(r'$\bar{R}_\pi$ (Long-term Social Welfare)')
    ax.set_title('(c) Average Long-term Social Welfare')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    if use_log_scale:
        ax.set_yscale('symlog', linthresh=0.1)
    
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
            
            # Inequality ratio
            rho = df_mean['rho_episode'].iloc[-1]
            rho_std = df_std['rho_episode'].iloc[-1]
            print(f"    Inequality Ratio (ρ): {rho:.4f} ± {rho_std:.4f}")
            
            # Long-term social welfare
            n_episodes = len(df_mean)
            mu_M_end_last = df_mean['mu_M_end'].iloc[-1]
            mu_M_start_first = df_mean['mu_M_start'].iloc[0]
            total_apps_M_last = df_mean['total_applications_M'].iloc[-1]
            mu_F_end_last = df_mean['mu_F_end'].iloc[-1]
            mu_F_start_first = df_mean['mu_F_start'].iloc[0]
            total_apps_F_last = df_mean['total_applications_F'].iloc[-1]
            
            if total_apps_M_last > 0:
                welfare_M = (mu_M_end_last - mu_M_start_first) / n_episodes / total_apps_M_last
            else:
                welfare_M = 0
            if total_apps_F_last > 0:
                welfare_F = (mu_F_end_last - mu_F_start_first) / n_episodes / total_apps_F_last
            else:
                welfare_F = 0
            welfare_avg = (welfare_M + welfare_F) / 2
            
            print(f"    Long-term Social Welfare (M): {welfare_M:.6f}")
            print(f"    Long-term Social Welfare (F): {welfare_F:.6f}")
            print(f"    Long-term Social Welfare (Avg): {welfare_avg:.6f}")


def print_aggregated_by_reward(aggregated: Dict):
    """Print results aggregated by reward function (across all constraints and seeds)."""
    print("\n" + "="*80)
    print("AGGREGATED RESULTS BY REWARD FUNCTION (Over All Constraints & Seeds)")
    print("="*80)
    
    # Collect data for table
    table_data = []
    
    for reward in JOB_MATRIX["rewards"]:
        if reward not in aggregated:
            continue
        
        apr_disparities = []
        wealth_gaps = []
        rho_values = []
        total_seeds = 0
        
        for constraint in aggregated[reward]:
            df_mean = aggregated[reward][constraint]['mean']
            n_seeds = aggregated[reward][constraint]['n_seeds']
            total_seeds += n_seeds
            
            apr_disp = df_mean['approval_rate_M_episode'].iloc[-1] - df_mean['approval_rate_F_episode'].iloc[-1]
            apr_disparities.append(apr_disp)
            
            wealth_gaps.append(df_mean['wealth_gap'].iloc[-1])
            rho_values.append(df_mean['rho_episode'].iloc[-1])
        
        row = {
            'reward': reward,
            'label': REWARD_LABELS[reward],
            'apr_disp_mean': np.mean(apr_disparities),
            'apr_disp_std': np.std(apr_disparities),
            'wealth_gap_mean': np.mean(wealth_gaps),
            'wealth_gap_std': np.std(wealth_gaps),
            'rho_mean': np.mean(rho_values),
            'rho_std': np.std(rho_values),
            'n_configs': len(apr_disparities),
            'total_seeds': total_seeds
        }
        table_data.append(row)
    
    # Print table header
    print(f"\n{'Reward Function':<22} | {'Apr. Rate Disp.':<20} | {'Wealth Gap':<20} | {'Inequality (ρ)':<20} | {'N'}")
    print("-" * 100)
    
    for row in table_data:
        print(f"{row['label']:<22} | "
              f"{row['apr_disp_mean']:>8.4f} ± {row['apr_disp_std']:<8.4f} | "
              f"{row['wealth_gap_mean']:>8.4f} ± {row['wealth_gap_std']:<8.4f} | "
              f"{row['rho_mean']:>8.4f} ± {row['rho_std']:<8.4f} | "
              f"{row['n_configs']}")
    
    print("-" * 100)
    print("Note: Mean ± Std computed across constraint configurations")
    
    return table_data


def main(directory_path: str, use_log_scale: bool = False):
    """Main function to generate all plots."""
    setup_plot_style()
    
    # Validate directory
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found: {directory_path}")
        return
    
    dir_name = os.path.basename(os.path.normpath(directory_path))
    if dir_name not in ['static_sequential_results', 'pepg_sequential_results']:
        print(f"Warning: Expected directory name to be 'static_sequential_results' or 'pepg_sequential_results', got '{dir_name}'")
    
    print(f"Loading results from: {directory_path}")
    print(f"Log scale: {'Enabled' if use_log_scale else 'Disabled'}")
    print("-" * 50)
    
    # Load results
    results = load_results(directory_path)
    
    if not results:
        print("No results found! Check if files match the expected pattern:")
        print("  test_episode_metrics_{reward}_{constraint}_seed{N}_{timestamp}.csv")
        return
    
    # Aggregate across seeds
    print("\n" + "="*80)
    print("AGGREGATING RESULTS ACROSS SEEDS")
    print("="*80)
    aggregated = aggregate_across_seeds(results)
    
    # Print detailed summary statistics
    print_summary_statistics(aggregated)
    
    # Print aggregated results by reward function
    table_data = print_aggregated_by_reward(aggregated)
    
    # Save aggregated results to CSV
    results_df = pd.DataFrame(table_data)
    results_csv_path = os.path.join(directory_path, 'aggregated_results_summary.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nAggregated results saved to: {results_csv_path}")
    
    # Create plots for each constraint
    print("\n" + "="*80)
    print("GENERATING PLOTS (4 plots × 3 constraints = 12 plots)")
    print("="*80)
    
    generated_plots = []
    for constraint in JOB_MATRIX["constraints"]:
        # Check if any data exists for this constraint
        has_data = any(constraint in aggregated[r] for r in aggregated)
        if has_data:
            print(f"\n--- Constraint: {CONSTRAINT_LABELS[constraint]} ---")
            create_plot_1(aggregated, constraint, directory_path, use_log_scale)
            create_plot_2(aggregated, constraint, directory_path, use_log_scale)
            create_plot_3(aggregated, constraint, directory_path, use_log_scale)
            create_plot_4(aggregated, constraint, directory_path, use_log_scale)
            generated_plots.append(f"plot_1_{constraint}_approval_wealth_disparity.png")
            generated_plots.append(f"plot_2_{constraint}_wealth_inequality_ratio.png")
            generated_plots.append(f"plot_3_{constraint}_profit_accuracy_precision.png")
            generated_plots.append(f"plot_4_{constraint}_longterm_social_welfare.png")
        else:
            print(f"\n--- Constraint: {CONSTRAINT_LABELS[constraint]} --- (no data, skipping)")
    
    print(f"\nAll outputs saved to: {directory_path}")
    print("Generated plots:")
    for plot in generated_plots:
        print(f"  - {plot}")
    print("  - aggregated_results_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate fairness metric plots from experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python plot_fairness_results.py ./static_sequential_results
    python plot_fairness_results.py ./pepg_sequential_results --log-scale
    
Expected file format:
    test_episode_metrics_{reward}_{constraint}_seed{N}_{timestamp}.csv
    
Reward functions: social_welfare, rawlsian_maximin, fairness_lagrangian, utilitarian_profit
Constraints: approval_rate, wealth, both

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
        (a) Cumulative Profit trajectory
        (b) Final Accuracy bar graph (avg M/F)
        (c) Final Precision bar graph (avg M/F)
      - plot_4_{constraint}_longterm_social_welfare.png
        (a) Red Group (M) Long-term Social Welfare bar graph
        (b) Blue Group (F) Long-term Social Welfare bar graph
        (c) Average Long-term Social Welfare bar graph
        """
    )
    
    parser.add_argument(
        "directory", 
        type=str, 
        help="Path to results directory (static_sequential_results or pepg_sequential_results)"
    )
    parser.add_argument(
        "--log-scale", 
        action="store_true",
        dest="log_scale",
        help="Use symmetric log scale for approval rate disparity and inequality ratio plots"
    )
    
    args = parser.parse_args()
    main(args.directory, args.log_scale)