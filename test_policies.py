#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from adult_income_training import (
    AdultIncomeDataLoader,
    TransitionParameterLearner,
    IncomeEnvironment,
)


def plot_profit(profits_dict, save_path="profit_per_episode.png"):
    plt.figure(figsize=(10, 6))
    for name, profits in profits_dict.items():
        plt.plot(profits, label=name)
    plt.xlabel("Episode")
    plt.ylabel("Total Profit")
    plt.title("Profit per Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_wealth_gap(wealth_gaps_dict, save_path="wealth_gap.png"):
    plt.figure(figsize=(10, 6))
    for name, gaps in wealth_gaps_dict.items():
        plt.plot(gaps, label=name)
    plt.xlabel("Episode")
    plt.ylabel("Wealth Gap (M - F)")
    plt.title("Wealth Gap per Episode")
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_approval_rate_diff(approval_diffs_dict, save_path="approval_rate_diff.png"):
    plt.figure(figsize=(10, 6))
    for name, diffs in approval_diffs_dict.items():
        plt.plot(diffs, label=name)
    plt.xlabel("Episode")
    plt.ylabel("Approval Rate Difference (M - F)")
    plt.title("Approval Rate Difference per Episode")
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def random_policy(obs):
    return np.random.uniform(0, 1)


def always_approve_policy(obs):
    return 1.0


def always_reject_policy(obs):
    return 0.0


def wealth_threshold_policy(obs):
    wealth = obs[0] * 100
    return 1.0 if wealth > 30 else 0.0


# placeholder for other policies
# def my_policy(obs):
#     return ...


def create_env(seed=42):
    loader = AdultIncomeDataLoader(filepath=None, sample_size=20000)
    loader.load_data()
    loader.preprocess()
    theta_learner = TransitionParameterLearner(default_rate_min=0.02, default_rate_max=0.15)
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
    return env


def run_policy(env, policy, num_episodes=200):
    profits = []
    wealth_gaps = []
    approval_diffs = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = policy(obs)
            obs, _, terminated, truncated, _ = env.step(np.array([action]))
            done = terminated or truncated

        profits.append(sum(env.history["profit"]))
        wealth_gaps.append(env.history["wealth_gap"][-1])
        approval_diffs.append(
            env.history["approval_rate_R"][-1] - env.history["approval_rate_B"][-1]
        )

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep + 1}/{num_episodes}")

    return profits, wealth_gaps, approval_diffs


if __name__ == "__main__":
    policies = {
        "random": random_policy,
        "always_approve": always_approve_policy,
        "always_reject": always_reject_policy,
        "wealth_threshold": wealth_threshold_policy,
    }

    all_profits = {}
    all_wealth_gaps = {}
    all_approval_diffs = {}

    env = create_env(seed=42)

    for name, policy in policies.items():
        print(f"\nRunning policy: {name}")
        profits, wealth_gaps, approval_diffs = run_policy(env, policy, num_episodes=200)
        all_profits[name] = profits
        all_wealth_gaps[name] = wealth_gaps
        all_approval_diffs[name] = approval_diffs

    plot_profit(all_profits)
    plot_wealth_gap(all_wealth_gaps)
    plot_approval_rate_diff(all_approval_diffs)

    print("\nPlots saved: profit_per_episode.png, wealth_gap.png, approval_rate_diff.png")
