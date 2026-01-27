#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from adult_income_training import (
    AdultIncomeDataLoader,
    TransitionParameterLearner,
    IncomeEnvironment,
)


def plot_all_metrics(results, save_path="policy_comparison.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    names = list(results.keys())
    colors = {name: plt.cm.tab10(i) for i, name in enumerate(names)}

    ax = axes[0, 0]
    for name, data in results.items():
        cumulative_profit = np.cumsum(data["profits"])
        ax.plot(cumulative_profit, label=name, color=colors[name])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Profit")
    ax.set_title("Cumulative Profit")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for name, data in results.items():
        ax.plot(data["wealth_gaps"], label=name, color=colors[name])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Wealth Gap (M - F)")
    ax.set_title("Wealth Gap Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    final_gaps = [results[n]["wealth_gaps"][-1] for n in names]
    ax.bar(names, final_gaps, color=[colors[n] for n in names])
    ax.set_ylabel("Final Wealth Gap (M - F)")
    ax.set_title("Final Wealth Gap")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1, 1]
    for name, data in results.items():
        ax.plot(data["approval_diffs"], label=name, color=colors[name])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Approval Rate Difference (M - F)")
    ax.set_title("Approval Rate Difference Trajectory")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

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

    obs, _ = env.reset()
    for ep in range(num_episodes):        
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

    results = {}
    env = create_env(seed=42)

    for name, policy in policies.items():
        print(f"\nRunning policy: {name}")
        profits, wealth_gaps, approval_diffs = run_policy(env, policy, num_episodes=200)
        results[name] = {
            "profits": profits,
            "wealth_gaps": wealth_gaps,
            "approval_diffs": approval_diffs,
        }

    plot_all_metrics(results)
    print("\nPlot saved: policy_comparison.png")
