from typing import Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_propagation(
    loss_history: dict,
    reward_function: str = "",
    constraint_type: str = "",
    seed: int = 0,
    save_path: str = None,
    show: bool = True,
    figsize: Tuple[int, int] = (20, 16),
):
    """
    Plot comprehensive loss propagation during training.

    Args:
        loss_history: Dictionary containing loss tracking data
        reward_function: Name of reward function used
        constraint_type: Type of constraint used
        seed: Random seed
        save_path: Path to save the figure
        show: Whether to display the plot
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25)

    episodes = range(len(loss_history.get("policy_loss", [])))
    if len(episodes) == 0:
        print("No loss history to plot")
        return

    def smooth(data, window=10):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window) / window, mode="valid")

    # 1. Policy Loss
    ax1 = fig.add_subplot(gs[0, 0])
    policy_loss = loss_history.get("policy_loss", [])
    ax1.plot(episodes, policy_loss, "b-", alpha=0.3, label="Raw")
    if len(policy_loss) > 10:
        smoothed = smooth(policy_loss)
        ax1.plot(range(9, len(episodes)), smoothed, "b-", linewidth=2, label="Smoothed")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Policy Loss")
    ax1.set_title("Policy Loss (Negative Expected Return)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Lambda Loss
    ax2 = fig.add_subplot(gs[0, 1])
    lambda_loss = loss_history.get("lambda_loss", [])
    ax2.plot(episodes, lambda_loss, "r-", alpha=0.3, label="Raw")
    if len(lambda_loss) > 10:
        smoothed = smooth(lambda_loss)
        ax2.plot(range(9, len(episodes)), smoothed, "r-", linewidth=2, label="Smoothed")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Lambda Loss")
    ax2.set_title("Lagrangian Multiplier Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Total Loss
    ax3 = fig.add_subplot(gs[0, 2])
    total_loss = loss_history.get("total_loss", [])
    ax3.plot(episodes, total_loss, "purple", alpha=0.3, label="Raw")
    if len(total_loss) > 10:
        smoothed = smooth(total_loss)
        ax3.plot(
            range(9, len(episodes)), smoothed, "purple", linewidth=2, label="Smoothed"
        )
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Total Loss")
    ax3.set_title("Combined Loss (Policy + Lambda)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Advantage Statistics
    ax4 = fig.add_subplot(gs[1, 0])
    adv_mean = loss_history.get("advantage_mean", [])
    adv_std = loss_history.get("advantage_std", [])
    ax4.plot(episodes, adv_mean, "g-", label="Mean", linewidth=1.5)
    if adv_std:
        adv_mean_arr = np.array(adv_mean)
        adv_std_arr = np.array(adv_std)
        ax4.fill_between(
            episodes,
            adv_mean_arr - adv_std_arr,
            adv_mean_arr + adv_std_arr,
            alpha=0.3,
            color="g",
            label="±1 Std",
        )
    ax4.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Advantage")
    ax4.set_title("Advantage Statistics")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Log Probability Mean
    ax5 = fig.add_subplot(gs[1, 1])
    log_prob = loss_history.get("log_prob_mean", [])
    ax5.plot(episodes, log_prob, "orange", alpha=0.5)
    if len(log_prob) > 10:
        smoothed = smooth(log_prob)
        ax5.plot(range(9, len(episodes)), smoothed, "darkorange", linewidth=2)
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Mean Log Probability")
    ax5.set_title("Policy Log Probability")
    ax5.grid(True, alpha=0.3)

    # 6. Policy Entropy
    ax6 = fig.add_subplot(gs[1, 2])
    entropy = loss_history.get("entropy", [])
    ax6.plot(episodes, entropy, "cyan", alpha=0.5)
    if len(entropy) > 10:
        smoothed = smooth(entropy)
        ax6.plot(range(9, len(episodes)), smoothed, "darkcyan", linewidth=2)
    ax6.set_xlabel("Episode")
    ax6.set_ylabel("Entropy")
    ax6.set_title("Policy Entropy (Exploration)")
    ax6.grid(True, alpha=0.3)

    # 7. Value Baseline
    ax7 = fig.add_subplot(gs[2, 0])
    baseline = loss_history.get("value_baseline", [])
    ax7.plot(episodes, baseline, "brown", alpha=0.5)
    if len(baseline) > 10:
        smoothed = smooth(baseline)
        ax7.plot(range(9, len(episodes)), smoothed, "saddlebrown", linewidth=2)
    ax7.set_xlabel("Episode")
    ax7.set_ylabel("Baseline Value")
    ax7.set_title("Value Function Baseline (Mean Return)")
    ax7.grid(True, alpha=0.3)

    # 8. Constraint Violations
    ax8 = fig.add_subplot(gs[2, 1])
    wealth_constraint = loss_history.get("constraint_wealth", [])
    approval_constraint = loss_history.get("constraint_approval", [])
    if wealth_constraint:
        ax8.plot(episodes, wealth_constraint, "b-", label="Wealth Gap", linewidth=1.5)
    if approval_constraint:
        ax8.plot(
            episodes, approval_constraint, "r-", label="Approval Gap", linewidth=1.5
        )
    ax8.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax8.set_xlabel("Episode")
    ax8.set_ylabel("Constraint Value")
    ax8.set_title("Fairness Constraint Violations")
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Loss Components Breakdown
    ax9 = fig.add_subplot(gs[2, 2])
    reward_comp = loss_history.get("reward_component", [])
    transition_comp = loss_history.get("transition_component", [])
    if reward_comp:
        ax9.plot(episodes, reward_comp, "g-", label="Reward Component", alpha=0.7)
    if transition_comp:
        ax9.plot(
            episodes, transition_comp, "m-", label="Transition Component", alpha=0.7
        )
    ax9.set_xlabel("Episode")
    ax9.set_ylabel("Component Value")
    ax9.set_title("Loss Component Breakdown")
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # 10. Loss Convergence Analysis (Rolling statistics)
    ax10 = fig.add_subplot(gs[3, 0])
    if len(policy_loss) > 20:
        window = 20
        rolling_mean = [
            np.mean(policy_loss[max(0, i - window) : i + 1])
            for i in range(len(policy_loss))
        ]
        rolling_std = [
            np.std(policy_loss[max(0, i - window) : i + 1])
            for i in range(len(policy_loss))
        ]
        ax10.plot(episodes, rolling_mean, "b-", label="Rolling Mean", linewidth=2)
        rolling_mean_arr = np.array(rolling_mean)
        rolling_std_arr = np.array(rolling_std)
        ax10.fill_between(
            episodes,
            rolling_mean_arr - rolling_std_arr,
            rolling_mean_arr + rolling_std_arr,
            alpha=0.3,
            color="b",
        )
    ax10.set_xlabel("Episode")
    ax10.set_ylabel("Loss")
    ax10.set_title("Loss Convergence (Rolling Window=20)")
    ax10.grid(True, alpha=0.3)

    # 11. Loss Change Rate
    ax11 = fig.add_subplot(gs[3, 1])
    if len(policy_loss) > 1:
        loss_change = np.diff(policy_loss)
        ax11.plot(range(1, len(episodes)), loss_change, "purple", alpha=0.3)
        if len(loss_change) > 10:
            smoothed = smooth(loss_change)
            ax11.plot(range(10, len(episodes)), smoothed, "purple", linewidth=2)
    ax11.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax11.set_xlabel("Episode")
    ax11.set_ylabel("Loss Change")
    ax11.set_title("Loss Change Rate (Gradient Descent Progress)")
    ax11.grid(True, alpha=0.3)

    # 12. Combined Summary Plot (normalized)
    ax12 = fig.add_subplot(gs[3, 2])

    def normalize(data):
        data = np.array(data)
        if len(data) == 0 or np.std(data) < 1e-8:
            return data
        return (data - np.mean(data)) / np.std(data)

    if policy_loss:
        ax12.plot(episodes, normalize(policy_loss), "b-", label="Policy Loss", alpha=0.7)
    if entropy:
        ax12.plot(episodes, normalize(entropy), "c-", label="Entropy", alpha=0.7)
    if baseline:
        ax12.plot(episodes, normalize(baseline), "brown", label="Baseline", alpha=0.7)
    ax12.set_xlabel("Episode")
    ax12.set_ylabel("Normalized Value")
    ax12.set_title("Normalized Metrics Comparison")
    ax12.legend(fontsize=8)
    ax12.grid(True, alpha=0.3)

    plt.suptitle(
        f"Loss Propagation Analysis: {reward_function} ({constraint_type}) - Seed {seed}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Loss plot saved to {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_loss_summary(
    loss_history: dict,
    save_path: str = None,
    show: bool = True,
):
    """
    Plot a compact summary of key loss metrics.

    Args:
        loss_history: Dictionary containing loss tracking data
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    episodes = range(len(loss_history.get("policy_loss", [])))
    if len(episodes) == 0:
        print("No loss history to plot")
        return

    def smooth(data, window=10):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window) / window, mode="valid")

    # 1. Total Loss
    ax = axes[0, 0]
    total_loss = loss_history.get("total_loss", [])
    ax.plot(episodes, total_loss, "b-", alpha=0.3)
    if len(total_loss) > 10:
        smoothed = smooth(total_loss)
        ax.plot(range(9, len(episodes)), smoothed, "b-", linewidth=2, label="Smoothed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    # 2. Advantage Mean
    ax = axes[0, 1]
    adv_mean = loss_history.get("advantage_mean", [])
    ax.plot(episodes, adv_mean, "g-", linewidth=1.5)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Advantage")
    ax.set_title("Advantage Estimation")
    ax.grid(True, alpha=0.3)

    # 3. Entropy
    ax = axes[1, 0]
    entropy = loss_history.get("entropy", [])
    ax.plot(episodes, entropy, "orange", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy")
    ax.grid(True, alpha=0.3)

    # 4. Constraints
    ax = axes[1, 1]
    wealth = loss_history.get("constraint_wealth", [])
    approval = loss_history.get("constraint_approval", [])
    if wealth:
        ax.plot(episodes, wealth, "b-", label="Wealth Gap")
    if approval:
        ax.plot(episodes, approval, "r-", label="Approval Gap")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Gap")
    ax.set_title("Fairness Constraints")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Loss summary saved to {save_path}")

    if show:
        plt.show()

    plt.close()


def _plot_pepg_v2_results(agent, reward_function, constraint_type, seed, save_dir):
    """Plot PePG V2 training results including gradient decomposition."""
    import os

    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = agent.gradient_metrics
    episodes = range(len(metrics["standard_grad_norm"]))

    # 1. Gradient norms decomposition
    ax = axes[0, 0]
    ax.plot(
        episodes, metrics["standard_grad_norm"], "b-", label="Standard PG", alpha=0.7
    )
    ax.plot(
        episodes,
        metrics.get("hawkes_grad_norm", [0] * len(episodes)),
        "g-",
        label="Hawkes",
        alpha=0.7,
    )
    ax.plot(
        episodes,
        metrics.get("wealth_grad_norm", [0] * len(episodes)),
        "r-",
        label="Wealth",
        alpha=0.7,
    )
    ax.plot(
        episodes,
        metrics.get("reward_grad_norm", [0] * len(episodes)),
        "m-",
        label="Reward",
        alpha=0.7,
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("PePG Gradient Decomposition")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Total gradient norm
    ax = axes[0, 1]
    ax.plot(episodes, metrics["total_grad_norm"], "purple", linewidth=2)
    if len(metrics["total_grad_norm"]) > 10:
        window = 10
        smoothed = np.convolve(
            metrics["total_grad_norm"], np.ones(window) / window, mode="valid"
        )
        ax.plot(
            range(window - 1, len(episodes)),
            smoothed,
            "orange",
            linewidth=2,
            label="Smoothed",
        )
        ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Gradient Norm")
    ax.set_title("Combined PePG Gradient")
    ax.grid(True, alpha=0.3)

    # 3. Episode rewards
    ax = axes[0, 2]
    ax.plot(agent.episode_rewards, "b-", alpha=0.5)
    if len(agent.episode_rewards) > 10:
        window = 10
        smoothed = np.convolve(
            agent.episode_rewards, np.ones(window) / window, mode="valid"
        )
        ax.plot(
            range(window - 1, len(agent.episode_rewards)), smoothed, "r-", linewidth=2
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Learning Curve")
    ax.grid(True, alpha=0.3)

    # 4. Decision counts
    ax = axes[1, 0]
    ax.plot(
        episodes, metrics["num_decisions"], "k-", label="Total Decisions", alpha=0.7
    )
    ax.plot(
        episodes, metrics["num_approvals_R"], "b-", label="Male Approvals", alpha=0.7
    )
    ax.plot(
        episodes, metrics["num_approvals_B"], "r-", label="Female Approvals", alpha=0.7
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Count")
    ax.set_title("Decisions per Episode")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Inequality ratio
    ax = axes[1, 1]
    rho = agent.episode_metrics["rho_episode"]
    rho_clipped = np.clip(rho, -10, 10)
    ax.plot(rho_clipped, "purple", linewidth=1.5)
    ax.axhline(y=1.0, color="red", linestyle="--", label="Equal growth")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("ρ = Δμ_M / Δμ_F")
    ax.set_title("Inequality Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Lambda evolution
    ax = axes[1, 2]
    if agent.lambda_history["wealth"]:
        ax.plot(agent.lambda_history["wealth"], "b-", label="λ_wealth", linewidth=2)
    if agent.lambda_history["approval"]:
        ax.plot(agent.lambda_history["approval"], "r-", label="λ_approval", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Lambda Value")
    ax.set_title("Learnable Lambda Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"PePG V2: {reward_function} ({constraint_type}) - Seed {seed}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    save_path = (
        f"{save_dir}/pepg_v2_{reward_function}_{constraint_type}_seed{seed}_results.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Results plot saved to {save_path}")
    plt.show()
