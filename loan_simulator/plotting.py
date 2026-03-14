import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def plot_episode_metrics(
    episode_metrics, reward_func, constraint_type, seed, save_path=None, show=True
):
    """Plot episode-level metrics: ρ, R_M, R_F, wealth gap, etc."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    episodes = episode_metrics.get("episode", [])
    if not episodes:
        print("  No episode metrics to plot")
        return None

    # 1. Inequality Ratio (ρ)
    ax = axes[0, 0]
    rho = episode_metrics.get("rho_episode", [])
    rho_clipped = np.clip(rho, -10, 10)
    ax.plot(episodes, rho_clipped, "b-", linewidth=2, label="ρ per episode")
    ax.axhline(y=1.0, color="red", linestyle="--", label="Equal growth (ρ=1)")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("ρ")
    ax.set_title("Inequality Ratio: ρ = Δμ_M / Δμ_F")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Long-term Social Welfare (R_M, R_F)
    ax = axes[0, 1]
    R_M = episode_metrics.get("R_M", [])
    R_F = episode_metrics.get("R_F", [])
    R_total = episode_metrics.get("R_total", [])
    ax.plot(episodes, R_M, "b-", linewidth=2, label="R_M (Male)")
    ax.plot(episodes, R_F, "r-", linewidth=2, label="R_F (Female)")
    ax.plot(episodes, R_total, "g--", linewidth=2, label="R_total")
    ax.axhline(y=0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("R_g = (μ_g,t - μ_g,0) / t")
    ax.set_title("Long-term Social Welfare")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Wealth Gap
    ax = axes[0, 2]
    wealth_gap = episode_metrics.get("wealth_gap", [])
    ax.plot(episodes, wealth_gap, "purple", linewidth=2)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Wealth Gap ($1000s)")
    ax.set_title("Wealth Gap: μ_M - μ_F")
    ax.grid(True, alpha=0.3)

    # 4. Mean Wealth Changes (Δμ)
    ax = axes[1, 0]
    delta_mu_M = episode_metrics.get("delta_mu_M", [])
    delta_mu_F = episode_metrics.get("delta_mu_F", [])
    ax.plot(episodes, delta_mu_M, "b-", linewidth=2, label="Δμ_M")
    ax.plot(episodes, delta_mu_F, "r-", linewidth=2, label="Δμ_F")
    ax.axhline(y=0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Δμ ($1000s)")
    ax.set_title("Wealth Change per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Approval Rates
    ax = axes[1, 1]
    approval_rate_M = episode_metrics.get("approval_rate_M", [])
    approval_rate_F = episode_metrics.get("approval_rate_F", [])
    ax.plot(episodes, approval_rate_M, "b-", linewidth=2, label="Male")
    ax.plot(episodes, approval_rate_F, "r-", linewidth=2, label="Female")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Approval Rate")
    ax.set_title("Approval Rates")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Success Probabilities
    ax = axes[1, 2]
    success_prob_M = episode_metrics.get("success_prob_M", [])
    success_prob_F = episode_metrics.get("success_prob_F", [])
    ax.plot(episodes, success_prob_M, "b-", linewidth=2, label="Male")
    ax.plot(episodes, success_prob_F, "r-", linewidth=2, label="Female")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Probability")
    ax.set_title("Success Probabilities (1 - Default Rate)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Episode-Level Metrics: {reward_func} ({constraint_type}) - Seed {seed}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Episode metrics plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_lambda_trajectory(
    lambda_history, reward_func, constraint_type, seed, save_path=None, show=True
):
    """Plot lambda multiplier trajectories across training episodes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    episodes = range(len(lambda_history.get("wealth", [])))

    # Plot lambda_wealth
    ax1 = axes[0]
    lambda_w = lambda_history.get("wealth", [])
    if lambda_w:
        ax1.plot(episodes, lambda_w, "b-", linewidth=2, label="λ_wealth")
        ax1.axhline(
            y=lambda_w[0],
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Initial: {lambda_w[0]:.4f}",
        )
        ax1.axhline(
            y=lambda_w[-1],
            color="green",
            linestyle="--",
            alpha=0.5,
            label=f"Final: {lambda_w[-1]:.4f}",
        )

        if len(lambda_w) > 10:
            window = min(20, len(lambda_w) // 5)
            smoothed = np.convolve(lambda_w, np.ones(window) / window, mode="valid")
            ax1.plot(
                range(window - 1, len(lambda_w)),
                smoothed,
                "r-",
                linewidth=1.5,
                alpha=0.7,
                label="Smoothed",
            )

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("λ_wealth")
    ax1.set_title(f"Lambda Wealth Trajectory\n{reward_func} ({constraint_type})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot lambda_approval
    ax2 = axes[1]
    lambda_a = lambda_history.get("approval", [])
    if lambda_a:
        ax2.plot(episodes, lambda_a, "purple", linewidth=2, label="λ_approval")
        ax2.axhline(
            y=lambda_a[0],
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Initial: {lambda_a[0]:.4f}",
        )
        ax2.axhline(
            y=lambda_a[-1],
            color="green",
            linestyle="--",
            alpha=0.5,
            label=f"Final: {lambda_a[-1]:.4f}",
        )

        if len(lambda_a) > 10:
            window = min(20, len(lambda_a) // 5)
            smoothed = np.convolve(lambda_a, np.ones(window) / window, mode="valid")
            ax2.plot(
                range(window - 1, len(lambda_a)),
                smoothed,
                "r-",
                linewidth=1.5,
                alpha=0.7,
                label="Smoothed",
            )

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("λ_approval")
    ax2.set_title(f"Lambda Approval Trajectory\n{reward_func} ({constraint_type})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Lagrangian Multiplier Evolution - Seed {seed}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Lambda trajectory plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_lambda_trajectories_comparison(
    results, constraint_type, seed, save_path=None, show=True
):
    """Plot lambda trajectories for multiple reward functions on same plot."""
    colors = {
        "social_welfare": "purple",
        "rawlsian_maximin": "blue",
        "fairness_lagrangian": "green",
        "utilitarian_profit": "red",
    }

    labels = {
        "social_welfare": "Social Welfare",
        "rawlsian_maximin": "Rawlsian Maximin",
        "fairness_lagrangian": "Fairness Lagrangian",
        "utilitarian_profit": "Utilitarian Profit",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot lambda_wealth
    ax1 = axes[0]
    for rf, data in results.items():
        if rf == "utilitarian_profit":
            continue
        lambda_history = data.get("lambda_history", {})
        lambda_w = lambda_history.get("wealth", [])
        if lambda_w:
            episodes = range(len(lambda_w))
            ax1.plot(
                episodes,
                lambda_w,
                color=colors[rf],
                linewidth=2,
                label=f"{labels[rf]} (final: {lambda_w[-1]:.4f})",
            )

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("λ_wealth")
    ax1.set_title("Lambda Wealth Trajectories")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot lambda_approval
    ax2 = axes[1]
    for rf, data in results.items():
        if rf == "utilitarian_profit":
            continue
        lambda_history = data.get("lambda_history", {})
        lambda_a = lambda_history.get("approval", [])
        if lambda_a:
            episodes = range(len(lambda_a))
            ax2.plot(
                episodes,
                lambda_a,
                color=colors[rf],
                linewidth=2,
                label=f"{labels[rf]} (final: {lambda_a[-1]:.4f})",
            )

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("λ_approval")
    ax2.set_title("Lambda Approval Trajectories")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Lagrangian Multiplier Comparison - {constraint_type} - Seed {seed}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Lambda comparison plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_single_seed_results(results, theta_learner, seed, constraint_type):
    """Plot results for a single seed."""

    fig = plt.figure(figsize=(28, 22))
    gs = gridspec.GridSpec(5, 4, figure=fig)

    colors = {
        "social_welfare": "purple",
        "rawlsian_maximin": "blue",
        "fairness_lagrangian": "green",
        "utilitarian_profit": "red",
    }

    labels = {
        "social_welfare": "Social Welfare",
        "rawlsian_maximin": "Rawlsian Maximin",
        "fairness_lagrangian": "Fairness Lagrangian",
        "utilitarian_profit": "Utilitarian Profit",
    }

    reward_funcs = list(results.keys())

    # Learning curves
    ax = fig.add_subplot(gs[0, :2])
    for rf in reward_funcs:
        rewards = results[rf]["episode_rewards"]
        window = 10
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        else:
            smoothed = rewards
        ax.plot(smoothed, label=labels[rf], color=colors[rf], linewidth=2.5)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(f"Learning Curves (Seed {seed}, Constraint: {constraint_type})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Lambda evolution
    ax = fig.add_subplot(gs[0, 2:])
    has_lambda_data = False
    for rf in reward_funcs:
        if (
            rf != "utilitarian_profit"
            and "lambda_history" in results[rf]
            and results[rf]["lambda_history"]
        ):
            lambda_w = results[rf]["lambda_history"].get("wealth", [])
            lambda_a = results[rf]["lambda_history"].get("approval", [])
            if constraint_type == "wealth" and lambda_w:
                ax.plot(
                    lambda_w,
                    label=f"{labels[rf]} λ_wealth",
                    color=colors[rf],
                    linewidth=2,
                    linestyle="-",
                )
                has_lambda_data = True
            elif constraint_type == "approval_rate" and lambda_a:
                ax.plot(
                    lambda_a,
                    label=f"{labels[rf]} λ_approval",
                    color=colors[rf],
                    linewidth=2,
                    linestyle="--",
                )
                has_lambda_data = True
            elif constraint_type == "both":
                if lambda_w:
                    ax.plot(
                        lambda_w,
                        label=f"{labels[rf]} λ_wealth",
                        color=colors[rf],
                        linewidth=2,
                        linestyle="-",
                    )
                    has_lambda_data = True
                if lambda_a:
                    ax.plot(
                        lambda_a,
                        label=f"{labels[rf]} λ_approval",
                        color=colors[rf],
                        linewidth=2,
                        linestyle="--",
                    )
                    has_lambda_data = True

    if has_lambda_data:
        ax.set_xlabel("Episode")
        ax.set_ylabel("Lambda Value")
        ax.set_title(f"Learnable Lambda Evolution ({constraint_type})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        for rf in reward_funcs:
            time = results[rf]["history"]["time"]
            gap = results[rf]["history"]["wealth_gap"]
            ax.plot(time, gap, label=labels[rf], color=colors[rf], linewidth=2)

        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("Time")
        ax.set_ylabel("Wealth Gap ($1000s)")
        ax.set_title("Gender Wealth Gap Evolution: μ_R(t) - μ_B(t)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Wealth gap
    ax = fig.add_subplot(gs[1, :2])
    for rf in reward_funcs:
        time = results[rf]["history"]["time"]
        gap = results[rf]["history"]["wealth_gap"]
        ax.plot(time, gap, label=labels[rf], color=colors[rf], linewidth=2)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Wealth Gap ($1000s)")
    ax.set_title("Gender Wealth Gap Evolution: μ_R(t) - μ_B(t)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Approval rates
    for i, (group, key) in enumerate(
        [("Male (R)", "approval_rate_R"), ("Female (B)", "approval_rate_B")]
    ):
        ax = fig.add_subplot(gs[1, 2 + i])
        for rf in reward_funcs:
            time = results[rf]["history"]["time"]
            rates = results[rf]["history"][key]
            ax.plot(time, rates, label=labels[rf], color=colors[rf], linewidth=2)

        ax.set_xlabel("Time")
        ax.set_ylabel("Approval Rate E[L_g(t)]")
        ax.set_title(f"{group} Approval Rates")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Approval disparity
    ax = fig.add_subplot(gs[2, :2])
    for rf in reward_funcs:
        time = results[rf]["history"]["time"]
        disp = results[rf]["history"]["approval_disparity"]
        ax.plot(time, disp, label=labels[rf], color=colors[rf], linewidth=2)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Approval Rate Gap (Male - Female)")
    ax.set_title("Approval Rate Disparity")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Mean wealth evolution
    ax = fig.add_subplot(gs[2, 2:])
    for rf in reward_funcs:
        time = results[rf]["history"]["time"]
        mu_R = results[rf]["history"]["mu_R"]
        mu_B = results[rf]["history"]["mu_B"]
        ax.plot(
            time,
            mu_R,
            label=f"{labels[rf]} Male",
            color=colors[rf],
            linewidth=2,
            linestyle="-",
        )
        ax.plot(
            time,
            mu_B,
            label=f"{labels[rf]} Female",
            color=colors[rf],
            linewidth=2,
            linestyle="--",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Mean Wealth ($1000s)")
    ax.set_title("Mean Wealth Evolution by Group")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Default rates
    for i, (group, key) in enumerate(
        [("Male (R)", "default_rate_R"), ("Female (B)", "default_rate_B")]
    ):
        ax = fig.add_subplot(gs[3, i])
        for rf in reward_funcs:
            time = results[rf]["history"]["time"]
            rates = results[rf]["history"][key]
            ax.plot(time, rates, label=labels[rf], color=colors[rf], linewidth=2)

        ax.set_xlabel("Time")
        ax.set_ylabel("Default Rate")
        ax.set_title(f"{group} Default Rates")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Cumulative profit
    ax = fig.add_subplot(gs[3, 2:])
    for rf in reward_funcs:
        time = results[rf]["history"]["time"]
        cum_profit = np.cumsum(results[rf]["history"]["profit"])
        ax.plot(time, cum_profit, label=labels[rf], color=colors[rf], linewidth=2.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Profit ($1000s)")
    ax.set_title("System Profit")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Final comparisons
    metrics = [
        ("Final Wealth Gap", "wealth_gap", -1),
        ("Final Approval Disparity", "approval_disparity", -1),
        ("Total Profit", "profit", None),
    ]

    for i, (title, key, idx) in enumerate(metrics):
        ax = fig.add_subplot(gs[4, i])

        if idx is not None:
            values = [results[rf]["history"][key][idx] for rf in reward_funcs]
        else:
            values = [sum(results[rf]["history"][key]) for rf in reward_funcs]

        x = np.arange(len(reward_funcs))
        ax.bar(x, values, color=[colors[rf] for rf in reward_funcs])
        ax.set_xticks(x)
        ax.set_xticklabels(
            [labels[rf].replace(" ", "\n") for rf in reward_funcs], fontsize=8
        )
        ax.set_ylabel("$1000s" if "Profit" in title or "Gap" in title else "Rate")
        ax.set_title(title)
        if "Gap" in title or "Disparity" in title:
            ax.axhline(y=0, color="black", linestyle="--")
        ax.grid(True, alpha=0.3, axis="y")

    # Parameters
    ax = fig.add_subplot(gs[4, 3])
    ax.axis("off")
    p_default = (theta_learner.default_rate_min + theta_learner.default_rate_max) / 2.0
    text = "Experiment Parameters:\n"
    text += "=" * 25 + "\n\n"
    text += f"Seed: {seed}\n"
    text += f"Constraint: {constraint_type}\n\n"
    text += f"θ_S = {theta_learner.theta_S:+.4f}\n"
    text += f"θ_X = {theta_learner.theta_X:+.4f}\n"
    text += f"b = {theta_learner.b:+.4f}\n\n"
    text += "Per-Individual (FIXED):\n"
    text += f"  Default: Bernoulli(p={p_default:.2f})\n"
    text += f"  Loan: N(${theta_learner.loan_amount_mean:.0f}k,${theta_learner.loan_amount_std:.0f}k)\n"

    ax.text(
        0.1,
        0.5,
        text,
        fontsize=9,
        family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.suptitle(
        f"Adult Income RL Analysis - Seed {seed}\nConstraint Type: {constraint_type}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        f"./results/adult_income_rl_{constraint_type}_seed{seed}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()
