import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd


def plot_testing_results(episode_metrics_df: pd.DataFrame, env, agent, save_path=None):
    """Plot comprehensive testing results with episode number on x-axis."""

    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(5, 3, figure=fig)

    episodes = episode_metrics_df["episode"].values

    # 1. Inequality Ratio (ρ) per episode
    ax = fig.add_subplot(gs[0, 0])
    rho_episode = episode_metrics_df["rho_episode"].values
    rho_clipped = np.clip(rho_episode, -10, 10)
    ax.plot(episodes, rho_clipped, "b-o", linewidth=2, markersize=4, label="ρ per episode")
    ax.axhline(y=1.0, color="red", linestyle="--", label="Equal growth (ρ=1)")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("ρ (Inequality Ratio)")
    ax.set_title("Inequality Ratio: ρ = Δμ_M / Δμ_F (per episode)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Mean Wealth at episode end
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(episodes, episode_metrics_df["mu_M_end"], "b-o", linewidth=2, markersize=4, label="Male (μ_M)")
    ax.plot(episodes, episode_metrics_df["mu_F_end"], "r-o", linewidth=2, markersize=4, label="Female (μ_F)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Wealth ($1000s)")
    ax.set_title("Mean Wealth at Episode End")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Long-term Social Welfare R_g(t)
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(episodes, episode_metrics_df["R_M"], "b-o", linewidth=2, markersize=4, label="R_M (Male)")
    ax.plot(episodes, episode_metrics_df["R_F"], "r-o", linewidth=2, markersize=4, label="R_F (Female)")
    ax.plot(episodes, episode_metrics_df["R_total"], "g--o", linewidth=2, markersize=4, label="R_total")
    ax.axhline(y=0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("R_g(t) = (μ_g,t - μ_g,0) / t")
    ax.set_title("Long-term Social Welfare")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Wealth Gap at episode end
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(episodes, episode_metrics_df["wealth_gap"], "purple", linewidth=2, marker="o", markersize=4)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Wealth Gap ($1000s)")
    ax.set_title("Wealth Gap: μ_M - μ_F (at episode end)")
    ax.grid(True, alpha=0.3)

    # 5. Episode Approval Rates
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(episodes, episode_metrics_df["approval_rate_M_episode"], "b-o", linewidth=2, markersize=4, label="Male (episode)")
    ax.plot(episodes, episode_metrics_df["approval_rate_F_episode"], "r-o", linewidth=2, markersize=4, label="Female (episode)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Approval Rate")
    ax.set_title("Approval Rates per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Cumulative Loans Approved
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(episodes, episode_metrics_df["total_loans_M"], "b-o", linewidth=2, markersize=4, label="Male")
    ax.plot(episodes, episode_metrics_df["total_loans_F"], "r-o", linewidth=2, markersize=4, label="Female")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Loans (cumulative)")
    ax.set_title("Cumulative Loans Approved")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7. Episode Success Probabilities
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(episodes, episode_metrics_df["success_prob_M_episode"], "b-o", linewidth=2, markersize=4, label="Male")
    ax.plot(episodes, episode_metrics_df["success_prob_F_episode"], "r-o", linewidth=2, markersize=4, label="Female")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Probability")
    ax.set_title("Success Probabilities per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8. Actual vs True Approvals — Male
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(episodes, episode_metrics_df["actual_approvals_M_episode"], "b-o", linewidth=2, markersize=4, label="Actual (Agent)")
    ax.plot(episodes, episode_metrics_df["true_approvals_M_episode"], "g--s", linewidth=2, markersize=4, label="True (Ground Truth)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Approvals per Episode")
    ax.set_title("Male: Actual vs True Approvals")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 9. Actual vs True Approvals — Female
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(episodes, episode_metrics_df["actual_approvals_F_episode"], "r-o", linewidth=2, markersize=4, label="Actual (Agent)")
    ax.plot(episodes, episode_metrics_df["true_approvals_F_episode"], "g--s", linewidth=2, markersize=4, label="True (Ground Truth)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Approvals per Episode")
    ax.set_title("Female: Actual vs True Approvals")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 10. Cumulative Profit
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(episodes, episode_metrics_df["cumulative_profit"], "green", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Profit ($1000s)")
    ax.set_title("Cumulative Bank Profit")
    ax.grid(True, alpha=0.3)

    # 11. Episode Profit
    ax = fig.add_subplot(gs[3, 1])
    ax.bar(episodes, episode_metrics_df["profit_episode"], color="green", alpha=0.7, edgecolor="darkgreen")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Profit ($1000s)")
    ax.set_title("Profit per Episode")
    ax.grid(True, alpha=0.3, axis="y")

    # 12. Accuracy & Precision over episodes (cumulative)
    ax = fig.add_subplot(gs[3, 2])
    ax.plot(episodes, episode_metrics_df["accuracy_M"], "b-o", linewidth=2, markersize=4, label="Accuracy M")
    ax.plot(episodes, episode_metrics_df["accuracy_F"], "r-o", linewidth=2, markersize=4, label="Accuracy F")
    ax.plot(episodes, episode_metrics_df["precision_M"], "b--s", linewidth=1.5, markersize=3, alpha=0.7, label="Precision M")
    ax.plot(episodes, episode_metrics_df["precision_F"], "r--s", linewidth=1.5, markersize=3, alpha=0.7, label="Precision F")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Metric Value")
    ax.set_title("Cumulative Accuracy & Precision")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 13. Confusion Matrix — Male (Final)
    ax = fig.add_subplot(gs[4, 0])
    cm_M = np.array([[env.tp_M, env.fn_M], [env.fp_M, env.tn_M]])
    ax.imshow(cm_M, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Approve", "Reject"])
    ax.set_yticklabels(["Should Approve", "Should Reject"])
    ax.set_xlabel("Agent Decision")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Male Confusion Matrix (Final)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_M[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")

    # 14. Confusion Matrix — Female (Final)
    ax = fig.add_subplot(gs[4, 1])
    cm_F = np.array([[env.tp_F, env.fn_F], [env.fp_F, env.tn_F]])
    ax.imshow(cm_F, cmap="Reds")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Approve", "Reject"])
    ax.set_yticklabels(["Should Approve", "Should Reject"])
    ax.set_xlabel("Agent Decision")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Female Confusion Matrix (Final)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_F[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")

    # 15. Summary Statistics
    ax = fig.add_subplot(gs[4, 2])
    ax.axis("off")

    total_M = env.tp_M + env.fp_M + env.tn_M + env.fn_M
    total_F = env.tp_F + env.fp_F + env.tn_F + env.fn_F
    accuracy_M = (env.tp_M + env.tn_M) / max(total_M, 1)
    accuracy_F = (env.tp_F + env.tn_F) / max(total_F, 1)
    precision_M = env.tp_M / max(env.tp_M + env.fp_M, 1)
    precision_F = env.tp_F / max(env.tp_F + env.fp_F, 1)

    if env.current_time > 1e-8:
        R_M_final = (env.mu_M - env.mu_M_0) / env.current_time
        R_F_final = (env.mu_F - env.mu_F_0) / env.current_time
    else:
        R_M_final = R_F_final = 0.0

    rho_values = episode_metrics_df["rho_episode"].values

    text = "Testing Summary\n"
    text += "=" * 50 + "\n\n"
    text += f"Model: {agent.reward_func_name}\n"
    text += f"Constraint: {agent.constraint_type}\n"
    text += f"Episodes: {env.total_episodes}\n"
    text += f"Total Time: {env.current_time:.1f}\n\n"
    text += "--- Long-term Social Welfare ---\n"
    text += f"R_M: {R_M_final:.6f}\n"
    text += f"R_F: {R_F_final:.6f}\n"
    text += (
        f"R_M/R_F: {R_M_final / R_F_final:.4f}\n\n"
        if abs(R_F_final) > 1e-8
        else "R_M/R_F: N/A\n\n"
    )
    text += "--- Inequality Ratio ---\n"
    text += f"Mean ρ: {np.mean(rho_values):.4f}\n"
    text += f"Std ρ:  {np.std(rho_values):.4f}\n\n"
    text += "--- Accuracy (Final) ---\n"
    text += f"Male:   {accuracy_M:.4f}\n"
    text += f"Female: {accuracy_F:.4f}\n"

    ax.text(
        0.05,
        0.5,
        text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.suptitle(
        f"Testing Results: {agent.reward_func_name} ({agent.constraint_type}) — Episode-wise Metrics",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot to {save_path}")

    plt.show()
    return fig
