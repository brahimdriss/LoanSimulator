#!/usr/bin/env python3
"""
Rule-based baseline policies on the performative TestingIncomeEnvironment.

Two ways to run it:

  * Local sweep (original behaviour): all 7 policies, one process each,
    default 50 episodes at N=3000/3000, summary table + plots.

  * Cluster deploy (matches pg_adapt / pepg_adapt's Phase 2 exactly):
        --policy NAME --seed S --episodes 3000 --N-male 12000 --N-female 12000
        --population-snapshot-episodes 100,500,... --deploy-artifacts-dir DIR
    runs ONE policy for the agents' deploy horizon on the same population
    sample (SAMPLE_SIZE = 100000, seeds 0..9) and writes the same artefacts
    the agents write, under the same names with agent tag "rule" and
    constraint tag "baseline":
        DIR/rule_{policy}__baseline__seed{S}_episodes.csv
        DIR/rule_{policy}__baseline__seed{S}_population.npz   (+ snapshots)
    so plots/plot_lorenz_run10.py / plot_reach_rate_run10.py can read them
    with agent="rule", reward=policy, constraint="baseline".
    --aggregate then averages the per-seed episodes CSVs into
    mean_{policy}__baseline_*.csv / std_*.csv next to DIR, the files the
    reach-rate plot reads.

Every policy is stepped through env.step_cohort (batched, one action per
applicant in the cohort) -- the same code path the agents deploy through --
so the baselines and the agents see identical environment dynamics.

None of the 7 policies has a learnable component in this script:
always_approve / always_reject / uniform_acceptance / rich_becomes_richer /
reverse_rich_becomes_richer are fixed rules; oracle reads each applicant's
true default probability (fixed per individual for the whole run); and
pattern_prediction applies the logistic credit model fit ONCE on the Adult
data (theta_S, theta_X, b) to the applicant's CURRENT wealth, so it tracks
wealth drift but never refits on loan outcomes.
"""

import argparse
import glob
import multiprocessing as mp
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loan_simulator.testing.data_loader import TestingAdultIncomeDataLoader
from loan_simulator.testing.environment import TestingIncomeEnvironment
from loan_simulator.transition_learner import TransitionParameterLearner

# -----------a----------------------------------------------------------------
# Observation index constants
# ---------------------------------------------------------------------------
OBS_X_NORM   = 0   # X / 100  (credit-score proxy)
OBS_S        = 1   # group: 1 = male, 0 = female
OBS_MU_M     = 2   # mu_M / 100
OBS_MU_F     = 3   # mu_F / 100
OBS_THETA_PROB = 8  # P(approve | X, S) from learned logistic model


# ---------------------------------------------------------------------------
# Plot style (mirrors plotting.py conventions)
# ---------------------------------------------------------------------------

POLICY_COLORS = {
    "always_approve":             "#1f77b4",
    "always_reject":              "#d62728",
    "uniform_acceptance":         "#ff7f0e",
    "oracle":                     "#17becf",
    "pattern_prediction":         "#9467bd",
    "rich_becomes_richer":        "#8c564b",
    "reverse_rich_becomes_richer":"#e377c2",
}

POLICY_LABELS = {
    "always_approve":             "Always Approve",
    "always_reject":              "Always Reject",
    "uniform_acceptance":         "Uniform Acceptance",
    "oracle":                     "Oracle",
    "pattern_prediction":         "Pattern Prediction",
    "rich_becomes_richer":        "Rich Becomes Richer",
    "reverse_rich_becomes_richer":"Reverse Rich→Richer",
}


def setup_plot_style():
    """Setup matplotlib style matching plotting.py conventions."""
    plt.rcParams.update({
        "font.family":       ["Liberation Serif"],
        "font.size":         10,
        "axes.labelsize":    11,
        "axes.titlesize":    12,
        "legend.fontsize":   8,
        "legend.framealpha": 0.9,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "figure.dpi":        200,
        "savefig.dpi":       200,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "axes.axisbelow":    True,
    })


# ---------------------------------------------------------------------------
# Policy base class
# Interface: get_action(obs, applicant) -> float in [0, 1]
#   obs       — 12-dim observation vector from the environment
#   applicant — env.current_applicant dict (may be None between timesteps)
# ---------------------------------------------------------------------------

class RuleBasedPolicy:
    name: str = "base"

    def __init__(self, seed: int = 0):
        # Own RNG so a randomised policy's draws never touch the environment's
        # stream (the env seeds its own np.random.default_rng).
        self.rng = np.random.default_rng(seed)

    def get_actions(self, obs: np.ndarray, applicants: list) -> np.ndarray:
        """One approval probability per row of obs (shape [n, 12]);
        applicants is the matching list of env application dicts."""
        raise NotImplementedError

    def get_action(self, obs: np.ndarray, applicant: dict | None) -> float:
        """Single-applicant convenience wrapper (env.step interface)."""
        if applicant is None:
            return 0.0
        return float(self.get_actions(np.asarray(obs).reshape(1, -1), [applicant])[0])

    def reset(self):
        """Called at the start of every episode."""
        pass


# ---------------------------------------------------------------------------
# The 7 policies
# ---------------------------------------------------------------------------

class AlwaysApprovePolicy(RuleBasedPolicy):
    """Approve all loans unconditionally."""
    name = "always_approve"

    def get_actions(self, obs, applicants):
        return np.ones(len(obs))


class AlwaysRejectPolicy(RuleBasedPolicy):
    """Reject all loans unconditionally."""
    name = "always_reject"

    def get_actions(self, obs, applicants):
        return np.zeros(len(obs))


class UniformAcceptancePolicy(RuleBasedPolicy):
    """Randomised policy — approval probability drawn uniformly from [0, 1]."""
    name = "uniform_acceptance"

    def get_actions(self, obs, applicants):
        return self.rng.uniform(0.0, 1.0, size=len(obs))


class OraclePolicy(RuleBasedPolicy):
    """
    Oracle: has access to the applicant's true default_prob (now a continuous,
    creditworthiness-derived probability -- see
    TransitionParameterLearner.initialize_individual_parameters -- not a
    pre-realized 0/1 latent outcome). Approves iff expected profit is
    positive: (1-default_prob)*loan_amount*interest - default_prob*loan_amount > 0,
    i.e. default_prob < interest/(1+interest). This is the theoretical upper
    bound on bank profit given perfect knowledge of risk.

    interest_rate defaults to 0.18, matching TestingIncomeEnvironment's own
    default (used consistently throughout this script) -- reused, not a new
    invented constant. At interest=0.18 the breakeven default prob is
    0.1525, which sits inside the deploy band [0.05, 0.25], so this policy
    is genuinely selective (~51% approved) rather than degenerating into
    AlwaysApprove or AlwaysReject.
    """
    name = "oracle"

    def __init__(self, seed: int = 0, interest_rate: float = 0.18):
        super().__init__(seed)
        self.breakeven_default_prob = interest_rate / (1 + interest_rate)

    def get_actions(self, obs, applicants):
        d = np.array([a.get("default_prob", 1.0) for a in applicants], dtype=float)
        return (d < self.breakeven_default_prob).astype(float)


class PatternPredictionPolicy(RuleBasedPolicy):
    """
    Pattern prediction: uses the learned logistic regression credit model
    (theta_approval_prob) as the approval signal.
    Ground truth is NOT available at inference time.
    """
    name = "pattern_prediction"

    def get_actions(self, obs, applicants):
        # obs[:, OBS_THETA_PROB] = sigma(theta_S * S + theta_X * X_norm + b),
        # evaluated on the applicant's CURRENT wealth with the coefficients
        # fit once at start-up (TransitionParameterLearner.learn_approval_model).
        return np.clip(obs[:, OBS_THETA_PROB], 0.0, 1.0)


class RichBecomesRicherPolicy(RuleBasedPolicy):
    """
    Group-biased policy favouring the advantaged group (male).
      Male   (S = 1) → always approve  (p = 1.0)
      Female (S = 0) → uniform random  (p ~ U(0, 1))
    """
    name = "rich_becomes_richer"

    def get_actions(self, obs, applicants):
        male = np.round(obs[:, OBS_S]) == 1
        return np.where(male, 1.0, self.rng.uniform(0.0, 1.0, size=len(obs)))


class ReverseRichBecomesRicherPolicy(RuleBasedPolicy):
    """
    Reverse of the above — favours the disadvantaged group (female).
      Female (S = 0) → always approve  (p = 1.0)
      Male   (S = 1) → uniform random  (p ~ U(0, 1))
    """
    name = "reverse_rich_becomes_richer"

    def get_actions(self, obs, applicants):
        female = np.round(obs[:, OBS_S]) == 0
        return np.where(female, 1.0, self.rng.uniform(0.0, 1.0, size=len(obs)))


POLICY_CLASSES = {
    "always_approve":              AlwaysApprovePolicy,
    "always_reject":               AlwaysRejectPolicy,
    "uniform_acceptance":          UniformAcceptancePolicy,
    "oracle":                      OraclePolicy,
    "pattern_prediction":          PatternPredictionPolicy,
    "rich_becomes_richer":         RichBecomesRicherPolicy,
    "reverse_rich_becomes_richer": ReverseRichBecomesRicherPolicy,
}


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _build_env(
    loader: TestingAdultIncomeDataLoader,
    theta_learner: TransitionParameterLearner,
    N_male: int,
    N_female: int,
    T: int,
    dt: float,
    seed: int,
) -> TestingIncomeEnvironment:
    return TestingIncomeEnvironment(
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


# ---------------------------------------------------------------------------
# Per-policy runner
# ---------------------------------------------------------------------------

def run_policy(
    policy: RuleBasedPolicy,
    env: TestingIncomeEnvironment,
    num_episodes: int,
    verbose: bool = True,
    snapshot_episodes: list | None = None,
    snapshots_out: dict | None = None,
) -> pd.DataFrame:
    """
    Run *policy* on *env* for *num_episodes* continuous episodes through
    env.step_cohort (the agents' deploy code path). Returns the
    episode-metrics DataFrame.

    env.pending_applications (the cohort's application dicts, same order as
    the observation rows) is passed to get_actions so the Oracle policy can
    read each applicant's default_prob.

    snapshot_episodes: 1-based episode numbers at which to copy the
    population state into snapshots_out[ep] (same keys as pg_adapt's
    deploy snapshots).
    """
    pending = sorted(e for e in (snapshot_episodes or []) if e <= num_episodes)
    for episode in range(num_episodes):
        policy.reset()
        obs, _ = env.reset_cohort()
        done = False

        while not done:
            applicants = list(env.pending_applications)
            actions = policy.get_actions(obs, applicants) if len(applicants) else np.zeros(0)
            obs, terminated, truncated, _ = env.step_cohort(actions)
            done = terminated or truncated

        if pending and (episode + 1) == pending[0] and snapshots_out is not None:
            snapshots_out[pending.pop(0)] = {
                "X_male": env.current_X_male.copy(),
                "X_female": env.current_X_female.copy(),
                "loan_counts_M": env.loan_counts_M.copy(),
                "loan_counts_F": env.loan_counts_F.copy(),
            }

        if verbose and ((episode + 1) % 10 == 0 or episode + 1 == num_episodes):
            app_rate_M = env.episode_loans_M / max(env.episode_applications_M, 1)
            app_rate_F = env.episode_loans_F / max(env.episode_applications_F, 1)
            print(
                f"    Ep {episode + 1:3d}/{num_episodes} | "
                f"μ_M={env.mu_M:.1f}k  μ_F={env.mu_F:.1f}k  "
                f"gap={env.mu_M - env.mu_F:+.1f}k | "
                f"appR M={app_rate_M:.3f} F={app_rate_F:.3f} | "
                f"profit=${env.episode_profit:.1f}k"
            )

    env.finalize_episode_metrics()
    return env.get_episode_metrics_dataframe()


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_policy_summary(policy_name: str, env: TestingIncomeEnvironment):
    t = env.current_time
    R_M = (env.mu_M - env.mu_M_0) / t if t > 1e-8 else 0.0
    R_F = (env.mu_F - env.mu_F_0) / t if t > 1e-8 else 0.0
    app_rate_M = env.total_loans_M / max(env.total_applications_M, 1)
    app_rate_F = env.total_loans_F / max(env.total_applications_F, 1)
    total_M = env.tp_M + env.fp_M + env.tn_M + env.fn_M
    total_F = env.tp_F + env.fp_F + env.tn_F + env.fn_F
    acc_M = (env.tp_M + env.tn_M) / max(total_M, 1)
    acc_F = (env.tp_F + env.tn_F) / max(total_F, 1)

    print(f"\n  Policy : {policy_name}")
    print(f"  μ_M={env.mu_M:.2f}k  μ_F={env.mu_F:.2f}k  gap={env.mu_M - env.mu_F:+.2f}k")
    print(f"  R_M={R_M:.5f}  R_F={R_F:.5f}")
    print(
        f"  Approval rate  M={app_rate_M:.4f}  F={app_rate_F:.4f}  "
        f"disparity={app_rate_M - app_rate_F:+.4f}"
    )
    print(f"  Accuracy  M={acc_M:.4f}  F={acc_F:.4f}")
    print(f"  Cumulative profit=${env.cumulative_profit:.2f}k")
    print(f"  Hawkes events  M={len(env.event_times_R)}  F={len(env.event_times_B)}")


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def plot_comparison(
    metrics: dict,
    results_dir: str,
    timestamp: str,
):
    """2×2 grid: wealth gap, approval disparity, episode profit, R_g trajectories."""
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Rule-Based Policies — Performative Environment",
                 fontsize=13, fontweight="bold")

    # (a) Wealth gap trajectory
    ax = axes[0, 0]
    for name, df in metrics.items():
        ax.plot(df["episode"], df["wealth_gap"],
                label=POLICY_LABELS.get(name, name),
                color=POLICY_COLORS.get(name),
                linewidth=1.5)
    ax.axhline(0, color="k", linewidth=0.8, ls="-")
    ax.set_xlabel("Episode")
    ax.set_ylabel("μ_M − μ_F  ($k)")
    ax.set_title("(a) Wealth Gap")
    ax.legend(loc="best")

    # (b) Approval rate disparity trajectory
    ax = axes[0, 1]
    for name, df in metrics.items():
        disp = df["approval_rate_M_episode"] - df["approval_rate_F_episode"]
        ax.plot(df["episode"], disp,
                label=POLICY_LABELS.get(name, name),
                color=POLICY_COLORS.get(name),
                linewidth=1.5)
    ax.axhline(0, color="k", linewidth=0.8, ls="-")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Approval rate  M − F")
    ax.set_title("(b) Approval Rate Disparity")
    ax.legend(loc="best")

    # (c) Episode profit trajectory
    ax = axes[1, 0]
    for name, df in metrics.items():
        ax.plot(df["episode"], df["profit_episode"],
                label=POLICY_LABELS.get(name, name),
                color=POLICY_COLORS.get(name),
                linewidth=1.5)
    ax.axhline(0, color="k", linewidth=0.8, ls="-")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode profit ($k)")
    ax.set_title("(c) Episode Profit")
    ax.legend(loc="best")

    # (d) Long-term social welfare R_g(t) — solid = M, dashed = F
    ax = axes[1, 1]
    for name, df in metrics.items():
        color = POLICY_COLORS.get(name)
        label = POLICY_LABELS.get(name, name)
        ax.plot(df["episode"], df["R_M"],
                label=f"{label} (M)", color=color, linewidth=1.5, ls="-")
        ax.plot(df["episode"], df["R_F"],
                color=color, linewidth=1.5, ls="--", alpha=0.7,
                label=f"{label} (F)")
    ax.axhline(0, color="k", linewidth=0.8, ls="-")
    ax.set_xlabel("Episode")
    ax.set_ylabel("R_g = (μ_g,t − μ_g,0) / t")
    ax.set_title("(d) Long-term Social Welfare R_g(t)")
    ax.legend(loc="best", ncol=2)

    plt.tight_layout()
    path = os.path.join(results_dir, f"policy_comparison_{timestamp}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved comparison plot → {path}")


def plot_wealth_trajectories(
    metrics: dict,
    results_dir: str,
    timestamp: str,
):
    """2×2 grid: μ_M, μ_F, cumulative profit, inequality ratio ρ."""
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Metric Trajectories — Performative Environment",
                 fontsize=13, fontweight="bold")

    # (a) μ_M trajectory
    ax = axes[0, 0]
    for name, df in metrics.items():
        ax.plot(df["episode"], df["mu_M_end"],
                label=POLICY_LABELS.get(name, name),
                color=POLICY_COLORS.get(name),
                linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean wealth ($k)")
    ax.set_title("(a) μ_M — Male Mean Wealth")
    ax.legend(loc="best")

    # (b) μ_F trajectory
    ax = axes[0, 1]
    for name, df in metrics.items():
        ax.plot(df["episode"], df["mu_F_end"],
                label=POLICY_LABELS.get(name, name),
                color=POLICY_COLORS.get(name),
                linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean wealth ($k)")
    ax.set_title("(b) μ_F — Female Mean Wealth")
    ax.legend(loc="best")

    # (c) Cumulative profit trajectory
    ax = axes[1, 0]
    for name, df in metrics.items():
        ax.plot(df["episode"], df["cumulative_profit"],
                label=POLICY_LABELS.get(name, name),
                color=POLICY_COLORS.get(name),
                linewidth=1.5)
    ax.axhline(0, color="k", linewidth=0.8, ls="-")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative profit ($k)")
    ax.set_title("(c) Cumulative Bank Profit")
    ax.legend(loc="best")

    # (d) Inequality ratio ρ = Δμ_M / Δμ_F per episode
    ax = axes[1, 1]
    for name, df in metrics.items():
        ax.plot(df["episode"], df["rho_episode"],
                label=POLICY_LABELS.get(name, name),
                color=POLICY_COLORS.get(name),
                linewidth=1.5)
    ax.axhline(1, color="k", linewidth=0.8, ls="-", label="ρ = 1 (equal growth)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("ρ = Δμ_M / Δμ_F")
    ax.set_title("(d) Inequality Ratio ρ (per episode)")
    ax.set_ylim(-5, 10)
    ax.legend(loc="best")

    plt.tight_layout()
    path = os.path.join(results_dir, f"metric_trajectories_{timestamp}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved metric trajectories → {path}")


def plot_social_welfare_trajectories(
    metrics: dict,
    results_dir: str,
    timestamp: str,
):
    """
    1×3 grid mirroring plotting.py plot_4:
      (a) R_M(t)  — male long-term social welfare
      (b) R_F(t)  — female long-term social welfare
      (c) R̄(t)   — population-weighted average
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Long-term Social Welfare Trajectories",
                 fontsize=13, fontweight="bold")

    # First pass: collect data and compute shared y range
    y_min, y_max = np.inf, -np.inf
    plot_data = []
    for name, df in metrics.items():
        N_M = df["total_applications_M"]
        N_F = df["total_applications_F"]
        total_N = (N_M + N_F).replace(0, np.nan)
        R_bar = (N_M * df["R_M"] + N_F * df["R_F"]) / total_N
        plot_data.append((name, df, R_bar))
        for series in [df["R_M"], df["R_F"], R_bar]:
            y_min = min(y_min, series.min())
            y_max = max(y_max, series.max())

    margin = (y_max - y_min) * 0.05
    ylim = (y_min - margin, y_max + margin)

    # Second pass: draw with shared y range
    for name, df, R_bar in plot_data:
        color = POLICY_COLORS.get(name)
        label = POLICY_LABELS.get(name, name)
        episodes = df["episode"]
        axes[0].plot(episodes, df["R_M"], label=label, color=color, linewidth=1.5)
        axes[1].plot(episodes, df["R_F"], label=label, color=color, linewidth=1.5)
        axes[2].plot(episodes, R_bar,     label=label, color=color, linewidth=1.5)

    for ax in axes:
        ax.axhline(0, color="k", linewidth=0.8, ls="-")
        ax.set_ylim(ylim)
        ax.set_xlabel("Episode")
        ax.legend(loc="best")

    axes[0].set_ylabel(r"$R_M$  (welfare / episode)")
    axes[1].set_ylabel(r"$R_F$  (welfare / episode)")
    axes[2].set_ylabel(r"$\bar{R}$  (welfare / episode)")
    axes[0].set_title("(a) Male Long-term Social Welfare $R_M$")
    axes[1].set_title("(b) Female Long-term Social Welfare $R_F$")
    axes[2].set_title(r"(c) Weighted Average $\bar{R}$")

    plt.tight_layout()
    path = os.path.join(results_dir, f"social_welfare_{timestamp}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved social welfare plot → {path}")


def plot_inequality_ratio_trajectories(
    metrics: dict,
    results_dir: str,
    timestamp: str,
):
    """
    1×3 grid mirroring plotting.py plot_2:
      (a) Wealth gap  μ_M − μ_F  trajectory
      (b) Approval rate disparity  trajectory
      (c) Cumulative inequality ratio  ρ(t) = (μ_M(t) − μ_M(0)) / (μ_F(t) − μ_F(0))
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Wealth Inequality & Disparity Trajectories",
                 fontsize=13, fontweight="bold")

    for name, df in metrics.items():
        color = POLICY_COLORS.get(name)
        label = POLICY_LABELS.get(name, name)
        episodes = df["episode"]

        # (a) Wealth gap
        axes[0].plot(episodes, df["wealth_gap"],
                     label=label, color=color, linewidth=1.5)

        # (b) Approval rate disparity
        disp = df["approval_rate_M_episode"] - df["approval_rate_F_episode"]
        axes[1].plot(episodes, disp,
                     label=label, color=color, linewidth=1.5)

        # (c) Cumulative ρ(t) = (μ_M(t) − μ_M(0)) / (μ_F(t) − μ_F(0))
        mu_M_0 = df["mu_M_start"].iloc[0]
        mu_F_0 = df["mu_F_start"].iloc[0]
        delta_M = df["mu_M_end"] - mu_M_0
        delta_F = df["mu_F_end"] - mu_F_0
        rho_cumul = delta_M / delta_F.replace(0, np.nan)
        axes[2].plot(episodes, rho_cumul,
                     label=label, color=color, linewidth=1.5)

    axes[0].axhline(0, color="k", linewidth=0.8, ls="-")
    axes[1].axhline(0, color="k", linewidth=0.8, ls="-")
    axes[2].axhline(1, color="red", linewidth=1.0, ls="--", alpha=0.7,
                    label="ρ = 1 (equal growth)")

    axes[0].set_xlabel("Episode")
    axes[1].set_xlabel("Episode")
    axes[2].set_xlabel("Episode")

    axes[0].set_ylabel("μ_M − μ_F  ($k)")
    axes[1].set_ylabel("Approval rate  M − F")
    axes[2].set_ylabel(r"$\rho(t)$")

    axes[0].set_title("(a) Wealth Gap Trajectory")
    axes[1].set_title("(b) Approval Rate Disparity Trajectory")
    axes[2].set_title(r"(c) Inequality Ratio $\rho(t)$")

    for ax in axes:
        ax.legend(loc="best")

    plt.tight_layout()
    path = os.path.join(results_dir, f"inequality_ratio_{timestamp}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved inequality ratio plot → {path}")


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------

def _policy_worker(args):
    """Run a single policy in a worker process. Returns (name, DataFrame)."""
    policy, loader, theta_learner, N_male, N_female, T, dt, seed, num_episodes = args
    np.random.seed(seed)
    env = _build_env(loader, theta_learner, N_male, N_female, T, dt, seed)
    df = run_policy(policy, env, num_episodes=num_episodes, verbose=False)
    return policy.name, df


# ---------------------------------------------------------------------------
# Cluster deploy: one policy, one seed, agents' artefact layout
# ---------------------------------------------------------------------------

AGENT_TAG = "rule"
CONSTRAINT_TAG = "baseline"


def deploy_one(args, loader, theta_learner):
    policy = POLICY_CLASSES[args.policy](seed=args.seed)
    stem = f"{AGENT_TAG}_{policy.name}__{CONSTRAINT_TAG}__seed{args.seed}"
    out_dir = args.deploy_artifacts_dir
    os.makedirs(out_dir, exist_ok=True)
    episodes_path = os.path.join(out_dir, f"{stem}_episodes.csv")
    if os.path.exists(episodes_path) and not args.force:
        print(f"  SKIP {stem}: {episodes_path} exists (pass --force to redo)")
        return

    snapshot_eps = [int(e) for e in args.population_snapshot_episodes.split(",") if e.strip()]
    np.random.seed(args.seed)
    env = _build_env(loader, theta_learner, args.N_male, args.N_female, args.T, args.dt, args.seed)
    snapshots = {}
    print(f"  DEPLOY {stem}: {args.episodes} episodes, snapshots at {snapshot_eps}")
    df = run_policy(policy, env, num_episodes=args.episodes, verbose=True,
                    snapshot_episodes=snapshot_eps, snapshots_out=snapshots)

    df.to_csv(episodes_path, index=False)
    npz_payload = {
        "X_male": env.current_X_male, "X_female": env.current_X_female,
        "loan_counts_M": env.loan_counts_M, "loan_counts_F": env.loan_counts_F,
    }
    for snap_ep, snap in snapshots.items():
        for arr_name, arr_val in snap.items():
            npz_payload[f"{arr_name}_ep{snap_ep}"] = arr_val
    np.savez_compressed(os.path.join(out_dir, f"{stem}_population.npz"), **npz_payload)
    row = df.iloc[-1]
    print(f"  DEPLOY OK {stem}: mu_M={row['mu_M_end']:.1f} mu_F={row['mu_F_end']:.1f} "
          f"appM={df['approval_rate_M_cumulative'].iloc[-1]:.3f} "
          f"appF={df['approval_rate_F_cumulative'].iloc[-1]:.3f} "
          f"profit={df['cumulative_profit'].iloc[-1]:.1f}")


def aggregate(args):
    """Per-policy mean/std over seeds of the deploy episodes CSVs, written
    next to the deploy_artifacts dir in the agents' aggregate layout:
        <parent>/mean_{policy}__baseline_<timestamp>.csv, std_...
    (plots/plot_reach_rate_run10.py globs mean_{reward}__{constraint}_*.csv
    under <root>/<agent>/, so <parent> should be <root>/rule)."""
    out_dir = args.deploy_artifacts_dir
    parent = os.path.dirname(os.path.normpath(out_dir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in POLICY_CLASSES:
        paths = sorted(glob.glob(os.path.join(out_dir, f"{AGENT_TAG}_{name}__{CONSTRAINT_TAG}__seed*_episodes.csv")))
        if not paths:
            print(f"  aggregate {name}: no seeds found, skipping")
            continue
        dfs = [pd.read_csv(p) for p in paths]
        n = min(len(d) for d in dfs)
        num_cols = [c for c in dfs[0].columns if np.issubdtype(dfs[0][c].dtype, np.number)]
        stack = np.stack([d[num_cols].iloc[:n].to_numpy(dtype=float) for d in dfs])
        mdf = pd.DataFrame(stack.mean(axis=0), columns=num_cols)
        sdf = pd.DataFrame(stack.std(axis=0, ddof=0), columns=num_cols)
        mdf["episode"] = dfs[0]["episode"].iloc[:n].to_numpy()
        sdf["episode"] = mdf["episode"]
        mdf.to_csv(os.path.join(parent, f"mean_{name}__{CONSTRAINT_TAG}_{timestamp}.csv"), index=False)
        sdf.to_csv(os.path.join(parent, f"std_{name}__{CONSTRAINT_TAG}_{timestamp}.csv"), index=False)
        print(f"  aggregate {name}: {len(paths)} seeds x {n} episodes -> {parent}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rule-Based Policy Baselines on Performative Testing Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--episodes", type=int, default=50,
                        help="Continuous episodes per policy (default: 50)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--N-male", type=int, default=3000)
    parser.add_argument("--N-female", type=int, default=3000)
    parser.add_argument("--T", type=int, default=100,
                        help="Time horizon per episode (default: 100)")
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--data", type=str, default=None,
                        help="Path to adult.csv (None = auto-download)")
    parser.add_argument("--credit-threshold", type=float, default=0.5,
                        help="Ground-truth creditworthiness threshold (default: 0.5)")
    parser.add_argument("--results-dir", type=str, default="./rule_policy_results")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--policy", type=str, default=None, choices=sorted(POLICY_CLASSES),
                        help="Cluster deploy mode: run ONLY this policy for --episodes and "
                             "write the agents' deploy artefacts (see module docstring)")
    parser.add_argument("--deploy-artifacts-dir", type=str, default=None,
                        help="Where --policy / --aggregate read and write artefacts "
                             "(e.g. ~/eutopia_runs/run11/rule/deploy_artifacts)")
    parser.add_argument("--population-snapshot-episodes", type=str, default="",
                        help="Comma-separated deploy episodes at which to snapshot the "
                             "population into population.npz (same as pg_adapt)")
    parser.add_argument("--aggregate", action="store_true",
                        help="Average the per-seed deploy artefacts into mean_/std_ CSVs")
    parser.add_argument("--force", action="store_true",
                        help="Redo a --policy run whose episodes CSV already exists")
    parser.add_argument("--sample-size", type=int, default=100000,
                        help="Adult rows to load (100000 = pg_adapt/pepg_adapt's SAMPLE_SIZE)")
    args = parser.parse_args()

    if args.aggregate:
        if not args.deploy_artifacts_dir:
            parser.error("--aggregate needs --deploy-artifacts-dir")
        aggregate(args)
        return

    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("RULE-BASED POLICY BASELINES — PERFORMATIVE TESTING ENVIRONMENT")
    print("=" * 70)
    print(f"  Episodes / policy : {args.episodes}")
    print(f"  Population        : {args.N_male}M + {args.N_female}F")
    print(f"  T={args.T}  dt={args.dt}  seed={args.seed}")
    print("=" * 70)

    print("\n[1] Loading and preprocessing data...")
    loader = TestingAdultIncomeDataLoader(
        filepath=args.data,
        sample_size=args.sample_size,
        credit_threshold=args.credit_threshold,
    )
    loader.load_data()
    loader.preprocess()

    print("\n[2] Fitting transition parameters...")
    theta_learner = TransitionParameterLearner(
        default_rate_min=0.05, default_rate_max=0.25
    )
    theta_learner.fit(loader.data)

    if args.policy:
        if not args.deploy_artifacts_dir:
            parser.error("--policy needs --deploy-artifacts-dir")
        deploy_one(args, loader, theta_learner)
        return

    # ------------------------------------------------------------------ #
    # Policies to evaluate
    # ------------------------------------------------------------------ #
    policies = [cls(seed=args.seed) for cls in POLICY_CLASSES.values()]

    # ------------------------------------------------------------------ #
    # Load existing checkpoint CSVs — skip those policies (pepg_adapt style)
    # ------------------------------------------------------------------ #
    all_metrics    = {}
    policies_to_run = []
    n_loaded = 0

    for policy in policies:
        ckpt_path = os.path.join(checkpoint_dir, f"policy_{policy.name}_seed{args.seed}.csv")
        if os.path.exists(ckpt_path):
            try:
                all_metrics[policy.name] = pd.read_csv(ckpt_path)
                n_loaded += 1
                print(f"  Loaded checkpoint: {policy.name}")
            except Exception:
                policies_to_run.append(policy)
        else:
            policies_to_run.append(policy)

    if n_loaded:
        print(f"\n  Loaded {n_loaded} checkpoint(s) — skipping those policies.")

    # ------------------------------------------------------------------ #
    # Run remaining policies in parallel (one process per policy)
    # ------------------------------------------------------------------ #
    if policies_to_run:
        worker_args = [
            (policy, loader, theta_learner,
             args.N_male, args.N_female, args.T, args.dt,
             args.seed, args.episodes)
            for policy in policies_to_run
        ]

        n_workers = min(len(policies_to_run), mp.cpu_count())
        print(f"\n[3] Running {len(policies_to_run)} policies in parallel ({n_workers} workers)...")

        with mp.Pool(processes=n_workers) as pool:
            for name, df in pool.imap_unordered(_policy_worker, worker_args):
                all_metrics[name] = df
                ckpt_path = os.path.join(checkpoint_dir, f"policy_{name}_seed{args.seed}.csv")
                df.to_csv(ckpt_path, index=False)
                print(f"  Checkpointed: {name}")
    else:
        print("\n[3] All policies already checkpointed — skipping to saving.")

    # Print summaries in original policy order
    for policy in policies:
        df = all_metrics[policy.name]
        row = df.iloc[-1]
        t = row["time_end"]
        R_M = row["R_M"]
        R_F = row["R_F"]
        app_rate_M = df["approval_rate_M_cumulative"].iloc[-1]
        app_rate_F = df["approval_rate_F_cumulative"].iloc[-1]
        print(f"\n  Policy : {policy.name}")
        print(f"  μ_M_end={row['mu_M_end']:.2f}k  μ_F_end={row['mu_F_end']:.2f}k  "
              f"gap={row['wealth_gap']:+.2f}k")
        print(f"  R_M={R_M:.5f}  R_F={R_F:.5f}")
        print(f"  Approval rate  M={app_rate_M:.4f}  F={app_rate_F:.4f}  "
              f"disparity={app_rate_M - app_rate_F:+.4f}")
        print(f"  Cumulative profit=${df['cumulative_profit'].iloc[-1]:.2f}k")

    # ------------------------------------------------------------------ #
    # Save per-policy CSVs
    # ------------------------------------------------------------------ #
    print(f"\n[4] Saving results to {args.results_dir}/")
    for name, df in all_metrics.items():
        csv_path = os.path.join(
            args.results_dir, f"policy_{name}_seed{args.seed}_{timestamp}.csv"
        )
        df.to_csv(csv_path, index=False)
        print(f"  {name} → {os.path.basename(csv_path)}")

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 76}")
    print(f"{'Policy':<32} {'R_M':>8} {'R_F':>8} {'appM':>7} {'appF':>7} "
          f"{'gap':>8} {'profit':>10}")
    print(f"{'—' * 76}")
    for name, df in all_metrics.items():
        row = df.iloc[-1]
        app_rate_M = df["approval_rate_M_cumulative"].iloc[-1]
        app_rate_F = df["approval_rate_F_cumulative"].iloc[-1]
        profit_total = df["cumulative_profit"].iloc[-1]
        wealth_gap = df["wealth_gap"].iloc[-1]
        print(
            f"  {name:<30} {row['R_M']:>8.5f} {row['R_F']:>8.5f} "
            f"{app_rate_M:>7.4f} {app_rate_F:>7.4f} "
            f"{wealth_gap:>+8.2f} {profit_total:>10.2f}k"
        )
    print(f"{'=' * 76}")

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    if not args.no_plots:
        print("\n[5] Generating plots...")
        plot_comparison(all_metrics, args.results_dir, timestamp)
        plot_wealth_trajectories(all_metrics, args.results_dir, timestamp)
        plot_social_welfare_trajectories(all_metrics, args.results_dir, timestamp)
        plot_inequality_ratio_trajectories(all_metrics, args.results_dir, timestamp)

    print("\nDone.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # required on macOS / Windows
    main()
