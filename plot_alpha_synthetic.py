#!/usr/bin/env python3
"""
Generates a synthetic alpha-trajectory plot: a stationary OU process around 0.5.
Used for illustration when real trained weights are not yet available.
"""

import os
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

OUTPUT_PATH = "./alpha_trajectory_synthetic.png"
N_EPISODES  = 1000
N_SEEDS     = 10
THETA       = 0.08   # mean-reversion speed
SIGMA       = 0.045  # noise amplitude
RNG_SEED    = 42

_available_fonts = {f.name for f in fm.fontManager.ttflist}
if "STIX Two Text" in _available_fonts:
    _FONT = "STIX Two Text"
elif "STIXGeneral" in _available_fonts:
    _FONT = "STIXGeneral"
else:
    _FONT = "DejaVu Serif"

plt.rcParams.update({
    "font.family":    _FONT,
    "font.size":      11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize":10,
    "ytick.labelsize":10,
    "figure.dpi":     200,
    "savefig.dpi":    200,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
})

REWARD_META = {
    "utilitarian_profit":  {"color": "#1f77b4", "label": "Utilitarian Profit"},
    "social_welfare":      {"color": "#2ca02c", "label": "Social Welfare"},
    "rawlsian_maximin":    {"color": "#ff7f0e", "label": "Rawlsian Maximin"},
    "fairness_lagrangian": {"color": "#9467bd", "label": "Fairness Lagrangian"},
}


def ou_trajectory(n, theta=THETA, sigma=SIGMA, mu=0.5, x0=0.5, rng=None):
    """Discrete-time Ornstein-Uhlenbeck process clipped to (0, 1)."""
    if rng is None:
        rng = np.random.default_rng()
    x = np.empty(n)
    x[0] = x0
    noise = rng.normal(0, sigma, size=n)
    for t in range(1, n):
        x[t] = x[t-1] + theta * (mu - x[t-1]) + noise[t]
    return np.clip(x, 0.0, 1.0)


def main():
    rng = np.random.default_rng(RNG_SEED)
    eps = np.arange(1, N_EPISODES + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(True, lw=0.4, color="gray", alpha=0.35, linestyle="--", zorder=0)

    for reward_fn, meta in REWARD_META.items():
        # Each reward function gets a slightly different starting point for variety
        x0 = rng.uniform(0.42, 0.58)
        seeds = np.stack([
            ou_trajectory(N_EPISODES, x0=x0, rng=rng)
            for _ in range(N_SEEDS)
        ])
        mean = seeds.mean(axis=0)
        std  = seeds.std(axis=0)

        ax.plot(eps, mean, color=meta["color"], lw=1.8, label=meta["label"])
        ax.fill_between(eps, mean - std, mean + std,
                        color=meta["color"], alpha=0.15)

    ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"$\alpha$")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
