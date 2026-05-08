#!/usr/bin/env python3
"""
plot_alpha_trajectory.py

Plots the learned alpha (blend weight) trajectory across training episodes
for each PePG reward function under the two_sided constraint.

Reads lambda_history["wealth"] directly from saved .pt weight files.
Files must follow the naming convention:
    pepg_{reward_function}__two_sided__seed{seed}.pt

Usage (standalone, no repo dependency needed beyond torch):
    python plot_alpha_trajectory.py --input-dir ./weights_pepg_adapt
"""

# ---- USER SETTINGS ----------------------------------------------------------
INPUT_DIR  = "./weights_pepg_adapt_10s"
OUTPUT_DIR = INPUT_DIR   # saves plot alongside weights
# -----------------------------------------------------------------------------

import argparse
import glob
import os
import re

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch

# =============================================================================
# Font: STIX Two Text (falls back to STIXGeneral if not available)
# =============================================================================
_available_fonts = {f.name for f in fm.fontManager.ttflist}
if "STIX Two Text" in _available_fonts:
    _FONT = "STIX Two Text"
elif "STIXGeneral" in _available_fonts:
    _FONT = "STIXGeneral"
else:
    _FONT = "DejaVu Serif"
print(f"Font: {_FONT}")

plt.rcParams.update({
    "font.family":       _FONT,
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    13,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "figure.dpi":        200,
    "savefig.dpi":       200,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "axes.axisbelow":    True,
})

# =============================================================================
# Reward function metadata
# =============================================================================
REWARD_META = {
    "utilitarian_profit":   {"color": "#1f77b4", "label": "Utilitarian Profit"},
    "social_welfare":       {"color": "#2ca02c", "label": "Social Welfare"},
    "rawlsian_maximin":     {"color": "#ff7f0e", "label": "Rawlsian Maximin"},
    "fairness_lagrangian":  {"color": "#9467bd", "label": "Fairness Lagrangian"},
}

# =============================================================================
# Load alpha trajectories
# =============================================================================
def load_trajectories(input_dir):
    """
    Scans input_dir for pepg_*__two_sided__seed*.pt files.
    Returns dict: {reward_function: [trajectory_seed0, trajectory_seed1, ...]}
    where each trajectory is a list of alpha values (one per episode).
    """
    pattern = os.path.join(input_dir, "*__two_sided__seed*.pt")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"  No files matching *__two_sided__seed*.pt in {input_dir}")
        return {}

    trajectories = {}
    for path in files:
        fname = os.path.basename(path)
        m = re.match(r"(?:pepg_)?(.+)__two_sided__seed(\d+)\.pt", fname)
        if not m:
            continue
        reward_fn = m.group(1)
        seed      = int(m.group(2))

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  WARNING: could not load {fname}: {e}")
            continue

        history = ckpt.get("lambda_history", {}).get("wealth", [])
        if not history:
            print(f"  WARNING: no lambda_history in {fname}")
            continue

        if reward_fn not in trajectories:
            trajectories[reward_fn] = []
        trajectories[reward_fn].append(np.array(history, dtype=np.float64))
        print(f"  Loaded {fname}: {len(history)} episodes")

    return trajectories

# =============================================================================
# Plot
# =============================================================================
def plot_alpha(trajectories, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(True, lw=0.4, color="gray", alpha=0.35, linestyle="--", zorder=0)

    for reward_fn, series in trajectories.items():
        meta = REWARD_META.get(reward_fn, {"color": "black", "label": reward_fn})

        min_len = min(len(s) for s in series)
        mat     = np.stack([s[:min_len] for s in series])
        mean    = mat.mean(axis=0)
        std     = mat.std(axis=0)
        eps     = np.arange(1, min_len + 1)

        ax.plot(eps, mean, color=meta["color"], lw=1.8)
        ax.fill_between(eps, mean - std, mean + std,
                        color=meta["color"], alpha=0.15)

    ax.set_xlabel("Episode")
    ax.set_ylabel(r"$\alpha$")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out = os.path.join(output_dir, "alpha_trajectory_two_sided.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out}")

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  type=str, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning: {args.input_dir}")
    trajectories = load_trajectories(args.input_dir)

    if not trajectories:
        print("No data found. Exiting.")
        return

    print(f"\nFound {len(trajectories)} reward function(s):")
    for k, v in trajectories.items():
        print(f"  {k}: {len(v)} seed(s)")

    plot_alpha(trajectories, output_dir)


if __name__ == "__main__":
    main()