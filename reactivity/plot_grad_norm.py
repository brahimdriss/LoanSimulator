from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT  = HERE / "results"


REWARD_LABELS = {
    "utilitarian_profit":  "Util. Profit",
    "social_welfare":      "Social Welfare",
    "rawlsian_maximin":    "Rawlsian Max-Min",
    "fairness_lagrangian": "Fairness Lagrangian",
}

REWARD_CONSTRAINT_COLORS = {
    ("utilitarian_profit",  "social"): "#1f77b4",
    ("utilitarian_profit",  "dm"):     "#17becf",
    ("social_welfare",      "social"): "#2ca02c",
    ("social_welfare",      "dm"):     "#bcbd22",
    ("rawlsian_maximin",    "social"): "#ff7f0e",
    ("rawlsian_maximin",    "dm"):     "#d62728",
    ("fairness_lagrangian", "social"): "#9467bd",
    ("fairness_lagrangian", "dm"):     "#e377c2",
}

ALGO_CONSTRAINT_STYLE = {
    ("pg",   "social"): "--",
    ("pg",   "dm"):     ":",
    ("pepg", "social"): "-",
    ("pepg", "dm"):     "-.",
}

ALGO_LABEL = {"pg": "PG", "pepg": "PePG"}

_RE = re.compile(
    r"^deploy_reactivity_(?P<reward>[a-z_]+)__(?P<constraint>[a-z_]+)__seed(?P<seed>\d+)\.csv$"
)


def load_all() -> pd.DataFrame:
    rows = []
    for algo in ("pg", "pepg"):
        for path in sorted((DATA / algo).glob("deploy_reactivity_*.csv")):
            m = _RE.match(path.name)
            if not m:
                continue
            df = pd.read_csv(path)
            df["algo"]       = algo
            df["reward"]     = m["reward"]
            df["constraint"] = m["constraint"]
            df["seed"]       = int(m["seed"])
            rows.append(df)
    if not rows:
        raise SystemExit(f"No CSVs found under {DATA}")
    return pd.concat(rows, ignore_index=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["algo", "reward", "constraint", "seed", "episode"]).copy()
    df["grad_norm_cum"] = (
        df.groupby(["algo", "reward", "constraint", "seed"])["grad_norm"].cumsum()
    )
    return (
        df.groupby(["algo", "reward", "constraint", "episode"], as_index=False)
          .agg(
              grad_norm_mu     =("grad_norm",     "mean"),
              grad_norm_sd     =("grad_norm",     "std"),
              grad_norm_cum_mu =("grad_norm_cum", "mean"),
              grad_norm_cum_sd =("grad_norm_cum", "std"),
              n_seeds          =("seed",          "nunique"),
          )
    )


def plot_grad_norm(agg: pd.DataFrame, out_path: Path, *, cumulative: bool):
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 8, "figure.dpi": 200, "savefig.dpi": 200,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    })

    mu_col = "grad_norm_cum_mu" if cumulative else "grad_norm_mu"
    sd_col = "grad_norm_cum_sd" if cumulative else "grad_norm_sd"
    title  = ("Cumulative " if cumulative else "Per-episode ") + r"$\|\partial\pi/\partial\mu\|$ over deploy"
    ylabel = (r"$\sum_{s \leq t}\,\|\partial\pi/\partial\mu\|_s$"
              if cumulative else r"$\|\partial\pi/\partial\mu\|_t$")

    rewards = list(REWARD_LABELS.keys())
    constraints = ("predictive", "social", "dm", "two_sided")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    fig.suptitle(f"{title}  (mean ± std across seeds)",
                 fontsize=12, fontweight="bold")

    for ax, reward in zip(axes.flat, rewards):
        any_curve = False
        for constraint in constraints:
            for algo in ("pg", "pepg"):
                ls = ALGO_CONSTRAINT_STYLE.get((algo, constraint))
                if ls is None:
                    continue
                sub = agg[(agg["algo"] == algo)
                          & (agg["reward"] == reward)
                          & (agg["constraint"] == constraint)]
                if sub.empty:
                    continue
                any_curve = True
                ep = sub["episode"]
                mu = sub[mu_col]
                sd = sub[sd_col].fillna(0.0)
                color = REWARD_CONSTRAINT_COLORS.get((reward, constraint), "#444444")
                ax.plot(ep, mu,
                        label=f"{ALGO_LABEL[algo]} / {constraint}",
                        color=color, linestyle=ls, linewidth=1.5, alpha=0.9)
                ax.fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.08)
        ax.set_title(REWARD_LABELS[reward])
        ax.set_xlabel("Deploy episode")
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        if any_curve:
            ax.legend(loc="best", ncol=2, fontsize=7)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved → {out_path}")


def main():
    df = load_all()
    print(f"Loaded {len(df)} rows: "
          f"{df['algo'].nunique()} algos, {df['seed'].nunique()} seeds, "
          f"{df['episode'].nunique()} episodes.")
    agg = aggregate(df)
    plot_grad_norm(agg, OUT / "grad_norm_per_episode.png", cumulative=False)
    plot_grad_norm(agg, OUT / "grad_norm_cumulative.png",  cumulative=True)


if __name__ == "__main__":
    main()
