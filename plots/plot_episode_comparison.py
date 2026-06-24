# =============================================================================
# Colab cell: nth episode vs final episode — histograms + Lorenz curves
# Standalone — no repo dependency, just point INPUT_DIR at your results folder
# =============================================================================

# ---- USER SETTINGS ----------------------------------------------------------
INPUT_DIR     = "/content/drive/MyDrive/LoanSimulator/results_individual_freq"
OUTPUT_DIR    = INPUT_DIR + "/plots_regen"
EPISODE_N     = 100        # episode to compare against final
FINAL_EPISODE = 1000
POLICY        = "pepg"     # "pg" or "pepg"
SEEDS         = [0, 1, 2]
N_MALE        = 3000
N_FEMALE      = 3000
N_BINS        = 30
RUN_HIST      = True       # Plot 1 — histograms
RUN_LORENZ    = True       # Plot 2 — Lorenz curves
RUN_REACH     = True       # Plot 3 — reach rate curves
# -----------------------------------------------------------------------------

import io, json, os, subprocess, urllib.request, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as fm

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Font: STIX Two Text  (falls back to STIXGeneral if download fails)
# =============================================================================
_STIX2_DIR   = "/usr/share/fonts/stix2"
_STIX2_FILES = ["STIXTwoText-Regular.otf", "STIXTwoText-Bold.otf",
                "STIXTwoText-Italic.otf",  "STIXTwoText-BoldItalic.otf"]
os.makedirs(_STIX2_DIR, exist_ok=True)

def _install_stix2():
    base = "https://raw.githubusercontent.com/stipub/stixfonts/main/fonts/static_otf/"
    try:
        for fname in _STIX2_FILES:
            dest = os.path.join(_STIX2_DIR, fname)
            if not os.path.exists(dest):
                print(f"  Downloading {fname}…")
                urllib.request.urlretrieve(base + fname, dest)
        return True
    except Exception as e:
        print(f"  Direct download failed ({e}) — trying release zip…")
    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/stipub/stixfonts/releases/latest",
            headers={"User-Agent": "python"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        zip_url = next((a["browser_download_url"] for a in data.get("assets", [])
                        if a["name"].endswith(".zip")), None)
        if zip_url:
            print("  Downloading release zip…")
            with urllib.request.urlopen(zip_url, timeout=120) as r:
                zf = zipfile.ZipFile(io.BytesIO(r.read()))
            for name in zf.namelist():
                if name.endswith(".otf") and "TwoText" in name:
                    dest = os.path.join(_STIX2_DIR, os.path.basename(name))
                    with open(dest, "wb") as f:
                        f.write(zf.read(name))
                    print(f"    Extracted: {os.path.basename(name)}")
            return True
    except Exception as e:
        print(f"  Release zip failed ({e}) — will use STIXGeneral.")
    return False

if not all(os.path.exists(os.path.join(_STIX2_DIR, f)) for f in _STIX2_FILES):
    _install_stix2()
    subprocess.run(["fc-cache", "-fv"], capture_output=True)
    fm.fontManager = fm.FontManager()

_FONT = "STIX Two Text" if "STIX Two Text" in {f.name for f in fm.fontManager.ttflist} \
        else "STIXGeneral"
print(f"Font: {_FONT}")

plt.rcParams.update({
    "font.family":       _FONT,
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    13,
    "legend.fontsize":   9,
    "legend.framealpha": 0.9,
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
# Colors
# =============================================================================
COLOR_M       = "#FF0000"   # bright red   — male final
COLOR_F       = "#0000FF"   # bright blue  — female final
COLOR_M_N     = "#FF9999"   # light red    — male nth
COLOR_F_N     = "#9999FF"   # light blue   — female nth
_GRID_KW      = dict(lw=0.4, color="gray", alpha=0.35, zorder=0)

# =============================================================================
# Helpers
# =============================================================================
policy_dir = os.path.join(INPUT_DIR, POLICY)

def _load_counts(group, ep):
    """Average cumulative loan counts across seeds. ep = int or 'final'."""
    arrays = []
    for seed in SEEDS:
        path = (os.path.join(policy_dir, f"seed{seed}", group, "final_loan_counts.csv")
                if ep == "final" else
                os.path.join(policy_dir, f"seed{seed}", group, "counts",
                             f"loan_counts_{ep:04d}.csv"))
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}")
            continue
        arrays.append(pd.read_csv(path)["loan_count"].values)
    return np.mean(arrays, axis=0) if arrays else None

def _load_lorenz(group, ep):
    """Average Lorenz curve across seeds."""
    curves, pop_ref = [], None
    for seed in SEEDS:
        path = os.path.join(policy_dir, f"seed{seed}", group, "lorenz",
                            f"episode_{ep:04d}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if pop_ref is None:
            pop_ref = df["cum_pop_fraction"].values
        curves.append(df["cum_loan_fraction"].values)
    return (pop_ref, np.mean(curves, axis=0)) if curves else (None, None)

def _bin(counts, N):
    bw    = N // N_BINS
    edges = [i * bw for i in range(N_BINS)] + [N]
    tots  = np.array([counts[edges[i]:edges[i+1]].sum() for i in range(N_BINS)])
    ctrs  = np.array([(edges[i] + edges[i+1]) / 2 for i in range(N_BINS)])
    return ctrs, tots, bw * 0.85

# shared episode list used by plots 2 & 3
_EPS      = sorted({ep for ep in [100, 400, 800, 1000] if ep <= FINAL_EPISODE} | {EPISODE_N})
_colors_m = cm.Reds(np.linspace(0.30, 0.95, len(_EPS)))
_colors_f = cm.Blues(np.linspace(0.30, 0.95, len(_EPS)))

# =============================================================================
# Plot 1 — Histograms: nth episode (light) vs final (bright)
# =============================================================================
if RUN_HIST:
    fig, axes = plt.subplots(2, 1, figsize=(11, 10))
    for ax, group, N, col_n, col_f, label in zip(
            axes,
            ("male",    "female"),
            (N_MALE,    N_FEMALE),
            (COLOR_M_N, COLOR_F_N),
            (COLOR_M,   COLOR_F),
            ("Male (Red Group)", "Female (Blue Group)"),
    ):
        ax.grid(True, axis="y", **_GRID_KW)
        c_n = _load_counts(group, EPISODE_N)
        c_f = _load_counts(group, "final")
        if c_n is not None:
            ctrs, tots, w = _bin(c_n, N)
            ax.bar(ctrs, tots, width=w, color=col_n, alpha=1.0, zorder=2,
                   edgecolor="none", label=f"Episode {EPISODE_N}")
        if c_f is not None:
            ctrs, tots, w = _bin(c_f, N)
            ax.bar(ctrs, tots, width=w, color=col_f, alpha=0.55, zorder=3,
                   edgecolor="none", label=f"Episode {FINAL_EPISODE} (final)")
        ax.set_title(label)
        ax.set_ylabel("Total loans received (cumulative)")
        ax.set_xlabel(f"Individual index  ({N // N_BINS} per bin)")
        ax.legend()
    fig.suptitle(
        f"{POLICY.upper()} — Episode {EPISODE_N} vs Episode {FINAL_EPISODE}"
        f"  (avg over {len(SEEDS)} seeds)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    _out = os.path.join(OUTPUT_DIR, f"{POLICY}_histogram_ep{EPISODE_N}_vs_ep{FINAL_EPISODE}.png")
    fig.savefig(_out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"Saved: {_out}")

# =============================================================================
# Plot 2 — Lorenz curves (male and female, separate figures)
# =============================================================================
if RUN_LORENZ:
    for group, label, _colors in [
            ("male",   "Male (Red Group)",    _colors_m),
            ("female", "Female (Blue Group)", _colors_f)]:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.grid(True, **_GRID_KW)
        for i, ep in enumerate(_EPS):
            pop, loan = _load_lorenz(group, ep)
            if pop is None:
                continue
            ax.plot(pop, loan,
                    color=_colors[i],
                    lw=2.4 if ep == EPISODE_N else 1.6,
                    ls="--" if ep == EPISODE_N else "-",
                    label=f"ep {ep}" + ("  ←" if ep == EPISODE_N else ""))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect equality")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Cumulative fraction of individuals")
        ax.set_ylabel("Cumulative fraction of loans received")
        ax.set_title(f"{POLICY.upper()} — {label}\nLoan Lorenz Curves"
                     f"  (avg over {len(SEEDS)} seeds)")
        ax.legend(fontsize=9)
        plt.tight_layout()
        _out = os.path.join(OUTPUT_DIR, f"{POLICY}_{group}_lorenz_regen.png")
        fig.savefig(_out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.show()
        print(f"Saved: {_out}")

# =============================================================================
# Plot 3 — Reach rate curves (male & female superposed)
# =============================================================================
def _load_reach_rates(group):
    """Full reach-rate trajectory averaged across seeds.
    reach_rate = unique loan recipients that episode / N  (from metrics.csv).
    Returns (episodes, mean, std) or (None, None, None) if no data."""
    series = []
    ep_ref = None
    for seed in SEEDS:
        path = os.path.join(policy_dir, f"seed{seed}", group, "metrics.csv")
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}")
            continue
        df = pd.read_csv(path).sort_values("episode")
        if ep_ref is None:
            ep_ref = df["episode"].values
        series.append(df["reach_rate"].values)
    if not series:
        return None, None, None
    mat  = np.stack(series)
    return ep_ref, mat.mean(axis=0), mat.std(axis=0)

if RUN_REACH:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(True, **_GRID_KW)
    for group, label, color in [
            ("male",   "Male (Red Group)",    COLOR_M),
            ("female", "Female (Blue Group)", COLOR_F)]:
        eps, mean, std = _load_reach_rates(group)
        if eps is None:
            continue
        ax.plot(eps, mean, color=color, lw=1.8, label=label)
        ax.fill_between(eps, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reach rate  (unique recipients this episode / N)")
    ax.set_title(
        f"{POLICY.upper()} — Episode Reach Rate"
        f"  (avg ± std over {len(SEEDS)} seeds)"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    _out = os.path.join(OUTPUT_DIR, f"{POLICY}_reach_rate_curves.png")
    fig.savefig(_out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"Saved: {_out}")

print("\nAll done.")
