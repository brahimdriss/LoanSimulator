"""
Publication plotting style and helpers.

House rules (fixed, do not override per-figure):
  * Times New Roman throughout (Liberation Serif fallback on the cluster)
  * ONE axes per figure -- no subplot grids, no fig.suptitle. Each
    (agent, constraint, reward) combo is its own file; group multiple
    combos in the LaTeX source (subfigure/minipage), not in matplotlib.
  * gridlines on, top/right spines off (open axes)
  * saved as BOTH .pdf and .png
  * NO legend drawn on the axes, NO axes title
  * x-axis and y-axis labels only
  * fonts sized to stay legible once the PDF is placed at column width or
    smaller in LaTeX -- err large, not "looks right at native matplotlib
    size"

Legends are emitted as a SEPARATE standalone file (`*_legend.pdf/.png`) so the
series can still be identified in the paper without cluttering the panel.
Reuse ONE legend file across every figure that shares the same series
(e.g. Male/Female) rather than re-emitting it per combo.

Every figure also writes the exact data behind it to CSV next to the image, so
any number in the paper can be traced back without re-running the experiment.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

def set_paper_style():
    """Apply the house style. Call once before plotting.

    Sizes are picked for a figure that gets shrunk to ~half or a third of
    its native size once placed in LaTeX (multiple combos side by side),
    not for how it looks at matplotlib's native render size -- err large.
    """
    plt.rcParams.update({
        # Times New Roman, with fallbacks so this still renders on the cluster
        # (where the MS font may be absent) rather than silently erroring.
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Times", "Nimbus Roman",
                              "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset":  "stix",          # matching serif math
        "font.size":         15,
        "axes.labelsize":    17,
        "xtick.labelsize":   14,
        "ytick.labelsize":   14,
        "legend.fontsize":   16,
        "axes.titlesize":    17,      # unused (no titles) but keeps mpl quiet
        # 200/200 to match the repo's existing plotting convention
        # (plots/post_process_rule_policies.py, test_rule_based_policies.py).
        # Only affects the PNG raster; the PDF stays vector regardless.
        "figure.dpi":        200,
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
        # 0.02 was too tight for a rotated y-axis label at this font size --
        # matplotlib's tight-bbox estimate for rotated text is occasionally
        # short, and 0.02" left no slack, clipping the top of the label
        # (e.g. "...unique recipients / N)" losing its closing paren).
        "savefig.pad_inches": 0.08,
        "axes.grid":         True,
        "grid.alpha":        0.30,
        "grid.linestyle":    "--",
        "grid.linewidth":    0.6,
        "axes.axisbelow":    True,    # grid behind the data
        "axes.linewidth":    1.0,
        "axes.spines.top":   False,   # open axes (Zucchet-style: no boxed
        "axes.spines.right": False,   # panel, direct left/bottom frame only)
        "lines.linewidth":   2.2,
        "lines.markersize":  6,
        "legend.frameon":    True,
        "legend.framealpha": 0.9,
        "legend.handlelength": 2.2,
        "legend.markerscale":  1.4,
        "legend.borderpad":    0.6,
        "xtick.major.width":  1.0,
        "ytick.major.width":  1.0,
        "pdf.fonttype":      42,      # embed TrueType, not Type3 -- required
        "ps.fonttype":       42,      # by most publishers
    })


def save_figure(fig, out_dir, stem, data: pd.DataFrame = None):
    """
    Save `fig` as both PDF and PNG under out_dir/stem, and (if given) the
    underlying data as out_dir/stem.csv.

    Returns the list of paths written.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Force a full layout pass before computing the tight bbox -- without
    # this, savefig's own internal draw() can compute the bbox from stale
    # text extents (rotated axis labels especially), which is what was
    # clipping the top of tall y-axis labels.
    fig.canvas.draw()
    written = []
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path)
        written.append(path)
    if data is not None:
        path = os.path.join(out_dir, f"{stem}.csv")
        data.to_csv(path, index=False)
        written.append(path)
    plt.close(fig)
    return written


def save_legend(handles, labels, out_dir, stem, ncol=None):
    """
    Emit a standalone legend figure. Kept OUT of the data panels per house
    style; pair it with the panels in the paper's figure caption.
    """
    if not handles:
        return []
    os.makedirs(out_dir, exist_ok=True)
    ncol = ncol or min(len(labels), 4)
    fig = plt.figure(figsize=(0.1, 0.1))
    leg = fig.legend(handles, labels, loc="center", ncol=ncol, frameon=True)
    fig.canvas.draw()
    bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    written = []
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{stem}_legend.{ext}")
        fig.savefig(path, bbox_inches=bbox)
        written.append(path)
    plt.close(fig)
    return written


# --------------------------------------------------------------------------
# Plot primitives
# --------------------------------------------------------------------------

def line_panel(x, series, xlabel, ylabel, out_dir, stem,
               bands=None, hlines=(), figsize=(4.6, 3.6), ylim=None):
    """
    One single-axes figure: several labelled lines, optional +/- std bands.

    series : dict {label: y-array}
    bands  : dict {label: std-array}  (optional, drawn as +/-1 std)
    hlines : iterable of (yvalue, style-dict) reference lines

    Per house style this draws NO legend and NO title -- only x/y labels and
    a grid. Legend handles are returned so the caller can emit a standalone
    legend via save_legend().
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    out = {"x": np.asarray(x)}
    handles, labels = [], []
    for label, y in series.items():
        y = np.asarray(y, dtype=float)
        (ln,) = ax.plot(out["x"], y)
        handles.append(ln); labels.append(label)
        out[label] = y
        if bands and label in bands:
            sd = np.asarray(bands[label], dtype=float)
            ax.fill_between(out["x"], y - sd, y + sd,
                            alpha=0.15, linewidth=0, color=ln.get_color())
            out[f"{label}__std"] = sd

    for yval, style in hlines:
        ax.axhline(yval, **style)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.margins(x=0.01)
    # explicitly: no ax.set_title(), no ax.legend()

    paths = save_figure(fig, out_dir, stem, pd.DataFrame(out))
    return handles, labels, paths
