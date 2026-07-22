"""Regenerate paper/structure.png as a clean matplotlib figure.

All flow indicators are drawn as vector arrow patches rather than unicode
arrow characters, so there is no font-glyph to go missing.

Layer 4 (bottom) separates out the experimental components -- `sdtbayes`
and `itmc` -- from the stable core (stdpy, analysis, CLI) in layer 3.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_PATH = Path(__file__).resolve().parent.parent / "paper" / "structure.png"

DARK = "#1b2733"
BLUE_BG = "#eaf3fb"
BLUE_HEAD = "#cfe6f8"
BLUE_TEXT = "#1a5276"
GOLD_BG = "#fdf6d8"
GOLD_BORDER = "#b8860b"
GOLD_TEXT = "#7a4a00"
GREEN_BG = "#e9f7ef"
GREEN_TEXT = "#1e8449"
PURPLE_BG = "#f5eef8"
PURPLE_TEXT = "#7d3c98"
AMBER_BG = "#fdf1e3"
AMBER_TEXT = "#a85b00"
GREY_ARROW = "#9aa7b2"
EXP_BORDER = "#c77b00"


def box(ax, xy, w, h, fc, ec, lw=1.4, radius=0.02, ls="-"):
    x, y = xy
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        linestyle=ls,
        zorder=2,
    )
    ax.add_patch(p)
    return p


def arrow(ax, start, end, color=GREY_ARROW, lw=1.6, connectionstyle=None) -> None:
    a = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        zorder=1,
        shrinkA=0,
        shrinkB=0,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(11, 9.2), dpi=200)
ax.set_xlim(0, 11)
ax.set_ylim(0, 9.2)
ax.axis("off")

# --- Enclosing frame around the whole diagram ---
box(ax, (0.15, 0.08), 10.7, 9.05, "none", "#4a4a4a", lw=1.6, radius=0.05)

# --- Input Data ---
box(ax, (3.7, 8.4), 3.6, 0.6, DARK, DARK)
ax.text(
    5.5,
    8.82,
    "Input Data",
    ha="center",
    va="center",
    color="white",
    fontsize=13,
    fontweight="bold",
)
ax.text(
    5.5,
    8.55,
    "stim · resp · conf · n_ratings",
    ha="center",
    va="center",
    color="#cfd8e3",
    fontsize=9,
    family="monospace",
)

arrow(ax, (5.5, 8.4), (5.5, 7.95))

# --- Layer 1: stdpy ---
box(ax, (0.4, 6.05), 10.2, 1.9, BLUE_BG, "#2980b9", lw=1.8)
ax.text(
    0.65,
    7.8,
    "stdpy",
    ha="left",
    va="center",
    color="#2471a3",
    fontsize=12,
    fontweight="bold",
    family="monospace",
)
ax.text(
    1.55, 7.8, "— Pure Python SDT", ha="left", va="center", color="#2471a3", fontsize=11
)

box(ax, (0.65, 6.25), 3.1, 1.35, "white", "#aed6f1")
box(ax, (0.65, 7.3), 3.1, 0.3, BLUE_HEAD, "#aed6f1")
ax.text(
    2.2,
    7.45,
    "SDT Core",
    ha="center",
    va="center",
    color=BLUE_TEXT,
    fontsize=10,
    fontweight="bold",
)
ax.text(
    2.2,
    7.0,
    "compute_sdt_resp\ntrials_to_counts",
    ha="center",
    va="center",
    family="monospace",
    fontsize=8.5,
    linespacing=2.2,
)
ax.text(
    2.2,
    6.45,
    "d′ · criterion c · count matrices",
    ha="center",
    va="center",
    fontsize=8,
    color="#444",
)

arrow(ax, (3.75, 6.95), (4.05, 6.95))

box(ax, (4.1, 6.25), 6.2, 1.35, "white", "#aed6f1")
box(ax, (4.1, 7.3), 6.2, 0.3, BLUE_HEAD, "#aed6f1")
ax.text(
    7.2,
    7.45,
    "Metacognitive Measures",
    ha="center",
    va="center",
    color=BLUE_TEXT,
    fontsize=10,
    fontweight="bold",
)
ax.text(
    5.7,
    7.0,
    "fit_meta_d_mle\ncompute_type2_auc\ncompute_gamma · compute_phi",
    ha="center",
    va="center",
    family="monospace",
    fontsize=8,
    linespacing=2.2,
)
ax.text(
    8.7,
    7.0,
    "compute_delta_conf\ncompute_meta_uncertainty\ncompute_meta_noise",
    ha="center",
    va="center",
    family="monospace",
    fontsize=8,
    linespacing=2.2,
)
ax.text(
    7.2,
    6.45,
    "meta-d′ · M-ratio · Type 2 AUC · γ · φ · Δconf · meta-uncertainty · meta-noise",
    ha="center",
    va="center",
    fontsize=7.6,
    color="#444",
)

arrow(ax, (5.5, 6.05), (5.5, 5.65))

# --- Layer 2: compute_all_measures ---
box(ax, (3.3, 5.05), 4.4, 0.6, GOLD_BG, GOLD_BORDER, lw=1.8)
ax.text(
    5.5,
    5.47,
    "compute_all_measures()",
    ha="center",
    va="center",
    color=GOLD_TEXT,
    fontsize=11,
    fontweight="bold",
    family="monospace",
)
ax.text(
    5.5,
    5.18,
    "26-element array: 20 measures + 6 fit diagnostics",
    ha="center",
    va="center",
    color=GOLD_TEXT,
    fontsize=8.3,
)

arrow(ax, (4.6, 5.05), (2.6, 4.55))
arrow(ax, (6.4, 5.05), (8.4, 4.55))

# --- Layer 3: analysis, CLI (stable core) ---
box(ax, (0.4, 3.2), 5.1, 1.35, GREEN_BG, "#27ae60")
ax.text(
    0.65,
    4.32,
    "analysis",
    ha="left",
    va="center",
    color=GREEN_TEXT,
    fontsize=11,
    fontweight="bold",
    family="monospace",
)
ax.text(
    1.75,
    4.32,
    "— Inferential Pipeline",
    ha="left",
    va="center",
    color=GREEN_TEXT,
    fontsize=9.5,
)
ax.text(
    2.95,
    3.8,
    "bootstrap_measure – percentile CI\npermutation_test – p-value\ngroup_summary – group statistics",
    ha="center",
    va="center",
    family="monospace",
    fontsize=8,
    linespacing=2.2,
)

box(ax, (5.7, 3.2), 4.9, 1.35, DARK, DARK)
ax.text(
    8.15,
    4.32,
    "metasignal compute",
    ha="center",
    va="center",
    color="white",
    fontsize=11,
    fontweight="bold",
    family="monospace",
)
ax.text(
    8.15,
    3.75,
    "CLI: CSV input, no code required",
    ha="center",
    va="center",
    color="#cfd8e3",
    fontsize=8.5,
)


# --- Layer 4 (new, bottom): Experimental components ---
box(ax, (0.4, 0.25), 10.2, 1.9, "#fffaf2", EXP_BORDER, lw=1.8, ls="--")
ax.text(
    0.65,
    1.97,
    "Experimental Components",
    ha="left",
    va="center",
    color=EXP_BORDER,
    fontsize=11.5,
    fontweight="bold",
)

# sdtbayes panel
box(ax, (0.65, 0.45), 4.85, 1.35, PURPLE_BG, "#8e44ad")
ax.text(
    0.9,
    1.6,
    "sdtbayes",
    ha="left",
    va="center",
    color=PURPLE_TEXT,
    fontsize=11,
    fontweight="bold",
    family="monospace",
)
ax.text(
    2.0,
    1.6,
    "— Bayesian Estimation",
    ha="left",
    va="center",
    color=PURPLE_TEXT,
    fontsize=9.5,
)
ax.text(
    3.05,
    0.98,
    "7 approaches: ordered logistic · two-stage · full HMeta-d ·\nsubject-level · beta-AUC · meta-regression · within-subject\ndiagnostics: posterior_summary, plot_trace, plot_posterior, plot_forest",
    ha="center",
    va="center",
    family="monospace",
    fontsize=7.3,
    linespacing=2.2,
)
ax.text(
    3.05,
    0.55,
    "optional install: pip install metasignal[sdtbayes]",
    ha="center",
    va="center",
    fontsize=7.4,
    color="#6b3a80",
    style="italic",
)

# itmc panel
box(ax, (5.65, 0.45), 4.6, 1.35, AMBER_BG, EXP_BORDER)
ax.text(
    5.9,
    1.6,
    "itmc",
    ha="left",
    va="center",
    color=AMBER_TEXT,
    fontsize=11,
    fontweight="bold",
    family="monospace",
)
ax.text(
    6.55,
    1.6,
    "— Information-Theoretic Metacognition",
    ha="left",
    va="center",
    color=AMBER_TEXT,
    fontsize=9,
)
ax.text(
    7.95,
    0.9,
    "meta_I · meta_Ir1 · meta_Ir1_acc · meta_Ir2 · RMI\npermtest_meta_I",
    ha="center",
    va="center",
    family="monospace",
    fontsize=7.6,
    linespacing=2.2,
)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
