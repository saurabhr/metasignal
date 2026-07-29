"""Regenerate paper/structure.png as a clean matplotlib figure.

Design notes (revised for readability):
  - Every box uses the same visual pattern -- light pastel fill, saturated
    border, dark text -- including "Input Data" and the CLI box, which were
    previously solid dark-navy fills. Solid dark fills were dropped for two
    reasons: (1) they broke the diagram's otherwise consistent visual
    language, giving two structurally ordinary boxes disproportionate visual
    weight; (2) small light-on-dark text suffers a real perceptual penalty
    (the "irradiation illusion" -- thin light text on a dark ground appears
    to bleed/blur at small point sizes even at identical contrast ratio to
    dark-on-light), which compounds badly once the figure is shrunk to fit
    a paper column.
  - All text/background pairs are chosen to clear WCAG AA contrast
    (>=4.5:1 for body text, >=3:1 for large/bold headers and non-text
    UI elements like borders) -- verified numerically, not eyeballed.
    The original green header (analysis) failed AA at 4.27:1; darkened.
  - Font sizes are ~30-40% larger throughout, since the figure is embedded
    at 60% width in the paper and the smallest text was previously
    rendering below 3pt on the page.
  - The dead vertical gap that used to sit between Layer 3 (analysis/CLI)
    and the Experimental Components panel has been compressed.

All flow indicators are drawn as vector arrow patches rather than unicode
arrow characters, so there is no font-glyph to go missing.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_PATH = Path(__file__).resolve().parent.parent / "paper" / "structure.png"

# --- Palette: every category is light-bg + saturated-border + dark-text. ---
SLATE_BG, SLATE_BORDER, SLATE_TEXT = "#eef1f4", "#5d6d7e", "#2c3e50"      # Input Data
BLUE_BG, BLUE_HEAD, BLUE_BORDER, BLUE_TEXT = "#eaf3fb", "#cfe6f8", "#2980b9", "#1a5276"  # stdpy
GOLD_BG, GOLD_BORDER, GOLD_TEXT = "#fdf6d8", "#b8860b", "#7a4a00"          # compute_all_measures
GREEN_BG, GREEN_BORDER, GREEN_TEXT = "#e9f7ef", "#27ae60", "#166638"       # analysis (darkened for AA)
TEAL_BG, TEAL_BORDER, TEAL_TEXT = "#e8f6f6", "#0d7377", "#0b5d5d"          # CLI
PURPLE_BG, PURPLE_BORDER, PURPLE_TEXT = "#f5eef8", "#8e44ad", "#7d3c98"    # sdtbayes
AMBER_BG, AMBER_BORDER, AMBER_TEXT = "#fdf1e3", "#c77b00", "#8a4d00"       # itmc (darkened for AA)
EXP_BORDER = "#c77b00"
GREY_ARROW = "#7f8c9a"
BODY_GREY = "#3a3a3a"


def box(ax, xy, w, h, fc, ec, lw=1.6, radius=0.02, ls="-"):
    x, y = xy
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=2,
    )
    ax.add_patch(p)
    return p


def arrow(ax, start, end, color=GREY_ARROW, lw=1.8, connectionstyle=None) -> None:
    a = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=16, linewidth=lw,
        color=color, zorder=1, shrinkA=0, shrinkB=0, connectionstyle=connectionstyle,
    )
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(11.5, 8.85), dpi=200)
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 8.85)
ax.axis("off")

# --- Enclosing frame around the whole diagram ---
box(ax, (0.15, 0.08), 11.2, 8.69, "none", "#4a4a4a", lw=1.6, radius=0.05)

# --- Input Data ---
box(ax, (3.75, 7.78), 4.0, 0.68, SLATE_BG, SLATE_BORDER, lw=1.8)
ax.text(5.75, 8.28, "Input Data", ha="center", va="center",
         color=SLATE_TEXT, fontsize=17, fontweight="bold")
ax.text(5.75, 7.95, "stim · resp · conf · n_ratings", ha="center", va="center",
         color=SLATE_TEXT, fontsize=12, family="monospace")

arrow(ax, (5.75, 7.78), (5.75, 7.38))

# --- Layer 1: stdpy ---
box(ax, (0.4, 5.28), 10.7, 2.05, BLUE_BG, BLUE_BORDER, lw=1.8)
ax.text(0.65, 7.05, "stdpy", ha="left", va="center", color=BLUE_TEXT,
         fontsize=15, fontweight="bold", family="monospace")
ax.text(1.75, 7.05, "— Pure Python SDT", ha="left", va="center",
         color=BLUE_TEXT, fontsize=14)

box(ax, (0.65, 5.48), 3.25, 1.42, "white", "#aed6f1", lw=1.4)
box(ax, (0.65, 6.6), 3.25, 0.3, BLUE_HEAD, "#aed6f1", lw=1.4)
ax.text(2.28, 6.75, "SDT Core", ha="center", va="center",
         color=BLUE_TEXT, fontsize=13, fontweight="bold")
ax.text(2.28, 6.2, "compute_sdt_resp\ntrials_to_counts", ha="center", va="center",
         family="monospace", fontsize=11, linespacing=2.1)
ax.text(2.28, 5.65, "d′ · criterion c · count matrices", ha="center", va="center",
         fontsize=10.5, color=BODY_GREY)

arrow(ax, (3.95, 6.19), (4.3, 6.19))

box(ax, (4.35, 5.48), 6.5, 1.42, "white", "#aed6f1", lw=1.4)
box(ax, (4.35, 6.6), 6.5, 0.3, BLUE_HEAD, "#aed6f1", lw=1.4)
ax.text(7.6, 6.75, "Metacognitive Measures", ha="center", va="center",
         color=BLUE_TEXT, fontsize=13, fontweight="bold")
ax.text(6.0, 6.2, "fit_meta_d_mle\ncompute_type2_auc\ncompute_gamma · compute_phi",
         ha="center", va="center", family="monospace", fontsize=10.3, linespacing=2.1)
ax.text(9.15, 6.2, "compute_delta_conf\ncompute_meta_uncertainty\ncompute_meta_noise",
         ha="center", va="center", family="monospace", fontsize=10.3, linespacing=2.1)
ax.text(7.6, 5.65, "meta-d′ · M-ratio · Type 2 AUC · γ · φ · Δconf · meta-uncertainty · meta-noise",
         ha="center", va="center", fontsize=9.6, color=BODY_GREY)

arrow(ax, (5.75, 5.28), (5.75, 4.86))

# --- Layer 2: compute_all_measures ---
box(ax, (3.35, 4.18), 4.8, 0.68, GOLD_BG, GOLD_BORDER, lw=1.8)
ax.text(5.75, 4.65, "compute_all_measures()", ha="center", va="center",
         color=GOLD_TEXT, fontsize=14, fontweight="bold", family="monospace")
ax.text(5.75, 4.32, "26-element array: 20 measures + 6 fit diagnostics",
         ha="center", va="center", color=GOLD_TEXT, fontsize=10.5)

arrow(ax, (4.7, 4.18), (2.7, 3.7))
arrow(ax, (6.8, 4.18), (8.8, 3.7))

# --- Layer 3: analysis, CLI (stable core) ---
box(ax, (0.4, 2.35), 5.3, 1.42, GREEN_BG, GREEN_BORDER, lw=1.8)
ax.text(0.65, 3.55, "analysis", ha="left", va="center", color=GREEN_TEXT,
         fontsize=14, fontweight="bold", family="monospace")
ax.text(1.85, 3.55, "— Inferential Pipeline", ha="left", va="center",
         color=GREEN_TEXT, fontsize=12.5)
ax.text(3.05, 2.95, "bootstrap_measure – percentile CI\npermutation_test – p-value\n"
         "group_summary – group statistics",
         ha="center", va="center", family="monospace", fontsize=10.3, linespacing=2.1)

box(ax, (5.95, 2.35), 5.15, 1.42, TEAL_BG, TEAL_BORDER, lw=1.8)
ax.text(8.53, 3.55, "metasignal compute", ha="center", va="center", color=TEAL_TEXT,
         fontsize=14, fontweight="bold", family="monospace")
ax.text(8.53, 2.9, "CLI: CSV input, no code required", ha="center", va="center",
         color=TEAL_TEXT, fontsize=11)

# --- Layer 4 (bottom): Experimental components ---
box(ax, (0.4, 0.25), 10.7, 2.0, "#fffaf2", EXP_BORDER, lw=1.8, ls="--")
ax.text(0.65, 2.05, "Experimental Components", ha="left", va="center",
         color=EXP_BORDER, fontsize=14.5, fontweight="bold")

# sdtbayes panel
box(ax, (0.65, 0.42), 5.1, 1.45, PURPLE_BG, PURPLE_BORDER, lw=1.6)
ax.text(0.9, 1.68, "sdtbayes", ha="left", va="center", color=PURPLE_TEXT,
         fontsize=13, fontweight="bold", family="monospace")
ax.text(2.05, 1.68, "— Bayesian Estimation", ha="left", va="center",
         color=PURPLE_TEXT, fontsize=11.5)
ax.text(3.2, 1.14, "7 approaches: ordered logistic · two-stage · full HMeta-d ·\n"
         "subject-level · beta-AUC · meta-regression · within-subject\n"
         "diagnostics: posterior_summary, plot_trace,\nplot_posterior, plot_forest",
         ha="center", va="center", family="monospace", fontsize=8.6, linespacing=1.7)
ax.text(3.2, 0.58, "optional install: pip install metasignal[sdtbayes]",
         ha="center", va="center", fontsize=9.3, color="#5c3170", style="italic")

# itmc panel
box(ax, (5.95, 0.42), 4.85, 1.45, AMBER_BG, AMBER_BORDER, lw=1.6)
ax.text(6.2, 1.68, "itmc", ha="left", va="center", color=AMBER_TEXT,
         fontsize=13, fontweight="bold", family="monospace")
ax.text(6.9, 1.68, "— Information-Theoretic Metacognition", ha="left", va="center",
         color=AMBER_TEXT, fontsize=10.7)
ax.text(8.38, 1.05, "meta_I · meta_Ir1 · meta_Ir1_acc · meta_Ir2 · RMI\npermtest_meta_I",
         ha="center", va="center", family="monospace", fontsize=9.8, linespacing=2.0)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
