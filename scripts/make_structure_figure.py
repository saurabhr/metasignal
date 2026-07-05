"""Regenerate paper/structure.png as a clean matplotlib figure.

Replaces the previous diagram, which had two labels rendering as tofu boxes
(missing glyphs) instead of arrows. All flow indicators here are drawn as
vector arrow patches rather than unicode arrow characters, so there is no
font-glyph to go missing.
"""

from pathlib import Path

import matplotlib.patches as mpatches
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
GREY_ARROW = "#9aa7b2"


def box(ax, xy, w, h, fc, ec, lw=1.4, radius=0.02):
    x, y = xy
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(p)
    return p


def arrow(ax, start, end, color=GREY_ARROW, lw=1.6):
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
    )
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(11, 7.3), dpi=200)
ax.set_xlim(0, 11)
ax.set_ylim(0, 7.3)
ax.axis("off")

# --- Input Data ---
box(ax, (3.7, 6.5), 3.6, 0.6, DARK, DARK)
ax.text(5.5, 6.92, "Input Data", ha="center", va="center", color="white",
         fontsize=13, fontweight="bold")
ax.text(5.5, 6.65, "stim · resp · conf · n_ratings", ha="center",
         va="center", color="#cfd8e3", fontsize=9, family="monospace")

arrow(ax, (5.5, 6.5), (5.5, 6.05))

# --- stdpy layer ---
box(ax, (0.4, 4.15), 10.2, 1.9, BLUE_BG, "#2980b9", lw=1.8)
ax.text(0.65, 5.9, "stdpy", ha="left", va="center", color="#2471a3",
         fontsize=12, fontweight="bold", family="monospace")
ax.text(1.55, 5.9, "— Pure Python SDT", ha="left", va="center",
         color="#2471a3", fontsize=11)

# SDT Core panel
box(ax, (0.65, 4.35), 3.1, 1.35, "white", "#aed6f1")
box(ax, (0.65, 5.4), 3.1, 0.3, BLUE_HEAD, "#aed6f1")
ax.text(2.2, 5.55, "SDT Core", ha="center", va="center", color=BLUE_TEXT,
         fontsize=10, fontweight="bold")
ax.text(2.2, 5.1, "compute_sdt_resp\ntrials_to_counts", ha="center",
         va="center", family="monospace", fontsize=8.5)
ax.text(2.2, 4.55, "d′ · criterion c · count matrices",
         ha="center", va="center", fontsize=8, color="#444")

arrow(ax, (3.75, 5.05), (4.05, 5.05))

# Metacognitive measures panel
box(ax, (4.1, 4.35), 6.2, 1.35, "white", "#aed6f1")
box(ax, (4.1, 5.4), 6.2, 0.3, BLUE_HEAD, "#aed6f1")
ax.text(7.2, 5.55, "Metacognitive Measures", ha="center", va="center",
         color=BLUE_TEXT, fontsize=10, fontweight="bold")
ax.text(
    5.7, 5.1,
    "fit_meta_d_mle\ncompute_type2_auc\ncompute_gamma · compute_phi",
    ha="center", va="center", family="monospace", fontsize=8,
)
ax.text(
    8.7, 5.1,
    "compute_delta_conf\ncompute_meta_uncertainty\ncompute_meta_noise",
    ha="center", va="center", family="monospace", fontsize=8,
)
ax.text(
    7.2, 4.55,
    "meta-d′ · M-ratio · Type 2 AUC · γ · φ · Δconf · meta-uncertainty · meta-noise",
    ha="center", va="center", fontsize=7.6, color="#444",
)

arrow(ax, (5.5, 4.15), (5.5, 3.75))

# --- compute_all_measures ---
box(ax, (3.3, 3.15), 4.4, 0.6, GOLD_BG, GOLD_BORDER, lw=1.8)
ax.text(5.5, 3.57, "compute_all_measures()", ha="center", va="center",
         color=GOLD_TEXT, fontsize=11, fontweight="bold", family="monospace")
ax.text(
    5.5, 3.28,
    "26-element array: 20 measures + 6 fit diagnostics",
    ha="center", va="center", color=GOLD_TEXT, fontsize=8.3,
)

arrow(ax, (4.2, 3.15), (2.6, 2.65))
arrow(ax, (5.5, 3.15), (5.5, 2.65))
arrow(ax, (6.8, 3.15), (8.6, 2.65))

# --- Layer 3: analysis, CLI, sdtbayes ---
box(ax, (0.4, 1.3), 4.0, 1.35, GREEN_BG, "#27ae60")
ax.text(0.65, 2.42, "analysis", ha="left", va="center", color=GREEN_TEXT,
         fontsize=11, fontweight="bold", family="monospace")
ax.text(1.75, 2.42, "— Inferential Pipeline", ha="left", va="center",
         color=GREEN_TEXT, fontsize=9.5)
ax.text(
    2.4, 1.9,
    "bootstrap_measure – percentile CI\npermutation_test – p-value\ngroup_summary – group statistics",
    ha="center", va="center", family="monospace", fontsize=8,
)

box(ax, (4.6, 1.3), 1.8, 1.35, DARK, DARK)
ax.text(5.5, 2.35, "metasignal\ncompute", ha="center", va="center",
         color="white", fontsize=9.5, fontweight="bold", family="monospace")
ax.text(5.5, 1.75, "CSV input, no\nPython code\nrequired", ha="center",
         va="center", color="#cfd8e3", fontsize=7.6)

box(ax, (6.6, 1.3), 4.0, 1.35, PURPLE_BG, "#8e44ad")
ax.text(6.85, 2.42, "sdtbayes", ha="left", va="center", color=PURPLE_TEXT,
         fontsize=11, fontweight="bold", family="monospace")
ax.text(8.0, 2.42, "— Bayesian Estimation", ha="left", va="center",
         color=PURPLE_TEXT, fontsize=9.5)
ax.text(
    8.6, 1.9,
    "fit_subject_level\nfit_two_stage_group\nfit_full_metad · diagnostics",
    ha="center", va="center", family="monospace", fontsize=8,
)

# Layer labels
for y, label in ((5.1, "LAYER 1"), (3.45, "LAYER 2"), (1.97, "LAYER 3")):
    ax.text(-0.05, y, label, ha="right", va="center", color="#8a97a3",
             fontsize=8, rotation=90)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"wrote {OUT_PATH}")
