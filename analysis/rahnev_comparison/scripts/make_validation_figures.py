#!/usr/bin/env python3
"""Publication-ready Paper · MATLAB · Python validation figures.

Designed for inclusion in a manuscript as the metasignal validation /
evaluation section. Prefer identity scatters and compact summary tables so
readers can compare the three sources at a glance.

Outputs (notebooks/figures/validation/):
  Fig1_identity_scatters.png/.pdf
  Fig2_profile_overlays.png/.pdf
  Fig3_error_summary.png/.pdf
  Fig4_protocol_caveats.png/.pdf
  validation_summary.csv
  validation_figures.pdf   (all pages)

Usage:
    python analysis/rahnev_comparison/scripts/make_validation_figures.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-metasignal")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Style — print-friendly, journal-like
# ---------------------------------------------------------------------------

PAPER_C = "#222222"
MATLAB_C = "#1f5aa6"
PYTHON_C = "#b33a2b"
GRID_C = "#e6e6e6"
TEXT_C = "#222222"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,  # editable text in Illustrator
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    }
)


def load_analysis_module(repo: Path):
    path = Path(__file__).resolve().parent / "generate_rahnev_comparison_plots.py"
    if not path.exists():
        path = repo / "scripts" / "generate_rahnev_comparison_plots.py"
    spec = importlib.util.spec_from_file_location("rahnev_cmp", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["rahnev_cmp"] = module
    spec.loader.exec_module(module)
    return module


def compute_all(module, results: Path, precomp: Path):
    task_m, task_p = module.compute_task_effects(results, precomp)
    bias_m, bias_p = module.compute_bias_effects(results, precomp)
    resp_m, resp_p = module.compute_response_effects(results, precomp)
    prec_m, prec_p = module.compute_precision(results, precomp)
    split_m, split_p = module.compute_split_half(results, precomp)
    retest_m, retest_p = module.compute_test_retest(results, precomp)
    paper = module.PAPER
    matlab = {
        "precision": prec_m,
        "task": task_m,
        "bias": bias_m,
        "response": resp_m,
        "split_half": split_m,
        "test_retest": retest_m,
    }
    python = {
        "precision": prec_p,
        "task": task_p,
        "bias": bias_p,
        "response": resp_p,
        "split_half": split_p,
        "test_retest": retest_p,
    }
    return paper, matlab, python, module.MEASURES


def agreement(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return dict(n=int(ok.sum()), r=np.nan, mae=np.nan, rmse=np.nan, bias=np.nan)
    xx, yy = x[ok], y[ok]
    diff = yy - xx
    r = float(np.corrcoef(xx, yy)[0, 1]) if np.std(xx) > 0 and np.std(yy) > 0 else np.nan
    return dict(
        n=int(ok.sum()),
        r=r,
        mae=float(np.mean(np.abs(diff))),
        rmse=float(np.sqrt(np.mean(diff**2))),
        bias=float(np.mean(diff)),
    )


def panel_letter(ax, letter: str):
    ax.text(
        -0.08,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="right",
        color=TEXT_C,
    )


def identity_limits(*arrays):
    vals = np.concatenate([np.asarray(a, float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return -1, 1
    lo, hi = float(np.min(vals)), float(np.max(vals))
    pad = 0.08 * (hi - lo if hi > lo else max(abs(hi), 1))
    return lo - pad, hi + pad


# ---------------------------------------------------------------------------
# Figure 1 — Identity scatters (primary validation)
# ---------------------------------------------------------------------------

PRIMARY = [
    ("task", "Task performance\n(Cohen's d)", True),
    ("bias", "Metacognitive bias\n(Cohen's d)", True),
    ("response", "Response bias\n(Pearson r)", True),
    ("test_retest", "Test–retest reliability\n(ICC)", False),
]


def fig1_identity(paper, matlab, python, out: Path, pdf: PdfPages):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.8))
    legend_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=MATLAB_C, markersize=7, label="MATLAB"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=PYTHON_C, markersize=7, label="Python"),
        Line2D([0], [0], linestyle="--", color="#888888", label="Identity"),
    ]

    for ax, (key, title, zero), letter in zip(axes.flat, PRIMARY, "abcd"):
        p, m, y = paper[key], matlab[key], python[key]
        lim = identity_limits(p, m, y)
        ax.plot(lim, lim, "--", color="#888888", lw=1, zorder=1)
        if zero:
            ax.axhline(0, color="#cccccc", lw=0.6, zorder=0)
            ax.axvline(0, color="#cccccc", lw=0.6, zorder=0)

        ok_m = np.isfinite(p) & np.isfinite(m)
        ok_y = np.isfinite(p) & np.isfinite(y)
        ax.scatter(p[ok_m], m[ok_m], s=28, marker="s", color=MATLAB_C, zorder=3, label="MATLAB")
        ax.scatter(p[ok_y], y[ok_y], s=32, marker="^", color=PYTHON_C, zorder=4, label="Python")

        sm = agreement(p, m)
        sy = agreement(p, y)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Paper (Rahnev 2025)")
        ax.set_ylabel("Replication")
        ax.set_title(title, pad=6)
        ax.grid(color=GRID_C, lw=0.5)
        ax.text(
            0.04,
            0.96,
            f"MATLAB  r={sm['r']:.3f}  MAE={sm['mae']:.3f}\n"
            f"Python   r={sy['r']:.3f}  MAE={sy['mae']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            family="DejaVu Sans Mono",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd", alpha=0.92),
        )
        panel_letter(ax, letter)

    axes[0, 1].legend(handles=legend_handles, loc="lower right", frameon=False)
    fig.suptitle(
        "Validation against Rahnev (2025): Paper vs MATLAB pipeline vs Python package",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out / "Fig1_identity_scatters.png", bbox_inches="tight")
    fig.savefig(out / "Fig1_identity_scatters.pdf", bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Profile overlays (easy visual comparison)
# ---------------------------------------------------------------------------

PROFILE = [
    ("task", "a  Task performance (Cohen's d)", True, None),
    ("bias", "b  Metacognitive bias (Cohen's d)", True, None),
    ("response", "c  Response bias (r with |c|)", True, None),
    ("test_retest", "d  Test–retest ICC", False, (0, 1.05)),
]


def fig2_profiles(paper, matlab, python, measures, out: Path, pdf: PdfPages):
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 9.2), sharex=True)
    x = np.arange(len(measures))
    offsets = {"Paper": -0.18, "MATLAB": 0.0, "Python": 0.18}

    for ax, (key, title, zero, ylim) in zip(axes, PROFILE):
        series = {
            "Paper": paper[key],
            "MATLAB": matlab[key],
            "Python": python[key],
        }
        colors = {"Paper": PAPER_C, "MATLAB": MATLAB_C, "Python": PYTHON_C}
        markers = {"Paper": "o", "MATLAB": "s", "Python": "^"}
        for name, vals in series.items():
            ax.plot(
                x + offsets[name],
                vals,
                color=colors[name],
                alpha=0.35,
                lw=1.1,
                zorder=1,
            )
            ax.scatter(
                x + offsets[name],
                vals,
                s=26 if name != "Paper" else 22,
                marker=markers[name],
                color=colors[name],
                label=name,
                zorder=3,
                edgecolors="white",
                linewidths=0.4,
            )
        if zero:
            ax.axhline(0, color="#999999", lw=0.7)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=4)
        ax.grid(axis="y", color=GRID_C, lw=0.5)
        ax.set_xticks(x)

    axes[0].legend(ncol=3, loc="upper right", frameon=False)
    axes[-1].set_xticklabels(measures, rotation=55, ha="right")
    axes[-1].set_xlabel("Measure")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    # Fix y labels manually for clarity
    axes[0].set_ylabel("Cohen's d")
    axes[1].set_ylabel("Cohen's d")
    axes[2].set_ylabel("Pearson r")
    axes[3].set_ylabel("ICC")

    fig.suptitle(
        "Measure-by-measure profiles: Paper · MATLAB · Python",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out / "Fig2_profile_overlays.png", bbox_inches="tight")
    fig.savefig(out / "Fig2_profile_overlays.pdf", bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Quantitative summary (table + bar of MAE)
# ---------------------------------------------------------------------------

SUMMARY_KEYS = [
    ("task", "Task d"),
    ("bias", "Bias d"),
    ("response", "Response r"),
    ("test_retest", "Test–retest ICC"),
    ("split_half", "Split-half r*"),
    ("precision", "Precision*"),
]


def fig3_summary(paper, matlab, python, out: Path, pdf: PdfPages):
    rows = []
    for key, label in SUMMARY_KEYS:
        rows.append(
            {
                "metric": label,
                "key": key,
                "PM": agreement(paper[key], matlab[key]),
                "PY": agreement(paper[key], python[key]),
                "MY": agreement(matlab[key], python[key]),
            }
        )

    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.38)

    # --- Table panel ---
    ax_t = fig.add_subplot(gs[0])
    ax_t.axis("off")
    ax_t.set_title("a  Quantitative agreement across analyses", loc="left", fontweight="bold", pad=8)

    cell = [
        [
            "Analysis",
            "Paper↔MATLAB\nr / MAE",
            "Paper↔Python\nr / MAE",
            "MATLAB↔Python\nr / MAE",
            "n",
        ]
    ]
    for row in rows:
        def fmt(s):
            if not np.isfinite(s["r"]):
                return "—"
            return f"{s['r']:.3f} / {s['mae']:.3f}"

        cell.append(
            [
                row["metric"],
                fmt(row["PM"]),
                fmt(row["PY"]),
                fmt(row["MY"]),
                str(row["MY"]["n"] if np.isfinite(row["MY"]["r"]) else row["PM"]["n"]),
            ]
        )

    table = ax_t.table(
        cellText=cell,
        cellLoc="center",
        loc="upper center",
        colWidths=[0.22, 0.20, 0.20, 0.22, 0.08],
        bbox=[0.02, 0.05, 0.96, 0.9],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.55)
    for j in range(5):
        table[0, j].set_facecolor("#1f4e79")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(cell)):
        for j in range(5):
            table[i, j].set_facecolor("#f7f9fc" if i % 2 else "white")
            if j == 0:
                table[i, j].set_text_props(ha="left")

    ax_t.text(
        0.02,
        -0.02,
        "* Split-half and precision are not fully protocol-matched across sources "
        "(see Fig. 4). Prefer Task / Bias / Response / Test–retest for formal validation.",
        transform=ax_t.transAxes,
        fontsize=7,
        color="#555555",
        va="top",
    )

    # --- MAE bars ---
    ax_b = fig.add_subplot(gs[1])
    labels = [r["metric"].replace("*", "") for r in rows]
    x = np.arange(len(labels))
    w = 0.27
    mae_pm = [r["PM"]["mae"] if np.isfinite(r["PM"]["mae"]) else 0 for r in rows]
    mae_py = [r["PY"]["mae"] if np.isfinite(r["PY"]["mae"]) else 0 for r in rows]
    mae_my = [r["MY"]["mae"] if np.isfinite(r["MY"]["mae"]) else 0 for r in rows]

    ax_b.bar(x - w, mae_pm, w, color=MATLAB_C, label="|MATLAB − Paper|", alpha=0.9)
    ax_b.bar(x, mae_py, w, color=PYTHON_C, label="|Python − Paper|", alpha=0.9)
    ax_b.bar(x + w, mae_my, w, color="#6a6a6a", label="|Python − MATLAB|", alpha=0.85)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, rotation=20, ha="right")
    ax_b.set_ylabel("Mean absolute error")
    ax_b.set_title("b  Absolute discrepancy by analysis", loc="left", fontweight="bold", pad=6)
    ax_b.legend(frameon=False, ncol=3, loc="upper right")
    ax_b.grid(axis="y", color=GRID_C, lw=0.5)

    fig.suptitle(
        "Validation summary for manuscript",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    fig.savefig(out / "Fig3_error_summary.png", bbox_inches="tight")
    fig.savefig(out / "Fig3_error_summary.pdf", bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # Write CSV for the paper SI
    csv_path = out / "validation_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "analysis",
                "paper_matlab_r",
                "paper_matlab_mae",
                "paper_python_r",
                "paper_python_mae",
                "matlab_python_r",
                "matlab_python_mae",
                "n",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["metric"],
                    row["PM"]["r"],
                    row["PM"]["mae"],
                    row["PY"]["r"],
                    row["PY"]["mae"],
                    row["MY"]["r"],
                    row["MY"]["mae"],
                    row["MY"]["n"],
                ]
            )
    return rows


# ---------------------------------------------------------------------------
# Figure 4 — Protocol caveats (precision + split-half)
# ---------------------------------------------------------------------------


def fig4_caveats(paper, matlab, python, measures, out: Path, pdf: PdfPages):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    x = np.arange(len(measures))

    # Precision
    ax = axes[0]
    ax.scatter(x - 0.15, paper["precision"], s=24, c=PAPER_C, marker="o", label="Paper")
    ax.scatter(x, matlab["precision"], s=26, c=MATLAB_C, marker="s", label="MATLAB")
    ax.scatter(x + 0.15, python["precision"], s=28, c=PYTHON_C, marker="^", label="Python")
    ax.plot(x - 0.15, paper["precision"], color=PAPER_C, alpha=0.3, lw=1)
    ax.plot(x, matlab["precision"], color=MATLAB_C, alpha=0.3, lw=1)
    ax.plot(x + 0.15, python["precision"], color=PYTHON_C, alpha=0.3, lw=1)
    ax.set_ylabel("Normalized precision")
    ax.set_title(
        "a  Precision (protocol differs: Python = Haddara-only; MLE measures uncached)",
        loc="left",
        fontweight="bold",
    )
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", color=GRID_C, lw=0.5)
    panel_letter(ax, "a")

    # Split-half
    ax = axes[1]
    ax.scatter(x - 0.15, paper["split_half"], s=24, c=PAPER_C, marker="o", label="Paper")
    ax.scatter(x, matlab["split_half"], s=26, c=MATLAB_C, marker="s", label="MATLAB")
    ax.scatter(x + 0.15, python["split_half"], s=28, c=PYTHON_C, marker="^", label="Python")
    ax.plot(x - 0.15, paper["split_half"], color=PAPER_C, alpha=0.3, lw=1)
    ax.plot(x, matlab["split_half"], color=MATLAB_C, alpha=0.3, lw=1)
    ax.plot(x + 0.15, python["split_half"], color=PYTHON_C, alpha=0.3, lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pearson r")
    ax.set_title(
        "b  Split-half (protocol differs: Paper/MATLAB ≈ 100-trial bins; Python = full odd/even)",
        loc="left",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(measures, rotation=55, ha="right")
    ax.set_xlabel("Measure")
    ax.grid(axis="y", color=GRID_C, lw=0.5)
    panel_letter(ax, "b")

    fig.suptitle(
        "Analyses with residual protocol mismatch (reported for transparency)",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out / "Fig4_protocol_caveats.png", bbox_inches="tight")
    fig.savefig(out / "Fig4_protocol_caveats.pdf", bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined one-page highlight for main text
# ---------------------------------------------------------------------------


def fig_main_highlight(paper, matlab, python, measures, out: Path, pdf: PdfPages):
    """Single page suitable as the main-text validation figure."""
    fig = plt.figure(figsize=(7.2, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.05], hspace=0.42, wspace=0.28)

    # Top row: identity scatters for task + bias
    for col, (key, title, zero), letter in zip(
        range(2),
        [("task", "Task performance (Cohen's d)", True), ("bias", "Metacognitive bias (Cohen's d)", True)],
        "ab",
    ):
        ax = fig.add_subplot(gs[0, col])
        p, m, y = paper[key], matlab[key], python[key]
        lim = identity_limits(p, m, y)
        ax.plot(lim, lim, "--", color="#888888", lw=1)
        if zero:
            ax.axhline(0, color="#cccccc", lw=0.6)
            ax.axvline(0, color="#cccccc", lw=0.6)
        ax.scatter(p, m, s=26, marker="s", color=MATLAB_C, label="MATLAB")
        ax.scatter(p, y, s=28, marker="^", color=PYTHON_C, label="Python")
        sm, sy = agreement(p, m), agreement(p, y)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Paper")
        ax.set_ylabel("Replication")
        ax.set_title(title, fontsize=9.5)
        ax.grid(color=GRID_C, lw=0.5)
        ax.text(
            0.04,
            0.96,
            f"MAT r={sm['r']:.3f}\nPY  r={sy['r']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            family="DejaVu Sans Mono",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#dddddd"),
        )
        panel_letter(ax, letter)
        if col == 1:
            ax.legend(frameon=False, loc="lower right", fontsize=7)

    # Middle row: response + test-retest
    for col, (key, title, zero, ylim), letter in zip(
        range(2),
        [
            ("response", "Response bias (r)", True, None),
            ("test_retest", "Test–retest (ICC)", False, (0, 1.05)),
        ],
        "cd",
    ):
        ax = fig.add_subplot(gs[1, col])
        p, m, y = paper[key], matlab[key], python[key]
        lim = identity_limits(p, m, y)
        if ylim:
            lim = (min(lim[0], ylim[0]), max(lim[1], ylim[1]))
        ax.plot(lim, lim, "--", color="#888888", lw=1)
        ax.scatter(p, m, s=26, marker="s", color=MATLAB_C)
        ax.scatter(p, y, s=28, marker="^", color=PYTHON_C)
        sm, sy = agreement(p, m), agreement(p, y)
        ax.set_xlim(lim)
        ax.set_ylim(lim if not ylim else ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Paper")
        ax.set_ylabel("Replication")
        ax.set_title(title, fontsize=9.5)
        ax.grid(color=GRID_C, lw=0.5)
        ax.text(
            0.04,
            0.96,
            f"MAT r={sm['r']:.3f}\nPY  r={sy['r']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            family="DejaVu Sans Mono",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#dddddd"),
        )
        panel_letter(ax, letter)

    # Bottom: compact profile for task (most diagnostic)
    ax = fig.add_subplot(gs[2, :])
    x = np.arange(len(measures))
    for name, vals, color, marker, off in [
        ("Paper", paper["task"], PAPER_C, "o", -0.18),
        ("MATLAB", matlab["task"], MATLAB_C, "s", 0.0),
        ("Python", python["task"], PYTHON_C, "^", 0.18),
    ]:
        ax.plot(x + off, vals, color=color, alpha=0.35, lw=1)
        ax.scatter(x + off, vals, s=22, color=color, marker=marker, label=name, edgecolors="white", lw=0.3)
    ax.axhline(0, color="#999999", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(measures, rotation=50, ha="right")
    ax.set_ylabel("Cohen's d")
    ax.set_title("e  Task-performance profile across all 17 measures", loc="left", fontweight="bold")
    ax.legend(ncol=3, frameon=False, loc="upper right")
    ax.grid(axis="y", color=GRID_C, lw=0.5)
    panel_letter(ax, "e")

    fig.suptitle(
        "metasignal validation: Rahnev (2025) · MATLAB pipeline · Python package",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )
    fig.savefig(out / "Fig_main_validation.png", bbox_inches="tight")
    fig.savefig(out / "Fig_main_validation.pdf", bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_readme(out: Path, rows):
    lines = [
        "# Publication-ready validation figures",
        "",
        "Generated by `analysis/rahnev_comparison/scripts/make_validation_figures.py`.",
        "",
        "## Recommended for the paper",
        "",
        "| File | Use |",
        "|---|---|",
        "| `Fig_main_validation.pdf` | **Main-text figure** (identity scatters + task profile) |",
        "| `Fig1_identity_scatters.pdf` | Expanded identity panels |",
        "| `Fig2_profile_overlays.pdf` | Measure-by-measure overlays |",
        "| `Fig3_error_summary.pdf` | Quantitative table + MAE bars |",
        "| `Fig4_protocol_caveats.pdf` | Transparency for non-matched protocols |",
        "| `validation_summary.csv` | SI numbers |",
        "| `validation_figures.pdf` | All pages combined |",
        "",
        "## Primary validation metrics (protocol-matched)",
        "",
        "Use Task, Bias, Response, and Test–retest for formal claims.",
        "Split-half and Precision differ in sampling protocol between sources.",
        "",
        "## Snapshot",
        "",
        "| Analysis | Paper↔MATLAB r | Paper↔Python r | MATLAB↔Python r |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['PM']['r']:.3f} | {row['PY']['r']:.3f} | {row['MY']['r']:.3f} |"
        )
    lines.append("")
    (out / "README.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    def _find_repo() -> Path:
        here = Path(__file__).resolve().parent
        for c in [here, *here.parents]:
            if (c / "src" / "metasignal").is_dir() and (c / "matlab" / "metasignal_mat" / "Results").is_dir():
                return c
        return here.parents[3]
    parser.add_argument("--repo", type=Path, default=_find_repo())
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    results = repo / "matlab" / "metasignal_mat" / "Results"
    precomp = repo / "notebooks" / "precomputed"
    out = (args.out or repo / "notebooks" / "figures" / "validation").resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("Loading analysis module and recomputing summaries...")
    module = load_analysis_module(repo)
    paper, matlab, python, measures = compute_all(module, results, precomp)

    pdf_path = out / "validation_figures.pdf"
    with PdfPages(pdf_path) as pdf:
        print("Fig_main_validation...")
        fig_main_highlight(paper, matlab, python, measures, out, pdf)
        print("Fig1_identity_scatters...")
        fig1_identity(paper, matlab, python, out, pdf)
        print("Fig2_profile_overlays...")
        fig2_profiles(paper, matlab, python, measures, out, pdf)
        print("Fig3_error_summary...")
        rows = fig3_summary(paper, matlab, python, out, pdf)
        print("Fig4_protocol_caveats...")
        fig4_caveats(paper, matlab, python, measures, out, pdf)

    write_readme(out, rows)
    print(f"\nWrote publication-ready figures to {out}")
    print(f"Main-text figure: {out / 'Fig_main_validation.pdf'}")
    print(f"Combined PDF:     {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
