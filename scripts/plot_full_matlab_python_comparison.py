#!/usr/bin/env python3
"""Plot a full MATLAB-vs-Python comparison for all matched result arrays.

Outputs one 20-panel scatter-grid page for every matched analysis, plus
summary heatmaps and a CSV containing correlation/error statistics.

Usage:
    python scripts/plot_full_matlab_python_comparison.py
    python scripts/plot_full_matlab_python_comparison.py --out path/to/output
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-metasignal")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib.backends.backend_pdf import PdfPages


MEASURES = [
    "meta-d'",
    "AUC2",
    "Gamma",
    "Phi",
    "ΔConf",
    "M-Ratio",
    "AUC2-Ratio",
    "Gamma-Ratio",
    "Phi-Ratio",
    "ΔConf-Ratio",
    "M-Diff",
    "AUC2-Diff",
    "Gamma-Diff",
    "Phi-Diff",
    "ΔConf-Diff",
    "meta-noise",
    "meta-uncertainty",
    "d'",
    "Criterion",
    "Confidence",
]
N_MEASURES = len(MEASURES)
META_NOISE_INDEX = 15
MATLAB_META_NOISE_ARTIFACT = 0.495934
ARTIFACT_TOLERANCE = 1e-3


@dataclass(frozen=True)
class ComparisonSpec:
    label: str
    matlab_file: str
    matlab_key: str
    python_file: str
    python_key: str


SPECS = [
    ComparisonSpec("Haddara raw", "results_Haddara.mat", "metas_raw", "haddara_mle.npz", "raw"),
    ComparisonSpec(
        "Haddara metacognitive bias",
        "results_Haddara.mat",
        "metas_confRecode",
        "haddara_mle.npz",
        "bias",
    ),
    ComparisonSpec(
        "Haddara odd-even",
        "results_Haddara.mat",
        "metas_oddEven",
        "haddara_mle.npz",
        "split",
    ),
    ComparisonSpec(
        "Maniscalco raw",
        "results_Maniscalco.mat",
        "metas_raw",
        "maniscalco_mle.npz",
        "raw",
    ),
    ComparisonSpec(
        "Maniscalco metacognitive bias",
        "results_Maniscalco.mat",
        "metas_confRecode",
        "maniscalco_mle.npz",
        "bias",
    ),
    ComparisonSpec(
        "Rouault1 difficulty",
        "results_Rouault1.mat",
        "metas_diff",
        "rouault1_mle.npz",
        "diff",
    ),
    ComparisonSpec(
        "Rouault2 difficulty",
        "results_Rouault2.mat",
        "metas_diff",
        "rouault2_mle.npz",
        "diff",
    ),
    ComparisonSpec(
        "Shekhar difficulty",
        "results_Shekhar.mat",
        "metas_diff",
        "shekhar_mle.npz",
        "diff",
    ),
    ComparisonSpec(
        "Shekhar metacognitive bias",
        "results_Shekhar.mat",
        "metas_confRecode",
        "shekhar_mle.npz",
        "bias",
    ),
    ComparisonSpec(
        "Locke response bias",
        "results_Locke.mat",
        "metas_bias",
        "locke_mle.npz",
        "rb",
    ),
]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_pair(
    results: Path, precomputed: Path, spec: ComparisonSpec
) -> tuple[np.ndarray, np.ndarray]:
    matlab = np.asarray(
        scipy.io.loadmat(results / spec.matlab_file, squeeze_me=True)[spec.matlab_key],
        dtype=float,
    )
    python = np.asarray(np.load(precomputed / spec.python_file)[spec.python_key], dtype=float)
    matlab = matlab[..., :N_MEASURES]
    python = python[..., :N_MEASURES]

    if matlab.shape != python.shape:
        raise ValueError(
            f"{spec.label}: MATLAB shape {matlab.shape} != Python shape {python.shape}"
        )

    # MATLAB goldenSearch returns this endpoint when meta-noise is
    # unidentifiable. It is an optimizer artefact rather than an estimate.
    matlab = matlab.copy()
    noise = matlab[..., META_NOISE_INDEX]
    noise[np.abs(noise - MATLAB_META_NOISE_ARTIFACT) < ARTIFACT_TOLERANCE] = np.nan
    return matlab, python


def stats(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    missing_mismatch = int(np.sum(np.isfinite(x) != np.isfinite(y)))
    x, y = x[valid], y[valid]
    if not len(x):
        return {
            "n": 0,
            "r": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "max_abs": np.nan,
            "nan_mismatch": missing_mismatch,
        }
    diff = y - x
    if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
        r = float(np.corrcoef(x, y)[0, 1])
    else:
        r = np.nan
    return {
        "n": int(len(x)),
        "r": r,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "max_abs": float(np.max(np.abs(diff))),
        "nan_mismatch": missing_mismatch,
    }


def robust_limits(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    combined = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    if not len(combined):
        return -1, 1
    if len(combined) > 20:
        low, high = np.quantile(combined, [.005, .995])
    else:
        low, high = np.min(combined), np.max(combined)
    if low == high:
        padding = max(abs(low) * .1, .1)
    else:
        padding = .08 * (high - low)
    return float(low - padding), float(high + padding)


def draw_scatter_grid(
    label: str,
    matlab: np.ndarray,
    python: np.ndarray,
    path: Path,
    pdf: PdfPages,
) -> list[dict[str, float | int | str]]:
    fig, axes = plt.subplots(4, 5, figsize=(15, 11.5))
    rows = []

    for index, (measure, ax) in enumerate(zip(MEASURES, axes.flat)):
        x = matlab[..., index].ravel()
        y = python[..., index].ravel()
        valid = np.isfinite(x) & np.isfinite(y)
        summary = stats(x, y)
        rows.append({"comparison": label, "measure": measure, **summary})

        if valid.any():
            ax.scatter(
                x[valid],
                y[valid],
                s=10,
                alpha=.5,
                color="#a33d32" if index == META_NOISE_INDEX else "#2766ad",
                edgecolors="none",
                rasterized=True,
            )
            low, high = robust_limits(x, y)
            ax.plot([low, high], [low, high], "--", color="#555555", linewidth=.9)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)

        r_text = "NA" if not np.isfinite(summary["r"]) else f"{summary['r']:.4f}"
        ax.set_title(
            f"{measure}\nr={r_text} · MAE={summary['mae']:.3g} · n={summary['n']}",
            fontsize=9,
        )
        if index // 5 == 3:
            ax.set_xlabel("MATLAB")
        if index % 5 == 0:
            ax.set_ylabel("Python")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)

    fig.suptitle(
        f"{label} — MATLAB vs Python, all 20 measures",
        fontsize=15,
        fontweight="bold",
        y=.99,
    )
    fig.text(
        .5,
        .955,
        "Dashed line: exact equality · red panel: meta-noise · MATLAB artefact bins masked",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0, 1, .925])
    fig.savefig(path, dpi=170, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return rows


def annotate_heatmap(ax: plt.Axes, matrix: np.ndarray, fmt: str) -> None:
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isfinite(value):
                ax.text(col, row, format(value, fmt), ha="center", va="center", fontsize=6)


def draw_summary(
    rows: list[dict[str, float | int | str]], path: Path, pdf: PdfPages
) -> None:
    labels = [spec.label for spec in SPECS]
    shape = (len(labels), N_MEASURES)
    correlation = np.full(shape, np.nan)
    mae = np.full(shape, np.nan)

    lookup = {(str(row["comparison"]), str(row["measure"])): row for row in rows}
    for row_index, label in enumerate(labels):
        for col_index, measure in enumerate(MEASURES):
            row = lookup[(label, measure)]
            correlation[row_index, col_index] = float(row["r"])
            mae[row_index, col_index] = float(row["mae"])

    # Log scale lets unlike measurement units share one heatmap while retaining
    # absolute-error meaning. Exact zeros are displayed at the numerical floor.
    log_mae = np.log10(np.maximum(mae, 1e-12))

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    image_r = axes[0].imshow(
        correlation, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto"
    )
    axes[0].set_title("Pearson correlation (MATLAB vs Python)", fontweight="bold")
    annotate_heatmap(axes[0], correlation, ".2f")
    fig.colorbar(image_r, ax=axes[0], fraction=.018, pad=.01, label="Pearson r")

    # Clip display only; CSV retains exact values.
    image_e = axes[1].imshow(
        log_mae, cmap="YlOrRd", vmin=-12, vmax=2, aspect="auto"
    )
    axes[1].set_title(
        "Absolute error (log10 MAE; exact equality shown at −12)",
        fontweight="bold",
    )
    annotate_heatmap(axes[1], log_mae, ".1f")
    fig.colorbar(image_e, ax=axes[1], fraction=.018, pad=.01, label="log10(MAE)")

    for ax in axes:
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
    axes[1].set_xticks(range(N_MEASURES))
    axes[1].set_xticklabels(MEASURES, rotation=50, ha="right")
    fig.suptitle("Full MATLAB–Python comparison summary", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(path, dpi=180, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_global_scatter(
    loaded: list[tuple[ComparisonSpec, np.ndarray, np.ndarray]],
    path: Path,
    pdf: PdfPages,
) -> None:
    """Aggregate standardized values so all analyses can share one plot."""
    figure, axes = plt.subplots(4, 5, figsize=(15, 11.5))
    for index, (measure, ax) in enumerate(zip(MEASURES, axes.flat)):
        all_x, all_y = [], []
        for _, matlab, python in loaded:
            x = matlab[..., index].ravel()
            y = python[..., index].ravel()
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            if not len(x):
                continue
            center = np.mean(x)
            spread = np.std(x)
            if spread == 0:
                continue
            all_x.append((x - center) / spread)
            all_y.append((y - center) / spread)
        x = np.concatenate(all_x) if all_x else np.array([])
        y = np.concatenate(all_y) if all_y else np.array([])
        summary = stats(x, y)
        if len(x):
            ax.hexbin(
                x,
                y,
                gridsize=35,
                mincnt=1,
                cmap="Blues" if index != META_NOISE_INDEX else "Reds",
            )
            low, high = robust_limits(x, y)
            ax.plot([low, high], [low, high], "--", color="#444444", linewidth=.9)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
        r_text = "NA" if not np.isfinite(summary["r"]) else f"{summary['r']:.3f}"
        ax.set_title(f"{measure} · pooled r={r_text}", fontsize=9)
        if index // 5 == 3:
            ax.set_xlabel("MATLAB (standardized)")
        if index % 5 == 0:
            ax.set_ylabel("Python (MATLAB scale)")
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Pooled MATLAB–Python agreement across all analyses",
        fontsize=15,
        fontweight="bold",
        y=.99,
    )
    figure.text(
        .5,
        .955,
        "Values standardized within each matched result array before pooling",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=[0, 0, 1, .925])
    figure.savefig(path, dpi=180, bbox_inches="tight")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    columns = [
        "comparison",
        "measure",
        "n",
        "r",
        "mae",
        "rmse",
        "bias",
        "max_abs",
        "nan_mismatch",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    results = repo / "matlab" / "metasignal_mat" / "Results"
    precomputed = repo / "notebooks" / "precomputed"
    output = (
        args.out or repo / "notebooks" / "figures" / "matlab_python_full_comparison"
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)

    loaded = [(spec, *load_pair(results, precomputed, spec)) for spec in SPECS]
    rows: list[dict[str, float | int | str]] = []
    pdf_path = output / "full_matlab_python_comparison.pdf"

    with PdfPages(pdf_path) as pdf:
        for page, (spec, matlab, python) in enumerate(loaded, start=1):
            print(f"[{page}/{len(loaded)}] {spec.label}")
            rows.extend(
                draw_scatter_grid(
                    spec.label,
                    matlab,
                    python,
                    output / f"{page:02d}_{slug(spec.label)}.png",
                    pdf,
                )
            )
        draw_summary(rows, output / "11_summary_heatmaps.png", pdf)
        draw_global_scatter(loaded, output / "12_pooled_measure_agreement.png", pdf)

    write_csv(output / "matlab_python_statistics.csv", rows)
    print(f"Wrote {len(SPECS)} scatter grids, 2 summaries, CSV, and PDF to {output}")
    print(f"Combined PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
