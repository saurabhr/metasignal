#!/usr/bin/env python3
"""Generate Rahnev (2025) paper-vs-MATLAB-vs-Python comparison plots.

The "Paper" series uses the values printed in Rahnev's Figure 7. MATLAB
series are recomputed from matlab/metasignal_mat/Results/*.mat; Python series
are recomputed from notebooks/precomputed/*.npz.

Outputs:
  notebooks/figures/rahnev_comparison/
    figure1_precision_comparison.png
    figure2_task_performance_comparison.png
    figure3_metacognitive_bias_comparison.png
    figure4_response_bias_comparison.png
    figure5_split_half_comparison.png
    figure6_test_retest_comparison.png
    figure7_summary_comparison.png
    matlab_python_agreement.png
    rahnev_all_comparison_plots.pdf
    comparison_values.csv

Usage:
    python scripts/generate_rahnev_comparison_plots.py
    python scripts/generate_rahnev_comparison_plots.py --out path/to/output
"""

from __future__ import annotations

import argparse
import csv
import os
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
]
N_META = len(MEASURES)

# Values printed in Rahnev (2025), Figure 7.
PAPER = {
    "precision": np.array(
        [.65, .54, .65, .61, .50, .61, .60, .61, .62, .58, .56, .59, .65, .62, .53, .63, .34]
    ),
    "task": np.array(
        [2.47, 2.29, 2.95, 1.34, 1.81, -.18, -.39, -.11, -.17, -.23, -.58, -.49, -.39, -.30, -.55, -.29, .06]
    ),
    "bias": np.array(
        [.44, .51, -.61, .81, .54, .27, .09, .001, .23, .42, .43, .10, .24, .11, .34, -.21, .27]
    ),
    "response": np.array(
        [-.04, .18, .12, .11, .18, .07, .13, .08, .01, .11, -.002, .12, .06, .001, .12, .03, .13]
    ),
    "split_half": np.array(
        [.89, .89, .88, .87, .90, .85, .85, .84, .84, .84, .87, .85, .85, .85, .85, .84, .86]
    ),
    "test_retest": np.array(
        [.71, .73, .71, .63, .75, .42, .36, .30, .28, .27, .47, .29, .43, .35, .31, .29, .21]
    ),
}

COLORS = {"Paper": "#4d4d4d", "MATLAB": "#2766ad", "Python": "#c4473a"}
MARKERS = {"Paper": "o", "MATLAB": "s", "Python": "^"}


def load_mat(results: Path, filename: str, key: str) -> np.ndarray:
    return np.asarray(
        scipy.io.loadmat(results / filename, squeeze_me=True, struct_as_record=False)[key],
        dtype=float,
    )


def load_npz(precomp: Path, filename: str, key: str) -> np.ndarray:
    return np.asarray(np.load(precomp / filename)[key], dtype=float)[..., :20]


def fisher_mean(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    values = np.clip(values, -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(values))))


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def cohen_d(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd) if sd > 0 else np.nan


def remove_3sd(arr: np.ndarray) -> np.ndarray:
    """Match ana_taskPerformance.m outlier removal."""
    out = np.asarray(arr, float).copy()
    for measure in range(out.shape[-1]):
        for level in range(out.shape[1]):
            col = out[:, level, measure]
            mu, sd = np.nanmean(col), np.nanstd(col, ddof=1)
            if np.isfinite(sd) and sd > 0:
                out[np.abs(col - mu) > 3 * sd, level, measure] = np.nan
        bad = np.any(~np.isfinite(out[:, :, measure]), axis=1)
        out[bad, :, measure] = np.nan
    return out


def compute_task_effects(results: Path, precomp: Path) -> tuple[np.ndarray, np.ndarray]:
    specs = [
        ("results_Shekhar.mat", "metas_diff", "shekhar_mle.npz", "diff", [0, 2]),
        ("results_Rouault1.mat", "metas_diff", "rouault1_mle.npz", "diff", [0, 1]),
        ("results_Rouault2.mat", "metas_diff", "rouault2_mle.npz", "diff", [0, 1]),
    ]
    mat_ds, py_ds = [], []
    for mf, mk, pf, pk, levels in specs:
        mat = remove_3sd(load_mat(results, mf, mk)[:, levels, :])
        py = remove_3sd(load_npz(precomp, pf, pk)[:, levels, :])
        mat_ds.append([cohen_d(mat[:, 1, m] - mat[:, 0, m]) for m in range(N_META)])
        py_ds.append([cohen_d(py[:, 1, m] - py[:, 0, m]) for m in range(N_META)])
    return np.nanmean(mat_ds, axis=0), np.nanmean(py_ds, axis=0)


def _bias_array(arr: np.ndarray, shekhar: bool) -> np.ndarray:
    # Shekhar: subject × contrast × recode × measure.
    return np.nanmean(arr, axis=1) if shekhar else arr


def compute_bias_effects(results: Path, precomp: Path) -> tuple[np.ndarray, np.ndarray]:
    specs = [
        ("results_Haddara.mat", "metas_confRecode", "haddara_mle.npz", "bias", False),
        ("results_Maniscalco.mat", "metas_confRecode", "maniscalco_mle.npz", "bias", False),
        ("results_Shekhar.mat", "metas_confRecode", "shekhar_mle.npz", "bias", True),
    ]
    mat_ds, py_ds = [], []
    for mf, mk, pf, pk, shekhar in specs:
        mat = _bias_array(load_mat(results, mf, mk), shekhar)
        py = _bias_array(load_npz(precomp, pf, pk), shekhar)
        mat_ds.append([cohen_d(mat[:, 1, m] - mat[:, 0, m]) for m in range(N_META)])
        py_ds.append([cohen_d(py[:, 1, m] - py[:, 0, m]) for m in range(N_META)])
    return np.nanmean(mat_ds, axis=0), np.nanmean(py_ds, axis=0)


def compute_response_effects(results: Path, precomp: Path) -> tuple[np.ndarray, np.ndarray]:
    mat = load_mat(results, "results_Locke.mat", "metas_bias")
    py = load_npz(precomp, "locke_mle.npz", "rb")

    def per_backend(arr: np.ndarray) -> np.ndarray:
        values = np.full(N_META, np.nan)
        criterion = np.abs(arr[:, :, 18])
        for m in range(N_META):
            per_subject = [
                correlation(criterion[s], arr[s, :, m]) for s in range(arr.shape[0])
            ]
            values[m] = fisher_mean(per_subject)
        return values

    return per_backend(mat), per_backend(py)


def _precision_dataset(cell_array: np.ndarray, haddara: bool) -> np.ndarray:
    """Port the normalized precision calculation in ana_precision.m."""
    by_size = []
    for raw in cell_array.flat:
        arr = np.asarray(raw, float).copy()
        if haddara:
            # subject × bins × days × alteration × measure
            if arr.ndim == 4:  # single bin at 400 trials
                arr = arr[:, None, :, :, :]
            arr = arr.reshape(arr.shape[0], -1, arr.shape[-2], arr.shape[-1])
        elif arr.ndim == 3:  # single Maniscalco bin
            arr = arr[:, None, :, :]

        arr[np.abs(arr) > 4.5] = np.nan
        arr[..., 15:17] *= -1
        sd_subject_measure = np.nanstd(arr[:, :, 0, :], axis=1, ddof=1)
        denominator = np.nanmean(sd_subject_measure, axis=0)
        per_alter = []
        for alteration in range(1, arr.shape[2]):
            drop = np.nanmean(arr[:, :, 0, :] - arr[:, :, alteration, :], axis=1)
            per_alter.append(drop / denominator)
        by_size.append(np.asarray(per_alter))

    # bin size × alteration × subject × measure -> subject × measure
    mean_subject = np.nanmean(np.asarray(by_size), axis=(0, 1))
    mean_measure = np.nanmean(mean_subject, axis=0)[:N_META]
    return mean_measure / np.nanmean(mean_measure[:16])


def compute_precision(results: Path, precomp: Path) -> tuple[np.ndarray, np.ndarray]:
    ha = scipy.io.loadmat(
        results / "results_Haddara.mat", squeeze_me=True, struct_as_record=False
    )["metas_precision"]
    ma = scipy.io.loadmat(
        results / "results_Maniscalco.mat", squeeze_me=True, struct_as_record=False
    )["metas_precision"]
    mat = np.nanmean([_precision_dataset(ha, True), _precision_dataset(ma, False)], axis=0)

    # Python cache currently contains Haddara only and intentionally omits the
    # slow MLE family (meta-d', M-Ratio, M-Diff).
    drops = np.asarray(np.load(precomp / "haddara_precision.npz")["drops"], float)
    # For noise/uncertainty, lower values mean better metacognition, so match
    # the sign flip used by Rahnev's MATLAB precision analysis.
    drops[:, 15:17] *= -1
    py = np.nanmean(drops[:, :N_META], axis=0)
    py /= np.nanmean(py[np.r_[1:5, 6:10, 11:16]])
    return mat, py


def _split_half_from_mat(cell: np.ndarray, dataset: str, bin_index: int = 1) -> np.ndarray:
    arr = np.asarray(cell.flat[bin_index], float)
    out = np.full(N_META, np.nan)
    for m in range(N_META):
        rs = []
        if dataset in {"Haddara", "Shekhar"}:
            # subject × bin × day/contrast × half × measure
            for b in range(arr.shape[1]):
                for d in range(arr.shape[2]):
                    rs.append(correlation(arr[:, b, d, 0, m], arr[:, b, d, 1, m]))
        else:
            # subject × bin × half × measure
            for b in range(arr.shape[1]):
                rs.append(correlation(arr[:, b, 0, m], arr[:, b, 1, m]))
        out[m] = fisher_mean(rs)
    return out


def compute_split_half(results: Path, precomp: Path) -> tuple[np.ndarray, np.ndarray]:
    mat_values, py_values = [], []
    specs = [
        ("Haddara", "results_Haddara.mat", "haddara_mle.npz"),
        ("Shekhar", "results_Shekhar.mat", "shekhar_mle.npz"),
        ("Maniscalco", "results_Maniscalco.mat", "maniscalco_mle.npz"),
    ]
    for dataset, mf, pf in specs:
        cell = scipy.io.loadmat(
            results / mf, squeeze_me=True, struct_as_record=False
        )["metas_splitHalf"]
        mat_values.append(_split_half_from_mat(cell, dataset, bin_index=1))
        split = load_npz(precomp, pf, "split")
        py_values.append(
            [correlation(split[:, 0, m], split[:, 1, m]) for m in range(N_META)]
        )
    return (
        np.array([fisher_mean(v) for v in np.asarray(mat_values).T]),
        np.array([fisher_mean(v) for v in np.asarray(py_values).T]),
    )


def icc_a1(two_columns: np.ndarray) -> float:
    """Two-rater absolute-agreement ICC(A,1), matching MATLAB ICC(..., 'A-1')."""
    x = np.asarray(two_columns, float)
    x = x[np.all(np.isfinite(x), axis=1)]
    n, k = x.shape if x.ndim == 2 else (0, 0)
    if n < 3 or k != 2:
        return np.nan
    grand = np.mean(x)
    row_mean = np.mean(x, axis=1)
    col_mean = np.mean(x, axis=0)
    msr = k * np.sum((row_mean - grand) ** 2) / (n - 1)
    msc = n * np.sum((col_mean - grand) ** 2) / (k - 1)
    residual = x - row_mean[:, None] - col_mean[None, :] + grand
    mse = np.sum(residual**2) / ((n - 1) * (k - 1))
    denominator = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denominator) if denominator else np.nan


def _remove_test_retest_outliers(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, float).copy()
    # subject × bin × day × measure
    for b in range(out.shape[1]):
        for d in range(out.shape[2]):
            for m in range(out.shape[3]):
                col = out[:, b, d, m]
                mu, sd = np.nanmean(col), np.nanstd(col, ddof=1)
                if np.isfinite(sd) and sd > 0:
                    bad = np.abs(col - mu) > 3 * sd
                    out[bad, b, d, m] = np.nan
    return out


def _test_retest_icc(arr: np.ndarray) -> np.ndarray:
    # Accept subject × day × measure or subject × bin × day × measure.
    if arr.ndim == 3:
        arr = arr[:, None, :, :]
    out = np.full(N_META, np.nan)
    for m in range(N_META):
        values = []
        for b in range(arr.shape[1]):
            for d1 in range(arr.shape[2] - 1):
                for d2 in range(d1 + 1, arr.shape[2]):
                    values.append(icc_a1(arr[:, b, [d1, d2], m]))
        out[m] = fisher_mean(values)
    return out


def compute_test_retest(results: Path, precomp: Path) -> tuple[np.ndarray, np.ndarray]:
    cells = scipy.io.loadmat(
        results / "results_Haddara.mat", squeeze_me=True, struct_as_record=False
    )["metas_testRetest"]
    # MATLAB script flips the order; cell index 3 is the 400-trial result.
    mat_arr = np.asarray(cells.flat[3], float)
    if mat_arr.ndim == 3:
        mat_arr = mat_arr[:, None, :, :]
    mat = _test_retest_icc(_remove_test_retest_outliers(mat_arr))

    py_arr = load_npz(precomp, "haddara_testRetest.npz", "data")
    # Apply the same ±3 SD cleaning by introducing a singleton bin dimension.
    py4 = _remove_test_retest_outliers(py_arr[:, None, :, :])
    py = _test_retest_icc(py4)
    return mat, py


def style_axis(ax: plt.Axes, zero: bool = False) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    if zero:
        ax.axhline(0, color="#777777", linewidth=.8)


def comparison_plot(
    key: str,
    matlab: np.ndarray,
    python: np.ndarray,
    title: str,
    ylabel: str,
    out_file: Path,
    pdf: PdfPages,
    *,
    zero: bool = False,
    ylim: tuple[float, float] | None = None,
    note: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.8))
    x = np.arange(N_META)
    offsets = {"Paper": -.22, "MATLAB": 0, "Python": .22}
    series = {"Paper": PAPER[key], "MATLAB": matlab, "Python": python}
    for label, values in series.items():
        ax.scatter(
            x + offsets[label],
            values,
            s=48,
            marker=MARKERS[label],
            color=COLORS[label],
            label=label,
            zorder=3,
        )
        ax.plot(x + offsets[label], values, color=COLORS[label], alpha=.35, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(MEASURES, rotation=50, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    if note:
        ax.text(
            .01,
            .98,
            note,
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="#555555",
        )
    ax.legend(ncol=3, frameon=False)
    style_axis(ax, zero=zero)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def summary_plot(
    matlab: dict[str, np.ndarray],
    python: dict[str, np.ndarray],
    out_file: Path,
    pdf: PdfPages,
) -> None:
    metrics = ["precision", "task", "bias", "response", "split_half", "test_retest"]
    labels = ["Precision", "Task d", "Bias d", "Response r", "Split-half r", "Test-retest ICC"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, source, values in zip(
        axes,
        ["Paper", "MATLAB", "Python"],
        [PAPER, matlab, python],
    ):
        matrix = np.vstack([values[k] for k in metrics])
        # Column-normalize only for visualization; numeric values are annotated.
        scale = np.nanmax(np.abs(matrix), axis=1, keepdims=True)
        normalized = np.divide(matrix, scale, where=scale != 0)
        image = ax.imshow(normalized, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(source, loc="left", fontweight="bold", color=COLORS[source])
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                text = "NA" if not np.isfinite(value) else f"{value:.2f}"
                ax.text(col, row, text, ha="center", va="center", fontsize=6)
    axes[-1].set_xticks(range(N_META))
    axes[-1].set_xticklabels(MEASURES, rotation=50, ha="right")
    color_axis = fig.add_axes([.955, .18, .012, .65])
    fig.colorbar(image, cax=color_axis, label="Row-normalized display scale")
    fig.suptitle(
        "Rahnev Figure 7 summary — published values vs MATLAB vs Python",
        fontweight="bold",
    )
    fig.subplots_adjust(left=.12, right=.94, bottom=.16, top=.93, hspace=.25)
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def agreement_plot(
    matlab: dict[str, np.ndarray],
    python: dict[str, np.ndarray],
    out_file: Path,
    pdf: PdfPages,
) -> None:
    keys = ["precision", "task", "bias", "response", "split_half", "test_retest"]
    labels = ["Precision", "Task d", "Bias d", "Response r", "Split-half r", "Test-retest ICC"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, key, label in zip(axes.flat, keys, labels):
        x, y = matlab[key], python[key]
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], color=COLORS["Python"], alpha=.8)
        if ok.any():
            lo = min(np.min(x[ok]), np.min(y[ok]))
            hi = max(np.max(x[ok]), np.max(y[ok]))
            ax.plot([lo, hi], [lo, hi], "--", color="#777777", linewidth=1)
        r = correlation(x, y)
        ax.set_title(f"{label} · r={r:.3f}", fontsize=10)
        ax.set_xlabel("MATLAB")
        ax.set_ylabel("Python")
        style_axis(ax, zero=key in {"task", "bias", "response"})
    fig.suptitle("MATLAB–Python agreement across paper analyses", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, matlab: dict[str, np.ndarray], python: dict[str, np.ndarray]) -> None:
    metrics = ["precision", "task", "bias", "response", "split_half", "test_retest"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["measure", "metric", "paper", "matlab", "python"])
        for index, measure in enumerate(MEASURES):
            for metric in metrics:
                writer.writerow(
                    [
                        measure,
                        metric,
                        PAPER[metric][index],
                        matlab[metric][index],
                        python[metric][index],
                    ]
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    results = repo / "matlab" / "metasignal_mat" / "Results"
    precomp = repo / "notebooks" / "precomputed"
    out = (args.out or repo / "notebooks" / "figures" / "rahnev_comparison").resolve()
    out.mkdir(parents=True, exist_ok=True)

    required = [
        results / "results_Haddara.mat",
        results / "results_Maniscalco.mat",
        results / "results_Shekhar.mat",
        results / "results_Rouault1.mat",
        results / "results_Rouault2.mat",
        results / "results_Locke.mat",
        precomp / "haddara_mle.npz",
        precomp / "maniscalco_mle.npz",
        precomp / "shekhar_mle.npz",
        precomp / "rouault1_mle.npz",
        precomp / "rouault2_mle.npz",
        precomp / "locke_mle.npz",
        precomp / "haddara_precision.npz",
        precomp / "haddara_testRetest.npz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    print("Computing paper-analysis summaries...")
    task_mat, task_py = compute_task_effects(results, precomp)
    bias_mat, bias_py = compute_bias_effects(results, precomp)
    response_mat, response_py = compute_response_effects(results, precomp)
    precision_mat, precision_py = compute_precision(results, precomp)
    split_mat, split_py = compute_split_half(results, precomp)
    retest_mat, retest_py = compute_test_retest(results, precomp)

    matlab = {
        "precision": precision_mat,
        "task": task_mat,
        "bias": bias_mat,
        "response": response_mat,
        "split_half": split_mat,
        "test_retest": retest_mat,
    }
    python = {
        "precision": precision_py,
        "task": task_py,
        "bias": bias_py,
        "response": response_py,
        "split_half": split_py,
        "test_retest": retest_py,
    }

    pdf_path = out / "rahnev_all_comparison_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        comparison_plot(
            "precision",
            precision_mat,
            precision_py,
            "Figure 1 — Validity and normalized precision",
            "Normalized precision",
            out / "figure1_precision_comparison.png",
            pdf,
            ylim=(0, max(1.5, np.nanmax([PAPER["precision"], precision_mat, precision_py]) * 1.1)),
            note="Python precision: Haddara only; meta-d′, M-Ratio and M-Diff were not cached (shown as NA).",
        )
        comparison_plot(
            "task",
            task_mat,
            task_py,
            "Figure 2 — Dependence on task performance",
            "Average Cohen's d",
            out / "figure2_task_performance_comparison.png",
            pdf,
            zero=True,
        )
        comparison_plot(
            "bias",
            bias_mat,
            bias_py,
            "Figure 3 — Dependence on metacognitive bias",
            "Average Cohen's d",
            out / "figure3_metacognitive_bias_comparison.png",
            pdf,
            zero=True,
        )
        comparison_plot(
            "response",
            response_mat,
            response_py,
            "Figure 4 — Dependence on absolute response bias",
            "Average within-subject Pearson r",
            out / "figure4_response_bias_comparison.png",
            pdf,
            zero=True,
        )
        comparison_plot(
            "split_half",
            split_mat,
            split_py,
            "Figure 5 — Split-half reliability",
            "Pearson r",
            out / "figure5_split_half_comparison.png",
            pdf,
            ylim=(0, 1.02),
            note="Paper and MATLAB use 100-trial bins; Python cache uses each participant's full odd/even split.",
        )
        comparison_plot(
            "test_retest",
            retest_mat,
            retest_py,
            "Figure 6 — Test-retest reliability",
            "ICC(A,1)",
            out / "figure6_test_retest_comparison.png",
            pdf,
            ylim=(0, 1.02),
            note="Paper/MATLAB use 400-trial bins; Python uses the full cached day-level estimates.",
        )
        summary_plot(matlab, python, out / "figure7_summary_comparison.png", pdf)
        agreement_plot(matlab, python, out / "matlab_python_agreement.png", pdf)

    write_csv(out / "comparison_values.csv", matlab, python)
    print(f"Wrote 8 plots, CSV, and PDF to {out}")
    print(f"Combined PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
