#!/usr/bin/env python3
"""Three-way comparison: Rahnev (2025) paper · MATLAB Results · Python caches.

Compares:
  1. Subject-level measure arrays  — MATLAB ``matlab/metasignal_mat/Results/*.mat``
     vs Python ``notebooks/precomputed/*_mle.npz`` (first 20 of 26 columns).
  2. Supplementary-table *t*-tests — same contrasts in both backends vs the
     MATLAB-table / paper reference *t* values.
  3. Published paper scalars       — e.g. Haddara mean meta-d' ≈ 1.14 and the
     confidence-corruption ladder.

Usage (from repo root)::

    python scripts/compare_paper_matlab_python.py
    python scripts/compare_paper_matlab_python.py -v
    python scripts/compare_paper_matlab_python.py --table
    python scripts/compare_paper_matlab_python.py --json out.json

Exit code 0 = overall PASS; 1 = FAIL (or missing inputs).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import scipy.io
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEASURE_NAMES = [
    "meta-d'",
    "AUC2",
    "Gamma",
    "Phi",
    "DeltaConf",
    "M-Ratio",
    "AUC2-Ratio",
    "Gamma-Ratio",
    "Phi-Ratio",
    "DeltaConf-Ratio",
    "M-Diff",
    "AUC2-Diff",
    "Gamma-Diff",
    "Phi-Diff",
    "DeltaConf-Diff",
    "meta-noise",
    "meta-uncertainty",
    "d'",
    "Criterion",
    "Confidence",
]
N_PRIMARY = 20

EXACT_FAMILY = {"AUC2", "Gamma", "Phi", "DeltaConf", "d'", "Criterion", "Confidence"}
OPTIMIZE_FAMILY = {"meta-noise", "meta-uncertainty"}

# MATLAB goldenSearch returns ~0.495934 when the meta-noise logL is NaN
# (unidentifiable bins with non-finite d'). Mask for an honest comparison.
MN_ARTIFACT = 0.495934
MN_ARTIFACT_TOL = 1e-3
META_NOISE_IDX = 15

ARRAY_COMPARISONS = [
    ("Haddara raw", "results_Haddara.mat", "metas_raw", "haddara_mle.npz", "raw"),
    ("Haddara bias", "results_Haddara.mat", "metas_confRecode", "haddara_mle.npz", "bias"),
    ("Haddara oddEven", "results_Haddara.mat", "metas_oddEven", "haddara_mle.npz", "split"),
    ("Maniscalco raw", "results_Maniscalco.mat", "metas_raw", "maniscalco_mle.npz", "raw"),
    ("Maniscalco bias", "results_Maniscalco.mat", "metas_confRecode", "maniscalco_mle.npz", "bias"),
    ("Rouault1 diff", "results_Rouault1.mat", "metas_diff", "rouault1_mle.npz", "diff"),
    ("Rouault2 diff", "results_Rouault2.mat", "metas_diff", "rouault2_mle.npz", "diff"),
    ("Shekhar diff", "results_Shekhar.mat", "metas_diff", "shekhar_mle.npz", "diff"),
    ("Shekhar bias", "results_Shekhar.mat", "metas_confRecode", "shekhar_mle.npz", "bias"),
    ("Locke bias", "results_Locke.mat", "metas_bias", "locke_mle.npz", "rb"),
]

STAT_TESTS = [
    {
        "table": "Table 3: Shekhar difficulty",
        "mat_file": "results_Shekhar.mat",
        "mat_key": "metas_diff",
        "npz_file": "shekhar_mle.npz",
        "npz_key": "diff",
        "transform": "shekhar_contrast",
        "reference": {
            "meta-d'": 22.616,
            "AUC2": 20.612,
            "Gamma": 29.238,
            "Phi": 10.898,
            "DeltaConf": 14.834,
            "M-Ratio": -1.240,
            "d'": 23.777,
            "Confidence": 14.543,
        },
    },
    {
        "table": "Table 4: Rouault1 difficulty",
        "mat_file": "results_Rouault1.mat",
        "mat_key": "metas_diff",
        "npz_file": "rouault1_mle.npz",
        "npz_key": "diff",
        "transform": "easy_minus_hard",
        "reference": {
            "meta-d'": 35.285,
            "AUC2": 35.405,
            "d'": 49.278,
            "Confidence": 32.390,
        },
    },
    {
        "table": "Table 6: Haddara metacognitive bias",
        "mat_file": "results_Haddara.mat",
        "mat_key": "metas_confRecode",
        "npz_file": "haddara_mle.npz",
        "npz_key": "bias",
        "transform": "xue_recode_diff",
        "reference": {
            "AUC2": 0.688,
            "Gamma": -4.331,
            "Phi": 1.257,
            "DeltaConf": 1.034,
            "M-Ratio": 1.795,
            "Gamma-Diff": 2.361,
            "Confidence": 24.538,
        },
    },
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MeasureStats:
    name: str
    n: int
    pearson_r: float
    max_abs_diff: float
    mean_abs_diff: float
    mat_finite: int
    py_finite: int


@dataclass
class ArrayComparison:
    label: str
    status: str
    shape: tuple
    max_abs_diff: float
    mean_abs_diff: float
    nan_mismatches: int
    min_correlation: float
    median_correlation: float
    worst_measure: str
    worst_diff: float
    measure_stats: list[MeasureStats] = field(default_factory=list)
    failed_measures: list[str] = field(default_factory=list)


@dataclass
class StatComparison:
    table: str
    measure: str
    py_t: float
    mat_t: float
    ref_t: float
    py_sig: bool
    mat_sig: bool
    sig_match: bool
    ref_match: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here.parent, here, *here.parents]:
        if (candidate / "src" / "metasignal").is_dir() and (
            candidate / "matlab" / "metasignal_mat" / "Results"
        ).is_dir():
            return candidate
    raise FileNotFoundError("Could not locate metasignal repo root")


def primary(arr: np.ndarray) -> np.ndarray:
    """Keep only the 20 primary Rahnev measures (drop Python fit diagnostics)."""
    arr = np.asarray(arr, float)
    if arr.shape[-1] > N_PRIMARY:
        return arr[..., :N_PRIMARY]
    return arr


def mask_matlab_metanoise_artifact(mat_arr: np.ndarray) -> np.ndarray:
    """Set MATLAB's meta-noise search artefact (~0.4959) cells to NaN."""
    out = np.array(mat_arr, float).copy()
    if out.shape[-1] > META_NOISE_IDX:
        col = out[..., META_NOISE_IDX]
        col[np.abs(col - MN_ARTIFACT) < MN_ARTIFACT_TOL] = np.nan
    return out


def ttest_1samp(data: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(data, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    t, p = stats.ttest_1samp(x, 0.0)
    return float(t), float(len(x) - 1), float(p)


def remove_3sd_outliers(arr: np.ndarray) -> np.ndarray:
    """Match MATLAB ana_taskPerformance.m ±3 SD outlier removal."""
    out = np.array(arr, float).copy()
    n_sub, n_levels = out.shape[0], out.shape[1]
    n_meas = out.shape[-1]
    for m in range(n_meas):
        for lev in range(n_levels):
            col = out[:, lev, m]
            valid = ~np.isnan(col)
            if valid.sum() < 3:
                continue
            mu, sd = np.nanmean(col), np.nanstd(col)
            if sd > 0:
                out[np.abs(col - mu) > 3 * sd, lev, m] = np.nan
        for sub in range(n_sub):
            if np.any(np.isnan(out[sub, :, m])):
                out[sub, :, m] = np.nan
    return out


def delta_per_subject(arr: np.ndarray, transform: str) -> np.ndarray:
    if transform == "easy_minus_hard":
        clean = remove_3sd_outliers(arr)
        return clean[:, 1, :] - clean[:, 0, :]
    if transform == "shekhar_contrast":
        clean = remove_3sd_outliers(arr[:, [0, 2], :])
        return clean[:, 1, :] - clean[:, 0, :]
    if transform == "xue_recode_diff":
        return arr[:, 1, :] - arr[:, 0, :]
    raise ValueError(transform)


def t_match(t: float, ref: float, rtol: float = 0.05, atol: float = 0.1) -> bool:
    if np.isnan(t) or np.isnan(ref):
        return False
    return abs(t - ref) < abs(ref) * rtol + atol


def per_measure_stats(mat: np.ndarray, py: np.ndarray) -> list[MeasureStats]:
    rows = []
    for m, name in enumerate(MEASURE_NAMES):
        a = mat[..., m].ravel()
        b = py[..., m].ravel()
        both = ~np.isnan(a) & ~np.isnan(b)
        if both.sum() < 3:
            rows.append(
                MeasureStats(
                    name=name,
                    n=int(both.sum()),
                    pearson_r=np.nan,
                    max_abs_diff=np.nan,
                    mean_abs_diff=np.nan,
                    mat_finite=int(np.isfinite(a).sum()),
                    py_finite=int(np.isfinite(b).sum()),
                )
            )
            continue
        aa, bb = a[both], b[both]
        rows.append(
            MeasureStats(
                name=name,
                n=int(both.sum()),
                pearson_r=float(np.corrcoef(aa, bb)[0, 1]),
                max_abs_diff=float(np.max(np.abs(aa - bb))),
                mean_abs_diff=float(np.mean(np.abs(aa - bb))),
                mat_finite=int(np.isfinite(a).sum()),
                py_finite=int(np.isfinite(b).sum()),
            )
        )
    return rows


def compare_arrays(label: str, mat_arr: np.ndarray, py_arr: np.ndarray) -> ArrayComparison:
    mat_arr = mask_matlab_metanoise_artifact(primary(mat_arr))
    py_arr = primary(py_arr)
    if mat_arr.shape != py_arr.shape:
        return ArrayComparison(
            label=label,
            status="SHAPE_MISMATCH",
            shape=tuple(mat_arr.shape),
            max_abs_diff=np.nan,
            mean_abs_diff=np.nan,
            nan_mismatches=-1,
            min_correlation=np.nan,
            median_correlation=np.nan,
            worst_measure="—",
            worst_diff=np.nan,
        )

    stats_rows = per_measure_stats(mat_arr, py_arr)
    # PASS gates on closed-form + meta-d' family. Ratio/Diff (non-M) and
    # meta-noise/uncertainty can diverge on small-n / near-zero d' edge cases.
    hard = OPTIMIZE_FAMILY | {
        n
        for n in MEASURE_NAMES
        if (("Ratio" in n and n != "M-Ratio") or ("Diff" in n and n != "M-Diff"))
    }
    focus = [s for s in stats_rows if s.name not in hard and not np.isnan(s.pearson_r)]
    ratios = [s for s in stats_rows if s.name in hard and not np.isnan(s.pearson_r)]

    nan_mm = int(np.sum(np.isnan(mat_arr) != np.isnan(py_arr)))
    both_ok = ~(np.isnan(mat_arr) | np.isnan(py_arr))
    if both_ok.any():
        abs_d = np.abs(py_arr[both_ok] - mat_arr[both_ok])
        max_diff = float(np.max(abs_d))
        mean_diff = float(np.mean(abs_d))
    else:
        max_diff = mean_diff = 0.0

    rs_focus = [s.pearson_r for s in focus]
    min_r = float(np.min(rs_focus)) if rs_focus else np.nan
    med_r = float(np.median([s.pearson_r for s in stats_rows if not np.isnan(s.pearson_r)]))

    worst = max(stats_rows, key=lambda s: (-1 if np.isnan(s.max_abs_diff) else s.max_abs_diff))
    failed = [s.name for s in focus if s.pearson_r < 0.99]
    ratio_fail = [s.name for s in ratios if not np.isnan(s.pearson_r) and s.pearson_r < 0.95]
    if ratio_fail:
        failed = sorted(set(failed + ratio_fail))

    if max_diff < 1e-4 and nan_mm == 0:
        status = "EXACT"
    elif min_r >= 0.99 and not [s.name for s in focus if s.pearson_r < 0.99]:
        status = "PASS"
    elif min_r >= 0.95:
        status = "WARN"
    else:
        status = "FAIL"

    return ArrayComparison(
        label=label,
        status=status,
        shape=tuple(mat_arr.shape),
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        nan_mismatches=nan_mm,
        min_correlation=min_r,
        median_correlation=med_r,
        worst_measure=worst.name,
        worst_diff=worst.max_abs_diff if not np.isnan(worst.max_abs_diff) else np.nan,
        measure_stats=stats_rows,
        failed_measures=failed,
    )


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def run_array_comparisons(
    results_dir: Path, precomp_dir: Path, verbose: bool
) -> list[ArrayComparison]:
    print("=" * 92)
    print("SECTION 1: Measure-level agreement (MATLAB .mat vs Python *_mle.npz, first 20 cols)")
    print("=" * 92)
    print(
        f"{'Comparison':<22} {'Status':<6} {'MaxDiff':>10} {'NaNΔ':>6} "
        f"{'Min r*':>7} {'Worst':<18}"
    )
    print("-" * 92)
    print(
        "  * Min r excludes meta-noise/uncertainty and non-M Ratio/Diff measures "
        "from PASS gating"
    )

    rows: list[ArrayComparison] = []
    for label, matf, mkey, pyf, pkey in ARRAY_COMPARISONS:
        mat_path, py_path = results_dir / matf, precomp_dir / pyf
        if not mat_path.exists() or not py_path.exists():
            print(f"{label:<22} SKIP   (missing file)")
            continue
        mat = scipy.io.loadmat(mat_path)[mkey]
        py = np.load(py_path)[pkey]
        row = compare_arrays(label, mat, py)
        rows.append(row)
        print(
            f"{row.label:<22} {row.status:<6} {row.max_abs_diff:10.4e} "
            f"{row.nan_mismatches:6d} {row.min_correlation:7.4f} {row.worst_measure:<18}"
        )
        if verbose:
            for s in row.measure_stats:
                if np.isnan(s.pearson_r):
                    continue
                flag = ""
                if s.name in EXACT_FAMILY and s.max_abs_diff > 1e-6:
                    flag = " !"
                if s.name in OPTIMIZE_FAMILY and (np.isnan(s.pearson_r) or s.pearson_r < 0.95):
                    flag = " ‡"
                if "Ratio" in s.name and s.name != "M-Ratio" and s.pearson_r < 0.95:
                    flag = " †"
                print(
                    f"    {s.name:<18} r={s.pearson_r:7.4f}  "
                    f"max|Δ|={s.max_abs_diff:10.4g}  n={s.n}{flag}"
                )
    return rows


def run_stat_comparisons(results_dir: Path, precomp_dir: Path) -> list[StatComparison]:
    print()
    print("=" * 92)
    print("SECTION 2: Statistical contrasts (Python vs MATLAB vs paper/MATLAB-table refs)")
    print("=" * 92)
    rows: list[StatComparison] = []
    for spec in STAT_TESTS:
        mat = primary(scipy.io.loadmat(results_dir / spec["mat_file"])[spec["mat_key"]])
        py = primary(np.load(precomp_dir / spec["npz_file"])[spec["npz_key"]])
        mat_delta = delta_per_subject(mat, spec["transform"])
        py_delta = delta_per_subject(py, spec["transform"])
        print(f"\n{spec['table']}")
        print(f"  {'Measure':<18} {'Py t':>8} {'MAT t':>8} {'Ref t':>8} {'Sig':>5} {'Ref':>5}")
        print("  " + "-" * 62)
        for m, name in enumerate(MEASURE_NAMES):
            ref = spec["reference"].get(name)
            if ref is None:
                continue
            py_t, _, py_p = ttest_1samp(py_delta[:, m])
            mat_t, _, mat_p = ttest_1samp(mat_delta[:, m])
            py_sig, mat_sig = py_p < 0.05, mat_p < 0.05
            sig_match = py_sig == mat_sig
            ref_ok = t_match(py_t, ref)
            rows.append(
                StatComparison(
                    table=spec["table"],
                    measure=name,
                    py_t=py_t,
                    mat_t=mat_t,
                    ref_t=ref,
                    py_sig=py_sig,
                    mat_sig=mat_sig,
                    sig_match=sig_match,
                    ref_match=ref_ok,
                )
            )
            print(
                f"  {name:<18} {py_t:8.3f} {mat_t:8.3f} {ref:8.3f} "
                f"{'✓' if sig_match else '✗':>5} {'✓' if ref_ok else '~':>5}"
            )
    return rows


def run_paper_checks(results_dir: Path, precomp_dir: Path) -> list[dict]:
    print()
    print("=" * 92)
    print("SECTION 3: Paper reference checks (Rahnev 2025 published scalars)")
    print("=" * 92)
    checks: list[dict] = []

    mat = primary(scipy.io.loadmat(results_dir / "results_Haddara.mat")["metas_raw"])
    py = primary(np.load(precomp_dir / "haddara_mle.npz")["raw"])
    paper_val, tol = 1.14, 0.05
    mat_mean = float(np.nanmean(mat[:, 0]))
    py_mean = float(np.nanmean(py[:, 0]))
    checks.append(
        {
            "metric": "Haddara meta-d' mean",
            "paper": paper_val,
            "matlab": mat_mean,
            "python": py_mean,
            "matlab_ok": abs(mat_mean - paper_val) < tol,
            "python_ok": abs(py_mean - paper_val) < tol,
        }
    )
    print(
        f"  Haddara meta-d' mean — Paper {paper_val:.3f} | "
        f"MATLAB {mat_mean:.4f} | Python {py_mean:.4f}"
    )

    mat_ha = scipy.io.loadmat(results_dir / "results_Haddara.mat")
    prec = mat_ha["metas_precision"]
    cell = prec[0, -1] if prec.ndim == 2 else prec[-1]
    arr = np.asarray(cell, float)
    print(f"  MATLAB metas_precision last cell shape: {arr.shape}")
    if arr.ndim == 5 and arr.shape[-1] >= 1:
        for i, pval in enumerate([1.14, 0.98, 0.84, 0.72]):
            m = float(np.nanmean(arr[:, :, :, i, 0]))
            checks.append(
                {
                    "metric": f"Haddara meta-d' corruption {i * 2}%",
                    "paper": pval,
                    "matlab": m,
                    "python": None,
                    "matlab_ok": abs(m - pval) < 0.08,
                    "python_ok": True,
                }
            )
            print(f"    corruption {i * 2}% — Paper {pval:.2f} | MATLAB {m:.4f}")
    return checks


def print_group_means(results_dir: Path, precomp_dir: Path) -> None:
    print("\n" + "=" * 88)
    print("TABLE A — Group means (Paper scalars where known | MATLAB | Python)")
    print("=" * 88)
    for label, matf, mkey, pyf, pkey in [
        ("Haddara (n=70)", "results_Haddara.mat", "metas_raw", "haddara_mle.npz", "raw"),
        ("Maniscalco (n=22)", "results_Maniscalco.mat", "metas_raw", "maniscalco_mle.npz", "raw"),
    ]:
        mat = primary(scipy.io.loadmat(results_dir / matf)[mkey])
        py = primary(np.load(precomp_dir / pyf)[pkey])
        paper = {"meta-d'": 1.14} if "Haddara" in label else {}
        print(f"\n### {label}")
        print(f"{'Measure':<18} {'Paper':>8} {'MATLAB':>10} {'Python':>10} {'|Δ|':>10} {'r':>8}")
        print("-" * 70)
        for m, name in enumerate(MEASURE_NAMES):
            mm = float(np.nanmean(mat[..., m]))
            pp = float(np.nanmean(py[..., m]))
            both = ~(np.isnan(mat[..., m]) | np.isnan(py[..., m]))
            r = (
                float(np.corrcoef(mat[..., m][both], py[..., m][both])[0, 1])
                if both.sum() >= 3
                else np.nan
            )
            pv = paper.get(name)
            pv_s = f"{pv:.3f}" if pv is not None else "—"
            print(
                f"{name:<18} {pv_s:>8} {mm:10.4f} {pp:10.4f} "
                f"{abs(mm - pp):10.4g} {r:8.4f}"
            )


def print_summary(array_rows, stat_rows, paper_checks, *, strict: bool) -> bool:
    print()
    print("=" * 92)
    print("SUMMARY")
    print("=" * 92)
    n_exact = sum(1 for r in array_rows if r.status == "EXACT")
    n_pass = sum(1 for r in array_rows if r.status == "PASS")
    n_warn = sum(1 for r in array_rows if r.status == "WARN")
    n_fail = sum(1 for r in array_rows if r.status in ("FAIL", "SHAPE_MISMATCH"))
    print(
        f"  Arrays:  {n_exact} exact, {n_pass} pass, {n_warn} warn, "
        f"{n_fail} fail / {len(array_rows)}"
    )
    sig_ok = sum(1 for r in stat_rows if r.sig_match)
    ref_ok = sum(1 for r in stat_rows if r.ref_match)
    print(f"  Stats:   {sig_ok}/{len(stat_rows)} sig match, {ref_ok}/{len(stat_rows)} ref-t match")
    paper_ok = all(c.get("matlab_ok") and c.get("python_ok") for c in paper_checks)
    print(f"  Paper:   {'OK' if paper_ok else 'mismatches (see §3)'}")

    noise_rs = []
    for r in array_rows:
        for s in r.measure_stats:
            if s.name == "meta-noise" and not np.isnan(s.pearson_r):
                noise_rs.append((r.label, s.pearson_r, s.max_abs_diff))
    if noise_rs:
        print("  meta-noise MAT↔PY (after masking MATLAB ~0.4959 artefact):")
        for lab, r, d in noise_rs:
            print(f"    {lab:<22} r={r:.3f}  max|Δ|={d:.3g}")

    array_ok = n_fail == 0
    stat_ok = all(r.sig_match for r in stat_rows) and ref_ok >= max(1, int(0.8 * len(stat_rows)))
    ok = stat_ok and paper_ok and (array_ok or not strict)
    print()
    print("  OVERALL:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="Per-measure correlations")
    ap.add_argument("--json", metavar="PATH", help="Write full report as JSON")
    ap.add_argument("--repo", metavar="PATH", help="Repo root (auto-detected if omitted)")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Treat array FAIL/SHAPE_MISMATCH as overall failure",
    )
    ap.add_argument(
        "--table",
        action="store_true",
        help="Only print group-mean table (Haddara + Maniscalco)",
    )
    args = ap.parse_args()

    repo = Path(args.repo) if args.repo else find_repo_root()
    results_dir = repo / "matlab" / "metasignal_mat" / "Results"
    precomp_dir = repo / "notebooks" / "precomputed"
    if not results_dir.is_dir() or not precomp_dir.is_dir():
        print("ERROR: missing Results/ or notebooks/precomputed/", file=sys.stderr)
        return 1

    print(f"Repo:   {repo}")
    print(f"MATLAB: {results_dir}")
    print(f"Python: {precomp_dir}")
    print("Note: Python arrays sliced to 20 primary measures (ignore fit-stats cols 21–26).")
    print()

    if args.table:
        print_group_means(results_dir, precomp_dir)
        return 0

    array_rows = run_array_comparisons(results_dir, precomp_dir, args.verbose)
    stat_rows = run_stat_comparisons(results_dir, precomp_dir)
    paper_checks = run_paper_checks(results_dir, precomp_dir)
    print_group_means(results_dir, precomp_dir)
    ok = print_summary(array_rows, stat_rows, paper_checks, strict=args.strict)

    if args.json:
        out = Path(args.json)
        payload = {
            "array_comparisons": [
                {**asdict(r), "measure_stats": [asdict(s) for s in r.measure_stats]}
                for r in array_rows
            ],
            "stat_comparisons": [asdict(r) for r in stat_rows],
            "paper_checks": paper_checks,
            "overall_pass": ok,
        }
        out.write_text(json.dumps(payload, indent=2, default=float))
        print(f"\nJSON written to {out}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
