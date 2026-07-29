# Replicability protocol — Rahnev comparison figures

**Date:** 2026-07-18  

This document records every step needed to regenerate the MATLAB–Python–paper
comparison from a clean checkout after the scientifically motivated estimator
fixes.

---

## 0. Prerequisites

```bash
cd metasignal  # repo root
# Python 3.10+ with metasignal editable / src on path
pip install -e ".[dev]"   # or: PYTHONPATH=src:notebooks
```

Required packages: NumPy, SciPy, Matplotlib, pandas.  
MATLAB Engine is **not** required — comparisons use shipped
`matlab/metasignal_mat/Results/*.mat`.

Optional environment (avoids BLAS thrashing during many serial fits):

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
```

---

## 1. Code fixes applied (scientific rationale)

| Step | File | What changed | Why |
|---|---|---|---|
| A | `src/metasignal/stdpy/metanoise.py` | Golden-section + IDW; evaluate Gaussian baseline at meta-noise = 0 | Match MATLAB `lognormalMetaNoise/`; previous SciPy search never evaluated the zero-noise anchor |
| B | `src/metasignal/stdpy/type2.py` | `sdt_expect_conf` returns **proportions** (sum to 1 per stimulus) | Match MATLAB `SDTexpectConf.m`; counts broke Ratio/Diff under unequal base rates (Locke, Rouault) |
| C | `src/metasignal/stdpy/uncertainty.py` | Same likelihood/bounds as MATLAB; **deterministic multi-start** (default 5), pick lowest NLL | MATLAB uses one random `fmincon` start; multi-start removes avoidable optimizer noise without changing the model |
| D | `src/metasignal/stdpy/metanoise.py` | **No HR/FAR clip**; Inf criteria → global grid; unclipped lookup; `==0` flooring | Match `compute_SDTcriteria.m` / `compute_metaNoise.m`; fixes Rouault meta-noise soft *r* |
| E | `uncertainty.py` / `metad.py` / `compute_all.py` | Optional `matlab_compat=True` | Single-start / high-budget mode to mimic MATLAB Result files |
| F | `scripts/rebuild_protocol_caches.py` | Precision / split-half / test–retest binning | Match `analysis_Haddara.m` / Maniscalco live script protocols |

Unit checks:

```bash
python -m pytest tests/test_type2.py tests/test_uncertainty.py -q
```

---

## 2. Refresh Python measure caches

Caches live in `notebooks/precomputed/*_mle.npz`.

### 2a. Meta-noise column only (index 15) — after Inf-criteria fix

```bash
python analysis/rahnev_comparison/scripts/refresh_metanoise_column.py --only all
# or --only rouault for a fast Rouault-only check
# log: analysis/rahnev_comparison/figures/refresh_metanoise_column_log.txt
```

Full recompute (all measures) if needed:

```bash
python analysis/rahnev_comparison/scripts/refresh_metanoise_caches.py
```

### 2b. Meta-uncertainty only (column index 16)

```bash
python analysis/rahnev_comparison/scripts/refresh_metauncertainty_caches.py --n-starts 5
# log: analysis/rahnev_comparison/figures/refresh_metauncertainty_log.txt
```

### 2c. Paper-protocol precision / split-half / test–retest

```bash
# Full Haddara + Maniscalco (slow: tens of thousands of compute_all calls)
python analysis/rahnev_comparison/scripts/rebuild_protocol_caches.py

# Faster subsets:
python analysis/rahnev_comparison/scripts/rebuild_protocol_caches.py --dataset haddara --only test_retest
python analysis/rahnev_comparison/scripts/rebuild_protocol_caches.py --dataset haddara --only precision
python analysis/rahnev_comparison/scripts/rebuild_protocol_caches.py --dataset maniscalco --only split
```

Writes `notebooks/precomputed/{haddara,maniscalco}_protocol.npz`. Plot scripts
prefer these over legacy odd/even / whole-day caches when present.

---

## 3. Numeric gate (paper · MATLAB · Python)

```bash
python analysis/rahnev_comparison/scripts/compare_paper_matlab_python.py \
  --repo . \
  --json analysis/rahnev_comparison/comparison_report.json \
  | tee analysis/rahnev_comparison/comparison_console.txt
```

Expect: **OVERALL PASS** (10/10 arrays, 19/19 Supp *t*, paper scalars OK).

---

## 4. Regenerate all comparison figures + PDFs

```bash
# Fig. 1–7 style overlays + combined PDF
python analysis/rahnev_comparison/scripts/generate_rahnev_comparison_plots.py \
  --repo . --out analysis/rahnev_comparison/figures/rahnev_style \
  2>&1 | tee analysis/rahnev_comparison/figures/rahnev_style_log.txt

# Validation panels + PDF
python analysis/rahnev_comparison/scripts/make_validation_figures.py \
  --repo . --out analysis/rahnev_comparison/figures/validation \
  2>&1 | tee analysis/rahnev_comparison/figures/validation_log.txt

# Subject-level identity grids + pooled agreement + PDF
python analysis/rahnev_comparison/scripts/plot_full_matlab_python_comparison.py \
  --repo . --out analysis/rahnev_comparison/figures/full_comparison \
  2>&1 | tee analysis/rahnev_comparison/figures/full_comparison_log.txt
```

### Outputs

| Folder | Key artifacts |
|---|---|
| `figures/rahnev_style/` | `figure1_…png` … `figure7_…png`, `rahnev_all_comparison_plots.pdf` |
| `figures/validation/` | `Fig_main_validation.png/.pdf`, `validation_figures.pdf`, `validation_summary.csv` |
| `figures/full_comparison/` | `01_….png` … `12_pooled_measure_agreement.png`, `full_matlab_python_comparison.pdf`, `matlab_python_statistics.csv` |

---

## 5. How the pooled agreement panel is defined

In `12_pooled_measure_agreement.png`:

1. For each analysis × measure, z-score on the **MATLAB** mean/SD.
2. Compute Pearson *r* **per analysis**.
3. Title reports **mean of those *r* values** (equal analysis weight).
4. N-weighted pooled *r* is shown in parentheses (can be dominated by Rouault
   difficulty halves, which contribute many points).

Soft measures (meta-noise, meta-uncertainty) are highlighted in red hexbins.

---

## 6. What is *not* claimed

- Bit-for-bit identity with a particular MATLAB `fmincon` random start on
  low-information Rouault difficulty halves.
- Exact Fig. 5 / Fig. 6 absolute levels when binning protocols differ
  (400-trial test–retest bins; MATLAB split-half packaging).
- Complete Python precision cache for all MLE measures.

Core claim: after fixes A–C, Python `stdpy` reproduces the scientific backbone
of Rahnev (2025) for Type-1/Type-2, meta-d′ family, task/bias contrasts, and
response-bias profiles.
