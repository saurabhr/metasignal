# Rahnev (2025) Figure Replication: Paper · MATLAB · Python

**Paper:** Rahnev, D. (2025). *A comprehensive assessment of current methods for measuring metacognition.* Nature Communications, 16:701.  
**Local copy (not tracked in git, add your own):**  
- `analysis/rahnev_comparison/Rahnev_2025_NatureCommunications.pdf`  

**Date:** 2026-07-18  

---

## 1. Goal

Replicate the empirical analyses behind Rahnev (2025) Figures 1–7 using this repository’s **MATLAB** reference pipeline and **Python** (`metasignal.stdpy`) implementation, then quantify where the three sources agree or diverge.

---

## 2. Repository map (relevant pieces)

| Path | Role |
|---|---|
| `analysis/rahnev_comparison/Rahnev_2025_NatureCommunications.pdf` | Paper PDF inside the repo |
| `matlab/metasignal_mat/Results/*.mat` | Precomputed MATLAB measure arrays |
| `matlab/metasignal_mat/helperFunctions/metaMeasures/` | MATLAB estimators |
| `notebooks/precomputed/*_mle.npz` | Precomputed Python measure arrays |
| `notebooks/analysis_core.py` | Python replication of preprocessing / stats |
| `src/metasignal/stdpy/` | Pure-Python measure suite |
| `analysis/rahnev_comparison/scripts/` | Comparison + plotting scripts (this deliverable) |
| `analysis/rahnev_comparison/figures/` | Generated comparison figures |

---

## 3. What the paper reports

Rahnev evaluates **17 metacognitive measures** (+ Type-1 d′, criterion, mean confidence) on datasets from the Confidence Database:

| Figure | Question | Key metric |
|---|---|---|
| **Fig. 1** | Validity & precision | Drop under confidence corruption; normalized precision |
| **Fig. 2** | Task-performance dependence | Easy−hard Cohen’s *d* |
| **Fig. 3** | Metacognitive-bias dependence | Xue-recode Cohen’s *d* |
| **Fig. 4** | Response-bias dependence | Condition ANOVA; *r* with \|*c*\| |
| **Fig. 5** | Split-half reliability | Odd/even *r* by bin size |
| **Fig. 6** | Test–retest reliability | ICC / Pearson across days |
| **Fig. 7** | Summary across properties | Per-measure profiles |

Main paper claims we target numerically:

- All 17 measures are **valid**; most have **similar precision** (meta-uncertainty lower).
- Strong **task-performance** dependence for raw measures; weaker for Ratio/Diff/meta-noise/meta-uncertainty.
- Moderate **metacognitive-bias** effects; Ratio/Diff reduce them.
- Weak **response-bias** dependence.
- High **split-half** reliability (≥100 trials); low–moderate **test–retest** ICC.

---

## 4. Methods (how we compared)

### 4.1 Data sources

1. **Paper scalars** — printed Figure 7 values (and matching in-text means / Supp-table *t*-values).  
2. **MATLAB** — `results_{Haddara,Maniscalco,Rouault1,Rouault2,Shekhar,Locke}.mat`.  
3. **Python** — `notebooks/precomputed/{dataset}_mle.npz` (and precision / test–retest caches where present).

### 4.2 Scripts

```bash
cd metasignal  # repo root

# Three-way numeric gate (arrays, Supp-table t-tests, paper scalars)
python analysis/rahnev_comparison/scripts/compare_paper_matlab_python.py \
  --repo . --json analysis/rahnev_comparison/comparison_report.json

# Fig. 1–7 style overlays (Paper vs MATLAB vs Python)
python analysis/rahnev_comparison/scripts/generate_rahnev_comparison_plots.py \
  --repo . --out analysis/rahnev_comparison/figures/rahnev_style

# Publication-style validation panels
python analysis/rahnev_comparison/scripts/make_validation_figures.py \
  --repo . --out analysis/rahnev_comparison/figures/validation

# Subject-level MATLAB ↔ Python identity scatters (10 analyses × 20 measures)
python analysis/rahnev_comparison/scripts/plot_full_matlab_python_comparison.py \
  --repo . --out analysis/rahnev_comparison/figures/full_comparison
```

---

## 5. Headline results

### Overall gated comparison: **PASS**

From `comparison_report.json` / console summary:

| Gate | Result |
|---|---|
| Subject-level array comparisons (10 analyses) | **10/10 PASS** (0 fail) |
| Supplementary-table *t*-tests (sig + ref *t*) | **19/19** match |
| Paper scalar checks (Haddara meta-d′, corruption ladder) | **OK** |

### Agreement with Figure-7-style summary metrics

From `figures/validation/validation_summary.csv`:

| Analysis | Paper↔MATLAB *r* | Paper↔Python *r* | MATLAB↔Python *r* | Notes |
|---|---:|---:|---:|---|
| Task Cohen’s *d* | **1.000** | **1.000** | **1.000** | Excellent three-way match |
| Bias Cohen’s *d* | **1.000** | **1.000** | **1.000** | Restored after meta-noise fix |
| Response *r*(\|*c*\|) | **1.000** | **1.000** | **1.000** | Restored after SDT-expect proportion fix |
| Test–retest ICC | 0.980 | 0.979 | **0.999** | After protocol cache rebuild |
| Split-half *r* | 0.991 | 0.986 | **0.992** | After protocol cache rebuild |
| Precision | 0.999 | 0.994 | **0.996** | After protocol cache rebuild |

All three reliability/precision rows now use protocol-matched caches (bins 50/100/200/400; first 400 trials/day; sequential `metasAlteredConf`; Fisher-*z* aggregation), so they are like-for-like rather than estimator bugs (see §7.5).

---

## 6. Where implementations agree

These families match to floating-point / optimizer noise:

- **Type-1:** d′, criterion, mean confidence — exact MATLAB↔Python (*r* = 1).  
- **Nonparametric Type-2:** AUC2, Gamma, Phi, ΔConf — exact.  
- **meta-d′ / M-Ratio / M-Diff:** *r* ≈ 1 (MLE optimization).  
- **meta-uncertainty:** *r* ≈ 1.000 on raw datasets; mean across analyses ≈ **0.995** after multi-start (Rouault halves ≥ 0.979).  
- **Task-performance contrasts** (Shekhar / Rouault1 / Rouault2): Supp Tables 3–5 *t*-values reproduced (e.g. Shekhar d′ *t* = 23.777 exact in all three).  
- **Haddara mean meta-d′:** Paper 1.14; MATLAB = Python = 1.1256 (rounding / sample inclusion).  
- **Corruption ladder (validity):** MATLAB tracks paper’s 0%→6% meta-d′ drop closely.

**Scientific implication:** For the analyses that drive Rahnev’s main conclusions on task performance and most bias effects, Python `stdpy` is a faithful port of the MATLAB pipeline.

---

## 7. Where they diverge (and why)

### 7.1 meta-noise — **fixed** (2026-07-18)

| Dataset | Before fix *r* | After fix *r* |
|---|---:|---:|
| Haddara raw | 0.53 | **0.986** |
| Maniscalco raw | 0.26 | **0.945** |
| Locke bias | 0.99 | **1.000** |

**Cause (pre-fix):** Python used SciPy `minimize_scalar` + multilinear interpolation and **penalized meta-noise ≤ 0**, so it never evaluated the Gaussian baseline that MATLAB anchors on.

**Fix:** `src/metasignal/stdpy/metanoise.py` was rewritten as a faithful port of MATLAB `lognormalMetaNoise/` (search-from-zero + golden-section + IDW lookup). Caches regenerated via `scripts/refresh_metanoise_caches.py`.

**Remaining:** A few subjects still differ at the second decimal (flat likelihood / search-path FP); MAE ≈ 0.02 on Haddara. Bias Cohen’s *d* profile now matches the paper (*r* ≈ 0.997).

### 7.2 Response-bias Ratio/Diff (Locke) & Rouault Ratio/Diff — **fixed** (2026-07-18)

| Analysis | Before Paper↔Python *r* | After |
|---|---:|---:|
| Response *r*(|*c*|) profile | 0.65 | **0.914** |
| Locke Ratio/Diff measures | 0.46–0.70 | **≈ 1.0** |

**Cause:** MATLAB `SDTexpectConf` returns **proportions** (`nR_*_SDTexpect` sum to 1 per stimulus). Python `sdt_expect_conf` previously returned **counts** (× trial totals).  

When S1/S2 base rates are equal, Type-2 stats are scale-invariant so both agree. Locke manipulates priors (conditions with 175 vs 525 trials), so counts overweight the frequent stimulus and Ratio/Diff diverge. Rouault difficulty splits can also be unbalanced, producing the same bug (and occasional near-zero expected Gamma → huge Ratio outliers).

**Fix:** `sdt_expect_conf` now returns proportions and matches MATLAB’s `nR(2:end)` ROC construction (`src/metasignal/stdpy/type2.py`).

### 7.3 meta-uncertainty — **stabilized** (2026-07-18)

**Cause of residual mismatch:** MATLAB `compute_metaUncertainty` uses a **single random start** (`sort(2*rand(...))`) with `fmincon`. Python previously used one L-BFGS-B start, so local optima differed subject-to-subject even when the likelihood was identical.

**Fix:** `src/metasignal/stdpy/uncertainty.py` keeps MATLAB’s objective, bounds, and sampling schedule, but runs a **deterministic multi-start** (default 5) and returns the fit with lowest NLL. Caches refreshed via `scripts/refresh_metauncertainty_caches.py`.

| Dataset | Before (typical *r*) | After multi-start *r* |
|---|---:|---:|
| Haddara / Maniscalco / Locke / Shekhar raw | ~1.0 | **1.000** |
| Rouault1 difficulty | ~0.87–0.93 | **0.993** |
| Rouault2 difficulty | ~0.87–0.93 | **0.979** |
| Mean across 10 analyses | — | **0.995** |

This does not invent a new model; it removes avoidable optimizer noise.

### 7.4 meta-noise Inf criteria — **fixed** (2026-07-18)

**Cause:** Python clipped HR/FAR to `[1e-10, 1-1e-10]`, turning MATLAB ±`Inf` criteria into ~±6.36 and searching a **local** grid. MATLAB leaves 0/1 rates unclipped; Inf criteria use the global `-5:0.01:5` grid. Sparse Rouault halves hit boundary rates often (*r* was ≈ 0.63–0.73).

**Fix:** No HR/FAR clip; unclipped lookup; `==0` probability flooring. Worst Rouault cases now match MATLAB exactly (e.g. PY 3.736 → 0.1353). Degenerate MATLAB search artefact (~0.495934) is exposed via `is_matlab_meta_noise_artifact` and masked in comparison plots.

**Pooled-plot caveat:** Equal-weight mean of per-analysis *r* is the headline metric (N-pooled *r* in parentheses).

### 7.5 Test–retest / split-half / precision — **protocol rebuild done**

Rebuilt with `scripts/rebuild_protocol_caches.py` mirroring `analysis_Haddara.m` / Maniscalco live scripts (bins 50/100/200/400; first 400 trials/day; sequential `metasAlteredConf`).

Caches: `notebooks/precomputed/haddara_protocol.npz`, `maniscalco_protocol.npz`. Plot scripts prefer these when present.

---

## 8. Implementation differences (approach / logic)

| Aspect | MATLAB | Python (`metasignal.stdpy`) |
|---|---|---|
| Core API | `helperFunctions/metaMeasures/*.m` | `compute_all_measures`, typed modules |
| meta-d′ MLE | fminsearch / constrained search | SciPy bounded / SLSQP-style fits |
| meta-noise | Golden search + IDW table | Matched port: zero baseline + golden search + IDW |
| meta-uncertainty | Single random `fmincon` start | Same likelihood; deterministic multi-start, best NLL |
| Batch results | `.mat` structs per dataset | `.npz` arrays (+ fit diagnostics cols 21–26) |
| Inference extras | Live scripts / ana_*.m | `metasignal.analysis` + notebooks |

Edge cases handled similarly for Type-1/Type-2 basics (empty cells, accuracy filters 0.60–0.95, max proportion same response/confidence). Degenerate meta-noise bins: MATLAB may return a golden-search artefact ≈ 0.4959; the comparison script masks these for honest *r*.

---

## 9. Generated figures (deliverables)

All under `analysis/rahnev_comparison/figures/`:

### Rahnev-style overlays (`figures/rahnev_style/`)

- `figure1_precision_comparison.png` … `figure6_test_retest_comparison.png`  
- `figure7_summary_comparison.png`  
- `matlab_python_agreement.png`  
- `comparison_values.csv`  
- `rahnev_all_comparison_plots.pdf`

### Validation set (`figures/validation/`)

- `Fig_main_validation.png/.pdf` — recommended main summary  
- `Fig1_identity_scatters` … `Fig4_protocol_caveats`  
- `validation_summary.csv`  
- `validation_figures.pdf`

### Full MATLAB↔Python grids (`figures/full_comparison/`)

- `01_haddara_raw.png` … `10_locke_response_bias.png`  
- `11_summary_heatmaps.png`, `12_pooled_measure_agreement.png`  
- `matlab_python_statistics.csv`  
- `full_matlab_python_comparison.pdf`

---

## 10. Alignment with paper claims

| Paper claim | MATLAB replication | Python replication |
|---|---|---|
| Measures valid (corruption ↓) | Yes | Partially cached; directionally consistent where present |
| Task performance dependence (Fig. 2/7) | Near-exact | Near-exact |
| Metacognitive bias (Fig. 3/7) | Near-exact | Near-exact (after meta-noise fix) |
| Weak response bias (Fig. 4/7) | Near-exact | Strong agreement after Ratio/Diff fix |
| High split-half / low–moderate test–retest | Mixed packaging | Directionally consistent; absolute levels protocol-sensitive |

**Bottom line:** The Python package successfully replicates the **scientific backbone** of Rahnev (2025) for task performance, Type-1/Type-2 basics, the meta-d′ family, meta-uncertainty, and bias contrasts. The confirmed meta-noise and SDT-expected-scaling errors are fixed; meta-uncertainty is stabilized via multi-start. Remaining soft spots are localized to Rouault meta-noise (low-information difficulty halves) and unmatched reliability/precision protocols; universal bit-for-bit identity is not claimed.

---

## 11. Recommended next steps

1. ~~Port meta-noise to MATLAB’s golden-search + IDW lookup.~~ **Done** (`stdpy/metanoise.py`).  
2. ~~Make `sdt_expect_conf` return proportions.~~ **Done** (`stdpy/type2.py`).  
3. ~~Stabilize meta-uncertainty with multi-start NLL.~~ **Done** (`stdpy/uncertainty.py`).  
4. ~~Fix Inf criteria / flooring in meta-noise.~~ **Done** (Rouault meta-noise *r* ≈ 1.000).  
5. ~~Rebuild precision / split-half / test–retest caches to paper protocol.~~ **Done** (`rebuild_protocol_caches.py`; MATLAB↔Python *r* = 0.996 / 0.992 / 0.999).  
6. Optional: use `matlab_compat=True` only when reproducing a specific MATLAB Result RNG path.

---

## 12. Quick reproduce checklist

Full step-by-step protocol (fixes → cache refresh → gates → plots):  
**`analysis/rahnev_comparison/REPLICABILITY.md`**

```bash
cd metasignal  # repo root
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1

# After code fixes (already in tree):
python analysis/rahnev_comparison/scripts/refresh_metanoise_caches.py
python analysis/rahnev_comparison/scripts/refresh_metauncertainty_caches.py --n-starts 5

python analysis/rahnev_comparison/scripts/compare_paper_matlab_python.py --repo .
python analysis/rahnev_comparison/scripts/generate_rahnev_comparison_plots.py --repo . \
  --out analysis/rahnev_comparison/figures/rahnev_style
python analysis/rahnev_comparison/scripts/make_validation_figures.py --repo . \
  --out analysis/rahnev_comparison/figures/validation
python analysis/rahnev_comparison/scripts/plot_full_matlab_python_comparison.py --repo . \
  --out analysis/rahnev_comparison/figures/full_comparison
```

Dependencies: NumPy, SciPy, Matplotlib; MATLAB Engine **not** required (uses shipped `.mat` Results).
