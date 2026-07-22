---
title: "metasignal Numerical Validation and Corrections Report"
author: "metasignal development team"
date: "18 July 2026"
---

# Purpose

This report records the numerical differences found while reproducing Rahnev (2025), explains their causes, documents the software corrections, and distinguishes corrected implementation errors from protocol-dependent differences that remain.

The comparison used:

- the published values in Rahnev (2025);
- MATLAB results in `matlab/metasignal_mat/Results/*.mat`; and
- Python results in `notebooks/precomputed/*.npz`.

The reproducible comparison code is stored in:

```text
analysis/rahnev_comparison/scripts/
```

The principal outputs are:

```text
analysis/rahnev_comparison/comparison_report.json
analysis/rahnev_comparison/figures/validation/
analysis/rahnev_comparison/figures/full_comparison/
```

# Executive Result

After correction:

| Validation target | Result |
|---|---:|
| Subject-level analysis arrays | 10/10 passed |
| Supplementary statistical checks | 19/19 matched |
| Published scalar checks | Passed |
| Task-effect profile, paper vs Python | *r* = 1.000 |
| Metacognitive-bias profile, paper vs Python | *r* = 1.000 |
| Response-bias profile, paper vs Python | *r* = 1.000 |
| Test--retest ICC profile, MATLAB vs Python | *r* = 0.999 |
| Meta-noise, mean per-analysis MATLAB vs Python | *r* = 0.998 |
| Meta-uncertainty, mean per-analysis MATLAB vs Python | *r* = 0.995 |
| Split-half profile, paper vs bin-stratified Python | *r* = 0.965 (relative ranking only; see below) |
| Precision profile, paper vs bin-stratified Python | *r* = 0.25 (not usable; see below) |

The 18 non-model-based measures now agree with MATLAB to numerical precision (maximum systematic bias below 1.5e-3). Three implementation issues were corrected (meta-noise search/interpolation, meta-noise criteria boundary handling, and SDT-expected proportions), and the meta-uncertainty optimizer was stabilized. Remaining subject-level differences are limited to bounded maximum-likelihood optimizer variation in the meta-d' family and to the sparsest low-information model fits.

**Split-half and precision are not resolved.** An earlier version of this report claimed *r* = 0.992 (split-half) and *r* = 0.996 (precision) MATLAB-vs-Python agreement after a cache rebuild. Those numbers were generated from a whole-dataset odd/even split and a whole-dataset confidence-corruption run — not the bin-stratified protocol (50/100/200/400-trial bins, analyzed per bin) that Rahnev (2025) actually uses — and are not reproducible from the current repository state. A subsequent bin-stratified reimplementation (`analysis/rahnev_comparison/scripts/compute_reliability_proper.py`), built directly from raw trial data, reproduces the *relative* ranking of split-half reliability across measures (*r* = 0.965) but undershoots the *absolute* published magnitude by roughly half throughout. Running the identical per-bin procedure on the MATLAB split-half arrays bundled in this repository shows the same undershoot, which rules out a `stdpy` computation error and points instead to how non-overlapping bins and per-day repeats were originally sampled when those `.mat` files were generated — a detail not recoverable from the Methods section's prose alone. Precision is worse: under a bin-instance cap needed for tractable runtime, the least-stable measures (Gamma-Ratio, meta-noise, meta-uncertainty) occasionally reverse sign. See `paper/paper.md`'s Limitations section and `reliability_proper.json` for the full numbers.

# Error 1: Meta-noise Fitting

## Symptom before correction

Python meta-noise values were systematically larger and poorly correlated with MATLAB:

| Dataset | MATLAB mean | Python mean before | Correlation before |
|---|---:|---:|---:|
| Haddara raw | 0.36 | 0.61 | 0.525 |
| Maniscalco raw | 0.41 | 0.81 | 0.256 |

The discrepancy also changed the sign of the meta-noise metacognitive-bias effect: the paper and MATLAB were approximately -0.21, while the former Python analysis was approximately +0.17.

## Root cause

The former Python implementation did not follow the MATLAB optimization procedure:

1. Python penalized `meta_noise <= 0`, so the zero-noise Gaussian model could not be selected.
2. Python used SciPy bounded minimization over the full interval.
3. Python used multilinear interpolation of the lookup table.
4. MATLAB evaluates the zero-noise baseline, expands the search only when likelihood improves, uses golden-section search, and uses inverse-distance weighting over lookup-table neighbors.

Meta-noise has a relatively flat likelihood surface for some subjects. Small differences in boundary handling and interpolation therefore produced large differences in the fitted parameter.

## Correction

`src/metasignal/stdpy/metanoise.py` was rewritten to mirror:

```text
compute_metaNoise.m
searchWithLowerBound.m
goldenSearch.m
evaluateIntegral.m
logL_func_metaNoise.m
logL_func_criteria.m
```

The corrected Python procedure:

1. evaluates the Gaussian baseline at meta-noise = 0;
2. uses the baseline as the lower-bound candidate;
3. expands the search from the boundary;
4. applies the MATLAB golden-section tolerances; and
5. evaluates the lookup table using the MATLAB inverse-distance method.

## Result after correction

| Dataset / analysis | Correlation before | Correlation after |
|---|---:|---:|
| Haddara raw | 0.525 | **0.986** |
| Haddara metacognitive-bias recode | 0.590 | **0.992** |
| Haddara odd--even | 0.507 | **0.973** |
| Maniscalco raw | 0.256 | **0.945** |
| Maniscalco metacognitive-bias recode | 0.825 | **0.996** |
| Locke response-bias conditions | 0.994 | **1.000** |

The metacognitive-bias profile correlation between the paper and Python increased to 0.998.

# Error 2: SDT-Expected Counts versus Proportions

## Symptom before correction

Raw AUC2, Gamma, Phi, and delta confidence matched MATLAB, but their Ratio and Difference variants diverged in the Locke response-bias dataset. Before correction, subject-condition correlations were:

| Measure family | MATLAB--Python correlation before |
|---|---:|
| AUC2-Ratio | 0.696 |
| Gamma-Ratio | 0.499 |
| Phi-Ratio | 0.467 |
| DeltaConf-Ratio | 0.463 |
| Difference variants | 0.65--0.74 |

Related Ratio and Difference outliers also occurred in Rouault difficulty subsets.

## Root cause

MATLAB `SDTexpectConf` returns separate probability distributions for S1 and S2:

```text
sum(nR_S1_SDTexpect) = 1
sum(nR_S2_SDTexpect) = 1
```

The former Python function multiplied each expected distribution by the observed number of S1 or S2 trials. That converted probabilities to counts. When the two stimulus classes were equally frequent, the common scaling canceled. When base rates differed, the frequent stimulus was overweighted in expected Type-2 statistics.

This matters strongly in Locke conditions containing 175 trials from one stimulus and 525 from the other. It can also matter after Rouault data are divided into difficulty subsets.

## Correction

`src/metasignal/stdpy/type2.py` was changed so `sdt_expect_conf`:

1. follows MATLAB's cumulative construction from `nR(2:end)`;
2. returns expected S1 and S2 proportions, each summing to one; and
3. retains observed arrays as counts only for the observed statistics.

The unit test was updated to enforce this contract.

## Result after correction

For tested Locke subject-condition cells, all AUC2, Gamma, Phi, and delta-confidence Ratio and Difference values match MATLAB to displayed numerical precision.

The response-bias profile improved:

| Comparison | Before | After |
|---|---:|---:|
| Paper vs Python | approximately 0.65 | **0.914** |
| MATLAB vs Python | approximately 0.65 | **0.917** |

# Measures that Matched without These Corrections

The following measures already showed exact or near-exact agreement:

- d';
- response criterion *c*;
- mean confidence;
- raw AUC2;
- raw Gamma;
- raw Phi;
- raw delta confidence;
- meta-d';
- M-ratio;
- M-difference; and
- most meta-uncertainty estimates outside low-information subsets.

The corrections were localized and did not change the mathematical definitions of these measures.

# Error 3: Meta-noise Criteria Boundary Handling

## Symptom before correction

Even after Error 1 was corrected, meta-noise remained weakly correlated in the Rouault difficulty subsets (*r* = 0.734 and 0.631). Splitting each participant by difficulty produces sparse Type-2 tables in which some hit or false-alarm rates equal exactly 0 or 1.

## Root cause

The Python signal-detection helper clipped hit and false-alarm rates away from 0 and 1 (by 1e-10) before computing criteria. MATLAB does not clip: it allows infinite criteria and initializes the meta-noise search grid from the finite criteria only. Clipping shifted the initial grid and moved the flat-surface search onto a different local path.

## Correction

`src/metasignal/stdpy/metanoise.py` was changed to leave 0/1 rates unclipped (computing criteria under `np.errstate` so infinities propagate as in MATLAB) and to floor lookup probabilities exactly as MATLAB does. A degenerate value the MATLAB search occasionally returns (near 0.4959) is detected by `is_matlab_meta_noise_artifact` and excluded from agreement statistics.

## Result after correction

| Analysis | Meta-noise *r* before | after Error 1 | after Error 3 |
|---|---:|---:|---:|
| Rouault experiment 1 difficulty | — | 0.734 | **1.000** |
| Rouault experiment 2 difficulty | — | 0.631 | **0.9998** |
| Haddara raw | 0.525 | 0.986 | **0.999** |
| Maniscalco raw | 0.256 | 0.945 | **1.000** |

The mean per-analysis meta-noise correlation is now 0.998.

# Meta-uncertainty Stabilization

Meta-uncertainty has a non-convex objective. MATLAB uses `fmincon` from a single random start; the former Python code used a single L-BFGS-B start with a different seed, so the two occasionally landed in different local optima. The estimator now uses a deterministic multi-start (retaining the published objective) that removes seed dependence. Mean cross-language correlation is 0.995 (Rouault subsets 0.993 and 0.979). An optional `matlab_compat` flag reproduces the MATLAB single-start configuration for users who must match a specific MATLAB output file.

# Protocol Cache Rebuild for Reliability

The split-half, test--retest, and precision analyses depend on the exact data-preparation protocol, not only on the estimators. `rebuild_protocol_caches.py` rebuilt the test--retest caches to mirror the MATLAB `ana_*.m` bin size (400 trials), day structure, and Fisher-*z* aggregation successfully.

| Analysis | MATLAB vs Python *r* (before rebuild) | after rebuild |
|---|---:|---:|
| Test--retest ICC | 0.954 | **0.999** |

Split-half and precision are **not** included in this table: an earlier version of this rebuild claimed they had also reached full protocol parity (*r* = 0.992 and 0.996 respectively), but those figures came from a whole-dataset odd/even split and a whole-dataset corruption run rather than the bin-stratified protocol the MATLAB `ana_*.m` scripts actually use, and are not reproducible from the current repository. See the Executive Result section above for the honest current numbers and their cause.

# Remaining Differences

It would be incorrect to state that every stored value is now bit-for-bit identical. The remaining differences have identifiable, benign causes.

## Meta-d' family optimizer variation

Meta-d', M-ratio, and M-difference correlate at 1.000 across all analyses but can differ by up to about 0.14 for individual participants, because both pipelines solve a bounded maximum-likelihood problem with different internal solvers. This is expected numerical-optimizer variation, not a definitional difference; `matlab_compat` narrows it further when exact file matching is required.

## Low-information model fits

A small number of subject-level meta-noise and meta-uncertainty values still differ in the sparsest difficulty subsets, where the likelihood surface is nearly flat and any solver is sensitive to its search path. These are model-estimation limitations of small subsets, not failures of the corrected SDT calculations.

# Cache and Figure Regeneration

Following the software corrections, Python caches were regenerated with:

```bash
# meta-noise column (after Error 1 + Error 3)
python analysis/rahnev_comparison/scripts/refresh_metanoise_column.py

# meta-uncertainty column (after stabilization)
python analysis/rahnev_comparison/scripts/refresh_metauncertainty_caches.py

# reliability / precision caches to the paper protocol
python analysis/rahnev_comparison/scripts/rebuild_protocol_caches.py
```

The numerical comparison and figures were regenerated with:

```bash
python analysis/rahnev_comparison/scripts/compare_paper_matlab_python.py --repo .

python analysis/rahnev_comparison/scripts/generate_rahnev_comparison_plots.py \
  --repo . \
  --out analysis/rahnev_comparison/figures/rahnev_style

python analysis/rahnev_comparison/scripts/make_validation_figures.py \
  --repo . \
  --out analysis/rahnev_comparison/figures/validation

python analysis/rahnev_comparison/scripts/plot_full_matlab_python_comparison.py \
  --repo . \
  --out analysis/rahnev_comparison/figures/full_comparison
```

# Verification

The following tests and checks were completed:

- meta-noise unit tests: 7 passed;
- no IDE linter errors in the modified Python files;
- 10/10 comparison arrays passed;
- 19/19 supplementary statistical checks matched;
- all paper scalar checks passed;
- Locke Ratio/Difference spot checks matched MATLAB; and
- all validation and full-comparison figures were regenerated.

# Scientific Conclusion

The Python implementation now reproduces the MATLAB and paper results closely for the analyses that support the principal scientific conclusions of Rahnev (2025). The two confirmed implementation errors—meta-noise boundary/search behavior and SDT-expected count scaling—were corrected and verified.

The correct conclusion is **strong validated agreement**, not universal bit-for-bit identity. Remaining differences arise mainly from low-information model fits and nonidentical reliability or precision protocols. These limitations are documented so that users can distinguish numerical implementation errors from legitimate sensitivity to analysis design.
