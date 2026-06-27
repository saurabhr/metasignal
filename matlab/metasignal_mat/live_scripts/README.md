# MATLAB Live Scripts — Rahnev (2025) Replication

These scripts replicate the full analysis pipeline from:

> Rahnev (2025). A comprehensive assessment of current methods for measuring metacognition. *Nature Communications*, 16(1), 701.

## What are these files?

Each `.m` file is structured as a **MATLAB cell-mode script** compatible with MATLAB's Live Editor:

- `%%` section headers create named sections visible in the Live Editor
- Run individual sections with **Ctrl+Enter** (or **Cmd+Enter** on Mac)
- Convert to a true Live Script (`.mlx`) via **File → Save As → MATLAB Live Script**

## File Overview

| File | Purpose | Figures |
|------|---------|---------|
| `00_run_all.m` | Master runner — executes all scripts in order | — |
| `01_analysis_Haddara.m` | Haddara (2022) dataset: reliability + precision | — |
| `02_analysis_Maniscalco.m` | Maniscalco (2017) dataset: split-half + precision | — |
| `03_analysis_Rouault1.m` | Rouault (2018) Expt 1: contrast-defined difficulty | — |
| `04_analysis_Rouault2.m` | Rouault (2018) Expt 2: median-split difficulty | — |
| `05_analysis_Shekhar.m` | Shekhar (2021) dataset: multi-contrast analysis | — |
| `06_ana_taskPerformance.m` | Difficulty dependence across 3 datasets | Figure 2 |
| `07_ana_respBias.m` | Response bias (Locke 2020): 7 conditions | — |
| `08_ana_metaBias.m` | Metacognitive bias dependence (Xue recoding) | Figure 3 |
| `09_ana_precision.m` | Precision: sensitivity to confidence alteration | Figure 1 |
| `10_ana_splitHalf.m` | Split-half reliability (odd vs. even trials) | — |
| `11_ana_testRetest.m` | Test-retest reliability (ICC + Pearson, 15 day pairs) | — |
| `12_ana_acrossMeasCorr.m` | Inter-measure correlation matrices | Figure 11 |

## How to Run

### Option A: Run everything
```matlab
cd /path/to/metasignal_mat/live_scripts
run('00_run_all.m')
```

### Option B: Run individual scripts
```matlab
cd /path/to/metasignal_mat/live_scripts
run('06_ana_taskPerformance.m')
```

### Option C: Use as Live Script in MATLAB Live Editor
1. Open any `.m` file in MATLAB
2. MATLAB will detect cell sections (`%%`) and offer to open in Live Editor
3. Run sections interactively with **Ctrl+Enter**
4. Save as `.mlx` for a full Live Script experience

## Recomputing vs. Loading Results

Each dataset script has a `recompute_measures` flag at the top:

```matlab
recompute_measures = 0;  % Load pre-saved results (fast)
recompute_measures = 1;  % Recompute from raw data (slow, ~hours)
```

Set to `1` only when you want to regenerate results from the raw `.mat` files in `Preprocess/`.

## Measure Order (all scripts)

| # | Measure | # | Measure |
|---|---------|---|---------|
| 1 | meta-d' | 11 | M-Diff |
| 2 | AUC2 | 12 | AUC2-Diff |
| 3 | Gamma | 13 | Gamma-Diff |
| 4 | Phi | 14 | Phi-Diff |
| 5 | ΔConf | 15 | ΔConf-Diff |
| 6 | M-Ratio | 16 | meta-noise |
| 7 | AUC2-Ratio | 17 | meta-uncertainty |
| 8 | Gamma-Ratio | 18 | d' |
| 9 | Phi-Ratio | 19 | Criterion |
| 10 | ΔConf-Ratio | 20 | Confidence |

## Dependencies

All scripts automatically add `helperFunctions/` to the MATLAB path using `addpath`. The helper functions include `compute_all_measures`, `type2_SDT_MLE`, `ICC`, `perform_ttest`, `r2z`, `z2r`, `xue_recode`, `metasAlteredConf`, and `good_colors_for_plotting`.

The scripts resolve paths relative to the `live_scripts/` directory, so they work regardless of MATLAB's current working directory.
