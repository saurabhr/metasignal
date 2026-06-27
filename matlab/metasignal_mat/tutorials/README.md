# MATLAB Tutorial Live Scripts

Cell-mode tutorial scripts (`%%` section delimiters) for the `metasignal_mat` toolbox.
Each file runs as a standalone script; `00_run_all.m` executes all seven in sequence.

## Prerequisites

- MATLAB R2016b or later (local function support in scripts)
- No additional toolboxes required beyond Statistics and Machine Learning Toolbox (used by `ttest`)
- The `helperFunctions/` tree is added to the MATLAB path automatically by each script

## Running

### Live Editor (recommended)

Open any `.m` file in the MATLAB Live Editor (`Home → New → Open`). Press **Run** or step through cells with **Run Section**.

### Command window

```matlab
cd matlab/metasignal_mat/tutorials
run('00_run_all.m')     % all tutorials
run('01_getting_started.m')   % individual
```

### From any working directory

```matlab
run('/path/to/metasignal_mat/tutorials/01_getting_started.m')
```

Each script resolves its own paths with `fileparts(mfilename('fullpath'))` so the working directory does not matter.

## Tutorial overview

| File | Topic |
|------|-------|
| `01_getting_started.m` | Install check, input format, compute_SDT_resp, full 20-measure vector |
| `02_computing_measures.m` | All 20 measures, individual APIs (SDTtype2AUC, SDTgamma, SDTphi, SDTdeltaConf, meta-noise, meta-uncertainty) |
| `03_statistical_inference.m` | Bootstrap CI, permutation test, perform_ttest |
| `04_difficulty_dependence.m` | Easy/hard split, ±3 SD outlier removal, difficulty t-tests |
| `05_metacognitive_bias.m` | xue_recode (Xue et al. 2021), bias sensitivity t-tests |
| `06_split_half_reliability.m` | Split-half correlation, Spearman-Brown correction, confidence corruption |
| `07_mle_fitting.m` | type2_SDT_MLE single-participant + group fits, rm dataset analysis |

## Key helper functions

| Function | Returns |
|----------|---------|
| `compute_all_measures(stim, resp, conf, nRatings)` | `[1×20]` measure vector |
| `compute_SDT_resp(stim, resp)` | `[dprime, c, ln_beta]` |
| `type2_SDT_MLE(stim, resp, conf, nRatings, [], 1)` | struct with `da`, `meta_da`, `M_ratio`, `logL`, … |
| `SDTtype2AUC / SDTgamma / SDTphi / SDTdeltaConf` | `[observed, ratio, diff]` |
| `SDTexpectConf(stim, resp, conf, nRatings)` | struct with actual and SDT-expected count arrays |
| `trials2counts(stim, resp, conf, nRatings)` | `[nR_S1, nR_S2]` count vectors |
| `perform_ttest(data, description, display)` | `[p, t, df, Cohen_d, CI]` |
| `xue_recode(conf, lowHighRecoding)` | recoded confidence (1 = lower bias, 2 = higher bias) |
| `compute_metaNoise / compute_metaUncertainty` | scalar noise/uncertainty estimates |

## Measure index reference (compute_all_measures output)

```
 1  meta-d'          6  M-Ratio          11  M-Diff           16  meta-noise
 2  AUC2             7  AUC2-Ratio       12  AUC2-Diff        17  meta-uncertainty
 3  Gamma            8  Gamma-Ratio      13  Gamma-Diff       18  d'
 4  Phi              9  Phi-Ratio        14  Phi-Diff         19  Criterion
 5  DeltaConf       10  DeltaConf-Ratio  15  DeltaConf-Diff   20  Confidence
```
