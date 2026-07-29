# itmc vs. statConfR (R) cross-validation

Validates `metasignal.itmc` (`backend='statconfr'`) against the actual R
`statConfR::estimateMetaI()` function it is ported from (Rausch et al., 2025,
JOSS), on 12 shared synthetic participants (5,350 trials total) spanning a
range of sensitivity (d' = 0.2-2.5), metacognitive bias (under-/over-confident),
confidence-scale granularity (2/4/6 points), and sample sizes (150-1200 trials).

Both sides consume the identical `data/shared_trials.csv` — any numeric
difference below reflects a real implementation discrepancy, not a data
difference.

A second, independent check reproduces Dayan (2023)'s own hand-worked
numerical example directly (`check_dayan_table1.py`) — see below.

## Verdict

**Deterministic core math (`bias_reduction=FALSE`): exact match.** All 5
measures agree with R to machine precision (Pearson r = 1.0000, MAE =
0.0000) across every participant. This confirms `backend='statconfr'` is a
faithful port of statConfR's algorithm, not just a similar approximation.

**Bias-corrected values (`bias_reduction=TRUE`): strong agreement**, with
one known instability at very low d' explained below (not a bug).

## Without bias correction (deterministic, `bias_reduction=FALSE`)

| Measure | Pearson r | MAE | Max abs diff | n |
|---|---:|---:|---:|---:|
| meta_I | 1.0000 | 0.0000 | 0.0000 | 12 |
| meta_Ir1 | 1.0000 | 0.0000 | 0.0000 | 12 |
| meta_Ir1_acc | 1.0000 | 0.0000 | 0.0000 | 12 |
| meta_Ir2 | 1.0000 | 0.0000 | 0.0000 | 12 |
| RMI | 1.0000 | 0.0000 | 0.0000 | 12 |

## With bias correction (Monte Carlo, `bias_reduction=TRUE`, seed=42)

| Measure | Pearson r | MAE | Max abs diff | n |
|---|---:|---:|---:|---:|
| meta_I | 0.9957 | 0.0071 | 0.0154 | 12 |
| meta_Ir1 | 0.8882 | 0.2629 | 1.9386 | 12 |
| meta_Ir1_acc | 0.9641 | 0.1158 | 0.2394 | 12 |
| meta_Ir2 | 0.9975 | 0.0103 | 0.0238 | 12 |
| RMI | 0.9933 | 0.0253 | 0.0512 | 12 |

Bias-corrected values use independent Monte Carlo resampling in R and Python
(different RNG streams entirely), so exact agreement is not expected even
with matched implementations — residual differences reflect sampling noise.

`meta_Ir1` and `meta_Ir1_acc` show the largest spread above, driven almost
entirely by participant `sim10_d0.2_meta0.2_r4_n400` (d' = 0.2, near chance): these two measures
normalize by the Gaussian-expected meta-I at the observed d', which is itself
tiny at low sensitivity, so any Monte Carlo noise in the numerator gets
amplified by dividing by a near-zero denominator. The deterministic
(non-bias-corrected) value for this same participant matched R **exactly**
(2.27714
both sides) — confirming this is a known instability of the ratio
measure itself at low d' (the same failure mode Rahnev 2025's own Figure 7
flags for M-Ratio/AUC2-Ratio/etc. as "Unstable for low d'"), not an R vs.
Python implementation discrepancy. Excluding that one participant:

| Measure | Pearson r | MAE | Max abs diff | n |
|---|---:|---:|---:|---:|
| meta_I | 0.9940 | 0.0073 | 0.0154 | 11 |
| meta_Ir1 | 0.9626 | 0.1105 | 0.3601 | 11 |
| meta_Ir1_acc | 0.9696 | 0.1045 | 0.1655 | 11 |
| meta_Ir2 | 0.9969 | 0.0107 | 0.0238 | 11 |
| RMI | 0.9911 | 0.0252 | 0.0512 | 11 |

## Direct check against Dayan (2023)'s own published example

Dayan (2023) works through a full hand calculation of meta-I in the paper
text (Table 1, p.397): subject 5, condition 3 of the Shekhar & Rahnev (2021)
dataset, confidence binarized at a single threshold. He reports the exact
trial counts (r=0,c=low: 138; r=0,c=high: 5; r=1,c=low: 440; r=1,c=high: 415;
n=998) and works the entropy calculation by hand to **meta-I = 0.094**
(matching the same value in his Figure 1 table, column `mI[2]`).

`check_dayan_table1.py` reconstructs those exact trial counts and computes
`metasignal.itmc.meta_I(backend='simple')` on them:

```
metasignal.itmc.meta_I (backend='simple'): 0.094320
Dayan (2023) published value:               0.094
Match (rounded to 3 d.p.): True
```

This is a direct reproduction of Dayan's own number — a stronger check than
the R cross-validation above, since it validates against the theory paper
itself rather than a third-party implementation of it.

Note: `meta_Ir1`, `meta_Ir1_acc`, and `RMI` were **not** checked against Dayan's
Figure 1 table (which also reports `mI1'[2]=0.775` for this same row). Those
normalized measures require simulating the expected meta-I of an idealized
first-order Bayesian rater at the reported d′ (Dayan's Eq. involving
α ~ N(d, 4/d′²)), which in turn requires a dataset with real stimulus-category
variability — the reconstruction above only has `stim=0` throughout (accuracy
and rating are all that matter for `meta_I` itself), which is not a valid
input for d′-dependent normalization. Extending this check would need
reconstructing (or obtaining) the actual Shekhar & Rahnev (2021) per-trial
data for subject 5, which is a distinct, larger effort from the plain
`meta_I` hand-check above.

Also worth noting, confirmed by reading the full Dayan (2023) paper text:
`meta_I`, `meta_Ir1`, and `meta_Ir2` are the exact measures Dayan proposes.
`meta_Ir1_acc` and `RMI` are **not** discussed anywhere in Dayan (2023) —
both are additions from Rausch et al. (2025)'s `statConfR` package (confirmed
independently via the R package's own help documentation and the statConfR
JOSS paper text, which describes RMI as "a novel measure... also derived
from information theory," not attributed to Dayan).

## How to reproduce

```bash
python analysis/itmc_comparison/scripts/generate_shared_data.py
python analysis/itmc_comparison/scripts/run_python_itmc.py
Rscript analysis/itmc_comparison/scripts/run_r_statconfr.R
python analysis/itmc_comparison/scripts/compare.py
python analysis/itmc_comparison/scripts/check_dayan_table1.py
```
