# Tutorial 2 — Computing All 26 Measures

This tutorial covers the full 26-measure battery in detail: what each measure captures, how to call it individually, and how to interpret the output.

## The 26-measure array

`stdpy.compute_all_measures` returns a single NumPy array with 26 elements organised into five conceptual blocks:

```
Index   Measure              Block
──────────────────────────────────────────────────────────
 0      meta-d'              Metacognitive sensitivity (MLE)
 1      AUC2                 Metacognitive sensitivity
 2      Gamma                Metacognitive sensitivity
 3      Phi                  Metacognitive sensitivity
 4      DeltaConf            Metacognitive sensitivity

 5      M-Ratio              Efficiency ratio  (obs / ideal)
 6      AUC2-Ratio           Efficiency ratio
 7      Gamma-Ratio          Efficiency ratio
 8      Phi-Ratio            Efficiency ratio
 9      DeltaConf-Ratio      Efficiency ratio

10      M-Diff               Efficiency difference (obs − ideal)
11      AUC2-Diff            Efficiency difference
12      Gamma-Diff           Efficiency difference
13      Phi-Diff             Efficiency difference
14      DeltaConf-Diff       Efficiency difference

15      metaNoise            Meta-noise model
16      metaUncertainty      Meta-uncertainty
17      d'                   Type-1 SDT sensitivity
18      c                    Type-1 SDT criterion
19      mean confidence      Mean raw confidence rating

20      logL                 Model-fit diagnostic (meta-d' MLE)
21      AIC                  Model-fit diagnostic
22      BIC                  Model-fit diagnostic
23      AICc                 Model-fit diagnostic
24      k                    Model-fit diagnostic (free parameters)
25      n                    Model-fit diagnostic (trials used in fit)
```

## Setup

```python
import numpy as np
from metasignal import stdpy

rng = np.random.default_rng(0)
n_trials, n_ratings = 400, 4

stim = rng.choice([0, 1], n_trials)
resp = np.where(rng.random(n_trials) < 0.78, stim, 1 - stim)
correct = stim == resp
conf = np.where(
    correct,
    rng.integers(3, n_ratings + 1, n_trials),
    rng.integers(1, 3, n_trials),
)
```

## Block 1 — Metacognitive sensitivity

These five measures ask: *how well does this participant's confidence track their accuracy?*

```python
meas = stdpy.compute_all_measures(stim, resp, conf, n_ratings=n_ratings)

print(f"meta-d'    = {meas[0]:.3f}")
print(f"AUC2       = {meas[1]:.3f}")
print(f"Gamma      = {meas[2]:.3f}")
print(f"Phi        = {meas[3]:.3f}")
print(f"DeltaConf  = {meas[4]:.3f}")
```

**meta-d'** (index 0) — the signal-detection-theory equivalent of d', but estimated from the confidence distribution rather than the hit/FA rates. Computed via maximum-likelihood estimation (slow on large datasets — see the precomputed files note below).

**AUC2** (index 1) — area under the Type-2 ROC curve. Ranges from 0.5 (chance) to 1.0 (perfect metacognition).

**Gamma / Phi** (indices 2–3) — rank-correlation measures of confidence-accuracy association (Goodman-Kruskal gamma and Matthews correlation coefficient).

**DeltaConf** (index 4) — mean confidence difference between correct and incorrect trials.

## Block 2 & 3 — Efficiency ratios and differences

Efficiency measures normalise the observed metacognitive performance by the *expected* performance of an ideal observer with the same d'. This removes the spurious dependence on task difficulty.

```python
nr_s1, nr_s2 = stdpy.trials_to_counts(stim, resp, conf, n_ratings=n_ratings)

# Expected counts for an ideal observer with the same d' and criterion
expected = stdpy.sdt_expect_conf(nr_s1, nr_s2)
nr_s1_exp = np.array(expected["nR_S1_exp"])
nr_s2_exp = np.array(expected["nR_S2_exp"])

auc2_obs = stdpy.compute_type2_auc(nr_s1, nr_s2)
auc2_exp = stdpy.compute_type2_auc(nr_s1_exp, nr_s2_exp)

print(f"AUC2 observed = {auc2_obs:.3f}")
print(f"AUC2 ideal    = {auc2_exp:.3f}")
print(f"AUC2-Ratio    = {meas[6]:.3f}   (= obs / ideal)")
print(f"AUC2-Diff     = {meas[11]:.3f}  (= obs − ideal)")
```

A ratio of 1.0 (or difference of 0.0) means the participant's metacognition is exactly as good as an ideal observer — perfectly efficient. Values below 1.0 indicate suboptimal metacognition.

## Individual measure functions

You can call each measure directly:

```python
# Type-2 AUC
auc2 = stdpy.compute_type2_auc(nr_s1, nr_s2)

# Goodman-Kruskal Gamma
gamma = stdpy.compute_gamma(nr_s1, nr_s2)

# Matthews correlation (Phi)
phi = stdpy.compute_phi(nr_s1, nr_s2)

# DeltaConf — returns a dict with raw, ratio, and difference
dc = stdpy.compute_delta_conf(nr_s1, nr_s2)
print(dc["delta_conf"], dc["delta_conf_ratio"], dc["delta_conf_diff"])

# meta-d' via MLE
result = stdpy.fit_meta_d_mle(nr_s1, nr_s2)
print(f"meta_da  = {result['meta_da']:.3f}")
print(f"M_ratio  = {result['M_ratio']:.3f}")
print(f"M_diff   = {result['M_diff']:.3f}")
```

## Block 4 — Meta-noise and meta-uncertainty

```python
# meta-noise (lognormal model)
noise_result = stdpy.compute_meta_noise(stim, resp, conf, n_ratings=n_ratings)
print(f"metaNoise        = {noise_result['meta_noise']:.3f}")

# meta-uncertainty
uncert = stdpy.compute_meta_uncertainty(stim, resp, conf, n_ratings=n_ratings)
print(f"metaUncertainty  = {uncert:.3f}")
```

**metaNoise** quantifies the dispersion in the confidence-generating process using a lognormal model. Higher values indicate noisier confidence ratings. **metaUncertainty** is a related model-free estimate.

## Handling NaN output

`compute_all_measures` returns NaN for a given measure when:

- d' is too small (< 0.2) — M-Ratio becomes undefined
- All confidence ratings are identical — no variance to estimate
- MLE optimisation fails to converge

```python
# Force a near-zero d' case
stim_flat = np.array([0, 1] * 100)
resp_flat = np.array([1, 0] * 100)   # chance performance
conf_flat = np.ones(200, dtype=int)   # all confidence = 1

meas_flat = stdpy.compute_all_measures(stim_flat, resp_flat, conf_flat, n_ratings=1)
nan_indices = np.where(np.isnan(meas_flat))[0]
print(f"NaN at indices: {nan_indices}")
```

## Performance note

`meta-d'`, `metaNoise`, and `metaUncertainty` each run an MLE optimisation (~0.1–4 seconds per call depending on trial count). For large datasets (hundreds of participants), use multiprocessing:

```python
from multiprocessing import Pool
from metasignal import stdpy

def _compute_one(args):
    stim, resp, conf, n_ratings = args
    return stdpy.compute_all_measures(stim, resp, conf, n_ratings)

participants = [(s["stim"], s["resp"], s["conf"], s["n_ratings"]) for s in dataset]

with Pool() as pool:
    results = pool.map(_compute_one, participants)

import numpy as np
all_measures = np.array(results)   # shape (n_participants, 26)
```

## Next steps

- **[Tutorial 3](statistical_inference.md)** — run bootstrap CIs and permutation tests over these measures
- **[Tutorial 4](difficulty_dependence.md)** — compute measures separately per difficulty level
