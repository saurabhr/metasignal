# Tutorial 1 — Getting Started

This tutorial verifies your metasignal installation and introduces the three input arrays every function expects.

## 1. Verify the install

```python
import numpy as np
import metasignal
from metasignal import stdpy

print(f"metasignal loaded from: {metasignal.__file__}")

# Quick smoke test on synthetic 4-rating data
rng  = np.random.default_rng(0)
stim = rng.choice([0, 1], 200)
resp = np.where(rng.random(200) < 0.75, stim, 1 - stim)  # 75% accuracy
conf = rng.integers(1, 5, 200)                             # ratings 1–4

meas = stdpy.compute_all_measures(stim, resp, conf, n_ratings=4)
print("Output shape:", meas.shape)   # (20,)
```

If you see `(20,)` you are good to go.

## 2. Input format

Every function in `stdpy` takes the same three arrays plus `n_ratings`:

| Argument | dtype | Values | Meaning |
| --- | --- | --- | --- |
| `stim` | int or float | 0 / 1 | Stimulus category — 0 = noise/S1, 1 = signal/S2 |
| `resp` | int or float | 0 / 1 | Participant response — 0 = "S1", 1 = "S2" |
| `conf` | int | 1 … n_ratings | Confidence rating — 1 = lowest, n_ratings = highest |
| `n_ratings` | int | — | Total number of confidence categories |

All three arrays must have the same length (one element per trial). NaN values in any array are silently dropped before computation.

## 3. Build a minimal dataset

```python
import numpy as np
from metasignal import stdpy

rng = np.random.default_rng(42)

n_trials  = 300
n_ratings = 4

# Stimulus: equal proportions of S1 (0) and S2 (1)
stim = rng.choice([0, 1], n_trials)

# Response: 80% correct
resp = np.where(rng.random(n_trials) < 0.80, stim, 1 - stim)

# Confidence: correlated with accuracy (correct → higher conf)
correct = (stim == resp)
conf = np.where(
    correct,
    rng.integers(3, n_ratings + 1, n_trials),   # correct: ratings 3–4
    rng.integers(1, 3, n_trials),               # incorrect: ratings 1–2
)

print(f"Trials: {n_trials}")
print(f"Accuracy: {correct.mean():.1%}")
print(f"Mean confidence: {conf.mean():.2f}")
```

## 4. Compute Type-1 SDT parameters

Before running the full 20-measure battery, compute the basic SDT summary:

```python
dprime, c, ln_beta = stdpy.compute_sdt_resp(stim, resp)
print(f"d'       = {dprime:.3f}")
print(f"criterion c = {c:.3f}")
```

`d'` measures perceptual sensitivity (how well the participant distinguished S1 from S2). `c` is the response criterion (negative = liberal, positive = conservative).

## 5. Convert trials to response counts

Several measures (including meta-d') operate on **response count matrices** rather than raw trials. `trials_to_counts` converts the three arrays:

```python
nr_s1, nr_s2 = stdpy.trials_to_counts(stim, resp, conf, n_ratings=n_ratings)

print("nr_s1 shape:", nr_s1.shape)   # (2 * n_ratings,)
print("nr_s2 shape:", nr_s2.shape)
print("nr_s1:", nr_s1)
print("nr_s2:", nr_s2)
```

Each array has `2 * n_ratings` elements:

- `nr_s1[:n_ratings]` — S1-response counts for S1 trials (from highest to lowest confidence)
- `nr_s1[n_ratings:]` — S2-response counts for S1 trials (from lowest to highest confidence)
- `nr_s2` — same layout for S2 trials

## 6. Inspect the 20-element output

```python
from metasignal import stdpy

MEASURE_NAMES = [
    "meta-d'",         # 0
    "AUC2",            # 1
    "Gamma",           # 2
    "Phi",             # 3
    "DeltaConf",       # 4
    "M-Ratio",         # 5
    "AUC2-Ratio",      # 6
    "Gamma-Ratio",     # 7
    "Phi-Ratio",       # 8
    "DeltaConf-Ratio", # 9
    "M-Diff",          # 10
    "AUC2-Diff",       # 11
    "Gamma-Diff",      # 12
    "Phi-Diff",        # 13
    "DeltaConf-Diff",  # 14
    "metaNoise",       # 15
    "metaUncertainty", # 16
    "d'",              # 17
    "c",               # 18
    "mean_conf",       # 19
]

meas = stdpy.compute_all_measures(stim, resp, conf, n_ratings=n_ratings)

for i, (name, val) in enumerate(zip(MEASURE_NAMES, meas)):
    print(f"  [{i:2d}] {name:<20s} = {val:.4f}" if not np.isnan(val)
          else f"  [{i:2d}] {name:<20s} = NaN")
```

!!! note "NaN values"
    Some measures return NaN when the data does not satisfy internal constraints — for example, when d' is near zero, M-Ratio is undefined. This is expected behaviour.

## Next steps

- **[Tutorial 2](computing_measures.md)** — computing and interpreting all 20 measures in detail
- **[Tutorial 3](statistical_inference.md)** — bootstrap confidence intervals and permutation tests
