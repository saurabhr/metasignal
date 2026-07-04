# Tutorial 3 — Statistical Inference

This tutorial shows how to run non-parametric statistical tests over the 26 measures using `metasignal.analysis`.

## Setup — simulate a multi-participant experiment

```python
import numpy as np
from metasignal import stdpy
from metasignal.analysis import bootstrap_measure, permutation_test, group_summary

rng = np.random.default_rng(1)
n_participants = 30
n_trials       = 200
n_ratings      = 4

def simulate_participant(seed, accuracy=0.78):
    r = np.random.default_rng(seed)
    stim    = r.choice([0, 1], n_trials)
    resp    = np.where(r.random(n_trials) < accuracy, stim, 1 - stim)
    correct = stim == resp
    conf    = np.where(
        correct,
        r.integers(3, n_ratings + 1, n_trials),
        r.integers(1, 3, n_trials),
    )
    return stim, resp, conf

participants = [simulate_participant(i) for i in range(n_participants)]
```

## 1. Group-level summary

`group_summary` runs `compute_all_measures` for every participant and returns group-level descriptive statistics:

```python
summary = group_summary(participants, n_ratings=n_ratings)

print("Measure names:", summary["labels"][:5], "...")
print("Group means (first 5):", summary["mean"][:5].round(3))
print("Group SEMs  (first 5):", summary["sem"][:5].round(3))
print("Valid n per measure:  ", summary["n_valid"][:5])
```

The returned dict has these keys:

| Key | Shape | Description |
| --- | --- | --- |
| `"individual"` | `(n_participants, 26)` | Per-participant measure values |
| `"mean"` | `(26,)` | nanmean across participants |
| `"median"` | `(26,)` | nanmedian |
| `"sem"` | `(26,)` | standard error of the mean |
| `"n_valid"` | `(26,)` | participants with non-NaN value per measure |
| `"labels"` | list of 26 str | measure names in index order |

```python
import numpy as np

# Print a formatted summary table
print(f"{'Measure':<20} {'Mean':>8} {'SEM':>8} {'n':>5}")
print("-" * 46)
for name, mean, sem, n in zip(
    summary["labels"], summary["mean"], summary["sem"], summary["n_valid"]
):
    if not np.isnan(mean):
        print(f"{name:<20} {mean:8.3f} {sem:8.3f} {n:5d}")
```

## 2. Bootstrap confidence intervals

`bootstrap_measure` estimates a confidence interval for one measure by resampling trials with replacement:

```python
stim, resp, conf = participants[0]

lo, hi = bootstrap_measure(
    stim, resp, conf,
    n_ratings=n_ratings,
    measure_index=5,   # M-Ratio
    n_boot=2000,
    ci=0.95,
    rng=np.random.default_rng(0),
)
print(f"M-Ratio 95% CI: [{lo:.3f}, {hi:.3f}]")
```

**`measure_index`** is the integer index into the 26-element output (see [Tutorial 2](computing_measures.md) for the full index→name mapping).

Bootstrap CIs for several measures at once:

```python
INDICES = {
    "meta-d'": 0,
    "AUC2":    1,
    "M-Ratio": 5,
    "d'":      17,
}

stim, resp, conf = participants[0]
rng_ci = np.random.default_rng(99)

print(f"{'Measure':<12} {'CI 95%':>20}")
print("-" * 35)
for name, idx in INDICES.items():
    lo, hi = bootstrap_measure(
        stim, resp, conf,
        n_ratings=n_ratings,
        measure_index=idx,
        n_boot=1000,
        rng=rng_ci,
    )
    print(f"{name:<12}  [{lo:.3f}, {hi:.3f}]")
```

!!! tip "Reproducibility"
    Pass an explicit `rng=np.random.default_rng(seed)` to get reproducible intervals.

## 3. Permutation test — comparing two conditions

`permutation_test` tests whether two trial sets differ on a given measure by shuffling condition labels:

```python
# Condition A: well-calibrated metacognition (accuracy 80%, conf tracks accuracy)
stim_a, resp_a, conf_a = simulate_participant(0, accuracy=0.80)

# Condition B: same accuracy, confidence does NOT track accuracy
rng_b = np.random.default_rng(200)
stim_b = rng_b.choice([0, 1], n_trials)
resp_b = np.where(rng_b.random(n_trials) < 0.80, stim_b, 1 - stim_b)
conf_b = rng_b.integers(1, n_ratings + 1, n_trials)   # random confidence

p_val, obs_diff = permutation_test(
    stim_a, resp_a, conf_a,
    stim_b, resp_b, conf_b,
    n_ratings=n_ratings,
    measure_index=1,   # AUC2
    n_perm=5000,
    rng=np.random.default_rng(42),
)
print(f"AUC2 observed difference (A − B): {obs_diff:.3f}")
print(f"Two-sided p-value:                {p_val:.4f}")
```

The function pools all trials, randomly shuffles the condition assignment `n_perm` times, and counts how often the shuffled difference exceeds the observed one in absolute value.

Scan all 26 measures for significant condition differences:

```python
from metasignal.analysis import permutation_test

print(f"{'Measure':<20} {'obs diff':>10} {'p-value':>10} {'sig':>4}")
print("-" * 50)

MEASURE_NAMES = [
    "meta-d'","AUC2","Gamma","Phi","DeltaConf",
    "M-Ratio","AUC2-Ratio","Gamma-Ratio","Phi-Ratio","DeltaConf-Ratio",
    "M-Diff","AUC2-Diff","Gamma-Diff","Phi-Diff","DeltaConf-Diff",
    "metaNoise","metaUncertainty","d'","c","mean_conf",
    "logL","AIC","BIC","AICc","k","n",
]

for idx, name in enumerate(MEASURE_NAMES):
    p, diff = permutation_test(
        stim_a, resp_a, conf_a,
        stim_b, resp_b, conf_b,
        n_ratings=n_ratings,
        measure_index=idx,
        n_perm=1000,
        rng=np.random.default_rng(idx),
    )
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{name:<20} {diff:10.3f} {p:10.4f} {stars:>4}")
```

## 4. One-sample t-test against zero

For group-level analyses (e.g. testing whether M-Ratio is significantly above 1.0), use SciPy:

```python
from scipy import stats

individual = summary["individual"]   # (n_participants, 26)

# Test M-Ratio against 1.0 (efficiency ≠ perfect)
m_ratio = individual[:, 5]
m_ratio_valid = m_ratio[~np.isnan(m_ratio)]

t, p = stats.ttest_1samp(m_ratio_valid, popmean=1.0)
n = len(m_ratio_valid)
cohen_d = t / np.sqrt(n)   # Cohen's d, matching Rahnev (2025) convention

print(f"M-Ratio: t({n-1}) = {t:.3f}, p = {p:.4f}, d = {cohen_d:.3f}")
```

!!! note "Cohen's d convention"
    Rahnev (2025) computes Cohen's d as `t / sqrt(n)` (not `t / sqrt(n-1)`). The `ttest_1samp` helper in the benchmark notebooks follows this convention.

## Next steps

- **[Tutorial 4](difficulty_dependence.md)** — compute measures per difficulty level and run repeated-measures ANOVA
- **[Tutorial 5](metacognitive_bias.md)** — test for sensitivity to confidence bias using Xue recoding
