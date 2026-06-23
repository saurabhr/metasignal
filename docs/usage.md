# Usage

## Input format

All computation functions expect three parallel arrays of the same length, one row per trial:

| Argument | Values | Description |
| --- | --- | --- |
| `stim` | 0 or 1 | Stimulus category (0 = S1 / noise, 1 = S2 / signal) |
| `resp` | 0 or 1 | Participant response (0 = "S1", 1 = "S2") |
| `conf` | 1 … n_ratings | Confidence rating (1 = lowest, n_ratings = highest) |
| `n_ratings` | int | Total number of confidence categories |

## Pure Python API (`stdpy`)

### Compute all 20 measures at once

```python
import numpy as np
from metasignal import stdpy

stim = np.array([0, 1, 0, 1] * 25)
resp = np.array([0, 1, 1, 0] * 25)
conf = np.array([1, 2, 2, 1] * 25)

results = stdpy.compute_all_measures(stim, resp, conf, n_ratings=2)
print(results)
```

The returned array has 20 elements, indexed as follows:

| Index | Measure |
| --- | --- |
| 0 | meta-d' |
| 1 | AUC2 |
| 2 | gamma |
| 3 | phi |
| 4 | deltaConf |
| 5 | M-ratio |
| 6 | AUC2-ratio |
| 7 | gamma-ratio |
| 8 | phi-ratio |
| 9 | deltaConf-ratio |
| 10 | M-diff |
| 11 | AUC2-diff |
| 12 | gamma-diff |
| 13 | phi-diff |
| 14 | deltaConf-diff |
| 15 | metaNoise |
| 16 | metaUncertainty |
| 17 | d' |
| 18 | c |
| 19 | mean confidence |

### Compute individual measures

**Basic SDT — d' and criterion c:**

```python
dprime, c, ln_beta = stdpy.compute_sdt_resp(stim, resp)
print(f"d' = {dprime:.3f}, c = {c:.3f}")
```

**Convert trials to response counts:**

```python
nr_s1, nr_s2 = stdpy.trials_to_counts(stim, resp, conf, n_ratings=2)
```

**Fit meta-d' via MLE:**

```python
nr_s1, nr_s2 = stdpy.trials_to_counts(stim, resp, conf, n_ratings=2)
result = stdpy.fit_meta_d_mle(nr_s1, nr_s2)
print(f"meta-d' = {result['meta_da']:.3f}")
print(f"M-ratio = {result['M_ratio']:.3f}")
```

**Type-2 measures:**

```python
auc2      = stdpy.compute_type2_auc(nr_s1, nr_s2)
gamma     = stdpy.compute_gamma(nr_s1, nr_s2)
phi       = stdpy.compute_phi(nr_s1, nr_s2)
dc        = stdpy.compute_delta_conf(nr_s1, nr_s2)
```

**metaNoise and metaUncertainty:**

```python
noise     = stdpy.compute_meta_noise(stim, resp, conf, n_ratings=2)
uncert    = stdpy.compute_meta_uncertainty(stim, resp, conf, n_ratings=2)
```

## MATLAB backend (`MetaSignal`)

If you have a MATLAB installation and have installed `pip install ".[matlab]"`, you can delegate computation to the original MATLAB code:

```python
import numpy as np
from metasignal import MetaSignal

ms = MetaSignal()                           # starts the MATLAB engine

stim = np.array([0, 1, 0, 1] * 20)
resp = np.array([0, 1, 1, 0] * 20)
conf = np.array([1, 2, 2, 1] * 20)

results = ms.compute_all_measures(stim, resp, conf, n_ratings=2)
print(results)

ms.stop()                                   # shut down the engine when done
```

The first call starts the MATLAB engine (may take a few seconds). Use `ms.stop()` or let the object go out of scope to release it.

## Statistical inference (`analysis`)

The `metasignal.analysis` sub-package provides tools for running inference over the point estimates:

### Bootstrap confidence intervals

```python
import numpy as np
from metasignal.analysis import bootstrap_measure

stim = np.array([0, 1] * 50)
resp = np.array([0, 1] * 50)
conf = np.random.default_rng(0).integers(1, 3, 100)

lo, hi = bootstrap_measure(
    stim, resp, conf,
    n_ratings=2,
    measure_index=5,   # M-ratio (index 5)
    n_boot=2000,
    ci=0.95,
)
print(f"M-ratio 95% CI: [{lo:.3f}, {hi:.3f}]")
```

### Permutation test (two-condition comparison)

```python
from metasignal.analysis import permutation_test

p_val, obs_diff = permutation_test(
    stim_a, resp_a, conf_a,
    stim_b, resp_b, conf_b,
    n_ratings=2,
    measure_index=5,   # M-ratio
    n_perm=5000,
)
print(f"p = {p_val:.4f}, observed difference = {obs_diff:.4f}")
```

### Group-level summary

```python
from metasignal.analysis import group_summary

# List of (stim, resp, conf) tuples — one per participant
participants = [(stim_p1, resp_p1, conf_p1), ...]

summary = group_summary(participants, n_ratings=2)
print(summary["mean"])    # (20,) array of group means
print(summary["sem"])     # (20,) array of standard errors
print(summary["labels"])  # list of 20 measure names
```

## CLI

For quick exploratory use without writing Python:

```bash
metasignal compute --stim "0,1,0,1" --resp "0,1,1,0" --conf "1,2,2,1" --n-ratings 2
```

Test the MATLAB engine connection:

```bash
metasignal test-matlab
```

See [CLI Reference](cli.md) for the full argument list.
