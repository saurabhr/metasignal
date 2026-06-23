# Tutorial 6 — Split-Half Reliability & Precision

Two further benchmarking criteria from Rahnev (2025):

- **Split-half reliability** — does a measure give consistent values when estimated from independent halves of the data?
- **Precision** — how quickly does a measure degrade when confidence ratings are artificially corrupted?

## Split-half reliability

The procedure splits each subject's trials into **odd** and **even** halves, computes the 20 measures on each half independently, correlates the two halves with Pearson *r*, then applies the **Spearman-Brown correction** to estimate the full-sample reliability:

$$r_{SB} = \frac{2r}{1 + r}$$

### Setup

```python
import numpy as np
from scipy.stats import pearsonr
from metasignal import stdpy

MEASURE_NAMES = [
    "meta-d'","AUC2","Gamma","Phi","DeltaConf",
    "M-Ratio","AUC2-Ratio","Gamma-Ratio","Phi-Ratio","DeltaConf-Ratio",
    "M-Diff","AUC2-Diff","Gamma-Diff","Phi-Diff","DeltaConf-Diff",
    "metaNoise","metaUncertainty","d'","c","mean_conf",
]
N_MEAS = 20

def simulate_subject(seed, n_trials=400, n_ratings=4, accuracy=0.78):
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

n_subjects = 30
n_ratings  = 4
dataset    = [simulate_subject(i, n_ratings=n_ratings) for i in range(n_subjects)]
```

### Computing split-half measures

```python
# split[subject, half (0=odd, 1=even), measure]
split = np.full((n_subjects, 2, N_MEAS), np.nan)

for s_idx, (stim, resp, conf) in enumerate(dataset):
    idx_odd  = np.arange(0, len(stim), 2)
    idx_even = np.arange(1, len(stim), 2)

    for half_idx, idx in enumerate([idx_odd, idx_even]):
        split[s_idx, half_idx] = stdpy.compute_all_measures(
            stim[idx], resp[idx], conf[idx], n_ratings=n_ratings
        )

print("Split array shape:", split.shape)   # (30, 2, 20)
```

### Spearman-Brown corrected correlations

```python
def spearman_brown(r):
    return 2 * r / (1 + r) if not np.isnan(r) else np.nan

print(f"{'Measure':<20} {'Pearson r':>10} {'SB-corrected':>14}")
print("-" * 48)

for m, name in enumerate(MEASURE_NAMES):
    x = split[:, 0, m]
    y = split[:, 1, m]
    ok = ~np.isnan(x) & ~np.isnan(y)

    if ok.sum() < 5:
        print(f"{name:<20} {'NaN':>10} {'NaN':>14}")
        continue

    r, p = pearsonr(x[ok], y[ok])
    sb   = spearman_brown(r)
    print(f"{name:<20} {r:10.3f} {sb:14.3f}")
```

A Spearman-Brown coefficient near 1.0 indicates high reliability. Values below ~0.7 suggest the measure requires more trials for stable estimation.

### Visualising reliability

```python
import matplotlib.pyplot as plt

sb_vals = []
for m in range(N_MEAS):
    x, y = split[:, 0, m], split[:, 1, m]
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() >= 5:
        r, _ = pearsonr(x[ok], y[ok])
        sb_vals.append(spearman_brown(r))
    else:
        sb_vals.append(np.nan)

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(N_MEAS)
colors = ["#0072b2" if v >= 0.7 else "#d55e00" if not np.isnan(v) else "#cccccc"
          for v in sb_vals]

ax.bar(x, sb_vals, color=colors, alpha=0.85)
ax.axhline(0.7, color="k", linewidth=1.0, linestyle="--", label="0.7 threshold")
ax.set_xticks(x)
ax.set_xticklabels(MEASURE_NAMES, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Spearman-Brown reliability")
ax.set_ylim(-0.1, 1.1)
ax.set_title("Split-Half Reliability (blue ≥ 0.7, orange < 0.7)")
ax.legend()
plt.tight_layout()
plt.savefig("split_half_reliability.png", dpi=120, bbox_inches="tight")
plt.show()
```

---

## Precision under confidence corruption

**Precision** tests how robust each measure is when confidence ratings are artificially degraded. A proportion `p` of trials have their confidence shifted by 1 in the **anti-metacognitive** direction: correct trials get *lower* confidence, incorrect trials get *higher* confidence.

The precision score is the resulting drop in the measure normalised by its across-subject standard deviation. A large drop indicates low precision — the measure is sensitive to noise in the confidence signal.

### Corruption function

```python
def corrupt_confidence(stim, resp, conf, proportion, rng):
    """Shift `proportion` of trials in the anti-metacognitive direction.

    Correct trials (stim==resp): confidence reduced by 1 (floored at 1)
    Incorrect trials: confidence raised by 1 (capped at n_ratings)
    """
    c       = conf.copy().astype(float)
    correct = (stim == resp)
    n       = len(c)
    n_corrupt = int(np.round(proportion * n))

    # Choose which trials to corrupt
    corrupt_idx = rng.choice(n, size=n_corrupt, replace=False)

    for i in corrupt_idx:
        if correct[i]:
            c[i] = max(1, c[i] - 1)
        else:
            c[i] = min(n_ratings, c[i] + 1)

    return c
```

### Computing precision

```python
PROPORTIONS = [0.0, 0.02, 0.04, 0.06]   # 0%, 2%, 4%, 6% corrupted

# base[subject, measure] — uncorrupted baseline
base = np.array([
    stdpy.compute_all_measures(s, r, c, n_ratings=n_ratings)
    for s, r, c in dataset
])

# across-subject SD of baseline (for normalisation)
base_sd = np.nanstd(base, axis=0, ddof=1)

# drops[proportion_idx, measure] — mean normalised drop
drops = np.full((len(PROPORTIONS), N_MEAS), np.nan)

for pi, prop in enumerate(PROPORTIONS):
    if prop == 0.0:
        drops[pi] = 0.0
        continue

    corrupted = np.full((n_subjects, N_MEAS), np.nan)
    for s_idx, (stim, resp, conf) in enumerate(dataset):
        conf_c = corrupt_confidence(
            stim, resp, conf, prop,
            rng=np.random.default_rng(s_idx * 100 + pi),
        )
        corrupted[s_idx] = stdpy.compute_all_measures(
            stim, resp, conf_c.astype(int), n_ratings=n_ratings
        )

    # normalised drop = (base − corrupted) / base_sd
    diff = base - corrupted
    drops[pi] = np.nanmean(diff / base_sd[np.newaxis, :], axis=0)

print("Precision drops (rows=proportions, cols=measures):")
print(np.round(drops, 3))
```

### Precision plot

```python
fig, axes = plt.subplots(4, 5, figsize=(18, 12), sharex=True)
axes = axes.flatten()

for m, (name, ax) in enumerate(zip(MEASURE_NAMES, axes)):
    ax.plot(PROPORTIONS, drops[:, m], marker="o", color="#d55e00")
    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.set_xlabel("Corrupted %", fontsize=7)
    ax.set_ylabel("Norm. drop", fontsize=7)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xticks(PROPORTIONS)
    ax.set_xticklabels([f"{int(p*100)}%" for p in PROPORTIONS], fontsize=7)

plt.suptitle("Precision: Normalised Drop Under Confidence Corruption", fontsize=13)
plt.tight_layout()
plt.savefig("precision.png", dpi=120, bbox_inches="tight")
plt.show()
```

Measures that remain near zero across all corruption proportions are the most precise — their estimated value is robust to random noise in the confidence ratings.

!!! note "Replicating Rahnev (2025)"
    The full split-half and precision analyses using real datasets are in `notebooks/08_split_half_precision.ipynb`. Note that meta-d', M-Ratio, and M-Diff are excluded from precision in the original paper due to the computational cost of repeated MLE fitting.

## Summary

You have now covered the complete benchmarking pipeline from Rahnev (2025):

1. Compute all 20 measures (`compute_all_measures`)
2. Run group-level statistics (bootstrap, permutation, ANOVA)
3. Test difficulty independence
4. Test bias sensitivity (Xue recoding)
5. Assess split-half reliability and precision

For the full dataset replication with all six published datasets, see the `notebooks/` directory in the repository.
