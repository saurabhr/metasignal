# Tutorial 5 — Metacognitive Bias

Some participants systematically use high or low confidence ratings regardless of their accuracy — a **response bias** in the metacognitive domain. A good metacognitive measure should be insensitive to this bias so that group comparisons are not confounded.

This tutorial implements the Xue et al. (2021) recoding method used in Supplementary Tables 6–8 of Rahnev (2025).

## The Xue recoding method

The test artificially induces two opposite biases and measures how much each measure changes:

| Recode | Effect | Mechanism |
| --- | --- | --- |
| **Recode 1** | High-confidence bias | Subtract 1 from all ratings; bump any trial that hits the minimum up by 1 |
| **Recode 2** | Low-confidence bias  | Replace all ratings at the maximum with max − 1 |

If a measure changes significantly when computing `recode2 − recode1`, it is sensitive to confidence bias.

## Setup

```python
import numpy as np
from scipy import stats
from metasignal import stdpy

MEASURE_NAMES = [
    "meta-d'","AUC2","Gamma","Phi","DeltaConf",
    "M-Ratio","AUC2-Ratio","Gamma-Ratio","Phi-Ratio","DeltaConf-Ratio",
    "M-Diff","AUC2-Diff","Gamma-Diff","Phi-Diff","DeltaConf-Diff",
    "metaNoise","metaUncertainty","d'","c","mean_conf",
    "logL","AIC","BIC","AICc","k","n",
]
N_MEAS = 26

def simulate_subject(seed, n_trials=300, n_ratings=4, accuracy=0.78):
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

n_subjects = 25
n_ratings  = 4
dataset    = [simulate_subject(i, n_ratings=n_ratings) for i in range(n_subjects)]
```

## The recoding function

```python
def xue_recode(conf, rtype):
    """Apply Xue et al. (2021) confidence recoding.

    rtype=1 — high-confidence bias: subtract 1, floor at min+1
    rtype=2 — low-confidence bias:  replace max with max-1
    Returns recoded confidence array with n_ratings - 1 effective levels.
    """
    c = conf.copy().astype(float)
    valid = c[~np.isnan(c)]
    if len(np.unique(valid)) < 3:
        return np.full_like(c, np.nan)

    if rtype == 1:
        c -= 1
        cmin = np.nanmin(c)
        c[c == cmin] = cmin + 1
    elif rtype == 2:
        cmax = np.nanmax(c)
        c[c == cmax] = cmax - 1
    return c

# Example on a 4-rating scale
ex = np.array([1, 2, 3, 4, 4, 3, 2, 1])
print("Original : ", ex)
print("Recode 1 : ", xue_recode(ex, 1).astype(int))   # → [2, 2, 3, 3, 3, 3, 2, 2]
print("Recode 2 : ", xue_recode(ex, 2).astype(int))   # → [1, 2, 3, 3, 3, 3, 2, 1]
```

After recode 1, the effective scale is 2–4 (high-confidence direction). After recode 2, the scale is 1–3 (low-confidence direction). Both recodings use `n_ratings - 1` effective levels.

## Computing measures under both recodings

```python
# bias[subject, recode_idx (0=recode1, 1=recode2), measure]
bias = np.full((n_subjects, 2, N_MEAS), np.nan)
n_ratings_recoded = n_ratings - 1

for s_idx, (stim, resp, conf) in enumerate(dataset):
    for rt in (1, 2):
        conf_rc = xue_recode(conf, rt)
        valid   = ~np.isnan(conf_rc)
        bias[s_idx, rt - 1] = stdpy.compute_all_measures(
            stim[valid], resp[valid], conf_rc[valid].astype(int),
            n_ratings=n_ratings_recoded,
        )

print("Bias array shape:", bias.shape)   # (25, 2, 26)
```

## Testing recode2 − recode1

A one-sample t-test against zero on the difference indicates whether a measure is influenced by the direction of confidence bias:

```python
def ttest_1samp(data):
    x = np.asarray(data, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan
    t, p = stats.ttest_1samp(x, 0)
    d    = t / np.sqrt(n)
    return t, n - 1, p, d

# d' and criterion are unaffected by confidence recoding — skip them
SKIP = {"d'", "c"}

delta = bias[:, 1, :] - bias[:, 0, :]   # recode2 − recode1

print(f"{'Measure':<20} {'t':>8} {'p':>10} {'Cohen d':>9} {'sig':>4}")
print("-" * 58)
for m, name in enumerate(MEASURE_NAMES):
    if name in SKIP:
        continue
    t, df, p, d = ttest_1samp(delta[:, m])
    if np.isnan(t):
        continue
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{name:<20} {t:8.3f} {p:10.4f} {d:9.3f} {stars:>4}")
```

A significant positive t-statistic means recode 2 (low-confidence bias) gives a *higher* value than recode 1 (high-confidence bias). The direction reverses depending on how the measure relates to mean confidence.

## Visualising the bias effect

```python
import matplotlib.pyplot as plt

means = np.nanmean(delta, axis=0)
sems  = np.nanstd(delta, axis=0, ddof=1) / np.sqrt(
    np.sum(~np.isnan(delta), axis=0)
)

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(N_MEAS)

colors = ["#d55e00" if s < 0.05 else "#999999"
          for s in [ttest_1samp(delta[:, m])[2] for m in range(N_MEAS)]]

ax.bar(x, means, yerr=sems, color=colors, alpha=0.85, capsize=3)
ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(MEASURE_NAMES, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Recode 2 − Recode 1 ± SEM")
ax.set_title("Metacognitive Bias Sensitivity (significant p < 0.05 shown in orange)")
plt.tight_layout()
plt.savefig("metacognitive_bias.png", dpi=120, bbox_inches="tight")
plt.show()
```

## Interpretation

Measures that are **not** significantly affected by the Xue recoding are preferable for studies that compare groups differing in mean confidence. In Rahnev (2025), efficiency ratios and differences (M-Ratio, AUC2-Ratio, etc.) tend to show weaker bias effects than absolute measures.

!!! note "Replicating Rahnev (2025) Supp Tables 6–8"
    The full replication with real datasets (Haddara 2022, Maniscalco 2017, Shekhar 2021) is in `notebooks/06_metacognitive_bias.ipynb`.

## Next steps

- **[Tutorial 6](split_half_reliability.md)** — split-half reliability and precision under confidence corruption
