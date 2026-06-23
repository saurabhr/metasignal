# Tutorial 4 — Difficulty Dependence

A key property required of a good metacognitive measure is **difficulty independence** — it should not change simply because the perceptual task is easier or harder. This tutorial replicates the analysis from Supplementary Tables 3–5 of Rahnev (2025).

## The question

For each of 20 measures we ask: does the measure differ between easy (high contrast) and hard (low contrast) trial blocks? A significant effect means the measure conflates metacognitive efficiency with task difficulty — undesirable when comparing groups.

## Setup

```python
import numpy as np
from scipy import stats
from metasignal import stdpy

rng = np.random.default_rng(7)

def simulate_subject(seed, difficulty_levels=(0.65, 0.75, 0.85), n_per_level=80, n_ratings=4):
    """Simulate one subject across multiple difficulty levels.

    difficulty_levels: per-level accuracy (easy → high number)
    Returns list of (stim, resp, conf) tuples, one per difficulty.
    """
    r = np.random.default_rng(seed)
    trials = []
    for acc in difficulty_levels:
        stim = r.choice([0, 1], n_per_level)
        resp = np.where(r.random(n_per_level) < acc, stim, 1 - stim)
        correct = stim == resp
        conf = np.where(
            correct,
            r.integers(3, n_ratings + 1, n_per_level),
            r.integers(1, 3, n_per_level),
        )
        trials.append((stim, resp, conf))
    return trials

n_subjects    = 20
n_ratings     = 4
difficulty    = (0.65, 0.75, 0.85)   # hard, medium, easy

dataset = [simulate_subject(i, difficulty, n_ratings=n_ratings)
           for i in range(n_subjects)]
```

## Computing measures per difficulty level

```python
N_MEAS      = 20
n_levels    = len(difficulty)

# raw[subject, level, measure]
raw = np.full((n_subjects, n_levels, N_MEAS), np.nan)

for s_idx, subject_trials in enumerate(dataset):
    for lv_idx, (stim, resp, conf) in enumerate(subject_trials):
        raw[s_idx, lv_idx] = stdpy.compute_all_measures(
            stim, resp, conf, n_ratings=n_ratings
        )

print("Computed array shape:", raw.shape)   # (20, 3, 20)
```

## 3-SD outlier removal

The Rahnev (2025) MATLAB pipeline (`ana_taskPerformance.m`) removes per-measure, per-level outliers beyond 3 SD, then sets **all levels** to NaN for any subject flagged at any level. This prevents inflation of effects from extreme values.

```python
def remove_3sd_outliers(arr):
    """arr: (n_subjects, n_levels, n_measures).
    For each measure and level, values > 3 SD from the mean become NaN.
    If any level is NaN for a subject, all levels are set to NaN.
    """
    out = arr.copy()
    n_sub, n_lev, n_meas = out.shape
    for m in range(n_meas):
        for lv in range(n_lev):
            col = out[:, lv, m]
            mu, sd = np.nanmean(col), np.nanstd(col, ddof=1)
            if not np.isnan(mu) and sd > 0:
                out[(col < mu - 3 * sd) | (col > mu + 3 * sd), lv, m] = np.nan
        bad = np.isnan(out[:, :, m]).any(axis=1)
        out[bad, :, m] = np.nan
    return out

clean = remove_3sd_outliers(raw)

removed = np.sum(np.isnan(clean) & ~np.isnan(raw))
print(f"Values set to NaN by 3-SD removal: {removed}")
```

## One-sample t-test on easy − hard difference

```python
MEASURE_NAMES = [
    "meta-d'","AUC2","Gamma","Phi","DeltaConf",
    "M-Ratio","AUC2-Ratio","Gamma-Ratio","Phi-Ratio","DeltaConf-Ratio",
    "M-Diff","AUC2-Diff","Gamma-Diff","Phi-Diff","DeltaConf-Diff",
    "metaNoise","metaUncertainty","d'","c","mean_conf",
]

# Compare hardest (level 0) vs easiest (level 2)
delta = clean[:, 2, :] - clean[:, 0, :]   # easy − hard

def ttest_1samp(data):
    """One-sample t-test vs 0 with Cohen's d = t / sqrt(n)."""
    x = np.asarray(data, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan
    t, p = stats.ttest_1samp(x, 0)
    d = t / np.sqrt(n)
    return t, n - 1, p, d

print(f"{'Measure':<20} {'t':>8} {'df':>5} {'p':>10} {'d':>8} {'sig':>4}")
print("-" * 60)
for m, name in enumerate(MEASURE_NAMES):
    t, df, p, d = ttest_1samp(delta[:, m])
    if np.isnan(t):
        continue
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{name:<20} {t:8.3f} {df:5.0f} {p:10.4f} {d:8.3f} {stars:>4}")
```

## Repeated-measures ANOVA across all three levels

For datasets with three or more difficulty levels, a repeated-measures ANOVA tests the overall effect of difficulty:

```python
def rm_anova_1way(data_2d):
    """data_2d: (n_subjects, n_levels). Returns (F, df_between, df_error, p, partial_eta2)."""
    n, k   = data_2d.shape
    grand  = np.nanmean(data_2d)
    row_m  = np.nanmean(data_2d, axis=1, keepdims=True)
    col_m  = np.nanmean(data_2d, axis=0, keepdims=True)
    ss_b   = n * np.sum((col_m - grand) ** 2)
    ss_s   = k * np.sum((row_m - grand) ** 2)
    ss_e   = np.sum((data_2d - grand) ** 2) - ss_b - ss_s
    df_b, df_e = k - 1, (n - 1) * (k - 1)
    F  = (ss_b / df_b) / (ss_e / df_e)
    p  = stats.f.sf(F, df_b, df_e)
    return F, df_b, df_e, p, ss_b / (ss_b + ss_e)

print(f"\nRepeated-measures ANOVA across {n_levels} difficulty levels")
print(f"{'Measure':<20} {'F':>8} {'df_b':>5} {'df_e':>6} {'p':>10} {'η²p':>8}")
print("-" * 62)
for m, name in enumerate(MEASURE_NAMES):
    col = clean[:, :, m]
    ok  = ~np.isnan(col).any(axis=1)
    if ok.sum() < 3:
        continue
    F, df_b, df_e, p, eta2 = rm_anova_1way(col[ok])
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{name:<20} {F:8.3f} {df_b:5.0f} {df_e:6.0f} {p:10.4f} {eta2:8.3f} {stars}")
```

## Visualising the difficulty effect

```python
import matplotlib.pyplot as plt

LEVEL_LABELS = ["Hard (65%)", "Medium (75%)", "Easy (85%)"]

fig, axes = plt.subplots(4, 5, figsize=(18, 12), sharex=True)
axes = axes.flatten()

for m, (name, ax) in enumerate(zip(MEASURE_NAMES, axes)):
    means = np.nanmean(clean[:, :, m], axis=0)
    sems  = np.nanstd(clean[:, :, m], axis=0, ddof=1) / np.sqrt(
        np.sum(~np.isnan(clean[:, :, m]), axis=0)
    )
    ax.errorbar(range(n_levels), means, yerr=sems, marker="o", capsize=4)
    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.set_xticks(range(n_levels))
    ax.set_xticklabels(["Hard", "Med", "Easy"], fontsize=7, rotation=30)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")

plt.suptitle("Effect of Difficulty on 20 Metacognitive Measures", fontsize=13)
plt.tight_layout()
plt.savefig("difficulty_dependence.png", dpi=120, bbox_inches="tight")
plt.show()
```

!!! note "Replicating Rahnev (2025) Supp Tables 3–5"
    The benchmark replication notebooks in `notebooks/05_difficulty_dependence.ipynb` reproduce the exact t-statistics from the paper using real datasets (Shekhar 2021, Rouault 2018 Expt 1 & 2). The code structure above mirrors that notebook.

## Next steps

- **[Tutorial 5](metacognitive_bias.md)** — test for sensitivity to confidence bias
