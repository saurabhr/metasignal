"""Create tutorial notebooks for the metasignal docs."""

import nbformat
from nbformat.v4 import new_code_cell as code, new_markdown_cell as md

TUTORIALS = {}

# ─────────────────────────────────────────────────────────────────────────────
# Tutorial 1 — Getting Started
# ─────────────────────────────────────────────────────────────────────────────
TUTORIALS["01_getting_started"] = [
    md("""# Tutorial 1 — Getting Started

This notebook verifies your metasignal installation and walks through the three
input arrays every function expects."""),

    md("## 1. Verify the install"),
    code("""\
import numpy as np
import metasignal
from metasignal import stdpy

print(f"metasignal loaded from: {metasignal.__file__}")

rng  = np.random.default_rng(0)
stim = rng.choice([0, 1], 200)
resp = np.where(rng.random(200) < 0.75, stim, 1 - stim)
conf = rng.integers(1, 5, 200)

meas = stdpy.compute_all_measures(stim, resp, conf, n_ratings=4)
print("Output shape:", meas.shape)   # (26,)
"""),

    md("""## 2. Input format

| Argument | Values | Meaning |
| --- | --- | --- |
| `stim` | 0 / 1 | Stimulus category (0 = S1/noise, 1 = S2/signal) |
| `resp` | 0 / 1 | Participant response |
| `conf` | 1 … n_ratings | Confidence rating (1 = lowest) |
| `n_ratings` | int | Total number of confidence categories |"""),

    md("## 3. Build a minimal dataset"),
    code("""\
rng = np.random.default_rng(42)
n_trials, n_ratings = 300, 4

stim = rng.choice([0, 1], n_trials)
resp = np.where(rng.random(n_trials) < 0.80, stim, 1 - stim)
correct = (stim == resp)
conf = np.where(
    correct,
    rng.integers(3, n_ratings + 1, n_trials),
    rng.integers(1, 3, n_trials),
)

print(f"Trials   : {n_trials}")
print(f"Accuracy : {correct.mean():.1%}")
print(f"Mean conf: {conf.mean():.2f}")
"""),

    md("## 4. Type-1 SDT parameters"),
    code("""\
dprime, c, ln_beta = stdpy.compute_sdt_resp(stim, resp)
print(f"d'        = {dprime:.3f}")
print(f"criterion = {c:.3f}")
"""),

    md("## 5. Convert trials to response counts"),
    code("""\
nr_s1, nr_s2 = stdpy.trials_to_counts(stim, resp, conf, n_ratings=n_ratings)
print("nr_s1 shape:", nr_s1.shape)
print("nr_s1:", nr_s1)
print("nr_s2:", nr_s2)
"""),

    md("## 6. Inspect the full 26-element output"),
    code("""\
MEASURE_NAMES = [
    "meta-d'", "AUC2", "Gamma", "Phi", "DeltaConf",
    "M-Ratio", "AUC2-Ratio", "Gamma-Ratio", "Phi-Ratio", "DeltaConf-Ratio",
    "M-Diff", "AUC2-Diff", "Gamma-Diff", "Phi-Diff", "DeltaConf-Diff",
    "metaNoise", "metaUncertainty", "d'", "c", "mean_conf",
    "logL", "AIC", "BIC", "AICc", "k", "n",
]

meas = stdpy.compute_all_measures(stim, resp, conf, n_ratings=n_ratings)

for i, (name, val) in enumerate(zip(MEASURE_NAMES, meas)):
    flag = "NaN" if np.isnan(val) else f"{val:.4f}"
    print(f"  [{i:2d}] {name:<20s} = {flag}")
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# Tutorial 2 — Computing All 26 Measures
# ─────────────────────────────────────────────────────────────────────────────
TUTORIALS["02_computing_measures"] = [
    md("""# Tutorial 2 — Computing All 26 Measures

A detailed walkthrough of each block of the 26-measure array, plus how
to call individual measures directly."""),

    md("## Setup"),
    code("""\
import numpy as np
from metasignal import stdpy

MEASURE_NAMES = [
    "meta-d'", "AUC2", "Gamma", "Phi", "DeltaConf",
    "M-Ratio", "AUC2-Ratio", "Gamma-Ratio", "Phi-Ratio", "DeltaConf-Ratio",
    "M-Diff", "AUC2-Diff", "Gamma-Diff", "Phi-Diff", "DeltaConf-Diff",
    "metaNoise", "metaUncertainty", "d'", "c", "mean_conf",
    "logL", "AIC", "BIC", "AICc", "k", "n",
]
N_MEAS = 26

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
print("Data ready:", n_trials, "trials,", n_ratings, "ratings")
"""),

    md("""## Block 1 — Metacognitive sensitivity (indices 0–4)

These five measures ask: *how well does confidence track accuracy?*"""),
    code("""\
meas = stdpy.compute_all_measures(stim, resp, conf, n_ratings=n_ratings)

labels = ["meta-d'", "AUC2", "Gamma", "Phi", "DeltaConf"]
for i, name in enumerate(labels):
    print(f"  {name:<12} = {meas[i]:.4f}")
"""),

    md("""## Block 2 & 3 — Efficiency ratios and differences (indices 5–14)

Normalise observed metacognition by the *expected* performance of an ideal
observer with the same d'. Removes spurious dependence on task difficulty."""),
    code("""\
nr_s1, nr_s2 = stdpy.trials_to_counts(stim, resp, conf, n_ratings=n_ratings)

expected   = stdpy.sdt_expect_conf(nr_s1, nr_s2)
nr_s1_exp  = np.array(expected["nR_S1_exp"])
nr_s2_exp  = np.array(expected["nR_S2_exp"])

auc2_obs = stdpy.compute_type2_auc(nr_s1, nr_s2)
auc2_exp = stdpy.compute_type2_auc(nr_s1_exp, nr_s2_exp)

print(f"AUC2 observed = {auc2_obs:.4f}")
print(f"AUC2 ideal    = {auc2_exp:.4f}")
print(f"AUC2-Ratio    = {meas[6]:.4f}  (obs / ideal)")
print(f"AUC2-Diff     = {meas[11]:.4f} (obs − ideal)")
"""),

    md("## Individual measure functions"),
    code("""\
gamma = stdpy.compute_gamma(nr_s1, nr_s2)
phi   = stdpy.compute_phi(nr_s1, nr_s2)
dc    = stdpy.compute_delta_conf(nr_s1, nr_s2)

print(f"Gamma      = {gamma:.4f}")
print(f"Phi        = {phi:.4f}")
print(f"DeltaConf  = {dc['delta_conf']:.4f}")

result = stdpy.fit_meta_d_mle(nr_s1, nr_s2)
print(f"meta_da    = {result['meta_da']:.4f}")
print(f"M_ratio    = {result['M_ratio']:.4f}")
"""),

    md("## Block 4 — Meta-noise and meta-uncertainty (indices 15–16)"),
    code("""\
noise_res = stdpy.compute_meta_noise(stim, resp, conf, n_ratings=n_ratings)
uncert    = stdpy.compute_meta_uncertainty(stim, resp, conf, n_ratings=n_ratings)

print(f"metaNoise       = {noise_res['meta_noise']:.4f}")
print(f"metaUncertainty = {uncert:.4f}")
"""),

    md("## Full 26-measure summary"),
    code("""\
print(f"{'Index':<6} {'Measure':<20} {'Value':>10}")
print("-" * 40)
for i, (name, val) in enumerate(zip(MEASURE_NAMES, meas)):
    vstr = f"{val:10.4f}" if not np.isnan(val) else "       NaN"
    print(f"[{i:2d}]   {name:<20} {vstr}")
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# Tutorial 3 — Statistical Inference
# ─────────────────────────────────────────────────────────────────────────────
TUTORIALS["03_statistical_inference"] = [
    md("""# Tutorial 3 — Statistical Inference

Bootstrap confidence intervals, permutation tests, and group-level summaries
using `metasignal.analysis`."""),

    md("## Setup — simulate a multi-participant experiment"),
    code("""\
import numpy as np
from metasignal import stdpy
from metasignal.analysis import bootstrap_measure, permutation_test, group_summary

MEASURE_NAMES = [
    "meta-d'", "AUC2", "Gamma", "Phi", "DeltaConf",
    "M-Ratio", "AUC2-Ratio", "Gamma-Ratio", "Phi-Ratio", "DeltaConf-Ratio",
    "M-Diff", "AUC2-Diff", "Gamma-Diff", "Phi-Diff", "DeltaConf-Diff",
    "metaNoise", "metaUncertainty", "d'", "c", "mean_conf",
    "logL", "AIC", "BIC", "AICc", "k", "n",
]

n_ratings = 4

def simulate_participant(seed, accuracy=0.78, n_trials=200):
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

participants = [simulate_participant(i) for i in range(20)]
print(f"Simulated {len(participants)} participants")
"""),

    md("""## 1. Group-level summary

`group_summary` runs `compute_all_measures` for every participant and returns
group-level descriptive statistics."""),
    code("""\
summary = group_summary(participants, n_ratings=n_ratings)

print(f"{'Measure':<20} {'Mean':>8} {'SEM':>8} {'n':>5}")
print("-" * 46)
for name, mean, sem, n in zip(
    summary["labels"], summary["mean"], summary["sem"], summary["n_valid"]
):
    if not np.isnan(mean):
        print(f"{name:<20} {mean:8.3f} {sem:8.3f} {n:5d}")
"""),

    md("""## 2. Bootstrap confidence intervals

Resample trials with replacement to estimate a CI for one measure."""),
    code("""\
stim, resp, conf = participants[0]

lo, hi = bootstrap_measure(
    stim, resp, conf,
    n_ratings=n_ratings,
    measure_index=5,        # M-Ratio
    n_boot=1000,
    ci=0.95,
    rng=np.random.default_rng(0),
)
print(f"M-Ratio 95% CI: [{lo:.3f}, {hi:.3f}]")
"""),
    code("""\
INDICES = {"meta-d'": 0, "AUC2": 1, "M-Ratio": 5, "d'": 17}

print(f"{'Measure':<12} {'95% CI':<25}")
print("-" * 38)
for name, idx in INDICES.items():
    lo, hi = bootstrap_measure(
        stim, resp, conf,
        n_ratings=n_ratings,
        measure_index=idx,
        n_boot=500,
        rng=np.random.default_rng(idx),
    )
    print(f"{name:<12}  [{lo:.3f}, {hi:.3f}]")
"""),

    md("""## 3. Permutation test — comparing two conditions

Tests whether two sets of trials differ on a measure by shuffling condition labels."""),
    code("""\
stim_a, resp_a, conf_a = simulate_participant(0, accuracy=0.80)

# Condition B: same accuracy, but random confidence (no metacognition)
rng_b = np.random.default_rng(200)
stim_b = rng_b.choice([0, 1], 200)
resp_b = np.where(rng_b.random(200) < 0.80, stim_b, 1 - stim_b)
conf_b = rng_b.integers(1, n_ratings + 1, 200)

p_val, obs_diff = permutation_test(
    stim_a, resp_a, conf_a,
    stim_b, resp_b, conf_b,
    n_ratings=n_ratings,
    measure_index=1,        # AUC2
    n_perm=1000,
    rng=np.random.default_rng(42),
)
print(f"AUC2 observed difference (A − B): {obs_diff:.3f}")
print(f"Two-sided p-value:                {p_val:.4f}")
"""),

    md("## 4. One-sample t-test against zero"),
    code("""\
from scipy import stats

individual = summary["individual"]
m_ratio = individual[:, 5]
valid   = m_ratio[~np.isnan(m_ratio)]

t, p = stats.ttest_1samp(valid, popmean=1.0)
d    = t / np.sqrt(len(valid))

print(f"M-Ratio vs 1.0:  t({len(valid)-1}) = {t:.3f}, p = {p:.4f}, Cohen's d = {d:.3f}")
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# Tutorial 4 — Difficulty Dependence
# ─────────────────────────────────────────────────────────────────────────────
TUTORIALS["04_difficulty_dependence"] = [
    md("""# Tutorial 4 — Difficulty Dependence

Tests whether each measure changes as task difficulty changes. A good metacognitive
measure should be difficulty-independent. Replicates Rahnev (2025) Supp Tables 3–5."""),

    md("## Setup"),
    code("""\
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from metasignal import stdpy

MEASURE_NAMES = [
    "meta-d'", "AUC2", "Gamma", "Phi", "DeltaConf",
    "M-Ratio", "AUC2-Ratio", "Gamma-Ratio", "Phi-Ratio", "DeltaConf-Ratio",
    "M-Diff", "AUC2-Diff", "Gamma-Diff", "Phi-Diff", "DeltaConf-Diff",
    "metaNoise", "metaUncertainty", "d'", "c", "mean_conf",
    "logL", "AIC", "BIC", "AICc", "k", "n",
]
N_MEAS = 26
n_ratings = 4
difficulty = (0.65, 0.75, 0.85)   # hard, medium, easy

def simulate_subject(seed, difficulty_levels=(0.65, 0.75, 0.85), n_per_level=80):
    r = np.random.default_rng(seed)
    trials = []
    for acc in difficulty_levels:
        stim    = r.choice([0, 1], n_per_level)
        resp    = np.where(r.random(n_per_level) < acc, stim, 1 - stim)
        correct = stim == resp
        conf    = np.where(
            correct,
            r.integers(3, n_ratings + 1, n_per_level),
            r.integers(1, 3, n_per_level),
        )
        trials.append((stim, resp, conf))
    return trials

n_subjects = 20
dataset = [simulate_subject(i) for i in range(n_subjects)]
print(f"Simulated {n_subjects} subjects × {len(difficulty)} difficulty levels")
"""),

    md("## Compute measures per difficulty level"),
    code("""\
n_levels = len(difficulty)
raw = np.full((n_subjects, n_levels, N_MEAS), np.nan)

for s_idx, subject_trials in enumerate(dataset):
    for lv_idx, (stim, resp, conf) in enumerate(subject_trials):
        raw[s_idx, lv_idx] = stdpy.compute_all_measures(
            stim, resp, conf, n_ratings=n_ratings
        )

print("Computed array shape:", raw.shape)
"""),

    md("""## 3-SD outlier removal

Matches MATLAB `ana_taskPerformance.m`: values beyond 3 SD per measure/level
are set to NaN, then propagated across all levels for that subject."""),
    code("""\
def remove_3sd_outliers(arr):
    out = arr.copy()
    _, n_lev, n_meas = out.shape
    for m in range(n_meas):
        for lv in range(n_lev):
            col = out[:, lv, m]
            mu, sd = np.nanmean(col), np.nanstd(col, ddof=1)
            if not np.isnan(mu) and sd > 0:
                out[(col < mu - 3*sd) | (col > mu + 3*sd), lv, m] = np.nan
        bad = np.isnan(out[:, :, m]).any(axis=1)
        out[bad, :, m] = np.nan
    return out

clean = remove_3sd_outliers(raw)
removed = int(np.sum(np.isnan(clean) & ~np.isnan(raw)))
print(f"Values set to NaN by 3-SD removal: {removed}")
"""),

    md("## One-sample t-test: easy − hard"),
    code("""\
def ttest_1samp(data):
    x = np.asarray(data, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan
    t, p = stats.ttest_1samp(x, 0)
    return t, n - 1, p, t / np.sqrt(n)

delta = clean[:, 2, :] - clean[:, 0, :]   # easy − hard

print(f"{'Measure':<20} {'t':>8} {'p':>10} {'d':>8} {'sig':>4}")
print("-" * 55)
for m, name in enumerate(MEASURE_NAMES):
    t, df, p, d = ttest_1samp(delta[:, m])
    if np.isnan(t):
        continue
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{name:<20} {t:8.3f} {p:10.4f} {d:8.3f} {stars:>4}")
"""),

    md("## Visualise difficulty effect"),
    code("""\
fig, axes = plt.subplots(4, 5, figsize=(16, 10), sharex=True)
axes = axes.flatten()
level_labels = ["Hard", "Med", "Easy"]

for m, (name, ax) in enumerate(zip(MEASURE_NAMES, axes)):
    col = clean[:, :, m]
    means = np.nanmean(col, axis=0)
    n_ok  = np.sum(~np.isnan(col), axis=0)
    sems  = np.nanstd(col, axis=0, ddof=1) / np.sqrt(np.maximum(n_ok, 1))
    ax.errorbar(range(n_levels), means, yerr=sems, marker="o", capsize=4, color="#0072b2")
    ax.set_title(name, fontsize=8, fontweight="bold")
    ax.set_xticks(range(n_levels))
    ax.set_xticklabels(level_labels, fontsize=7)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")

plt.suptitle("Effect of Difficulty on 20 Metacognitive Measures", fontsize=12)
plt.tight_layout()
plt.show()
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# Tutorial 5 — Metacognitive Bias
# ─────────────────────────────────────────────────────────────────────────────
TUTORIALS["05_metacognitive_bias"] = [
    md("""# Tutorial 5 — Metacognitive Bias

Tests whether each measure is sensitive to confidence *bias* — a systematic
tendency to use high or low ratings regardless of accuracy. Uses the Xue et al.
(2021) recoding method from Rahnev (2025) Supp Tables 6–8."""),

    md("## Setup"),
    code("""\
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from metasignal import stdpy

MEASURE_NAMES = [
    "meta-d'", "AUC2", "Gamma", "Phi", "DeltaConf",
    "M-Ratio", "AUC2-Ratio", "Gamma-Ratio", "Phi-Ratio", "DeltaConf-Ratio",
    "M-Diff", "AUC2-Diff", "Gamma-Diff", "Phi-Diff", "DeltaConf-Diff",
    "metaNoise", "metaUncertainty", "d'", "c", "mean_conf",
    "logL", "AIC", "BIC", "AICc", "k", "n",
]
N_MEAS = 26
n_ratings = 4

def simulate_subject(seed, n_trials=300, accuracy=0.78):
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
dataset    = [simulate_subject(i) for i in range(n_subjects)]
print(f"Simulated {n_subjects} subjects")
"""),

    md("""## The Xue recoding function

| Recode | Effect | Mechanism |
| --- | --- | --- |
| 1 | High-confidence bias | Subtract 1 from all ratings; bump minimum up by 1 |
| 2 | Low-confidence bias  | Replace maximum with max − 1 |"""),
    code("""\
def xue_recode(conf, rtype):
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

ex = np.array([1, 2, 3, 4, 4, 3, 2, 1])
print("Original : ", ex)
print("Recode 1 : ", xue_recode(ex, 1).astype(int))
print("Recode 2 : ", xue_recode(ex, 2).astype(int))
"""),

    md("## Compute measures under both recodings"),
    code("""\
n_ratings_rc = n_ratings - 1
bias = np.full((n_subjects, 2, N_MEAS), np.nan)

for s_idx, (stim, resp, conf) in enumerate(dataset):
    for rt in (1, 2):
        conf_rc = xue_recode(conf, rt)
        valid   = ~np.isnan(conf_rc)
        bias[s_idx, rt - 1] = stdpy.compute_all_measures(
            stim[valid], resp[valid], conf_rc[valid].astype(int),
            n_ratings=n_ratings_rc,
        )

print("Bias array shape:", bias.shape)
"""),

    md("## Test recode2 − recode1 against zero"),
    code("""\
SKIP = {"d'", "c"}

def ttest_1samp(data):
    x = np.asarray(data, float)[~np.isnan(np.asarray(data, float))]
    n = len(x)
    if n < 2: return np.nan, np.nan, np.nan, np.nan
    t, p = stats.ttest_1samp(x, 0)
    return t, n - 1, p, t / np.sqrt(n)

delta = bias[:, 1, :] - bias[:, 0, :]

print(f"{'Measure':<20} {'t':>8} {'p':>10} {'d':>9} {'sig':>4}")
print("-" * 56)
for m, name in enumerate(MEASURE_NAMES):
    if name in SKIP: continue
    t, df, p, d = ttest_1samp(delta[:, m])
    if np.isnan(t): continue
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{name:<20} {t:8.3f} {p:10.4f} {d:9.3f} {stars:>4}")
"""),

    md("## Visualise bias effect"),
    code("""\
means = np.nanmean(delta, axis=0)
n_ok  = np.sum(~np.isnan(delta), axis=0)
sems  = np.nanstd(delta, axis=0, ddof=1) / np.sqrt(np.maximum(n_ok, 1))

ps = [ttest_1samp(delta[:, m])[2] for m in range(N_MEAS)]
colors = ["#d55e00" if (p is not None and not np.isnan(p) and p < 0.05) else "#999999"
          for p in ps]

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(N_MEAS)
ax.bar(x, means, yerr=sems, color=colors, alpha=0.85, capsize=3)
ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(MEASURE_NAMES, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Recode 2 − Recode 1 ± SEM")
ax.set_title("Metacognitive Bias Sensitivity  (orange = significant p < 0.05)")
plt.tight_layout()
plt.show()
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# Tutorial 6 — Split-Half Reliability & Precision
# ─────────────────────────────────────────────────────────────────────────────
TUTORIALS["06_split_half_reliability"] = [
    md("""# Tutorial 6 — Split-Half Reliability & Precision

**Split-half reliability**: does a measure give consistent values on independent
halves of the data? Uses Spearman-Brown correction.

**Precision**: how quickly does a measure degrade when confidence ratings are
artificially corrupted toward the anti-metacognitive direction?"""),

    md("## Setup"),
    code("""\
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from metasignal import stdpy

MEASURE_NAMES = [
    "meta-d'", "AUC2", "Gamma", "Phi", "DeltaConf",
    "M-Ratio", "AUC2-Ratio", "Gamma-Ratio", "Phi-Ratio", "DeltaConf-Ratio",
    "M-Diff", "AUC2-Diff", "Gamma-Diff", "Phi-Diff", "DeltaConf-Diff",
    "metaNoise", "metaUncertainty", "d'", "c", "mean_conf",
    "logL", "AIC", "BIC", "AICc", "k", "n",
]
N_MEAS = 26
n_ratings = 4

def simulate_subject(seed, n_trials=400, accuracy=0.78):
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
dataset    = [simulate_subject(i) for i in range(n_subjects)]
print(f"Simulated {n_subjects} subjects, 400 trials each")
"""),

    md(r"""## Split-half reliability

Split each subject into **odd** and **even** trials, compute measures on each
half, then apply Spearman-Brown correction: $r_{SB} = \frac{2r}{1+r}$"""),
    code("""\
split = np.full((n_subjects, 2, N_MEAS), np.nan)

for s_idx, (stim, resp, conf) in enumerate(dataset):
    idx_odd  = np.arange(0, len(stim), 2)
    idx_even = np.arange(1, len(stim), 2)
    for half_idx, idx in enumerate([idx_odd, idx_even]):
        split[s_idx, half_idx] = stdpy.compute_all_measures(
            stim[idx], resp[idx], conf[idx], n_ratings=n_ratings
        )

print("Split array shape:", split.shape)
"""),
    code("""\
def spearman_brown(r):
    return 2 * r / (1 + r) if not np.isnan(r) else np.nan

print(f"{'Measure':<20} {'Pearson r':>10} {'SB-corrected':>14}")
print("-" * 48)
sb_vals = []
for m, name in enumerate(MEASURE_NAMES):
    x, y = split[:, 0, m], split[:, 1, m]
    ok   = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() >= 5:
        r, _ = pearsonr(x[ok], y[ok])
        sb   = spearman_brown(r)
        sb_vals.append(sb)
        print(f"{name:<20} {r:10.3f} {sb:14.3f}")
    else:
        sb_vals.append(np.nan)
        print(f"{name:<20} {'NaN':>10} {'NaN':>14}")
"""),

    md("### Reliability bar chart"),
    code("""\
fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(N_MEAS)
colors = ["#0072b2" if (v is not None and not np.isnan(v) and v >= 0.7)
          else "#d55e00" if (v is not None and not np.isnan(v))
          else "#cccccc" for v in sb_vals]

ax.bar(x, [v if v is not None and not np.isnan(v) else 0 for v in sb_vals],
       color=colors, alpha=0.85)
ax.axhline(0.7, color="k", linewidth=1.0, linestyle="--", label="0.7 threshold")
ax.set_xticks(x)
ax.set_xticklabels(MEASURE_NAMES, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Spearman-Brown reliability")
ax.set_ylim(-0.1, 1.1)
ax.set_title("Split-Half Reliability  (blue ≥ 0.7, orange < 0.7)")
ax.legend()
plt.tight_layout()
plt.show()
"""),

    md("""## Precision under confidence corruption

Shift a proportion of trial confidence ratings in the anti-metacognitive
direction and measure how much each measure degrades."""),
    code("""\
def corrupt_confidence(stim, resp, conf, proportion, rng):
    c       = conf.copy().astype(float)
    correct = (stim == resp)
    n       = len(c)
    n_c     = int(np.round(proportion * n))
    idx     = rng.choice(n, size=n_c, replace=False)
    for i in idx:
        c[i] = max(1, c[i] - 1) if correct[i] else min(n_ratings, c[i] + 1)
    return c

PROPORTIONS = [0.0, 0.02, 0.04, 0.06]

base    = np.array([stdpy.compute_all_measures(s, r, c, n_ratings=n_ratings)
                    for s, r, c in dataset])
base_sd = np.nanstd(base, axis=0, ddof=1)
base_sd[base_sd == 0] = np.nan

drops = np.zeros((len(PROPORTIONS), N_MEAS))
for pi, prop in enumerate(PROPORTIONS):
    if prop == 0.0:
        continue
    corrupted = np.array([
        stdpy.compute_all_measures(
            stim, resp,
            corrupt_confidence(stim, resp, conf, prop,
                               np.random.default_rng(s_idx * 100 + pi)).astype(int),
            n_ratings=n_ratings,
        )
        for s_idx, (stim, resp, conf) in enumerate(dataset)
    ])
    drops[pi] = np.nanmean((base - corrupted) / base_sd[np.newaxis, :], axis=0)

print("Precision drops computed for proportions:", PROPORTIONS)
"""),
    code("""\
fig, axes = plt.subplots(4, 5, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for m, (name, ax) in enumerate(zip(MEASURE_NAMES, axes)):
    ax.plot(PROPORTIONS, drops[:, m], marker="o", color="#d55e00")
    ax.set_title(name, fontsize=8, fontweight="bold")
    ax.set_xlabel("Corrupted %", fontsize=6)
    ax.set_ylabel("Norm. drop", fontsize=6)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xticks(PROPORTIONS)
    ax.set_xticklabels([f"{int(p*100)}%" for p in PROPORTIONS], fontsize=7)

plt.suptitle("Precision: Normalised Drop Under Confidence Corruption", fontsize=12)
plt.tight_layout()
plt.show()
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# Write all notebooks
# ─────────────────────────────────────────────────────────────────────────────
import pathlib

OUT = pathlib.Path(__file__).parent

for name, cells in TUTORIALS.items():
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13",
        },
    }
    path = OUT / f"{name}.ipynb"
    nbformat.write(nb, path)
    print(f"Written: {path.name}")
