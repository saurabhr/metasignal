# %% [markdown]
# # Tutorial 3 — Statistical Inference
#
# Bootstrap confidence intervals, permutation tests, and group-level summaries.
#
# > **Speed note** — `bootstrap_measure` and `permutation_test` in
# > `metasignal.analysis` call `compute_all_measures` internally, which includes
# > slow MLE fits (~3–5 s each) for meta-d', metaNoise, and metaUncertainty.
# > This notebook instead demonstrates the same concepts using the fast
# > non-MLE functions directly so it runs in seconds. In production scripts where
# > runtime is less constrained, use the convenience API shown at the end.

# %% [markdown]
# ## Setup

# %%
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from metasignal import stdpy

n_ratings = 4

def simulate_participant(seed, accuracy=0.78, n_trials=200):
    r = np.random.default_rng(seed)
    stim    = r.choice([0, 1], n_trials)
    resp    = np.where(r.random(n_trials) < accuracy, stim, 1 - stim)
    correct = stim == resp
    conf    = np.where(
        correct, r.integers(3, n_ratings + 1, n_trials),
        r.integers(1, 3, n_trials),
    )
    return stim, resp, conf

# Fast non-MLE measures: compute directly without compute_all_measures
def fast_measures(stim, resp, conf):
    """Return (dprime, AUC2, Gamma, Phi) — all O(n), no MLE."""
    dprime, c, _ = stdpy.compute_sdt_resp(stim, resp)
    nr_s1, nr_s2 = stdpy.trials_to_counts(stim, resp, conf, n_ratings)
    auc2  = stdpy.compute_type2_auc(nr_s1, nr_s2)
    gamma = stdpy.compute_gamma(nr_s1, nr_s2)
    phi   = stdpy.compute_phi(nr_s1, nr_s2)
    return np.array([float(dprime), auc2, gamma, phi])

FAST_LABELS = ["d'", "AUC2", "Gamma", "Phi"]

participants = [simulate_participant(i) for i in range(20)]
print(f"Simulated {len(participants)} participants")

# %% [markdown]
# ## 1. Group-level summary

# %%
# Compute fast measures for every participant
individual = np.array([fast_measures(*p) for p in participants])
print("individual shape:", individual.shape)   # (20, 4)

mean   = np.nanmean(individual, axis=0)
sem    = np.nanstd(individual, axis=0, ddof=1) / np.sqrt(individual.shape[0])

print(f"\n{'Measure':<8} {'Mean':>8} {'SEM':>8}")
print("-" * 28)
for name, m, s in zip(FAST_LABELS, mean, sem):
    print(f"{name:<8} {m:8.3f} {s:8.3f}")

# %%
# metasignal.analysis.group_summary wraps this for all 20 measures.
# Production usage (slower due to MLE fits):
#
#   from metasignal.analysis import group_summary
#   summary = group_summary(participants, n_ratings=n_ratings)
#   print(summary["mean"])    # shape (20,)
#   print(summary["labels"])  # 20 measure names
print("See docstring: metasignal.analysis.group_summary")

# %% [markdown]
# ## 2. Bootstrap confidence interval
#
# Resample trials with replacement and recompute the measure each time.
# The percentile interval of the resampled values is the CI.

# %%
def bootstrap_ci(stim, resp, conf, measure_fn, n_boot=2000, ci=0.95, seed=0):
    """Percentile-bootstrap CI for one scalar measure."""
    rng = np.random.default_rng(seed)
    n   = len(stim)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v   = measure_fn(stim[idx], resp[idx], conf[idx])
        if not np.isnan(v):
            vals.append(v)
    alpha = 1 - ci
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))

def auc2_fn(stim, resp, conf):
    nr_s1, nr_s2 = stdpy.trials_to_counts(stim, resp, conf, n_ratings)
    return stdpy.compute_type2_auc(nr_s1, nr_s2)

stim, resp, conf = participants[0]
lo, hi = bootstrap_ci(stim, resp, conf, auc2_fn, n_boot=2000)
print(f"AUC2 95% CI: [{lo:.3f}, {hi:.3f}]")

# %%
# CIs for all four fast measures
def gamma_fn(s, r, c):
    nr_s1, nr_s2 = stdpy.trials_to_counts(s, r, c, n_ratings)
    return stdpy.compute_gamma(nr_s1, nr_s2)

def phi_fn(s, r, c):
    nr_s1, nr_s2 = stdpy.trials_to_counts(s, r, c, n_ratings)
    return stdpy.compute_phi(nr_s1, nr_s2)

def dprime_fn(s, r, c):
    d, _, _ = stdpy.compute_sdt_resp(s, r)
    return d

fns = [dprime_fn, auc2_fn, gamma_fn, phi_fn]
print(f"{'Measure':<8} {'95% CI':<25}")
print("-" * 34)
for name, fn in zip(FAST_LABELS, fns):
    lo, hi = bootstrap_ci(stim, resp, conf, fn, n_boot=2000, seed=hash(name) % 1000)
    print(f"{name:<8}  [{lo:.3f}, {hi:.3f}]")

# %% [markdown]
# ## 3. Permutation test — two-condition comparison
#
# Pool all trials, shuffle condition labels `n_perm` times, measure the
# difference under each shuffle to build a null distribution.

# %%
def permutation_test(stim_a, resp_a, conf_a, stim_b, resp_b, conf_b,
                     measure_fn, n_perm=5000, seed=0):
    """Two-sided permutation test for a difference in one measure."""
    rng    = np.random.default_rng(seed)
    obs_a  = measure_fn(stim_a, resp_a, conf_a)
    obs_b  = measure_fn(stim_b, resp_b, conf_b)
    obs_diff = obs_a - obs_b

    all_stim = np.concatenate([stim_a, stim_b])
    all_resp = np.concatenate([resp_a, resp_b])
    all_conf = np.concatenate([conf_a, conf_b])
    n_a = len(stim_a)
    n   = len(all_stim)

    null = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        d = measure_fn(all_stim[perm[:n_a]], all_resp[perm[:n_a]], all_conf[perm[:n_a]]) -             measure_fn(all_stim[perm[n_a:]], all_resp[perm[n_a:]], all_conf[perm[n_a:]])
        if not np.isnan(d):
            null.append(d)

    p = float(np.mean(np.abs(null) >= abs(obs_diff))) if null else np.nan
    return p, obs_diff

# Condition A: metacognitive (conf tracks accuracy)
stim_a, resp_a, conf_a = simulate_participant(0, accuracy=0.80)

# Condition B: same accuracy, random confidence (no metacognition)
rng_b = np.random.default_rng(200)
stim_b = rng_b.choice([0, 1], 200)
resp_b = np.where(rng_b.random(200) < 0.80, stim_b, 1 - stim_b)
conf_b = rng_b.integers(1, n_ratings + 1, 200)

p_val, obs_diff = permutation_test(
    stim_a, resp_a, conf_a,
    stim_b, resp_b, conf_b,
    auc2_fn, n_perm=5000, seed=42,
)
print(f"AUC2 observed difference (A − B): {obs_diff:.3f}")
print(f"Two-sided p-value:                {p_val:.4f}")

# %% [markdown]
# ## 4. One-sample t-test across participants

# %%
auc2_vals = individual[:, 1]   # AUC2 for all 20 participants

t, p   = stats.ttest_1samp(auc2_vals[~np.isnan(auc2_vals)], popmean=0.5)
n_ok   = np.sum(~np.isnan(auc2_vals))
d      = t / np.sqrt(n_ok)

print(f"AUC2 vs chance (0.5): t({n_ok-1}) = {t:.3f}, p = {p:.4f}, Cohen's d = {d:.3f}")

# %% [markdown]
# ## Production API
#
# For full 20-measure results (including meta-d', metaNoise, metaUncertainty),
# use the convenience wrappers in `metasignal.analysis`. They are ideal for
# offline scripts with parallelism but will be slow in a notebook:
#
# ```python
# from metasignal.analysis import bootstrap_measure, permutation_test, group_summary
#
# # ~seconds per call — best run with multiprocessing for many participants
# lo, hi = bootstrap_measure(stim, resp, conf, n_ratings=4,
#                            measure_index=0,   # meta-d'
#                            n_boot=2000)
#
# p, diff = permutation_test(stim_a, resp_a, conf_a,
#                            stim_b, resp_b, conf_b,
#                            n_ratings=4, measure_index=5,   # M-Ratio
#                            n_perm=5000)
#
# summary = group_summary(participants, n_ratings=4)
# ```
