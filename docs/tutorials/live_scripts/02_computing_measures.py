# %% [markdown]
# # Tutorial 2 — Computing All 26 Measures
#
# A detailed walkthrough of each block of the 26-measure array, plus how
# to call individual measures directly.

# %% [markdown]
# ## Setup

# %%
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

# %% [markdown]
# ## Block 1 — Metacognitive sensitivity (indices 0–4)
#
# These five measures ask: *how well does confidence track accuracy?*

# %%
meas = stdpy.compute_all_measures(stim, resp, conf, n_ratings=n_ratings)

labels = ["meta-d'", "AUC2", "Gamma", "Phi", "DeltaConf"]
for i, name in enumerate(labels):
    print(f"  {name:<12} = {meas[i]:.4f}")

# %% [markdown]
# ## Block 2 & 3 — Efficiency ratios and differences (indices 5–14)
#
# Normalise observed metacognition by the *expected* performance of an ideal
# observer with the same d'. Removes spurious dependence on task difficulty.

# %%
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

# %% [markdown]
# ## Individual measure functions

# %%
gamma = stdpy.compute_gamma(nr_s1, nr_s2)
phi   = stdpy.compute_phi(nr_s1, nr_s2)
dc    = stdpy.compute_delta_conf(nr_s1, nr_s2)

print(f"Gamma      = {gamma:.4f}")
print(f"Phi        = {phi:.4f}")
print(f"DeltaConf  = {dc['delta_conf']:.4f}")

result = stdpy.fit_meta_d_mle(nr_s1, nr_s2)
print(f"meta_da    = {result['meta_da']:.4f}")
print(f"M_ratio    = {result['M_ratio']:.4f}")

# %% [markdown]
# ## Block 4 — Meta-noise and meta-uncertainty (indices 15–16)

# %%
noise_res = stdpy.compute_meta_noise(stim, resp, conf, n_ratings=n_ratings)
uncert    = stdpy.compute_meta_uncertainty(stim, resp, conf, n_ratings=n_ratings)

print(f"metaNoise       = {noise_res['meta_noise']:.4f}")
print(f"metaUncertainty = {uncert:.4f}")

# %% [markdown]
# ## Full 26-measure summary

# %%
print(f"{'Index':<6} {'Measure':<20} {'Value':>10}")
print("-" * 40)
for i, (name, val) in enumerate(zip(MEASURE_NAMES, meas)):
    vstr = f"{val:10.4f}" if not np.isnan(val) else "       NaN"
    print(f"[{i:2d}]   {name:<20} {vstr}")
