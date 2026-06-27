# %% [markdown]
# # Tutorial 1 — Getting Started
#
# This notebook verifies your metasignal installation and walks through the three
# input arrays every function expects.

# %% [markdown]
# ## 1. Verify the install

# %%
import numpy as np
import metasignal
from metasignal import stdpy

print(f"metasignal loaded from: {metasignal.__file__}")

rng  = np.random.default_rng(0)
stim = rng.choice([0, 1], 200)
resp = np.where(rng.random(200) < 0.75, stim, 1 - stim)
conf = rng.integers(1, 5, 200)

meas = stdpy.compute_all_measures(stim, resp, conf, n_ratings=4)
print("Output shape:", meas.shape)   # (20,)

# %% [markdown]
# ## 2. Input format
#
# | Argument | Values | Meaning |
# | --- | --- | --- |
# | `stim` | 0 / 1 | Stimulus category (0 = S1/noise, 1 = S2/signal) |
# | `resp` | 0 / 1 | Participant response |
# | `conf` | 1 … n_ratings | Confidence rating (1 = lowest) |
# | `n_ratings` | int | Total number of confidence categories |

# %% [markdown]
# ## 3. Build a minimal dataset

# %%
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

# %% [markdown]
# ## 4. Type-1 SDT parameters

# %%
dprime, c, ln_beta = stdpy.compute_sdt_resp(stim, resp)
print(f"d'        = {dprime:.3f}")
print(f"criterion = {c:.3f}")

# %% [markdown]
# ## 5. Convert trials to response counts

# %%
nr_s1, nr_s2 = stdpy.trials_to_counts(stim, resp, conf, n_ratings=n_ratings)
print("nr_s1 shape:", nr_s1.shape)
print("nr_s1:", nr_s1)
print("nr_s2:", nr_s2)

# %% [markdown]
# ## 6. Inspect the full 20-element output

# %%
MEASURE_NAMES = [
    "meta-d'", "AUC2", "Gamma", "Phi", "DeltaConf",
    "M-Ratio", "AUC2-Ratio", "Gamma-Ratio", "Phi-Ratio", "DeltaConf-Ratio",
    "M-Diff", "AUC2-Diff", "Gamma-Diff", "Phi-Diff", "DeltaConf-Diff",
    "metaNoise", "metaUncertainty", "d'", "c", "mean_conf",
]

meas = stdpy.compute_all_measures(stim, resp, conf, n_ratings=n_ratings)

for i, (name, val) in enumerate(zip(MEASURE_NAMES, meas)):
    flag = "NaN" if np.isnan(val) else f"{val:.4f}"
    print(f"  [{i:2d}] {name:<20s} = {flag}")
