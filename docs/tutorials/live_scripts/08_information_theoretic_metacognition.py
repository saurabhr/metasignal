# %% [markdown]
# # Tutorial 8 — Information-Theoretic Metacognition (`metasignal.itmc`)
#
# `itmc` implements the information-theoretic metacognition measures of
# [Dayan (2023)](https://doi.org/10.1162/opmi_a_00091), which treat
# metacognitive sensitivity as the mutual information between accuracy and
# confidence rather than assuming a Gaussian SDT model. `meta_Ir1_acc` and
# `RMI` extend the framework via
# [Rausch et al. (2025), statConfR](https://doi.org/10.21105/joss.06966).
#
# > **Experimental** — `itmc` is a pre-1.0 component; see the
# > [API Reference](../api.md#itmc-information-theoretic-metacognition-experimental)
# > for validation details against the R `statConfR` package.
#
# | Measure | Meaning |
# | --- | --- |
# | `meta_I` | Mutual information between confidence and accuracy (bits) |
# | `meta_Ir1` | Efficiency relative to an ideal Gaussian observer with the same d′ |
# | `meta_Ir1_acc` | Same, normalised by observed accuracy instead of d′ |
# | `meta_Ir2` | Fraction of the maximum possible information, range [0, 1] |
# | `RMI` | Range-normalised information; 0 = worst, 1 = best metacognition |
#
# | `backend=` | Best for |
# | --- | --- |
# | `'simple'` (default) | Fast exploration; `meta_I`, `meta_Ir1`, `meta_Ir2` |
# | `'statconfr'` | Exact reproduction of statConfR results; required for `RMI` |

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metasignal import stdpy
from metasignal.itmc import (
    meta_I, meta_Ir1, meta_Ir1_acc, meta_Ir2, RMI,
    estimate_meta_I, fit_group, MEASURE_COLS,
)

COLORS = {"simple": "#4C72B0", "statconfr": "#DD8452"}

# %% [markdown]
# ## 1. Single-participant quick start
#
# Simulate one participant and compute all five measures with both backends.

# %%
rng = np.random.default_rng(0)
df = stdpy.trialSimulation(d=1.5, metad=1.2, nTrials=600, rng=rng)

stim = df["Stimuli"].to_numpy(int)
resp = df["Responses"].to_numpy(int)
conf = df["Confidence"].to_numpy(int)

print(f"Trials: {len(stim)}   Accuracy: {(stim == resp).mean():.2f}")

pd.DataFrame([
    {
        "backend": backend,
        "meta_I": meta_I(stim, resp, conf, backend=backend),
        "meta_Ir1": meta_Ir1(stim, resp, conf, backend=backend),
        "meta_Ir1_acc": meta_Ir1_acc(stim, resp, conf, backend=backend),
        "meta_Ir2": meta_Ir2(stim, resp, conf, backend=backend),
        "RMI": RMI(stim, resp, conf, backend=backend),
    }
    for backend in ["simple", "statconfr"]
]).set_index("backend").round(4)

# %% [markdown]
# ## 2. meta-I is ≈ 0 for uninformative confidence
#
# Shuffling confidence breaks the accuracy-confidence link, so meta-I should
# collapse to approximately zero.

# %%
rng2 = np.random.default_rng(1)
conf_random = rng2.integers(1, 5, size=len(stim))

mi_real = meta_I(stim, resp, conf, backend="statconfr")
mi_random = meta_I(stim, resp, conf_random, backend="statconfr")

print(f"meta-I (informative confidence): {mi_real:.4f} bits")
print(f"meta-I (random confidence):      {mi_random:.4f} bits  <- should be ~ 0")

# %% [markdown]
# ## 3. meta-Ir1 tracks meta-d′ at fixed d′
#
# `meta_Ir1` measures efficiency relative to an ideal Gaussian observer.
# Values below 1 indicate sub-ideal metacognition.

# %%
dprime_fixed = 1.5
metad_levels = [0.5, 0.8, 1.0, 1.2, 1.5]

ir1_vals = []
for md in metad_levels:
    d = stdpy.trialSimulation(d=dprime_fixed, metad=md, nTrials=2000,
                               rng=np.random.default_rng(99))
    ir1_vals.append(meta_Ir1(d["Stimuli"].to_numpy(int), d["Responses"].to_numpy(int),
                              d["Confidence"].to_numpy(int), backend="statconfr"))

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(metad_levels, ir1_vals, "o-", color=COLORS["statconfr"])
ax.axhline(1.0, ls="--", color="gray", lw=0.8, label="Ideal observer")
ax.set_xlabel("meta-d'")
ax.set_ylabel("meta-Ir1")
ax.set_title(f"meta-Ir1 vs meta-d' (d' fixed at {dprime_fixed})")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. meta-Ir2 and RMI — bounded [0, 1] alternatives
#
# `meta_Ir2` is the fraction of the *maximum possible* information
# transmitted. `RMI` rescales between the analytic min/max I(S;R) at the
# observed accuracy level (Dayan 2023, Theorem 3) — always use
# `backend='statconfr'` for RMI, the `'simple'`-backend bounds are not tight.

# %%
print(f"meta-Ir2 (simple):    {meta_Ir2(stim, resp, conf, backend='simple'):.4f}")
print(f"meta-Ir2 (statconfr): {meta_Ir2(stim, resp, conf, backend='statconfr'):.4f}")
print(f"RMI      (statconfr): {RMI(stim, resp, conf, backend='statconfr'):.4f}")

# %% [markdown]
# ## 5. Bias correction for small samples
#
# meta-I has a positive sampling bias at low N (mutual information is always
# ≥ 0, even under the null). `bias_correction=True` subtracts an estimated
# bias — permutation of accuracy labels for `'simple'`, multinomial
# resampling of the contingency table for `'statconfr'` (matching
# statConfR's `bias_reduction=TRUE`).

# %%
n_trials_list = [50, 100, 200, 400, 800]
raw_vals, bc_vals = [], []
for n in n_trials_list:
    d = stdpy.trialSimulation(d=1.0, metad=0.3, nTrials=n, rng=np.random.default_rng(7))
    s = d["Stimuli"].to_numpy(int)
    r = d["Responses"].to_numpy(int)
    c = d["Confidence"].to_numpy(int)
    raw_vals.append(meta_I(s, r, c, backend="simple", bias_correction=False))
    bc_vals.append(meta_I(s, r, c, backend="simple", bias_correction=True, seed=0))

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(n_trials_list, raw_vals, "o-", label="Raw meta-I", color=COLORS["simple"])
ax.plot(n_trials_list, bc_vals, "s--", label="Bias-corrected", color="#55A868")
ax.set_xlabel("N trials")
ax.set_ylabel("meta-I (bits)")
ax.set_title("Bias correction effect at small N")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Group-level analysis with `estimate_meta_I`
#
# `estimate_meta_I(df)` mirrors statConfR's `estimateMetaI()`: one row per
# participant, all five measures. Expected columns: `stimulus`, `response`,
# `rating`, `participant` (overridable via `*_col=`).

# %%
records = []
for pid in range(10):
    d = stdpy.trialSimulation(d=1.2 + 0.1 * pid, metad=0.8 + 0.05 * pid, nTrials=400,
                               rng=np.random.default_rng(pid))
    d = d.rename(columns={"Stimuli": "stimulus", "Responses": "response", "Confidence": "rating"})
    d["participant"] = f"p{pid:02d}"
    records.append(d[["stimulus", "response", "rating", "participant"]])

df_group = pd.concat(records, ignore_index=True)
result = estimate_meta_I(df_group, backend="statconfr")
result.round(4)

# %%
fig, ax = plt.subplots(figsize=(5, 4))
ax.barh(result["participant"], result["meta_I"], color=COLORS["statconfr"], height=0.6)
ax.set_xlabel("meta-I (bits)")
ax.set_title("meta-I by participant")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Factorial designs with `fit_group`
#
# `fit_group` accepts `subject`, `within`, and `between` columns — including
# lists of multiple factors — and returns one row per cell.

# %%
records2 = []
for pid in range(8):
    for cond, label in [(0, "easy"), (1, "hard")]:
        d_level = 2.0 if cond == 0 else 1.0
        p = stdpy.trialSimulation(d=d_level, metad=d_level * 0.8, nTrials=300,
                                   rng=np.random.default_rng(pid * 10 + cond))
        p = p.rename(columns={"Stimuli": "stimulus", "Responses": "response", "Confidence": "rating"})
        p["subject"] = f"p{pid:02d}"
        p["condition"] = label
        p["group"] = "control" if pid < 4 else "patient"
        records2.append(p[["stimulus", "response", "rating", "subject", "condition", "group"]])

df2 = pd.concat(records2, ignore_index=True)

out = fit_group(df2, stimuli="stimulus", responses="response", confidence="rating",
                 subject="subject", within="condition", between="group", backend="statconfr")
print(f"Shape: {out.shape}  (one row per subject x condition)")
out.round(4)

# %% [markdown]
# Use `measures=` to request only specific columns, e.g. `measures=['meta_I', 'RMI']`.
# All available measures are listed in `MEASURE_COLS`.

# %%
print("All available measures:", MEASURE_COLS)

# %% [markdown]
# ## 8. Backend comparison
#
# Both backends closely agree on meta-I, meta-Ir1 and meta-Ir2. The main
# differences are in RMI (analytic bounds vs. approximation) and the
# Gaussian reference for meta-Ir1 (numerical integration vs. Monte Carlo).

# %%
both = pd.merge(
    estimate_meta_I(df_group, backend="simple").add_suffix("_simple")
        .rename(columns={"participant_simple": "participant"}),
    estimate_meta_I(df_group, backend="statconfr").add_suffix("_sc")
        .rename(columns={"participant_sc": "participant"}),
    on="participant",
)

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, m in zip(axes, ["meta_I", "meta_Ir2", "RMI"]):
    x, y = both[f"{m}_simple"], both[f"{m}_sc"]
    ax.scatter(x, y, color="#555", zorder=3)
    mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=0.8)
    ax.set_xlabel("simple")
    ax.set_ylabel("statconfr")
    ax.set_title(m)
plt.suptitle("Backend comparison across 10 participants", y=1.02)
plt.tight_layout()
plt.show()

for m in ["meta_I", "meta_Ir1", "meta_Ir2"]:
    r = both[f"{m}_simple"].corr(both[f"{m}_sc"])
    print(f"  {m}: r = {r:.4f}")

# %% [markdown]
# ## 9. IT measures vs. traditional meta-d′ measures
#
# IT measures are model-free — they don't assume a Gaussian SDT model.
# `meta_I` correlates with M-ratio across participants but captures
# different variance.

# %%
df_group_std = df_group.rename(
    columns={"stimulus": "Stimuli", "response": "Responses", "rating": "Confidence"})

trad = stdpy.fit_group(df_group_std, subject="participant", nRatings=4,
                        measures=["dprime", "meta_d", "M_ratio"])
it = estimate_meta_I(df_group, backend="statconfr")
merged = pd.merge(trad, it, on="participant")

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
axes[0].scatter(merged["M_ratio"], merged["meta_I"], color=COLORS["statconfr"])
axes[0].set_xlabel("M-ratio (meta-d' / d')")
axes[0].set_ylabel("meta-I (bits)")
axes[0].set_title("meta-I vs M-ratio")

axes[1].scatter(merged["M_ratio"], merged["RMI"], color=COLORS["statconfr"])
axes[1].set_xlabel("M-ratio")
axes[1].set_ylabel("RMI")
axes[1].set_title("RMI vs M-ratio")

plt.tight_layout()
plt.show()

print(f"Pearson r(M-ratio, meta-I) = {merged['M_ratio'].corr(merged['meta_I']):.3f}")
print(f"Pearson r(M-ratio, RMI)    = {merged['M_ratio'].corr(merged['RMI']):.3f}")

# %% [markdown]
# ## Summary
#
# | Task | Function |
# | --- | --- |
# | Single measure, one participant | `meta_I(stim, resp, conf, backend=)` |
# | All five measures, one participant | `estimate_meta_I(df, backend=, bias_correction=)` |
# | All five measures, factorial design | `fit_group(df, subject=, within=, between=, backend=)` |
#
# **Backend choice** — `'simple'` for exploration/speed; `'statconfr'` to
# reproduce statConfR results or whenever RMI is needed; add
# `bias_correction=True` for small N / publication.
#
# ### References
# - Dayan P (2023). Metacognitive Information Theory. *Open Mind*, 7, 392–411. doi:10.1162/opmi_a_00091
# - Rausch M et al. (2025). statConfR. *JOSS*. doi:10.21105/joss.06966
