# Bayesian Analysis

`metasignal.sdtbayes` fits hierarchical Bayesian metacognition models using
[brmspy](https://github.com/kaitumisuuringute-keskus/brmspy) (a Python
interface to the [brms](https://paul-buerkner.github.io/brms/) R package)
and analyses posteriors with [ArviZ](https://python.arviz.org).

See **[Tutorial 7 — Bayesian Hierarchical Meta-d'](tutorials/07_bayesian_hierarchical.ipynb)**
for a hands-on walkthrough using the canonical HMeta-d dataset and a
side-by-side comparison with [metadpy](https://github.com/embodied-computation-group/metadpy).

## Installation

The Bayesian submodule is an optional extra — install it alongside the base package:

```bash
pip install metasignal[sdtbayes]
```

Then run the one-time runtime setup (downloads Stan and installs brms, ~5 min):

```python
from metasignal.sdtbayes import setup_runtime
setup_runtime()
```

## Modelling approaches

| # | Approach | Primary function(s) | Description |
|---|----------|---------------------|-------------|
| 1 | **Ordered logistic** | `fit_hierarchical_metad` | Trial-level; confidence ratings as cumulative outcome. Fast and extensible. |
| 2 | **Two-stage Bayesian** | `fit_two_stage_group` | MLE per participant → Bayesian on log M-ratio. Fast Stage 2; no Stage-1 uncertainty propagation. |
| 3 | **Full HMeta-d** | `fit_full_metad` | Single-stage port of Fleming (2017) JAGS model to Stan via brms `stanvar()`. Identical SDT likelihood to MLE; most accurate. |
| 4 | **Subject-level Bayesian** | `fit_subject_level` | Single-participant model. Matches `metadpy.hmetad(nR_S1, nR_S2)` API and prior structure; runs via brmspy/Stan. |
| 5 | **Robust HMeta-d** | `fit_robust_metad` | Like Approach 3 but with a Student-t hyperprior on log M-ratio. Downweights outlier participants and estimates degrees of freedom. |
| 6 | **Variational inference** | `fit_full_metad_vi`, `fit_robust_metad_vi` | Fast approximate posteriors for Approaches 3 and 5 via Pathfinder or ADVI. |
| 7 | **Beta AUC** | `fit_beta_auc_group` | Non-parametric alternative to meta-d'. Models Type-2 AUC directly with a Beta likelihood; no Gaussian SDT assumption. |
| 8 | **Gaussian mixture** | `fit_mixture_group` | Identifies latent metacognitive subpopulations via a K-component Gaussian mixture on log M-ratio. |
| 9 | **Bivariate hierarchical** | `fit_multivariate_mratio` | Models log M-ratio and d' jointly, estimating their cross-participant correlation. |
| 10 | **Meta-regression** | `fit_two_stage_regression`, `fit_full_metad_regression` | Regresses log M-ratio on participant-level covariates (age, scores, etc.) via either the two-stage or full hierarchical path. |
| 11 | **State-space** | `fit_statespace_metad` | Models how group-level meta-d' evolves across repeated sessions via a random walk with free process noise. |
| 12 | **Within-subject comparison** | `fit_within_subject_comparison` | Two-stage paired model for within-subject designs: same participants in two conditions; participant random intercepts absorb stable individual differences. |

Approaches 1–3, 5, 7, and 9 have matching two-group comparison counterparts:
`fit_group_comparison`, `fit_two_stage_comparison`, `fit_full_metad_comparison`,
`fit_robust_metad_comparison`, `fit_beta_auc_comparison`, `fit_multivariate_mratio_comparison`.

## API Reference

### Approach 1 — Ordered Logistic

::: metasignal.sdtbayes.fit_hierarchical_metad

::: metasignal.sdtbayes.fit_group_comparison

### Approach 2 — Two-Stage Bayesian

::: metasignal.sdtbayes.fit_two_stage_group

::: metasignal.sdtbayes.fit_two_stage_comparison

### Approach 3 — Full HMeta-d

::: metasignal.sdtbayes.fit_full_metad

::: metasignal.sdtbayes.fit_full_metad_comparison

### Approach 4 — Subject-Level Bayesian (metadpy API)

::: metasignal.sdtbayes.fit_subject_level

### Approach 5 — Robust HMeta-d

::: metasignal.sdtbayes.fit_robust_metad

::: metasignal.sdtbayes.fit_robust_metad_comparison

### Approach 6 — Variational Inference

::: metasignal.sdtbayes.fit_full_metad_vi

::: metasignal.sdtbayes.fit_robust_metad_vi

### Approach 7 — Beta Regression on Type-2 AUC

::: metasignal.sdtbayes.fit_beta_auc_group

::: metasignal.sdtbayes.fit_beta_auc_comparison

### Approach 8 — Gaussian Mixture on log M-ratio

::: metasignal.sdtbayes.fit_mixture_group

### Approach 9 — Bivariate Hierarchical Model

::: metasignal.sdtbayes.fit_multivariate_mratio

::: metasignal.sdtbayes.fit_multivariate_mratio_comparison

### Approach 10 — Bayesian Meta-Regression

::: metasignal.sdtbayes.fit_two_stage_regression

::: metasignal.sdtbayes.fit_full_metad_regression

### Approach 11 — State-Space Model

::: metasignal.sdtbayes.fit_statespace_metad

### Approach 12 — Within-Subject Condition Comparison

::: metasignal.sdtbayes.fit_within_subject_comparison

### Runtime and Result Types

::: metasignal.sdtbayes.setup_runtime

::: metasignal.sdtbayes.FitResult

### Diagnostics and Posteriors

::: metasignal.sdtbayes.posterior_summary

::: metasignal.sdtbayes.plot_trace

::: metasignal.sdtbayes.plot_posterior

::: metasignal.sdtbayes.plot_forest

::: metasignal.sdtbayes.convergence_diagnostics

## Quick-start example (Option A — two-stage)

```python
import numpy as np
from metasignal.sdtbayes import (
    fit_two_stage_group,
    posterior_summary,
    plot_posterior,
    convergence_diagnostics,
)

rng = np.random.default_rng(0)

# 20 participants: (stim, resp, conf) per participant
participants = [
    (rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.integers(1, 5, 200))
    for _ in range(20)
]

# Fit
fit, mle_df = fit_two_stage_group(participants, n_ratings=4)

# Check convergence
diag = convergence_diagnostics(fit)
print(diag[~diag["converged"]])   # empty if all converged

# Group mean M-ratio posterior (exp converts log scale → M-ratio)
print(posterior_summary(fit, var_names=["b_Intercept", "sigma"]))

# Plot
plot_posterior(fit, var_names=["b_Intercept"], ref_val=0)
```

## Full HMeta-d example (Option B)

```python
from metasignal.sdtbayes import fit_full_metad, plot_trace

fit = fit_full_metad(participants, n_ratings=4)

# Key parameters
print(posterior_summary(fit,
      var_names=["mu_logMratio", "sigma_logMratio", "mu_d1", "Mratio"]))

plot_trace(fit, var_names=["mu_logMratio", "sigma_logMratio"])
```

## Group comparison example

```python
from metasignal.sdtbayes import fit_full_metad_comparison
import arviz as az

fit_cmp = fit_full_metad_comparison(group_a, group_b, n_ratings=4)

# Posterior of delta_logMratio = group_b - group_a
delta = az.extract(fit_cmp.idata)["delta_logMratio"].values
print(f"P(group B > group A): {(delta > 0).mean():.3f}")
```
