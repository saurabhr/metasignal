# Bayesian Analysis

`metasignal.sdtbayes` fits hierarchical Bayesian metacognition models and
analyses posteriors with [ArviZ](https://python.arviz.org).  Two backends are
in use, depending on the approach:

- **cmdstanpy** (recommended, no R required) — the primary backend, used by
  [`fit_meta_formula`](#formula-interface-and-runtime) and the approaches
  redirected to it below.
- **brms via [brmspy](https://github.com/kaitumisuuringute-keskus/brmspy)**
  (requires R) — used by the plain-regression approaches below.

See **[Tutorial 7 — Bayesian Hierarchical Meta-d'](tutorials/07_bayesian_hierarchical.ipynb)**
for a hands-on walkthrough using the canonical HMeta-d dataset and a
side-by-side comparison with [metadpy](https://github.com/embodied-computation-group/metadpy).

## Installation

The Bayesian submodule is an optional extra — install it alongside the base package:

```bash
pip install metasignal[sdtbayes]
```

For the cmdstanpy-backed approaches (3, 4, and Approach 10's full path),
install CmdStan once:

```python
import cmdstanpy
cmdstanpy.install_cmdstan()
```

For the brms-backed approaches (1, 2, 7, 10's two-stage path, 12), run the
one-time R/brms runtime setup instead (downloads Stan and installs brms, ~5 min):

```python
from metasignal.sdtbayes import setup_runtime
setup_runtime()
```

## Modelling approaches

| # | Approach | Backend | Status | Primary function(s) | Description |
|---|----------|---------|--------|---------------------|-------------|
| 1 | **Ordered logistic** | brms | ✅ working | `fit_hierarchical_metad` | Trial-level; confidence ratings as cumulative outcome. Fast and extensible. |
| 2 | **Two-stage Bayesian** | brms | ✅ working | `fit_two_stage_group` | MLE per participant → Bayesian on log M-ratio. Fast Stage 2; no Stage-1 uncertainty propagation. |
| 3 | **Full HMeta-d** | cmdstanpy | ✅ working | `fit_full_metad` | Single-stage port of Fleming (2017), delegating to `fit_meta_formula`. Numerically stable log-space likelihood; most accurate. |
| 4 | **Subject-level Bayesian** | cmdstanpy | ✅ working | `fit_subject_level` | Single-participant model. Matches `metadpy.hmetad(nR_S1, nR_S2)` API; reproduces metadpy's reference values to within MCMC noise. |
| 5 | **Robust HMeta-d** | brms | ❌ unavailable | `fit_robust_metad` | Student-t hyperprior on log M-ratio. Blocked by an upstream brmspy stanvar-list conversion limitation; no cmdstanpy port exists yet. |
| 6 | **Variational inference** | — | ❌ unavailable | `fit_full_metad_vi`, `fit_robust_metad_vi` | cmdstanpy exposes VI via dedicated methods, not the `algorithm=` kwarg this wrapper used. |
| 7 | **Beta AUC** | brms | ✅ working | `fit_beta_auc_group` | Non-parametric alternative to meta-d'. Models Type-2 AUC directly with a Beta likelihood; no Gaussian SDT assumption. |
| 8 | **Gaussian mixture** | brms | ❌ unavailable | `fit_mixture_group` | brms's `mixfamily` object loses its R S3 class round-tripping through brmspy. |
| 9 | **Bivariate hierarchical** | brms | ❌ unavailable | `fit_multivariate_mratio` | brms's `rescor` flag cannot be threaded through brmspy via any tried channel. |
| 10 | **Meta-regression** | brms + cmdstanpy | ✅ working | `fit_two_stage_regression`, `fit_full_metad_regression` | Regresses log M-ratio on participant-level covariates via either the two-stage (brms) or full hierarchical (cmdstanpy) path. |
| 11 | **State-space** | brms | ❌ unavailable | `fit_statespace_metad` | Same stanvar-list conversion issue as Approach 5; no cmdstanpy port exists. |
| 12 | **Within-subject comparison** | brms | ✅ working | `fit_within_subject_comparison` | Paired model for within-subject designs: same participants in two conditions; participant random intercepts absorb stable individual differences. |

Unavailable approaches raise a `RuntimeError` with a specific explanation and,
where one exists, a pointer to a working alternative — see each function's
docstring for the exact upstream brmspy/rpy2 limitation encountered.

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

## Full HMeta-d example (Option B, cmdstanpy)

```python
from metasignal.sdtbayes import fit_full_metad, plot_trace

fit = fit_full_metad(participants, n_ratings=4)

# Key parameters (mu_logMratio is an alias for alpha_logMratio)
print(posterior_summary(fit,
      var_names=["mu_logMratio", "sigma_logMratio", "mu_d1", "Mratio"]))

plot_trace(fit, var_names=["mu_logMratio", "sigma_logMratio"])
```

## Group comparison example

```python
from metasignal.sdtbayes import fit_full_metad_comparison
import arviz as az

fit_cmp = fit_full_metad_comparison(group_a, group_b, n_ratings=4)

# beta_logMratio[0] = group B - group A difference in log M-ratio
# (the covariate is mean-centred 0/1 -> -0.5/+0.5, spanning exactly one unit)
delta = az.extract(fit_cmp.idata)["beta_logMratio"].values[:, 0]
print(f"P(group B > group A): {(delta > 0).mean():.3f}")
```
