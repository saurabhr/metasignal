# Bayesian Analysis

`metasignal.bayesian` fits hierarchical Bayesian metacognition models using
[brmspy](https://github.com/kaitumisuuringute-keskus/brmspy) (a Python
interface to the [brms](https://paul-buerkner.github.io/brms/) R package)
and analyses posteriors with [ArviZ](https://python.arviz.org).

## Installation

The Bayesian submodule is an optional extra — install it alongside the base package:

```bash
pip install metasignal[bayesian]
```

Then run the one-time runtime setup (downloads Stan and installs brms, ~5 min):

```python
from metasignal.bayesian import setup_runtime
setup_runtime()
```

## Model overview

Confidence ratings are modelled as an **ordered (cumulative logistic)**
outcome. The key predictor is `correct` (1 = accurate, 0 = error): its
population-level coefficient is proportional to group-mean meta-d', while
the random slope `(correct | participant)` captures between-subject
variability. This mirrors the HMeta-d formulation of Fleming (2017) but is
expressed as a standard mixed-effects ordered regression, making it
extensible with any brms syntax.

## API Reference

::: metasignal.bayesian.fit_hierarchical_metad

::: metasignal.bayesian.fit_group_comparison

::: metasignal.bayesian.posterior_summary

::: metasignal.bayesian.plot_trace

::: metasignal.bayesian.plot_posterior

::: metasignal.bayesian.plot_forest

::: metasignal.bayesian.convergence_diagnostics

## Example workflow

```python
import numpy as np
from metasignal.bayesian import (
    fit_hierarchical_metad,
    fit_group_comparison,
    posterior_summary,
    plot_trace,
    plot_posterior,
    plot_forest,
    convergence_diagnostics,
)

rng = np.random.default_rng(0)

# Simulate 20 participants
participants = [
    (rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.integers(1, 5, 200))
    for _ in range(20)
]

# 1. Fit hierarchical meta-d' model
fit = fit_hierarchical_metad(participants, n_ratings=4)

# 2. Check convergence
diag = convergence_diagnostics(fit)
print(diag[~diag["converged"]])   # any non-converged parameters

# 3. Posterior summary
print(posterior_summary(fit))

# 4. Plots
plot_trace(fit, var_names=["b_correct", "sd_participant__correct"])
plot_posterior(fit, var_names=["b_correct"], ref_val=0)
plot_forest(fit, var_names=["b_correct", "r_participant"])
```

## Group comparison

```python
# Simulate two groups
group_a = [
    (rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.integers(1, 5, 200))
    for _ in range(15)
]
group_b = [
    (rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.integers(1, 5, 200))
    for _ in range(15)
]

fit_cmp = fit_group_comparison(group_a, group_b, n_ratings=4)

# Posterior probability that group B has lower meta-d' than group A
import arviz as az
post = az.extract(fit_cmp.idata)["b_correct:group1"].values
print(f"P(group B < group A): {(post < 0).mean():.3f}")

# Visualise the group difference
plot_posterior(fit_cmp, var_names=["b_correct:group1"], ref_val=0)
```
