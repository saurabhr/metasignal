# Tutorials

These tutorials walk through the full metasignal workflow step by step — from verifying your install to replicating the statistical analyses from [Rahnev (2025)](https://doi.org/10.1038/s41467-025-56117-0).

All code examples use **synthetic data** generated with NumPy, so you can run every tutorial without the original datasets.

## Tutorial sequence

| # | Tutorial | What you'll learn |
| --- | --- | --- |
| 1 | [Getting Started](01_getting_started.ipynb) | Install check, input format, smoke test |
| 2 | [Computing All 26 Measures](02_computing_measures.ipynb) | `compute_all_measures`, index→measure mapping, NaN handling |
| 3 | [Statistical Inference](03_statistical_inference.ipynb) | Bootstrap CIs, permutation tests, group summaries |
| 4 | [Difficulty Dependence](04_difficulty_dependence.ipynb) | Per-difficulty computation, 3-SD outlier removal, ANOVA |
| 5 | [Metacognitive Bias](05_metacognitive_bias.ipynb) | Xue recoding, bias sensitivity testing |
| 6 | [Split-Half Reliability](06_split_half_reliability.ipynb) | Odd/even splits, Spearman-Brown correction |
| 7 | [Bayesian Hierarchical Meta-d'](07_bayesian_hierarchical.ipynb) | Hierarchical Bayesian estimation via `sdtbayes` (requires `pip install "metasignal[sdtbayes] @ git+https://github.com/saurabhr/metasignal.git"`) |

## Background

The 26 measures fall into five conceptual groups (the original 20, plus 6 model-fit diagnostics from the meta-d' MLE):

```
Metacognitive sensitivity meta-d', AUC2, Gamma, Phi, DeltaConf
                          (how well does confidence track accuracy?)

Efficiency ratios         M-ratio, AUC2-ratio, Gamma-ratio, Phi-ratio, DeltaConf-ratio
                          (performance relative to an ideal observer)

Efficiency differences    M-diff, AUC2-diff, Gamma-diff, Phi-diff, DeltaConf-diff
                          (same idea, additive rather than multiplicative)

Noise & uncertainty       metaNoise, metaUncertainty
                          (dispersion in the confidence-generating process)

Model-fit diagnostics     logL, AIC, BIC, AICc, k, n
                          (from the meta-d' MLE fit)
```

Plus the three Type-1 SDT basics returned for convenience: **d'**, **c** (criterion), **mean confidence**.

## Replication notebooks

If you have the original datasets, the `notebooks/` directory contains a complete 10-step replication of Rahnev (2025), including pre-processing, figure generation, and validation against the reference outputs. See [`notebooks/README.md`](https://github.com/saurabhr/metasignal/blob/main/notebooks/README.md) for the full run order.
