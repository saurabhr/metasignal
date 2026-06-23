# Tutorials

These tutorials walk through the full metasignal workflow step by step — from verifying your install to replicating the statistical analyses from [Rahnev (2025)](https://doi.org/10.1038/s41467-025-56117-0).

All code examples use **synthetic data** generated with NumPy, so you can run every tutorial without the original datasets.

## Tutorial sequence

| # | Tutorial | What you'll learn |
| --- | --- | --- |
| 1 | [Getting Started](getting_started.md) | Install check, input format, smoke test |
| 2 | [Computing All 20 Measures](computing_measures.md) | `compute_all_measures`, index→measure mapping, NaN handling |
| 3 | [Statistical Inference](statistical_inference.md) | Bootstrap CIs, permutation tests, group summaries |
| 4 | [Difficulty Dependence](difficulty_dependence.md) | Per-difficulty computation, 3-SD outlier removal, ANOVA |
| 5 | [Metacognitive Bias](metacognitive_bias.md) | Xue recoding, bias sensitivity testing |
| 6 | [Split-Half Reliability](split_half_reliability.md) | Odd/even splits, Spearman-Brown correction |

## Background

The 20 measures fall into four conceptual groups:

```
Absolute sensitivity   meta-d', AUC2, Gamma, Phi, DeltaConf
                       (how well does confidence track accuracy?)

Efficiency ratios      M-ratio, AUC2-ratio, Gamma-ratio, Phi-ratio, DeltaConf-ratio
                       (performance relative to an ideal observer)

Efficiency differences M-diff, AUC2-diff, Gamma-diff, Phi-diff, DeltaConf-diff
                       (same idea, additive rather than multiplicative)

Noise & uncertainty    metaNoise, metaUncertainty
                       (dispersion in the confidence-generating process)
```

Plus the three Type-1 SDT basics returned for convenience: **d'**, **c** (criterion), **mean confidence**.

## Replication notebooks

If you have the original datasets, the `notebooks/` directory contains a complete 10-step replication of Rahnev (2025), including pre-processing, figure generation, and validation against the reference outputs. See [`notebooks/README.md`](https://github.com/saurabhr/metasignal/blob/main/notebooks/README.md) for the full run order.
