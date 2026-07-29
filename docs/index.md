# metasignal

**metasignal** is a Python library for Signal Detection Theory (SDT) and metacognitive measures that implements the comprehensive assessment framework from [Rahnev (2025)](https://doi.org/10.1038/s41467-025-56117-0) — _A comprehensive assessment of current methods for measuring metacognition_, Nature Communications, 16(1), 701.

## What is metasignal?

Metacognition — the ability to reflect on one's own cognitive processes — is measured across cognitive neuroscience, psychology, and clinical research using many different methods. **metasignal** provides a unified Python interface to compute all major metacognitive measures from the same trial-level data, making it straightforward to compare measures and replicate the Rahnev (2025) benchmarking study.

A single call to `compute_all_measures` returns a 26-element array: twenty metacognitive/SDT measures organised into five categories, followed by six meta-d' model-fit diagnostics.

| Category                  | Measures                                                     |
| ------------------------- | ------------------------------------------------------------ |
| Metacognitive sensitivity | meta-d', AUC2, gamma, phi, deltaConf                         |
| Efficiency ratios         | M-ratio, AUC2-ratio, gamma-ratio, phi-ratio, deltaConf-ratio |
| Efficiency differences    | M-diff, AUC2-diff, gamma-diff, phi-diff, deltaConf-diff      |
| Noise & uncertainty       | metaNoise, metaUncertainty                                   |
| Type-1 SDT                | d', c, mean confidence                                       |
| Model-fit diagnostics     | logL, AIC, BIC, AICc, k, n                                   |

## Quick start

```python
import numpy as np
from metasignal import stdpy

rng = np.random.default_rng(42)
n, n_ratings = 200, 4
stim = rng.choice([0, 1], n)
resp = np.where(rng.random(n) < 0.78, stim, 1 - stim)   # 78% accuracy
correct = stim == resp
conf = np.where(correct, rng.integers(2, n_ratings + 1, n),
                         rng.integers(1, n_ratings, n))   # higher conf when correct

results = stdpy.compute_all_measures(stim, resp, conf, n_ratings=n_ratings)
print(results)  # array of 26 float values

# Or labeled by name instead of position:
results = stdpy.compute_all_measures(stim, resp, conf, n_ratings=n_ratings, return_type="dict")
print(results["meta_d"])
```

For group-level inference, `metasignal.analysis` provides bootstrap confidence intervals, permutation tests, and group summaries over that same 26-element array. For fully Bayesian modelling, `metasignal.sdtbayes` offers 7 hierarchical approaches ranging from fast two-stage estimation to full HMeta-d and meta-regression (optional install: `pip install metasignal[sdtbayes]`).

**Experimental:** `metasignal.itmc` implements the information-theoretic metacognition framework of Dayan (2023), measuring metacognitive sensitivity as mutual information between accuracy and confidence (`meta_I`, `meta_Ir1`, `meta_Ir1_acc`, `meta_Ir2`, `RMI`, `permtest_meta_I`). As a pre-1.0 component, its API may still change between releases — see [API Reference](api.md#itmc-information-theoretic-metacognition-experimental). `itmc`'s `backend='statconfr'` is cross-validated against the real R `statConfR` package it was ported from (Rausch et al., 2025) — see [`analysis/itmc_comparison/`](https://github.com/saurabhr/metasignal/tree/main/analysis/itmc_comparison). Deterministic core math matches R to machine precision (Pearson r = 1.0000, all 5 measures); Monte Carlo bias-corrected values agree well, aside from a known ratio-measure instability at very low d′ present in both implementations (not an R↔Python discrepancy — see the report).

See [Installation](installation.md) to set up metasignal and [Usage](usage.md) for worked examples. For Bayesian modelling, see [Bayesian Analysis](sdtbayes.md).

## Contributing

See [Contributing](contributing.md) for development setup and the PR process, and [Future Development](roadmap.md) for known gaps and planned directions.
