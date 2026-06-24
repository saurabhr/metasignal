# metasignal

Python library for Signal Detection Theory (SDT) and metacognitive measures, implementing the comprehensive assessment framework from [Rahnev (2025)](https://doi.org/10.1038/s41467-025-56117-0) — *A comprehensive assessment of current methods for measuring metacognition*, Nature Communications, 16(1), 701.

## What is metasignal?

Metacognition — the ability to reflect on one's own cognitive processes — is measured across cognitive neuroscience, psychology, and clinical research using many different methods. **metasignal** provides a unified Python interface to compute all major metacognitive measures from the same trial-level data, making it straightforward to compare measures and replicate the Rahnev (2025) benchmarking study.

A single call to `compute_all_measures` returns 20 measures organised into four categories:

| Category | Measures |
| --- | --- |
| Absolute sensitivity | meta-d', AUC2, gamma, phi, deltaConf |
| Efficiency ratios | M-ratio, AUC2-ratio, gamma-ratio, phi-ratio, deltaConf-ratio |
| Efficiency differences | M-diff, AUC2-diff, gamma-diff, phi-diff, deltaConf-diff |
| Noise & uncertainty | metaNoise, metaUncertainty |
| Type-1 SDT | d', c, mean confidence |

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
print(results)  # array of 20 float values
```

See [Installation](installation.md) to set up metasignal and [Usage](usage.md) for worked examples.
