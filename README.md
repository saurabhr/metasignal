# metasignal

[![Tests](https://github.com/saurabhr/metasignal/actions/workflows/test.yml/badge.svg)](https://github.com/saurabhr/metasignal/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/saurabhr/metasignal/blob/main/LICENSE)
[![Docs](https://github.com/saurabhr/metasignal/actions/workflows/docs.yml/badge.svg)](https://github.com/saurabhr/metasignal/actions/workflows/docs.yml)

<!-- start docs-include-index -->

Python interface for the Signal Detection Theory (SDT) and meta-measures analysis from Rahnev (2025), [A comprehensive assessment of current methods for measuring metacognition.](https://www.nature.com/articles/s41467-025-56117-0) _Nature Communications_, 16(1), 701.

<!-- end docs-include-index -->

![Architecture of metasignal. Trial-level data enter the stable stdpy layer. Analysis and command-line layers provide inference and batch use; Bayesian and information-theoretic components are optional.](paper/structure.png)

## Measures

A single call to `compute_all_measures` returns a 26-element result: twenty metacognitive/SDT measures organised into five categories, followed by six meta-d' model-fit diagnostics.

| Category               | Measures                                                      |
| ----------------------- | -------------------------------------------------------------- |
| Absolute sensitivity    | meta-d', AUC2, gamma, phi, deltaConf                            |
| Efficiency ratios       | M-ratio, AUC2-ratio, gamma-ratio, phi-ratio, deltaConf-ratio    |
| Efficiency differences  | M-diff, AUC2-diff, gamma-diff, phi-diff, deltaConf-diff         |
| Noise & uncertainty     | metaNoise, metaUncertainty                                      |
| Type-1 SDT              | d', c, mean confidence                                          |
| Model-fit diagnostics   | logL, AIC, BIC, AICc, k, n                                      |

## Validation

`metasignal` is validated against the Rahnev (2025) MATLAB pipeline across six datasets. See [`analysis/rahnev_comparison/`](analysis/rahnev_comparison/) for the full replication workflow.

![Cross-implementation validation against Rahnev (2025). Panels show task-performance effects, metacognitive-bias effects, response-bias correlations, test-retest ICC, and the task-performance profile across all 17 measures, comparing published values with MATLAB and Python replications.](paper/validation_main.png)

## Installation

<!-- start docs-include-installation -->

The base install includes `stdpy` (SDT + all 26 measures), `analysis` (bootstrap CIs,
permutation tests, group summaries), `itmc` (information-theoretic metacognition), and the CLI.

**Directly from GitHub, no clone needed:**

```bash
pip install git+https://github.com/saurabhr/metasignal.git
```

**From a downloaded/cloned copy of the source:**

```bash
git clone https://github.com/saurabhr/metasignal.git
cd metasignal
pip install .
```

**Optional subpackages** are installed as extras on top of either method above:

```bash
pip install "metasignal[sdtbayes] @ git+https://github.com/saurabhr/metasignal.git"  # from GitHub
pip install ".[sdtbayes]"       # from a local clone — adds metasignal.sdtbayes (hierarchical Bayesian models)
pip install ".[matlab]"         # adds the deprecated MATLAB engine wrapper
pip install ".[sdtbayes,matlab]"  # everything
```

<!-- end docs-include-installation -->

## Usage

<!-- start docs-include-usage -->

### Python API

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

# Compute all measures at once, labeled by name (see Measures table above)
results = stdpy.compute_all_measures(stim, resp, conf, n_ratings=n_ratings, return_type="dict")
print(results["meta_d"], results["M_ratio"], results["AUC2"])

# Or as a positional array (26 values: 20 measures + 6 fit diagnostics) — the default
results = stdpy.compute_all_measures(stim, resp, conf, n_ratings=n_ratings)

# Compute basic SDT
dprime, c, _ = stdpy.compute_sdt_resp(stim, resp)

# Fit meta-d' MLE
nr_s1, nr_s2 = stdpy.trials_to_counts(stim, resp, conf, n_ratings=n_ratings)
meta = stdpy.fit_meta_d_mle(nr_s1, nr_s2)
print(f"Meta-d' M-Ratio: {meta['M_ratio']}")
```

### CLI

```bash
metasignal compute \
  --stim "0,1,0,1,1,0,1,0,0,1" \
  --resp "0,1,1,1,1,0,0,0,0,1" \
  --conf "2,3,1,4,4,3,2,1,3,4" \
  --n-ratings 4
```

<!-- end docs-include-usage -->

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests, and the PR process, and
[docs/roadmap.md](docs/roadmap.md) for known gaps and planned directions.
