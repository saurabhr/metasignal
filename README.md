# metasignal

[![Tests](https://github.com/saurabhr/metasignal/actions/workflows/test.yml/badge.svg)](https://github.com/saurabhr/metasignal/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/saurabhr/metasignal/blob/main/LICENSE)
[![Docs](https://github.com/saurabhr/metasignal/actions/workflows/docs.yml/badge.svg)](https://github.com/saurabhr/metasignal/actions/workflows/docs.yml)

<!-- start docs-include-index -->

Python interface for the Signal Detection Theory (SDT) and meta-measures analysis from Rahnev (2025), [A comprehensive assessment of current methods for measuring metacognition.](https://www.nature.com/articles/s41467-025-56117-0) _Nature Communications_, 16(1), 701.

<!-- end docs-include-index -->

## Installation

<!-- start docs-include-installation -->

```bash
pip install .
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

# Compute all measures at once (26-element array: 20 measures + 6 fit diagnostics)
results = stdpy.compute_all_measures(stim, resp, conf, n_ratings=n_ratings)
print(results)

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
