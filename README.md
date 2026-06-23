# metasignal

[![Tests](https://github.com/saurabhr/metasignal/actions/workflows/test.yml/badge.svg)](https://github.com/saurabhr/metasignal/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/saurabhr/metasignal/blob/main/LICENSE)
[![Docs](https://readthedocs.org/projects/metasignal/badge/?version=latest)](https://metasignal.readthedocs.io/en/latest/)

<!-- start docs-include-index -->
Python interface for the Signal Detection Theory (SDT) and meta-measures analysis from Rahnev (2025), [A comprehensive assessment of current methods for measuring metacognition.](https://www.nature.com/articles/s41467-025-56117-0) *Nature Communications*, 16(1), 701.
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

# Sample trial data
stim = np.array([0, 1, 0, 1] * 25)
resp = np.array([0, 1, 1, 0] * 25)
conf = np.array([1, 2, 2, 1] * 25)

# Compute all 20 measures at once
results = stdpy.compute_all_measures(stim, resp, conf, n_ratings=2)
print(results)

# Compute basic SDT
dprime, c, _ = stdpy.compute_sdt_resp(stim, resp)

# Fit meta-d' MLE
nr_s1, nr_s2 = stdpy.trials_to_counts(stim, resp, conf, n_ratings=2)
meta = stdpy.fit_meta_d_mle(nr_s1, nr_s2)
print(f"Meta-d' M-Ratio: {meta['M_ratio']}")
```

### CLI

```bash
metasignal compute --stim "0,1,0,1" --resp "0,1,1,0" --conf "1,2,2,1" --n-ratings 2
```
<!-- end docs-include-usage -->
