# metasignal

Python interface for the Signal Detection Theory (SDT) and meta-measures analysis from Rahnev (2025).

## Installation

This package requires a local MATLAB installation and the MATLAB Engine API for Python.

```bash
pip install .
```

## Usage

### Python API

```python
import numpy as np
from metasignal import MetaSignal

# Initialize the engine
ms = MetaSignal()

# Sample data
stim = np.array([0, 1, 0, 1] * 20)
resp = np.array([0, 1, 1, 0] * 20)
conf = np.array([1, 2, 2, 1] * 20)
n_ratings = 2

# Compute measures
results = ms.compute_all_measures(stim, resp, conf, n_ratings)
print(results)

# Stop the engine
ms.stop()
```

### Pure Python API (stdpy)

For users without a MATLAB installation, use the pure Python implementation:

```python
import numpy as np
from metasignal import stdpy

# Sample trial data
stim = np.array([0, 1, 0, 1] * 25)
resp = np.array([0, 1, 1, 0] * 25)
conf = np.array([1, 2, 2, 1] * 25)

# Compute basic SDT
dprime, c, _ = stdpy.compute_sdt_resp(stim, resp)

# Fit meta-d' MLE
nr_s1, nr_s2 = stdpy.trials_to_counts(stim, resp, conf, n_ratings=2)
results = stdpy.fit_meta_d_mle(nr_s1, nr_s2)
print(f"Meta-d' M-Ratio: {results['M_ratio']}")
```

### CLI

```bash
metasignal compute --stim "0,1,0,1" --resp "0,1,1,0" --conf "1,2,2,1" --n-ratings 2
```

## Reference

Rahnev, D. (2025). [A comprehensive assessment of current methods for measuring metacognition.](https://www.nature.com/articles/s41467-025-56117-0) *Nature Communications*, 16(1), 701.
