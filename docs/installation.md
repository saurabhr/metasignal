# Installation

## Requirements

- Python 3.9 or later
- NumPy, SciPy, matplotlib (installed automatically)
- **Optional:** a local [MATLAB](https://www.mathworks.com/products/matlab.html) installation for the MATLAB-backed `MetaSignal` class

## Pure Python install (no MATLAB)

Install the latest release from the project source:

```bash
pip install .
```

This gives you the `stdpy` module — a complete pure-Python implementation of all 20 measures — and the CLI.

## With MATLAB support

If you have MATLAB installed and want to use the `MetaSignal` wrapper class (which delegates to the original MATLAB code), also install the MATLAB Engine for Python:

```bash
pip install ".[matlab]"
```

You must have a compatible MATLAB version already installed on your system. See [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) for version compatibility.

## Development install

To install metasignal in editable mode with all development tools (tests, docs):

```bash
pip install -e ".[docs]"
pip install -e ".[matlab]"  # optional
```

Or use [nox](https://nox.thea.codes) to create a complete dev environment:

```bash
pip install nox
nox -s dev
```

## Verify the installation

```python
import metasignal
import numpy as np

stim = np.array([0, 1, 0, 1] * 25)
resp = np.array([0, 1, 1, 0] * 25)
conf = np.array([1, 2, 2, 1] * 25)

results = metasignal.stdpy.compute_all_measures(stim, resp, conf, n_ratings=2)
print(results.shape)  # (20,)
```

If you see `(20,)` printed, the installation is working correctly.
