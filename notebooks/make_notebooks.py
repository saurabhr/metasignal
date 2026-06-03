"""Generate all tutorial .ipynb notebooks for the metasignal benchmark replication."""
import json, os

def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": cells
    }

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src, "id": "md"}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src, "id": "cd"}

OUT = os.path.dirname(__file__)

# ════════════════════════════════════════════════════════════════════════════
# 00_setup.ipynb
# ════════════════════════════════════════════════════════════════════════════
nb00 = nb([
md("""# 00 · Environment Setup

**Paper**: *A comprehensive assessment of current methods for measuring metacognition*  
Rahnev, *Nature Communications* 2025

This notebook confirms your environment is ready and explains the repository layout.

## Quick-start
```bash
# From the metasignal/ root directory:
uv venv .venv && source .venv/bin/activate
uv pip install -e . scipy pingouin jupyter ipykernel nbformat
jupyter notebook notebooks/
```
Or with conda:
```bash
conda create -n metasignal python=3.10
conda activate metasignal
pip install -e . scipy pingouin jupyter ipykernel
```
"""),
code("""\
import sys, importlib

REQUIRED = {
    "numpy":      "2.0+",
    "scipy":      "1.10+",
    "pandas":     "2.0+",
    "matplotlib": "3.7+",
    "pingouin":   "0.5+",
}

all_ok = True
for pkg, ver in REQUIRED.items():
    try:
        mod = importlib.import_module(pkg)
        print(f"  ✓ {pkg:12s} {mod.__version__}  (need {ver})")
    except ImportError:
        print(f"  ✗ {pkg:12s} NOT FOUND  → pip install {pkg}")
        all_ok = False

# metasignal itself
try:
    import metasignal
    print(f"  ✓ metasignal   {metasignal.__version__}")
except ImportError:
    print("  ✗ metasignal   NOT FOUND  → pip install -e .")
    all_ok = False

print("\\n" + ("All dependencies satisfied ✓" if all_ok else "⚠ Install missing packages above"))
"""),
md("""## Repository layout
```
metasignal/
├── src/metasignal/          ← Python package (pure-Python MATLAB replicas)
│   └── stdpy/
│       ├── compute_all.py   ← compute_all_measures() — main entry point
│       ├── metad.py         ← meta-d' MLE fitting
│       ├── type2.py         ← AUC2, Gamma, Phi, DeltaConf
│       ├── metanoise.py     ← meta-noise model
│       └── uncertainty.py   ← meta-uncertainty model
├── notebooks/
│   ├── analysis_core.py     ← shared preprocessing + stats helpers
│   ├── 00_setup.ipynb       ← this notebook
│   ├── 01_preprocessing.ipynb
│   ├── 02_compute_measures.ipynb
│   ├── 03_statistical_tables.ipynb
│   └── 04_figures.ipynb
└── matlab/metasignal_mat/Preprocess/orig_csv_files/
    └── data_*.csv           ← raw trial-level data for all datasets
```

## The 20 measures
| Index | Measure | Type |
|-------|---------|------|
| 0 | meta-d' | absolute |
| 1 | AUC2 | absolute |
| 2 | Gamma | absolute |
| 3 | Phi | absolute |
| 4 | DeltaConf | absolute |
| 5-9 | M/AUC2/Gamma/Phi/DeltaConf-Ratio | ratio |
| 10-14 | M/AUC2/Gamma/Phi/DeltaConf-Diff | difference |
| 15 | meta-noise | model-based |
| 16 | meta-uncertainty | model-based |
| 17 | d' | task performance |
| 18 | Criterion | response bias |
| 19 | Confidence | metacognitive bias |
"""),
code("""\
# Verify a single compute_all_measures call works
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "src"))
# if running from notebooks/ subdirectory:
sys.path.insert(0, os.path.join(os.getcwd(), "..", "src"))

import numpy as np
from metasignal.stdpy.compute_all import compute_all_measures

MEASURE_NAMES = [
    "meta-d'", "AUC2", "Gamma", "Phi", "DeltaConf",
    "M-Ratio", "AUC2-Ratio", "Gamma-Ratio", "Phi-Ratio", "DeltaConf-Ratio",
    "M-Diff", "AUC2-Diff", "Gamma-Diff", "Phi-Diff", "DeltaConf-Diff",
    "meta-noise", "meta-uncertainty", "d'", "Criterion", "Confidence",
]

np.random.seed(42)
n = 400
stim = np.random.randint(0, 2, n)
resp = np.where(np.random.rand(n) < 0.75, stim, 1 - stim)
conf = np.random.randint(1, 5, n)

result = compute_all_measures(stim, resp, conf, n_ratings=4)
print(f"{'Measure':<20} Value")
print("-" * 30)
for name, val in zip(MEASURE_NAMES, result):
    print(f"  {name:<18} {val:.4f}")
"""),
])

with open(os.path.join(OUT, "00_setup.ipynb"), "w") as f:
    json.dump(nb00, f, indent=1)
print("Written: 00_setup.ipynb")
