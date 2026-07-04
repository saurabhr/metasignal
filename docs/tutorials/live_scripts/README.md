# Python Live Scripts — metasignal Tutorials

Cell-mode Python scripts (`.py`) equivalent to the Jupyter notebooks in `docs/tutorials/`.
Each `# %%` delimiter creates a runnable cell, and `# %% [markdown]` cells contain prose.

## File overview

| File | Tutorial | Key topics |
|------|----------|------------|
| `00_run_all.py` | — | Runs all scripts in sequence |
| `01_getting_started.py` | Tutorial 1 | Install check, input format, SDT basics |
| `02_computing_measures.py` | Tutorial 2 | All 26 measures, individual APIs |
| `03_statistical_inference.py` | Tutorial 3 | Bootstrap CI, permutation test, t-test |
| `04_difficulty_dependence.py` | Tutorial 4 | Difficulty split, outlier removal, plot |
| `05_metacognitive_bias.py` | Tutorial 5 | Xue recoding, bias dependence, plot |
| `06_split_half_reliability.py` | Tutorial 6 | Split-half r, Spearman-Brown, precision |
| `07_bayesian_hierarchical.py` | Tutorial 7 | MLE vs Bayesian, HMeta-d, rm dataset |

## How to use

### VS Code (Interactive Window)
Open any `.py` file — VS Code detects `# %%` cells and shows **Run Cell** buttons.
Run a single cell with **Shift+Enter** or the whole file with **Run All Cells**.

### Spyder
Open in the editor. Each `# %%` block is a Spyder cell. Press **Ctrl+Enter** to run
the current cell, **F5** to run the entire script.

### Jupyter (convert to notebook)
```bash
pip install jupytext
jupytext --to notebook 01_getting_started.py
jupyter notebook 01_getting_started.ipynb
```

### Plain Python
```bash
cd docs/tutorials/live_scripts
python 01_getting_started.py
```

### Run all
```bash
python 00_run_all.py
```

## Dependencies

```
metasignal   # this package
numpy
scipy
matplotlib
```

Tutorial 7 additionally needs `brmspy`, `pystan`, and `arviz` for the Bayesian
sections, but those sections are guarded and will print a skip message if the
packages are absent.
