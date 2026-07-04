# Benchmark Replication Notebooks

Pure-Python replication of the MATLAB benchmark analyses from:

> Rahnev, D. (2025). *A comprehensive assessment of current methods for measuring metacognition.*  
> Nature Communications. https://doi.org/10.1038/s41467-025-56117-0

---

## Setup

```bash
# From the repo root
uv venv .venv && source .venv/bin/activate
uv pip install -e . scipy pingouin jupyter ipykernel
jupyter notebook notebooks/
```

Or with conda/pip:
```bash
conda create -n metasignal python=3.10
conda activate metasignal
pip install -e . -r notebooks/requirements.txt
jupyter notebook notebooks/
```

---

## Notebook order

| Notebook | Purpose | Runtime |
|----------|---------|---------|
| `00_setup.ipynb` | Check environment, explain layout | < 1 min |
| `01_preprocessing.ipynb` | Load CSVs → filter subjects → save `.npz` | 1–2 min |
| `02_compute_measures.ipynb` | Compute all 26 measures per dataset | **2–4 hours** (see below) |
| `03_statistical_tables.ipynb` | Reproduce Supp Tables 3–9 | < 1 min (loads from disk) |
| `04_figures.ipynb` | Reproduce Supp Figures 1–4 | < 1 min (loads from disk) |

> **Note on runtime**: `meta-d'`, `meta-noise`, and `meta-uncertainty` each require MLE fitting (~3–4 s per call). For 466 Rouault1 subjects × 2 conditions = 932 calls ≈ 1 hour. Use `multiprocessing.Pool` to parallelise if needed.

---

## Shared utilities: `analysis_core.py`

All notebooks import from `analysis_core.py`, which provides:

| Function | Purpose |
|----------|---------|
| `preprocess_*()` | Load CSV, filter subjects, transform confidence |
| `xue_recode(conf, type)` | Xue et al. (2021) bias recoding |
| `ttest_1samp(data)` | One-sample t-test matching MATLAB `perform_ttest.m` |
| `rm_anova_1way(data_2d)` | Repeated-measures ANOVA |
| `compute_difficulty_*()` | Per-subject measures at each difficulty level |
| `compute_bias()` | Per-subject measures under both Xue recodings |
| `metas_altered_conf()` | Artificially corrupt confidence for precision analysis |

---

## Validation summary

| Table | Measure | Python t | MATLAB t | Match |
|-------|---------|----------|----------|-------|
| T3 Shekhar diff | d' | 23.777 | 23.777 | ✓ exact |
| T3 Shekhar diff | Criterion | 0.166 | 0.166 | ✓ exact |
| T3 Shekhar diff | Confidence | 14.544 | 14.543 | ✓ < 0.01% |
| T4 Rouault1 diff | d' | matches | — | ✓ (466 subjects) |
| T5 Rouault2 diff | d' | 48.583 | 48.583 | ✓ exact (df=469) |
| T6 Haddara bias | Confidence | 24.538 | 24.538 | ✓ exact |
| T9 Locke ANOVA | Criterion | F=12.185 | F=12.185 | ✓ exact |
| T9 Locke ANOVA | Confidence | F=0.482 | F=0.482 | ✓ exact |

Subject counts match MATLAB exactly: Haddara=70, Maniscalco=22, Shekhar=20, Rouault1=466, Rouault2=484, Locke=10.

## Methodological fixes applied (vs original Python draft)

1. **Maniscalco NaN responses**: NaN responses now count as incorrect (matching MATLAB `correct = (stim==resp)+0`). Excludes subject 18 (acc=0.590 with NaN→wrong), giving n=22.
2. **Rouault1 conf filter**: Confidence stereotypy filter now applied on *raw* conf (1–11) before the `conf−5; clip≥1` transformation. Gives n=466.
3. **Difficulty table outlier removal**: `difficulty_table()` now applies MATLAB's ±3 SD outlier removal per measure per difficulty level, then propagates NaN across levels (matching `ana_taskPerformance.m`). This reduces effective n for some measures (e.g. Rouault2 d': 484→470).

---

## Confidence transformations by dataset

| Dataset | Raw conf | Transformation | n_ratings |
|---------|----------|----------------|-----------|
| Haddara | 1–4 | none | 4 |
| Maniscalco | 1–4 | none | 4 |
| Shekhar | 50–100% | `digitize(linspace(50,100,7))` → 1–6 | 6 |
| Rouault1 | 1–11 | `conf - 5`, clip ≥ 1 | 6 |
| Rouault2 | 1–6 | none | 6 |
| Locke | 0–1 | `+ 1` → 1–2 | 2 |

---

## Key methodological notes

**Xue recoding** (`xue_recode`):  
- Type 1: removes lowest rating (biases toward **high** confidence)  
- Type 2: removes highest rating (biases toward **low** confidence)  
- Test statistic: `recode2 − recode1` (does low-conf bias give higher scores?)  

**Cohen's d**: computed as `t / sqrt(n)`, matching MATLAB `perform_ttest.m`.

**Split-half reliability**: odd/even trial interleaving within each bin.

**Precision**: SD-normalised drop under artificial confidence corruption.
