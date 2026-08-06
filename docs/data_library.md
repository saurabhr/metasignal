# Data Library

`metasignal.data_library` bundles real trial-level confidence/perceptual-decision
datasets from published studies for quick loading during development,
analysis, and tutorials — no separate download step.

```python
from metasignal.data_library import list_datasets, load_dataset, describe_dataset

list_datasets()
# ['haddara_2020', 'locke_2020', 'maniscalco_2017', 'metadpy_rm',
#  'rouault_2018_expt1', 'rouault_2018_expt2', 'shekhar_2021']

print(describe_dataset("shekhar_2021"))

df = load_dataset("shekhar_2021")
```

## Availability

These are a **development-checkout convenience**, not required at runtime
by the rest of the package — the data files (~16 MB total) are excluded
from the built wheel (`pyproject.toml`'s `wheel-exclude`). They're present
whenever you install from a git clone:

```bash
git clone https://github.com/saurabhr/metasignal.git
cd metasignal
pip install -e .
```

Calling `load_dataset`/`dataset_path` on an install that doesn't have the
data files raises `DatasetNotAvailableError` with this same instruction.

## Datasets

| Name | Description | Citation |
|---|---|---|
| `haddara_2020` | X-vs-O grid task, 75 subjects, feedback manipulation | Haddara & Rahnev (2020), PsyArXiv |
| `locke_2020` | Gabor tilt discrimination, 10 subjects, priors/payoffs | Locke et al. (2020), *Atten Percept Psychophys* |
| `maniscalco_2017` | Grating detection, 30 subjects, vigilance trade-off | Maniscalco et al. (2017), *J Neurosci* |
| `rouault_2018_expt1` | Dot-counting, N=498, 11-pt confidence | Rouault et al. (2018), *Biol Psychiatry*, Expt 1 |
| `rouault_2018_expt2` | Dot-counting, N=497, 6-pt confidence | Rouault et al. (2018), *Biol Psychiatry*, Expt 2 |
| `shekhar_2021` | Gabor orientation discrimination, continuous confidence | Shekhar & Rahnev, PsyArXiv preprint |
| `metadpy_rm` | Repeated-measures example (20 subjects x 2 conditions) | Bundled with [metadpy](https://github.com/embodied-computation-group/metadpy) |

Full descriptions and citations are in `metasignal.data_library.DATASETS`
(also returned by `describe_dataset`). Each dataset except `metadpy_rm` also
has a `<name>_readme.txt` in `src/metasignal/data_library/data/` — a copy of
the original contributor readme (task design, subject population, feedback,
etc.).

These are **copies**; the existing MATLAB validation pipeline in
`matlab/metasignal_mat/` and `analysis/` reads its own canonical copies
directly and is unaffected — see
[`src/metasignal/data_library/data/README.md`](https://github.com/saurabhr/metasignal/blob/main/src/metasignal/data_library/data/README.md)
for the mapping back to each original file.

## API Reference

::: metasignal.data_library.list_datasets

::: metasignal.data_library.load_dataset

::: metasignal.data_library.describe_dataset

::: metasignal.data_library.dataset_path
