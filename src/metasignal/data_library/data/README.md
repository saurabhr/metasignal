# Bundled datasets

These are copies of real trial-level confidence/perceptual-decision datasets
used elsewhere in this repo (MATLAB↔Python validation in `analysis/`,
tutorials in `docs/tutorials/`) — kept together here so they can be loaded
in one line via `metasignal.data_library`, without hunting through the repo
for the canonical copy.

These are **copies**. Each dataset's original location (used by the existing
MATLAB pipeline and validation scripts — do not remove) is:

| File here | Original location |
|---|---|
| `haddara_2020.csv` | `matlab/metasignal_mat/Preprocess/orig_csv_files/data_Haddara_2022_Expt2.csv` |
| `locke_2020.csv` | `matlab/metasignal_mat/Preprocess/orig_csv_files/data_Locke_2020.csv` |
| `maniscalco_2017.csv` | `matlab/metasignal_mat/Preprocess/orig_csv_files/data_Maniscalco_2017_expt1.csv` |
| `rouault_2018_expt1.csv` | `matlab/metasignal_mat/Preprocess/orig_csv_files/data_Rouault_2018_Expt1.csv` |
| `rouault_2018_expt2.csv` | `matlab/metasignal_mat/Preprocess/orig_csv_files/data_Rouault_2018_Expt2.csv` |
| `shekhar_2021.csv` | `matlab/metasignal_mat/Preprocess/orig_csv_files/data_Shekhar_2021.csv` |
| `metadpy_rm.txt` | `docs/tutorials/rm.txt` |

Each `*_readme.txt` here is a copy of that dataset's original contributor
readme (citation, task design, subject population, etc.).

See `metasignal.data_library.DATASETS` for descriptions and citations in
Python, or `list_datasets()` / `describe_dataset(name)` / `load_dataset(name)`
to use them directly.

Only available in a development checkout (`git clone` + `pip install -e .`)
— excluded from the built wheel (`pyproject.toml`'s `wheel-exclude`) since
these are a development/analysis convenience, not needed at runtime by the
package's compute/analysis API.
