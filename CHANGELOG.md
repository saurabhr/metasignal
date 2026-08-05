# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Calendar Versioning](https://calver.org/).

The **first number** of the version is the year.
The **second number** is incremented with each release, starting at 1 for each year.
The **third number** is for emergencies when we need to start branches for older releases.

## [Unreleased]

### Added

- `metasignal compute` now accepts trial data from a CSV file via `--csv`,
  as an alternative to typing `--stim`/`--resp`/`--conf` inline.
  `--stim-col`/`--resp-col`/`--conf-col` override the default column names
  (`stim`/`resp`/`conf`).
- `compute_all_measures` accepts a `return_type` keyword (`'array'`
  *(default, unchanged)*, `'dict'`, or `'dataframe'`) to get the 26 values
  labeled by measure name instead of by position. The canonical name list is
  exported as `metasignal.stdpy.MEASURE_NAMES`.
- [Installing with GitHub Desktop](docs/installation.md#installing-with-github-desktop-no-command-line-experience-needed),
  a point-and-click walkthrough for users without command-line experience.
- New `metasignal itmc` CLI command: computes meta-I, meta-Ir1, meta-Ir1_acc,
  meta-Ir2, and RMI per participant from a trial-level CSV, mirroring the
  `metasignal.itmc.estimate_meta_I` Python API. Supports `--backend`
  (`simple`/`statconfr`) and `--bias-correction`.

### Fixed

- `icc()` raised `ValueError` for every call: McGraw & Wong letter-based
  `icc_type` codes (e.g. `'C-k'`) were formatted into strings like
  `"ICC(C,k)"` that pingouin 0.5.5 never emits — it labels rows `ICC1`/
  `ICC2`/`ICC3`/`ICC1k`/`ICC2k`/`ICC3k`. `icc_type` is now mapped to
  pingouin's actual labels explicitly.
- Install instructions for the optional `sdtbayes`/`matlab` extras
  (`pip install metasignal[sdtbayes]`) failed with "No matching
  distribution found" since metasignal isn't published to PyPI. Docs, CLI
  help, and runtime error messages now show the working
  `pip install "metasignal[sdtbayes] @ git+https://github.com/saurabhr/metasignal.git"` form.

- `sdt_expect_conf` (SDT-expected confidence distributions) now returns
  proportions matching MATLAB's `SDTexpectConf`, instead of counts rescaled
  by trial totals. This corrects Ratio/Diff-family measures whenever S1 and
  S2 base rates differ.
- `compute_meta_noise` now follows the MATLAB search/interpolation procedure
  and preserves signal-detection criteria boundary behavior (hit/false-alarm
  rates of 0 or 1 are no longer clipped), fixing large MATLAB↔Python
  divergences in sparse difficulty subsets.
- `compute_meta_uncertainty` now uses a deterministic multi-start optimizer,
  removing optimizer-seed-dependent noise in the fitted estimate.
- `compute_meta_noise` no longer produces a spurious `RuntimeWarning:
  invalid value encountered in log`. Tiny floating-point negatives from
  criterion-grid differencing could reach a `log()` call unclipped, in rare
  cases corrupting the search bounds for the meta-noise optimizer on sparse
  difficulty subsets.

See [docs/MATLAB_PYTHON_CORRECTIONS_REPORT.md](docs/MATLAB_PYTHON_CORRECTIONS_REPORT.md)
for full validation numbers.

## [2025.1.0] - 2026-07-04

### Added

- Initial release of metasignal: Signal Detection Theory and metacognitive
  measures for Python (`stdpy` pure-Python backend, optional MATLAB engine
  wrapper, CLI, and analysis notebooks).
- `sdtbayes` subpackage for hierarchical Bayesian meta-d'/meta-noise models:
  7 estimation approaches (ordered logistic, two-stage, full HMeta-d,
  subject-level, beta-AUC, meta-regression, within-subject comparison) on
  cmdstanpy/Stan and brms backends, a shared formula interface, and
  ArviZ-based diagnostics.
- `itmc` subpackage (experimental, pre-1.0 API) implementing the
  information-theoretic metacognition framework of Dayan (2023) — `meta_I`,
  `meta_Ir1`, `meta_Ir1_acc`, `meta_Ir2`, `RMI`, and `permtest_meta_I`.
- Documentation site (MkDocs + Read the Docs) with tutorials and API
  reference; CI workflows for tests, linting, docs, and Docker builds.
