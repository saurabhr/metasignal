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

### Fixed

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
