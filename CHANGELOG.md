# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Calendar Versioning](https://calver.org/).

The **first number** of the version is the year.
The **second number** is incremented with each release, starting at 1 for each year.
The **third number** is for emergencies when we need to start branches for older releases.

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
