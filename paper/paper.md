---
title: 'metasignal: A Python Package for Signal Detection Theory and Metacognitive Measures'
tags:
  - Python
  - metacognition
  - signal detection theory
  - meta-d prime
  - confidence ratings
  - neuroscience
  - cognitive psychology
authors:
  - name: Saurabh Ranjan
    orcid: 0000-0002-7868-7223
    affiliation: 1
affiliations:
  - name: University of Florida, USA
    index: 1
date: 22 June 2026
bibliography: paper.bib
---

# Summary

`metasignal` is a Python package that provides implementations of Signal Detection Theory (SDT) measures and metacognitive efficiency metrics. The package implements the comprehensive suite of twenty methods evaluated in @rahnev2025, covering first-order perceptual sensitivity (d'), response bias (criterion *c*), and a range of second-order metacognitive measures including meta-d', M-ratio, meta-uncertainty, meta-noise, Type 2 AUC, gamma, phi, and delta confidence. The package is implemented entirely in Python (`stdpy` submodule) and can be used in any Python environment including cloud-computing and automated analysis pipelines.

# Statement of Need

Metacognition—the capacity to monitor and evaluate one's own cognitive performance—is a central topic in cognitive neuroscience and psychology [@flemingdolan2012]. The breadth of the field is illustrated by a 26-researcher consensus initiative that established shared computational modelling goals for visual metacognition [@rahnev2022]. Quantifying metacognitive efficiency requires tools grounded in SDT [@green1966], which separates sensitivity from response bias and allows second-order confidence reports to be analysed independently of first-order performance. The most widely adopted metacognitive efficiency measure, meta-d' [@maniscalco2012], involves non-trivial numerical procedures that have historically been distributed as MATLAB scripts [@maniscalco2014]. The broader landscape of metacognitive measures—from nonparametric Type 2 statistics to model-based uncertainty and meta-noise estimates—has until recently lacked a unified, systematic comparison; @rahnev2025 provided such a benchmark across twenty measures.

`metasignal` addresses the need for a single, maintained Python package that:

1. **Covers the full benchmark** — all twenty measures from @rahnev2025 (seventeen metacognitive measures plus d', criterion *c*, and mean confidence as Type 1 reference values) are available through a single `compute_all_measures` call.
2. **Pure Python implementation** — the `stdpy` submodule is a NumPy/SciPy reimplementation suitable for automated pipelines and environments of any kind, requiring no proprietary software.
3. **Provides a command-line interface** — all twenty measures can be computed directly from comma-separated trial-level data via `metasignal compute`, without writing any Python code, facilitating integration with experiment-control software and shell-based batch analyses.

# State of the Field

Several existing tools overlap with `metasignal`. The canonical MATLAB implementation by @maniscalco2014 implements meta-d' MLE and has been widely used since 2012 but requires a MATLAB licence and does not cover the broader measure landscape benchmarked by @rahnev2025. `metadpy` [@legrand2021] is a Python library focused on meta-d' and its hierarchical Bayesian variant (HMeta-d), and `hmeta-d` [@fleming2017] provides the hierarchical model in MATLAB and R. Neither package implements the full set of twenty measures from @rahnev2025, nor provides a command-line interface or the meta-uncertainty and meta-noise estimators introduced in recent work.

`metasignal` is therefore distinguished by (a) implementing the complete Rahnev benchmark suite in pure Python with no proprietary software dependencies, and (b) packaging these tools with a CLI, tutorial notebooks, and API documentation suitable for researchers who are not primarily Python programmers.

# Software Design

`metasignal` accepts a common input convention (NumPy arrays of stimulus labels, responses, and confidence ratings) and exposes all measures through the `stdpy` submodule, which re-implements the benchmark methods using NumPy [@harris2020] and SciPy [@virtanen2020]. Core routines include:

- `compute_sdt_resp` — d' and criterion *c* from trial-level stimulus and response vectors.
- `trials_to_counts` — conversion of trial-level data into Type 2 rating-scale count matrices.
- `fit_meta_d_mle` — maximum-likelihood estimation of meta-d' and M-ratio via bounded nonlinear optimisation.
- `compute_type2_auc`, `compute_gamma`, `compute_phi`, `compute_delta_conf` — nonparametric Type 2 statistics.
- `compute_meta_uncertainty` — model-based meta-uncertainty estimation.
- `compute_meta_noise` — meta-noise estimation via a lookup-table interpolation approach.
- `compute_all_measures` — a convenience wrapper that computes all twenty measures in a single call, returning a fixed-length NumPy array whose index-to-measure mapping is documented in the API reference.

The `metasignal.analysis` sub-package extends the point-estimation layer with a complete inferential pipeline for group-level metacognition research.

`bootstrap_measure` estimates a percentile-bootstrap confidence interval for any single element of the twenty-measure array. It resamples trials with replacement a user-specified number of times (default 2000), recomputes the requested measure on each resample, and returns the lower and upper percentile bounds. An optional `numpy.random.Generator` argument ensures reproducibility. This allows researchers to quantify estimation uncertainty for any measure, including MLE-based ones such as meta-d', without assuming a parametric sampling distribution.

`permutation_test` implements a two-sided permutation test for between-condition differences. It pools trials from two conditions, shuffles condition assignment a user-specified number of times (default 5000), and recomputes the measure difference on each shuffle to build an empirical null distribution. The returned p-value is the proportion of null differences whose absolute value meets or exceeds the observed difference. This is the recommended non-parametric alternative to parametric t-tests when comparing metacognitive measures across conditions or groups, as the sampling distributions of several measures (particularly M-ratio and meta-d') are non-Gaussian at typical sample sizes.

`group_summary` aggregates `compute_all_measures` across an arbitrary list of participants, each supplied as a `(stim, resp, conf)` tuple. It returns a dictionary containing the full per-participant matrix (shape *n* × 20), group means, medians, standard errors, and per-measure valid-participant counts. NaN values—arising when a measure cannot be estimated for a participant due to degenerate response patterns—are excluded measure-wise, so valid-participant counts may differ across the twenty measures. The returned labels list maps each array index to its measure name, enabling direct construction of pandas DataFrames for downstream statistical analysis.

The `metasignal compute` command-line entry point accepts comma-separated stimulus, response, and confidence arrays and prints all twenty measures as a named table, enabling integration with experiment-control software and shell-based batch pipelines without writing Python code.

All functions return results as Python dictionaries or NumPy arrays, making them straightforward to use with pandas DataFrames, matplotlib, and standard scientific-Python workflows. The package ships with six tutorial Jupyter notebooks that walk through progressively advanced analyses: basic SDT computation, the full measure suite, statistical inference, difficulty-dependence testing, metacognitive bias, and split-half reliability.

# Acknowledgements

The meta-d' MLE algorithm was originally described by @maniscalco2012. The benchmark suite implemented in `metasignal` was established by Dobromir Rahnev and colleagues [@rahnev2025].

# AI Usage Disclosure

Claude (Anthropic) was used to assist in writing portions of this paper and in generating initial drafts of source code and documentation. All content was reviewed, verified, and edited by the author. The software logic, numerical methods, and validation against reference outputs are the intellectual contribution of the author.

# References
