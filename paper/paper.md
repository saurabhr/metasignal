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

`metasignal` is a Python package that provides implementations of Signal Detection Theory (SDT) measures and metacognitive efficiency metrics. The package implements the comprehensive suite of methods evaluated in @rahnev2025, covering first-order perceptual sensitivity (d'), response bias (criterion *c*), and a range of second-order metacognitive measures including meta-d', M-ratio, meta-uncertainty, meta-noise, Type 2 AUC, gamma, phi, and delta confidence. The package is implemented entirely in Python (`stdpy` submodule) and can be used in any Python environment including cloud-computing and automated analysis pipelines.

# Statement of Need

Metacognition—the capacity to monitor and evaluate one's own cognitive performance—is a central topic in cognitive neuroscience and psychology [@flemingdolan2012]. Quantifying metacognitive efficiency requires tools grounded in SDT [@green1966], which separates sensitivity from response bias and allows second-order confidence reports to be analysed independently of first-order performance. The most widely adopted metacognitive efficiency measure, meta-d' [@maniscalco2012], involves non-trivial numerical procedures (maximum-likelihood estimation over rating-scale confusion matrices) that have historically been distributed as MATLAB scripts [@maniscalco2014]. The broader landscape of metacognitive measures—from nonparametric Type 2 statistics to model-based uncertainty and meta-noise estimates—has until recently lacked a unified, systematic comparison; @rahnev2025 provided such a benchmark across seventeen measures.

`metasignal` addresses the need for a single, maintained Python package that:

1. **Covers the full benchmark** — all seventeen measures from @rahnev2025 are available through a single `compute_all_measures` call.
2. **Pure Python implementation** — the `stdpy` submodule is a NumPy/SciPy reimplementation suitable for automated pipelines and environments of any kind, requiring no proprietary software.
3. **Provides a command-line interface** — measures can be computed directly from count vectors without writing any Python code, facilitating integration with experiment-control software and shell-based batch analyses.

# State of the Field

Several existing tools overlap with `metasignal`. The canonical MATLAB implementation by @maniscalco2014 implements meta-d' MLE and has been widely used since 2012 but requires a MATLAB licence and does not cover the broader measure landscape benchmarked by @rahnev2025. `metadpy` [@legrand2021] is a Python library focused on meta-d' and its hierarchical Bayesian variant (HMeta-d), and `hmeta-d` [@fleming2017] provides the hierarchical model in MATLAB and R. Neither package implements the full set of seventeen measures from @rahnev2025, and neither provides a command-line interface or the meta-uncertainty and meta-noise estimators introduced in recent work.

`metasignal` is therefore distinguished by (a) implementing the complete Rahnev benchmark suite in pure Python with no proprietary software dependencies, (b) packaging these tools with a CLI, tutorial notebooks, and API documentation suitable for researchers who are not primarily Python programmers.

# Software Design

`metasignal` accepts a common input convention (NumPy arrays of stimulus labels, responses, and confidence ratings) and exposes all measures through the `stdpy` submodule, which re-implements the benchmark methods using NumPy [@harris2020] and SciPy [@virtanen2020]. Core routines include:

- `compute_sdt_resp` — d' and criterion *c* from trial-level stimulus and response vectors.
- `trials_to_counts` — conversion of trial-level data into Type 2 rating-scale count matrices.
- `fit_meta_d_mle` — maximum-likelihood estimation of meta-d' and M-ratio via bounded nonlinear optimisation.
- `compute_type2_auc`, `compute_gamma`, `compute_phi`, `compute_delta_conf` — nonparametric Type 2 statistics.
- `compute_meta_uncertainty` — model-based meta-uncertainty estimation.
- `compute_meta_noise` — meta-noise estimation via a lookup-table interpolation approach.
- `compute_all_measures` — a convenience wrapper that computes the full benchmark suite in a single call.

All functions return results as Python dictionaries or NumPy arrays, making them straightforward to use with pandas DataFrames, matplotlib, and standard scientific-Python workflows. The package ships with six tutorial Jupyter notebooks that walk through progressively advanced analyses: basic SDT computation, the full measure suite, statistical inference, difficulty-dependence testing, metacognitive bias, and split-half reliability.

# Research Impact Statement

`metasignal` directly implements the benchmark introduced by @rahnev2025, which has been cited as a reference comparison for metacognitive measure selection in ongoing cognitive neuroscience research. The accompanying tutorial series provides worked examples across all seventeen measures, lowering the barrier for researchers adopting SDT-based metacognition analyses in Python. The package is openly licensed (MIT) and hosted on GitHub with continuous integration, public issue tracking, and versioned releases on PyPI, meeting the open-science infrastructure requirements expected of research software in cognitive neuroscience.

# AI Usage Disclosure

Large language model tools (Claude, Anthropic) were used to assist in writing portions of this paper and in generating initial drafts of source code and documentation. All content was reviewed, verified, and edited by the author. The software logic, numerical methods, and validation against reference outputs are the intellectual contribution of the author.

# Acknowledgements

The meta-d' MLE algorithm was originally described by @maniscalco2012. The benchmark suite implemented in `metasignal` was established by Dobromir Rahnev and colleagues [@rahnev2025].

# References
