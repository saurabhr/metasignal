---
title: "metasignal: A Python Package for Signal Detection Theory and Metacognitive Measures for Decision-Making"
tags:
    - Python
    - metacognition
    - signal detection theory
    - meta-d prime
    - confidence ratings
    - decision-making
    - cognitive psychology
authors:
    - name: Saurabh Ranjan
      orcid: 0000-0002-7868-7223
      affiliation: 1
    - name: Mukesh Makwana
      orcid: 0000-0003-2018-7768
      affiliation: 2
    - name: Konstantina Sokratous
      orcid: 0000-0003-4489-5494
      affiliation: 3
    - name: Brian Odegaard
      orcid: 0000-0002-5459-1884
      affiliation: 1
affiliations:
    - name: University of Florida, USA
      index: 1
    - name: Brown University, USA
      index: 2
    - name: University of Missouri, USA
      index: 3
date: 25 June 2026
bibliography: paper.bib
---

# Summary

Understanding how accurately people know what they know is among the most fundamental questions in cognitive neuroscience. `metasignal` is a Python package that makes measuring this capacity — metacognition — accessible to any researcher with a Python environment. It implements all twenty measures evaluated in the benchmark by @rahnev2025, spanning first-order perceptual sensitivity (d') and response bias (criterion _c_) from signal detection theory [@green1966], and seventeen second-order metacognitive measures including meta-d', M-ratio, meta-uncertainty, meta-noise, Type 2 AUC, gamma, phi, and delta confidence. The full suite is computable from a single function call, with no proprietary software required.

# Statement of Need

Metacognition shapes learning, clinical outcomes, and adaptive decision-making across virtually every domain of cognition [@flemingdolan2012] from perceptual [@yeungsummerfield2012] to economic decision-making [@lebreton2015]. A 26-researcher consensus initiative identified developing falsifiable computational models of visual metacognition as a primary long-term goal [@rahnev2022] — an agenda that depends critically on reliable, standardised implementations of the measures used to evaluate those models. Yet the measurement toolkit has remained fragmented and inaccessible. The gold-standard measure, meta-d' [@maniscalco2012], requires solving a constrained maximum-likelihood problem that has historically been available only as a MATLAB script [@maniscalco2014], placing it out of reach for the growing majority of researchers working in Python. The broader landscape of metacognitive measures lacked any systematic comparison until @rahnev2025 — and even then, no open Python implementation of the full benchmark followed.

`metasignal` closes that gap with a single, maintained package that:

1. **Covers the full benchmark** — all twenty measures from @rahnev2025 (seventeen metacognitive measures plus d', criterion _c_, and mean confidence as Type 1 reference values) are available through a single `compute_all_measures` call.
2. **Requires no proprietary software** — the `stdpy` submodule is a pure NumPy/SciPy implementation that runs in any Python environment, including cloud notebooks and automated pipelines.
3. **Lowers the barrier to entry** — a command-line interface lets researchers compute all twenty measures from raw trial data in a single shell command, without writing any Python code.

# State of the Field

The canonical meta-d' implementation by @maniscalco2014 has been the field's workhorse since 2012 but is MATLAB-only and covers a single measure. `metadpy` [@legrand2021] and `hmeta-d` [@fleming2017] brought hierarchical Bayesian meta-d' to Python and R respectively, addressing estimation reliability but still covering only a fraction of the measures benchmarked by @rahnev2025. No existing package provides the meta-uncertainty and meta-noise estimators, the full suite of ratio and difference normalisation variants, a CLI-based workflow, or a unified interface that combines the full frequentist benchmark with hierarchical Bayesian estimation. The result is that researchers wishing to compare measures — as recommended by @rahnev2025 — have had to stitch together code from multiple repositories in multiple languages.

`metasignal` consolidates this fragmented landscape into a single, documented, tested Python package. Its optional `sdtbayes` subpackage further adds full hierarchical Bayesian inference — including a Stan port of the HMeta-d model of @fleming2017 — making it the only Python package that addresses both the breadth of the @rahnev2025 benchmark and the estimation reliability concerns raised by @fleming2017.

# Software Design

![Computational architecture of `metasignal`. Trial-level arrays flow top-down through four layers: (1) `stdpy` computes SDT statistics and all twenty metacognitive measures; (2) `compute_all_measures` returns the full 26-element output; (3) `analysis` provides bootstrap CIs, permutation tests, and group summaries, and the CLI exposes the full measure suite without Python code; (4) two experimental, pre-1.0 components — `sdtbayes`, offering seven Bayesian estimation approaches that return a `FitResult` with shared diagnostics, and `itmc`, an information-theoretic metacognition module.](structure.png)

`metasignal` accepts NumPy arrays of stimulus labels, responses, and confidence ratings as its universal input and exposes all measures through the `stdpy` submodule, implemented in NumPy [@harris2020] and SciPy [@virtanen2020]. Core estimation routines include:

- `compute_sdt_resp` — d' and criterion _c_ from trial-level stimulus and response vectors.
- `trials_to_counts` — conversion of trial-level data into Type 2 rating-scale count matrices.
- `fit_meta_d_mle` — maximum-likelihood estimation of meta-d' and M-ratio via bounded nonlinear optimisation.
- `compute_type2_auc`, `compute_gamma`, `compute_phi`, `compute_delta_conf` — nonparametric Type 2 statistics.
- `compute_meta_uncertainty` — model-based meta-uncertainty estimation.
- `compute_meta_noise` — meta-noise estimation via lookup-table interpolation.
- `compute_all_measures` — computes all twenty measures in a single call, returning a 26-element NumPy array (the twenty measures followed by six meta-d' model-fit diagnostics — logL, AIC, BIC, AICc, k, n) whose index-to-measure mapping is documented in the API reference.

Beyond point estimation, the `metasignal.analysis` sub-package provides a complete inferential pipeline for group-level research. `bootstrap_measure` estimates percentile-bootstrap confidence intervals for any element of the twenty-measure array (default 2000 resamples, reproducible via an optional `numpy.random.Generator`), enabling uncertainty quantification without parametric assumptions. `permutation_test` implements a two-sided permutation test for between-condition differences (default 5000 shuffles), which is the recommended alternative to parametric t-tests given that the sampling distributions of measures such as M-ratio and meta-d' are non-Gaussian at typical sample sizes [@fleming2017; @rahnev2025]. `group_summary` aggregates results across participants, returning per-participant matrices, group means, medians, standard errors, and per-measure valid counts that handle NaN values arising from degenerate response patterns.

For researchers requiring full Bayesian inference, the optional `metasignal.sdtbayes` subpackage (installed via `pip install metasignal[sdtbayes]`) provides seven hierarchical estimation approaches backed by cmdstanpy/Stan and brms [@burkner2017], with results reported through ArviZ [@kumar2019]:

- `fit_subject_level` — subject-level Bayesian SDT, yielding full posteriors over d' and criterion _c_ for individual participants.
- `fit_two_stage_group` — a two-stage approach that first computes per-participant MLE M-ratios (Stage 1) then fits a hierarchical Bayesian model over log M-ratio across participants (Stage 2), providing a group-level posterior mean M-ratio with uncertainty.
- `fit_full_metad` — a full hierarchical HMeta-d model that ports the JAGS implementation of @fleming2017 to Stan, jointly estimating group-level and per-subject meta-d', M-ratio, d', and criterion from raw count matrices in a single pass.
- `fit_hierarchical_metad` — a trial-level ordered-logistic model in which confidence ratings are the outcome of a cumulative logistic regression and metacognitive discrimination is indexed by the `correct` predictor's coefficient (a log-odds-scale quantity, not meta-d'); supports crossed item random effects.
- `fit_beta_auc_group` — a non-parametric alternative to meta-d' that models Type 2 AUC directly with a Beta likelihood, avoiding the Gaussian SDT assumption.
- `fit_two_stage_regression` / `fit_full_metad_regression` — Bayesian meta-regression of log M-ratio on participant-level covariates, via either the two-stage or full hierarchical path.
- `fit_within_subject_comparison` — a paired model for within-subject designs, where participant random intercepts absorb stable individual differences and the condition effect is estimated directly.

Group comparison variants (`fit_group_comparison`, `fit_two_stage_comparison`, `fit_full_metad_comparison`, `fit_beta_auc_comparison`) extend the corresponding approaches to two-group designs. All fits return a `FitResult` wrapping an ArviZ `InferenceData` object, and the `diagnostics` module provides `posterior_summary`, `convergence_diagnostics`, `plot_trace`, `plot_posterior`, and `plot_forest` for routine MCMC quality checks. Together, these seven estimation approaches let `sdtbayes` users move from raw trial data to full hierarchical Bayesian inference without leaving `metasignal`.

`metasignal` also ships an experimental `itmc` subpackage implementing the information-theoretic metacognition framework of @dayan2023, which measures metacognitive sensitivity as mutual information between accuracy and confidence rather than via signal-detection assumptions. It offers `meta_I`, `meta_Ir1`, `meta_Ir1_acc`, and `meta_Ir2` — variants normalising raw mutual information against first-order performance — plus `RMI` [@meyen2025], a relative mutual-information efficiency measure, and `permtest_meta_I`, a permutation test of whether meta-I is significantly above chance. Two backends are available: a direct entropy calculation following @dayan2023, and an exact Python port of the `estimateMetaI` routine from the `statConfR` R package [@rausch2025] for numerical parity with that implementation. As a pre-1.0 component, `itmc`'s API may still change between releases.

The `metasignal compute` CLI prints all 26 values (20 measures plus 6 meta-d' fit diagnostics) as a named table from comma-separated trial data, and a `metasignal bayes` subcommand exposes the two-stage Bayesian group and comparison fits directly from a CSV, enabling shell-level integration with experiment-control software and batch pipelines. Seven tutorial Jupyter notebooks accompany the package, covering basic SDT computation, the full measure suite, statistical inference, difficulty-dependence testing, metacognitive bias, split-half reliability, and Bayesian hierarchical meta-d' — making the package accessible to researchers regardless of programming background.

# AI Usage Disclosure

Claude (Anthropic) was used to assist in writing portions of this paper and in generating initial drafts of source code and documentation. All content was reviewed, verified, and edited by the author. The software logic, numerical methods, and validation against reference outputs are the intellectual contribution of the author.

# References
