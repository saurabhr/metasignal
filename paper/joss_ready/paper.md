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
date: 18 July 2026
bibliography: paper.bib
---

# Summary

Understanding how accurately people know what they know is among the most fundamental questions in cognitive neuroscience. `metasignal` is a Python package that makes measuring this capacity — metacognition — accessible in any Python environment. It implements the twenty measures evaluated in the benchmark by @rahnev2025, spanning first-order perceptual sensitivity (d') and response bias (criterion _c_) from signal detection theory [@green1966], mean confidence, and seventeen second-order metacognitive measures including meta-d', M-ratio, meta-uncertainty, meta-noise, Type 2 AUC, gamma, phi, and delta confidence. The full suite is computable from a single function call, with no proprietary software required.

# Statement of Need

Metacognition shapes learning, clinical outcomes, and adaptive decision-making across virtually every domain of cognition [@flemingdolan2012], from perceptual [@yeungsummerfield2012] to economic [@lebreton2015] decision-making, and increasingly the evaluation of large language models, whose growing use in high-stakes domains such as medical reasoning depends on reliable metacognitive uncertainty communication [@steyvers2025; @griot2025]. A 26-researcher consensus initiative identified developing falsifiable computational models of visual metacognition as a primary long-term goal [@rahnev2022] — an agenda that depends critically on reliable, standardised implementations of the measures used to evaluate those models. Yet the measurement toolkit has remained fragmented and inaccessible. The gold-standard measure, meta-d' [@maniscalco2012], requires solving a constrained maximum-likelihood problem that has historically been available only as a MATLAB script [@maniscalco2014], placing it out of reach for the growing majority of researchers working in Python. The broader landscape of metacognitive measures lacked any systematic comparison until @rahnev2025 — and even then, no open Python implementation of the full benchmark followed.

`metasignal` closes that gap with a single, maintained package that:

1. **Covers the Rahnev (2025) measure suite** — all twenty measures (seventeen metacognitive measures plus d', criterion _c_, and mean confidence as Type 1 reference values) are available through a single `compute_all_measures` call.
2. **Requires no proprietary software** — the `stdpy` submodule is a pure NumPy/SciPy implementation that runs in any Python environment, including cloud notebooks and automated pipelines.
3. **Lowers the barrier to entry** — a command-line interface lets researchers compute all twenty measures from raw trial data in a single shell command, without writing any Python code.

# State of the Field

The canonical meta-d' implementation by @maniscalco2014 has been the field's workhorse since 2012 but is MATLAB-only and covers a single measure. `metadpy` [@legrand2021] and `hmeta-d` [@fleming2017] brought hierarchical Bayesian meta-d' to Python and R respectively, addressing estimation reliability but still covering only a fraction of the measures benchmarked by @rahnev2025. Researchers wishing to compare measures — as recommended by @rahnev2025 — have therefore had to stitch together code from multiple repositories in multiple languages.

To our knowledge, `metasignal` is the first maintained Python package that jointly (i) exposes the full @rahnev2025 frequentist measure suite in pure NumPy/SciPy and (ii) optionally pairs that suite with hierarchical Bayesian estimation via a Stan port of HMeta-d [@fleming2017] and related group-level models. Complementary tools remain preferable for specialised workflows (e.g., `metadpy` for Bayesian meta-d' alone; MATLAB for legacy pipelines), but they do not remove the need for a unified, documented Python entry point to the broader benchmark.

# Software Design

![Computational architecture of `metasignal`. Trial-level arrays flow through `stdpy` (stable frequentist measures), optional `analysis` and CLI layers, and two experimental pre-1.0 components (`sdtbayes`, `itmc`).](structure.png)

Three design decisions shape the package.

**1. Pure-Python core, optional Bayesian extras.** The stable public surface is `metasignal.stdpy`: NumPy [@harris2020] / SciPy [@virtanen2020] implementations of Type 1 SDT and the twenty @rahnev2025 measures, culminating in `compute_all_measures` (26-element output: 20 measures + 6 meta-d' fit diagnostics). This keeps installation lightweight and reproducible on machines without MATLAB or Stan. Hierarchical Bayesian inference lives in the optional `metasignal.sdtbayes` extra (`pip install metasignal[sdtbayes]`), which uses cmdstanpy/Stan and brms [@burkner2017] with ArviZ diagnostics [@kumar2019]. Information-theoretic measures [@dayan2023; @meyen2025], including a port aligned with `statConfR` [@rausch2025], live in experimental `itmc`. Marking `sdtbayes` and `itmc` as pre-1.0 deliberately separates a frozen frequentist API from components whose dependencies and interfaces may still evolve.

**2. Trial arrays as the universal input.** All core routines accept stimulus, response, and confidence vectors (or count matrices derived from them). A thin `analysis` layer adds bootstrap confidence intervals, permutation tests, and group summaries; the `metasignal compute` CLI exposes the same measure suite from CSV for shell pipelines. This favours composability with existing experiment code over a heavyweight domain-specific language.

**3. Replication notebooks as the validation contract.** Rather than treating MATLAB parity as an informal claim, the repository ships tutorial notebooks that recompute @rahnev2025-style analyses (task performance, metacognitive bias, response bias, reliability) from publicly released datasets. Primary Type 1 and most Type 2 measures agree closely with the original MATLAB reference outputs under matched protocols; documented exceptions (notably meta-noise optimisation details, and SDT-expected confidence scaling that affects some Ratio/Diff variants under unequal base rates) are tracked in the package documentation so users know when numerical identity should not be assumed.

# Research Impact Statement

`metasignal` addresses a concrete barrier identified by the visual-metacognition consensus agenda [@rahnev2022]: standardised, open implementations of the measures used to evaluate computational models. Near-term scholarly significance rests on three community-readiness signals.

**Reproducible benchmark materials.** Seven tutorial notebooks and accompanying documentation walk users from raw trials to the full measure suite, inferential tests, difficulty dependence, metacognitive bias, split-half reliability, and hierarchical Bayesian meta-d'. These materials let laboratories reproduce @rahnev2025-style comparisons without proprietary MATLAB licences.

**Cross-implementation validation.** The pure-Python `stdpy` estimators are checked against the MATLAB reference pipeline shipped with the package and against published summary contrasts from @rahnev2025 (e.g., task-performance and metacognitive-bias effect-size profiles). Agreement is near-exact for d', criterion, meta-d'/M-ratio family measures, and most nonparametric Type 2 indices under matched analysis protocols; remaining discrepancies are documented rather than hidden, which is itself a prerequisite for trustworthy comparative work.

**Open development posture.** The software is MIT-licensed, accompanied by automated tests, continuous-integration workflows, API documentation, and contributor guidelines. The intended audience includes cognitive neuroscientists, psychophysicists, and AI-evaluation researchers who need a single Python dependency for metacognitive measurement rather than an ad-hoc collage of scripts.

# AI Usage Disclosure

Claude (Anthropic) was used to assist in writing portions of this paper and in generating initial drafts of source code and documentation. All content was reviewed, verified, and edited by the authors. The software logic, numerical methods, and validation against reference outputs are the intellectual contribution of the authors.

# References
