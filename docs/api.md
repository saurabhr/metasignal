# API Reference

## stdpy — Python measures

The `stdpy` sub-module is a full NumPy/SciPy implementation of every measure.

### Entry point

::: metasignal.stdpy.compute_all_measures

### Signal Detection Theory

::: metasignal.stdpy.compute_sdt_resp

::: metasignal.stdpy.trials_to_counts

### Meta-d' (MLE)

::: metasignal.stdpy.fit_meta_d_mle

### Type-2 measures

::: metasignal.stdpy.compute_type2_auc

::: metasignal.stdpy.compute_gamma

::: metasignal.stdpy.compute_phi

::: metasignal.stdpy.compute_delta_conf

::: metasignal.stdpy.sdt_expect_conf

### Noise and uncertainty

::: metasignal.stdpy.compute_meta_noise

::: metasignal.stdpy.compute_meta_uncertainty

### Group-level fitting

::: metasignal.stdpy.fit_group

### Plotting

Note: `stdpy.plot_forest` and [`sdtbayes.plot_forest`](sdtbayes.md#diagnostics-and-posteriors) are unrelated functions that happen to share a name — the former plots any scalar measure across subjects/conditions, the latter plots MCMC posterior forest plots.

::: metasignal.stdpy.plot_confidence

::: metasignal.stdpy.plot_type2roc

::: metasignal.stdpy.plot_sanity_check

::: metasignal.stdpy.plot_forest

::: metasignal.stdpy.plot_measures

### Simulation

::: metasignal.stdpy.type2_SDT_simuation

::: metasignal.stdpy.type2_SDT_simuation_bayes

::: metasignal.stdpy.ratings2df

::: metasignal.stdpy.trialSimulation

::: metasignal.stdpy.responseSimulation

::: metasignal.stdpy.pairedResponseSimulation

::: metasignal.stdpy.discreteRatings

---

## analysis — Statistical inference

The `analysis` sub-package provides non-parametric inference tools that operate over the 26-element array (20 measures + 6 meta-d' model-fit diagnostics) produced by `compute_all_measures`.

::: metasignal.analysis.bootstrap_measure

::: metasignal.analysis.permutation_test

::: metasignal.analysis.group_summary

---

## itmc — Information-theoretic metacognition (experimental)

The `itmc` sub-package implements the information-theoretic metacognition framework of Dayan (2023) — *Metacognitive Information Theory*, Open Mind, 7, 392–411, [doi:10.1162/opmi_a_00091](https://doi.org/10.1162/opmi_a_00091) — measuring metacognitive sensitivity as mutual information between accuracy and confidence. `meta_I`, `meta_Ir1`, and `meta_Ir2` are Dayan's own proposed measures; `meta_Ir1_acc` and `RMI` are related measures introduced by Rausch et al. (2025) — *statConfR: An R Package for Static Models of Decision Confidence and Metacognition*, Journal of Open Source Software, 10(106), 6966, [doi:10.21105/joss.06966](https://doi.org/10.21105/joss.06966) — that extend Dayan's framework but aren't discussed in his original paper. It is a pre-1.0 component and its API may still change between releases. Its test suite (`tests/test_itmc.py`) covers bounds, monotonicity, and internal-consistency properties; `backend='statconfr'` is additionally cross-validated against the real R `statConfR` package it was ported from, **and** `meta_I` independently reproduces Dayan (2023)'s own hand-worked numerical example exactly (0.0943 vs. his published 0.094) — see [`analysis/itmc_comparison/ANALYSIS_REPORT.md`](https://github.com/saurabhr/metasignal/blob/main/analysis/itmc_comparison/ANALYSIS_REPORT.md) for the full agreement report.

::: metasignal.itmc.meta_I

::: metasignal.itmc.meta_Ir1

::: metasignal.itmc.meta_Ir1_acc

::: metasignal.itmc.meta_Ir2

::: metasignal.itmc.RMI

::: metasignal.itmc.permtest_meta_I
