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

The `itmc` sub-package implements the information-theoretic metacognition framework of Dayan (2023), measuring metacognitive sensitivity as mutual information between accuracy and confidence. It is a pre-1.0 component and its API may still change between releases. Its test suite (`tests/test_itmc.py`) covers bounds, monotonicity, and internal-consistency properties only — it has not yet been cross-checked against reference values from Dayan (2023) or the R `statConfR` package (Rausch et al., 2025) it was ported from.

::: metasignal.itmc.meta_I

::: metasignal.itmc.meta_Ir1

::: metasignal.itmc.meta_Ir1_acc

::: metasignal.itmc.meta_Ir2

::: metasignal.itmc.RMI

::: metasignal.itmc.permtest_meta_I
