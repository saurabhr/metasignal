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

---

## analysis — Statistical inference

The `analysis` sub-package provides non-parametric inference tools that operate over the 26-measure array produced by `compute_all_measures`.

::: metasignal.analysis.bootstrap_measure

::: metasignal.analysis.permutation_test

::: metasignal.analysis.group_summary
