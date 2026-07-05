# MATLAB to Python Code Translation Hash Map

This table represents the 1:1 conceptual and exact mathematical translation mapping of all MATLAB scripts in the `/metasignal/matlab/metasignal_mat/` repository to their new python locations.

| MATLAB Source (`metasignal_mat/`) | Python Destination (`metasignal/`) | Notes |
| :--- | :--- | :--- |
| **Dataset Analytical Pipelines** | | |
| [analysis_Haddara.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/analysis_Haddara.m) | [analysis/analysis_Haddara.py](file:///Users/saurabhext/Documents/metasignal/analysis/analysis_Haddara.py) | Full multi-processing dataset wrapper |
| [analysis_Maniscalco.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/analysis_Maniscalco.m) | [analysis/analysis_Maniscalco.py](file:///Users/saurabhext/Documents/metasignal/analysis/analysis_Maniscalco.py) | |
| [analysis_Shekhar.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/analysis_Shekhar.m) | [analysis/analysis_Shekhar.py](file:///Users/saurabhext/Documents/metasignal/analysis/analysis_Shekhar.py) | |
| [analysis_Rouault1.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/analysis_Rouault1.m) | [analysis/analysis_Rouault1.py](file:///Users/saurabhext/Documents/metasignal/analysis/analysis_Rouault1.py) | |
| [analysis_Rouault2.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/analysis_Rouault2.m) | [analysis/analysis_Rouault2.py](file:///Users/saurabhext/Documents/metasignal/analysis/analysis_Rouault2.py) | |
| **Preprocessing & Data Imports** | | |
| [Preprocess/step1_importDataToMatlab.m](file:///Users/saurabhext/Documents/metasignal/src/metasignal/matlab/Preprocess/step1_importDataToMatlab.m) | [analysis/preprocess.py](file:///Users/saurabhext/Documents/metasignal/analysis/preprocess.py) | Consolidates `.csv` parsing to NumPy |
| [Preprocess/step2_preprocessData.m](file:///Users/saurabhext/Documents/metasignal/src/metasignal/matlab/Preprocess/step2_preprocessData.m) | [analysis/preprocess.py](file:///Users/saurabhext/Documents/metasignal/analysis/preprocess.py) | Accuracy and mode exclusions |
| **Mass Aggregation Plotters** | | |
| [ana_respBias.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/ana_respBias.m) | [analysis/ana_figures.py](file:///Users/saurabhext/Documents/metasignal/analysis/ana_figures.py) | Central visualization mapper via matplotlib |
| [ana_metaBias.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/ana_metaBias.m) | [analysis/ana_figures.py](file:///Users/saurabhext/Documents/metasignal/analysis/ana_figures.py) | |
| [ana_taskPerformance.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/ana_taskPerformance.m) | [analysis/ana_figures.py](file:///Users/saurabhext/Documents/metasignal/analysis/ana_figures.py) | |
| [ana_precision.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/ana_precision.m) | [analysis/ana_figures.py](file:///Users/saurabhext/Documents/metasignal/analysis/ana_figures.py) | |
| [ana_testRetest.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/ana_testRetest.m) | [analysis/ana_figures.py](file:///Users/saurabhext/Documents/metasignal/analysis/ana_figures.py) | |
| [ana_acrossMeasCorr.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/ana_acrossMeasCorr.m) | [analysis/ana_figures.py](file:///Users/saurabhext/Documents/metasignal/analysis/ana_figures.py) | |
| [ana_splitHalf.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/ana_splitHalf.m) | [analysis/ana_figures.py](file:///Users/saurabhext/Documents/metasignal/analysis/ana_figures.py) | |
| **Core Measures (`helperFunctions/metaMeasures/`)** | | |
| [SDTdeltaConf.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/helperFunctions/metaMeasures/SDTdeltaConf.m) | [src/metasignal/stdpy/type2.py](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/type2.py) | Extracted into [compute_delta_conf](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/type2.py#148-189) |
| `SDTexpectConf.m` | [src/metasignal/stdpy/type2.py](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/type2.py) | Extracted into [sdt_expect_conf](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/type2.py#11-62) |
| [SDTtype2AUC.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/helperFunctions/metaMeasures/SDTtype2AUC.m) | [src/metasignal/stdpy/type2.py](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/type2.py) | Extracted into [compute_type2_auc](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/type2.py#65-89) |
| [SDTphi.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/helperFunctions/metaMeasures/SDTphi.m) | [src/metasignal/stdpy/type2.py](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/type2.py) | Mathematical refactor for fractional inputs |
| [SDTgamma.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/helperFunctions/metaMeasures/SDTgamma.m) | [src/metasignal/stdpy/type2.py](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/type2.py) | Extracted into [compute_gamma](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/type2.py#92-113) |
| `compute_metaUncertainty.m` | [src/metasignal/stdpy/uncertainty.py](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/uncertainty.py) | Includes bounds/clip optimizations |
| `compute_SDT_resp.m` | [src/metasignal/stdpy/core.py](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/core.py) | |
| [compute_all_measures.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/helperFunctions/compute_all_measures.m) | [src/metasignal/stdpy/compute_all.py](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/compute_all.py) | Matches 20-var array dimensionality |
| **Meta-d' Methods (`Mfunctions/`)** | | |
| `fit_meta_d_MLE.m` | [src/metasignal/stdpy/metad.py](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/metad.py) | Extends `type2agati` with `scipy.optimize` |
| `type2_SDT_MLE.m` | [src/metasignal/stdpy/metad.py](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/metad.py) | |
| `type2ag.m` | [src/metasignal/stdpy/metad.py](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/metad.py) | |
| `SDT_MLE_fit.m` | [src/metasignal/stdpy/metad.py](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/metad.py) | |
| `trials2counts.m` | [src/metasignal/stdpy/core.py](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/core.py) | Extracted into `trials_to_counts` |
| **Lognormal Meta Noise (`lognormalMetaNoise/`)** | | |
| [compute_metaNoise.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/helperFunctions/metaMeasures/lognormalMetaNoise/compute_metaNoise.m) | [src/metasignal/stdpy/metanoise.py](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/metanoise.py) | Main entry converted iteratively |
| [goldenSearch.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/helperFunctions/metaMeasures/lognormalMetaNoise/goldenSearch.m) | `scipy.optimize.minimize_scalar` | Translated loop structures to Scipy |
| [searchWithLowerBound.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/helperFunctions/metaMeasures/lognormalMetaNoise/searchWithLowerBound.m) | `scipy.optimize.minimize_scalar` | |
| [findInterval.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/helperFunctions/metaMeasures/lognormalMetaNoise/findInterval.m) | `scipy.optimize.minimize_scalar` | |
| [evaluateIntegral.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/helperFunctions/metaMeasures/lognormalMetaNoise/evaluateIntegral.m) | [src/metasignal/stdpy/metanoise.py](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/metanoise.py) | Condensed via `RegularGridInterpolator` |
| [logL_func_criteria.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/helperFunctions/metaMeasures/lognormalMetaNoise/logL_func_criteria.m) | [src/metasignal/stdpy/metanoise.py](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/metanoise.py) | Extracted to helper [_logl_func_criteria](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/metanoise.py#88-101) |
| [logL_func_metaNoise.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/helperFunctions/metaMeasures/lognormalMetaNoise/logL_func_metaNoise.m) | [src/metasignal/stdpy/metanoise.py](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/metanoise.py) | Subfunction target criteria logic |
| [compute_SDTcriteria.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/helperFunctions/metaMeasures/lognormalMetaNoise/compute_SDTcriteria.m) | [src/metasignal/stdpy/metanoise.py](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/metanoise.py) | Subfunction [_compute_sdt_criteria](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/metanoise.py#58-86) |
| [lookupTable.mat](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/helperFunctions/metaMeasures/lognormalMetaNoise/lookupTable.mat) (External Data) | [src/metasignal/stdpy/lookupTable.npz](file:///Users/saurabhext/Documents/metasignal/src/metasignal/stdpy/lookupTable.npz) | Compressed via NumPy serialization |
| **Data Alterations & Scripts** | | |
| [xue_recode.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/helperFunctions/xue_recode.m) | [analysis/helpers.py](file:///Users/saurabhext/Documents/metasignal/analysis/helpers.py) | Shared recoding routines |
| [metasAlteredConf.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/helperFunctions/metasAlteredConf.m) | [analysis/helpers.py](file:///Users/saurabhext/Documents/metasignal/analysis/helpers.py) | Pre-noise mutation injection |
| **Statistical & Graphic Utilities** | | |
| [ICC.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/helperFunctions/ICC.m) | [analysis/stats_helpers.py](file:///Users/saurabhext/Documents/metasignal/analysis/stats_helpers.py) | Uses `pingouin.intraclass_corr` core logic |
| [perform_ttest.m](file:///Users/saurabhext/Documents/metasignal/matlab/metasignal_mat/helperFunctions/perform_ttest.m) | [analysis/stats_helpers.py](file:///Users/saurabhext/Documents/metasignal/analysis/stats_helpers.py) | Replicated single-sample specs |
| `z2r.m` | [analysis/stats_helpers.py](file:///Users/saurabhext/Documents/metasignal/analysis/stats_helpers.py) | |
| `r2z.m` | [analysis/stats_helpers.py](file:///Users/saurabhext/Documents/metasignal/analysis/stats_helpers.py) | |
| `good_colors_for_plotting.m` | [analysis/ana_figures.py](file:///Users/saurabhext/Documents/metasignal/analysis/ana_figures.py) | Staged as Global config |
| `suplabel.m` | `matplotlib.pyplot.suptitle` | Replaced natively |
| `plot_corrTables.m` | [analysis/ana_figures.py](file:///Users/saurabhext/Documents/metasignal/analysis/ana_figures.py) | Matrix heatplots logic |
