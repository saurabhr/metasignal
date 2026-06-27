%% Run All Analyses — Rahnev (2025) Replication
% Executes the complete analysis pipeline from:
%   Rahnev (2025). A comprehensive assessment of current methods for
%   measuring metacognition. *Nature Communications*, 16(1), 701.
%
% STEP 1: Dataset-level analyses (compute measures, save results)
%   01_analysis_Haddara   — multi-day perceptual learning (Days 2-7)
%   02_analysis_Maniscalco — single-session visual discrimination
%   03_analysis_Rouault1  — Expt 1: contrast-defined difficulty split
%   04_analysis_Rouault2  — Expt 2: median-split difficulty
%   05_analysis_Shekhar   — multi-day, multi-contrast (3 days × 3 contrasts)
%   07_ana_respBias       — Locke (2020): 7 response-bias conditions
%
% STEP 2: Aggregate figure analyses (load results, plot figures)
%   06_ana_taskPerformance — Figure 2: difficulty dependence
%   08_ana_metaBias        — Figure 3: metacognitive bias dependence
%   09_ana_precision       — Figure 1: precision (sensitivity to alteration)
%   10_ana_splitHalf       — split-half reliability
%   11_ana_testRetest      — test-retest reliability (ICC + Pearson)
%   12_ana_acrossMeasCorr  — Figure 11: inter-measure correlation matrices
%
% USAGE:
%   Run this script from within the live_scripts/ directory, or set the
%   working directory to live_scripts/ before calling individual scripts.
%
% NOTE: Set recompute_measures = 1 in each dataset script to recompute
%   from raw data. The default (0) loads pre-saved results from Results/.

%% Setup
clear; close all; clc

script_dir = fileparts(mfilename('fullpath'));
cd(script_dir);
fprintf('Working directory: %s\n\n', pwd);

%% Phase 1: Dataset-Level Analyses
fprintf('=== Phase 1: Dataset analyses ===\n\n');

fprintf('[1/6] Haddara dataset...\n');
run('01_analysis_Haddara.m');

fprintf('[2/6] Maniscalco dataset...\n');
run('02_analysis_Maniscalco.m');

fprintf('[3/6] Rouault Expt 1...\n');
run('03_analysis_Rouault1.m');

fprintf('[4/6] Rouault Expt 2...\n');
run('04_analysis_Rouault2.m');

fprintf('[5/6] Shekhar dataset...\n');
run('05_analysis_Shekhar.m');

fprintf('[6/6] Locke dataset (response bias)...\n');
run('07_ana_respBias.m');

%% Phase 2: Aggregate Figure Analyses
fprintf('\n=== Phase 2: Aggregate analyses ===\n\n');

fprintf('[1/6] Task performance (difficulty dependence)...\n');
run('06_ana_taskPerformance.m');

fprintf('[2/6] Metacognitive bias dependence...\n');
run('08_ana_metaBias.m');

fprintf('[3/6] Precision analysis...\n');
run('09_ana_precision.m');

fprintf('[4/6] Split-half reliability...\n');
run('10_ana_splitHalf.m');

fprintf('[5/6] Test-retest reliability...\n');
run('11_ana_testRetest.m');

fprintf('[6/6] Across-measure correlations...\n');
run('12_ana_acrossMeasCorr.m');

%% Done
fprintf('\n=== All analyses complete. ===\n');
fprintf('Figures are open in separate windows.\n');
fprintf('Save them with print() or exportgraphics() as needed.\n');
