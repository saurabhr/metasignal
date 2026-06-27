%% Rouault (2018) Experiment 2 — Difficulty Dependence Analysis
% Replicates the analysis from Rahnev (2025), *Nature Communications*, 16(1), 701.
%
% Dataset: Rouault et al. (2018) Experiment 2 — visual contrast discrimination.
% Trials are split into LOW and HIGH difficulty using a median split on
% each subject's contrast values, to assess how metacognitive measures
% depend on task difficulty across subjects and measures.

%% Setup
clear; close all; clc

% Set to 1 to recompute from raw data; 0 to load saved results
recompute_measures = 0;

%% Parameters
nRatings = 6; % number of confidence rating levels

%% Load Data and Helper Functions
root_dir = fileparts(fileparts(mfilename('fullpath')));
load(fullfile(root_dir, 'Preprocess', 'dataset_Rouault_2018_Expt2'));
addpath(genpath(fullfile(root_dir, 'helperFunctions')));

fprintf('Loaded %d subjects.\n', length(data));

%% Measure Labels
variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', '\DeltaConf', ...
    'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', '\DeltaConf-Ratio', ...
    'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', '\DeltaConf-Diff', ...
    'meta-noise', 'meta-uncertainty', 'd''', 'Criterion', 'Confidence'};

%% Compute Difficulty-Split Measures (loop over subjects)
% Each subject's contrast median is used to define low vs. high difficulty.
% metas_diff(:,1,:) = low difficulty (below-median contrast)
% metas_diff(:,2,:) = high difficulty (above-median contrast)

if recompute_measures
    for sub = 1:length(data)
        fprintf('Processing subject %d / %d\n', sub, length(data));

        filt_high = data{sub}.contrast > median(data{sub}.contrast);

        %% Low difficulty
        metas_diff(sub,1,:) = compute_all_measures( ...
            data{sub}.stim(~filt_high), data{sub}.resp(~filt_high), data{sub}.conf(~filt_high), nRatings);

        %% High difficulty
        metas_diff(sub,2,:) = compute_all_measures( ...
            data{sub}.stim(filt_high), data{sub}.resp(filt_high), data{sub}.conf(filt_high), nRatings);
    end
end

%% Save or Load Results
results_path = fullfile(root_dir, 'Results', 'results_Rouault2');

if recompute_measures
    save(results_path, 'metas_*', 'variable_names');
    fprintf('Results saved to %s\n', results_path);
else
    load(results_path);
    fprintf('Results loaded from %s\n', results_path);
end

%% Summary: Difficulty Effect per Measure
fprintf('\n--- Difficulty Effect (high minus low difficulty, group mean) ---\n');
for m = 1:20
    diff_effect = nanmean(metas_diff(:,2,m) - metas_diff(:,1,m));
    fprintf('  %-20s : %+.4f\n', variable_names{m}, diff_effect);
end
