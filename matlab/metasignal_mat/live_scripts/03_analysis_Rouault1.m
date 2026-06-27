%% Rouault (2018) Experiment 1 — Difficulty Dependence Analysis
% Replicates the analysis from Rahnev (2025), *Nature Communications*, 16(1), 701.
%
% Dataset: Rouault et al. (2018) Experiment 1 — visual contrast discrimination.
% Trials are split into LOW difficulty (contrast ≤ 35) and HIGH difficulty
% (contrast > 35) to assess how metacognitive measures depend on task difficulty.
% Confidence ratings use a 1–6 scale; ratings are shifted so minimum = 1.

%% Setup
clear; close all; clc

% Set to 1 to recompute from raw data; 0 to load saved results
recompute_measures = 0;

%% Parameters
nRatings = 6; % number of confidence levels after recoding

%% Load Data and Helper Functions
root_dir = fileparts(fileparts(mfilename('fullpath')));
load(fullfile(root_dir, 'Preprocess', 'dataset_Rouault_2018_Expt1'));
addpath(genpath(fullfile(root_dir, 'helperFunctions')));

fprintf('Loaded %d subjects.\n', length(data));

%% Measure Labels
variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', '\DeltaConf', ...
    'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', '\DeltaConf-Ratio', ...
    'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', '\DeltaConf-Diff', ...
    'meta-noise', 'meta-uncertainty', 'd''', 'Criterion', 'Confidence'};

%% Compute Difficulty-Split Measures (loop over subjects)
% metas_diff(:,1,:) = low difficulty (contrast ≤ 35)
% metas_diff(:,2,:) = high difficulty (contrast > 35)

if recompute_measures
    for sub = 1:length(data)
        fprintf('Processing subject %d / %d\n', sub, length(data));

        stim = data{sub}.stim;
        resp = data{sub}.resp;
        conf = data{sub}.conf - 5;  % shift ratings: original range 6-10 → 1-5 (floor at 1)
        conf(conf < 1) = 1;

        contrast = data{sub}.contrast;

        %% Low difficulty (easier trials: low contrast threshold)
        filt_low  = contrast <= 35;
        metas_diff(sub,1,:) = compute_all_measures( ...
            stim(filt_low), resp(filt_low), conf(filt_low), nRatings);

        %% High difficulty (harder trials: high contrast threshold)
        filt_high = contrast > 35;
        metas_diff(sub,2,:) = compute_all_measures( ...
            stim(filt_high), resp(filt_high), conf(filt_high), nRatings);
    end
end

%% Save or Load Results
results_path = fullfile(root_dir, 'Results', 'results_Rouault1');

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
