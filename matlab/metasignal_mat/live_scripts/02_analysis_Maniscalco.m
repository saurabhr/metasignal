%% Maniscalco (2017) Dataset Analysis
% Replicates the analysis from Rahnev (2025), *Nature Communications*, 16(1), 701.
%
% Dataset: Maniscalco & Lau (2017) — single-session perceptual task.
% Subjects performed a visual discrimination task with 4-level confidence ratings.
% This script computes split-half reliability, precision, metacognitive bias
% dependence, and across-measure correlations.

%% Setup
clear; close all; clc

% Set to 1 to recompute from raw data; 0 to load saved results
recompute_measures = 0;

%% Parameters
nRatings     = 4;                   % number of confidence rating levels
binSize_SH   = [50, 100, 200, 400]; % split-half bin sizes (trials per half)
prop_altered = [.02, .04, .06];     % confidence alteration proportions for precision

%% Load Data and Helper Functions
root_dir = fileparts(fileparts(mfilename('fullpath')));
load(fullfile(root_dir, 'Preprocess', 'dataset_Maniscalco_2017_expt1'));
addpath(genpath(fullfile(root_dir, 'helperFunctions')));

fprintf('Loaded %d subjects.\n', length(data));

%% Measure Labels
variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', '\DeltaConf', ...
    'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', '\DeltaConf-Ratio', ...
    'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', '\DeltaConf-Diff', ...
    'meta-noise', 'meta-uncertainty', 'd''', 'Criterion', 'Confidence'};

fprintf(['Measure order: 1:meta-d'', 2:AUC2, 3:gamma, 4:phi, 5:deltaConf\n' ...
    '               6:M_ratio  ... 10:deltaConf_ratio\n' ...
    '               11:M_diff  ... 15:deltaConf_diff\n' ...
    '               16:meta-noise, 17:meta-uncertainty, 18:d'', 19:c, 20:conf\n\n']);

%% Compute All Measures (loop over subjects)
if recompute_measures
    for sub = 1:length(data)
        fprintf('Processing subject %d / %d\n', sub, length(data));

        stim = data{sub}.stim;
        resp = data{sub}.resp;
        conf = data{sub}.conf;

        %% Raw Metacognitive Measures (all trials)
        metas_raw(sub,:) = compute_all_measures(stim, resp, conf, nRatings);

        %% Precision (original + altered-confidence versions, windowed)
        for bs = 1:length(binSize_SH)
            bin_size = binSize_SH(bs);
            for bin_num = 1:floor(length(conf)/bin_size)
                filt = [false(1,bin_size*(bin_num-1)), true(1,bin_size)];
                metas_precision{bs}(sub,bin_num,1,:) = compute_all_measures( ...
                    stim(filt), resp(filt), conf(filt), nRatings);
                for alter = 1:length(prop_altered)
                    metas_precision{bs}(sub,bin_num,alter+1,:) = metasAlteredConf( ...
                        stim(filt), resp(filt), conf(filt), nRatings, prop_altered(alter));
                end
            end
        end

        %% Metacognitive Bias Dependence
        metas_confRecode(sub,1,:) = compute_all_measures(stim, resp, xue_recode(conf,1), nRatings-1);
        metas_confRecode(sub,2,:) = compute_all_measures(stim, resp, xue_recode(conf,2), nRatings-1);

        %% Across-Measure Correlations (odd vs. even trials)
        metas_oddEven(sub,1,:) = compute_all_measures(stim(1:2:end), resp(1:2:end), conf(1:2:end), nRatings);
        metas_oddEven(sub,2,:) = compute_all_measures(stim(2:2:end), resp(2:2:end), conf(2:2:end), nRatings);

        %% Split-Half Reliability (windowed over the full session)
        for bs = 1:length(binSize_SH)
            bin_size = 2 * binSize_SH(bs);
            for bin_num = 1:floor(length(conf)/bin_size)
                filter_odds  = [false(1,bin_size*(bin_num-1)), repmat([true,false], 1,bin_size/2)];
                filter_evens = [false(1,bin_size*(bin_num-1)), repmat([false,true], 1,bin_size/2)];
                metas_splitHalf{bs}(sub,bin_num,1,:) = compute_all_measures( ...
                    stim(filter_odds),  resp(filter_odds),  conf(filter_odds),  nRatings);
                metas_splitHalf{bs}(sub,bin_num,2,:) = compute_all_measures( ...
                    stim(filter_evens), resp(filter_evens), conf(filter_evens), nRatings);
            end
        end
    end
end

%% Save or Load Results
results_path = fullfile(root_dir, 'Results', 'results_Maniscalco');

if recompute_measures
    save(results_path, 'metas_*', 'variable_names');
    fprintf('Results saved to %s\n', results_path);
else
    load(results_path);
    fprintf('Results loaded from %s\n', results_path);
end

%% Summary: Average Measures Across Subjects
fprintf('\n--- Group Means (all trials) ---\n');
for m = 1:20
    fprintf('  %-20s : %.4f\n', variable_names{m}, nanmean(metas_raw(:,m)));
end
