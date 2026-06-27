%% Haddara (2022) Dataset Analysis
% Replicates the analysis from Rahnev (2025), *Nature Communications*, 16(1), 701.
%
% Dataset: Haddara & Rahnev (2022) — multi-day perceptual learning task.
% Subjects completed Days 2-7 of training (6 days, ~400 trials/day).
% This script computes all 20 metacognitive and SDT measures for each
% subject, then saves/loads results for downstream aggregate analyses.

%% Setup
clear; close all; clc

% Set to 1 to recompute from raw data; 0 to load saved results
recompute_measures = 0;

%% Parameters
nRatings       = 4;                   % number of confidence rating levels
binSize_SH     = [50, 100, 200, 400]; % split-half bin sizes (trials per half)
binSize_TR     = [50, 100, 200, 400]; % test-retest bin sizes; first 400 trials/day
prop_altered   = [.02, .04, .06];     % confidence alteration levels for precision

%% Load Data and Helper Functions
% The dataset contains one cell per subject with fields: stim, resp, conf, day
root_dir = fileparts(fileparts(mfilename('fullpath')));
load(fullfile(root_dir, 'Preprocess', 'dataset_Haddara_2022_Expt2'));
addpath(genpath(fullfile(root_dir, 'helperFunctions')));

fprintf('Loaded %d subjects.\n', length(data));

%% Measure Labels
variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', '\DeltaConf', ...
    'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', '\DeltaConf-Ratio', ...
    'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', '\DeltaConf-Diff', ...
    'meta-noise', 'meta-uncertainty', 'd''', 'Criterion', 'Confidence'};

fprintf(['Measure order: 1:meta-d'', 2:AUC2, 3:gamma, 4:phi, 5:deltaConf\n' ...
    '               6:M_ratio, 7:AUC2_ratio, 8:gamma_ratio, 9:phi_ratio, 10:deltaConf_ratio\n' ...
    '               11:M_diff, 12:AUC2_diff, 13:gamma_diff, 14:phi_diff, 15:deltaConf_diff\n' ...
    '               16:meta-noise, 17:meta-uncertainty, 18:d'', 19:c, 20:conf\n\n']);

%% Compute All Measures (loop over subjects)
% Skipped if recompute_measures == 0 (loads from Results/ instead).

if recompute_measures
    for sub = 1:length(data)
        fprintf('Processing subject %d / %d\n', sub, length(data));

        stim           = data{sub}.stim;
        resp           = data{sub}.resp;
        conf           = data{sub}.conf;
        day_of_testing = data{sub}.day;

        % Segment data into Days 2-7 (recoded as 1-6)
        for day = 2:7
            stim_day{day-1} = stim(day_of_testing == day);
            resp_day{day-1} = resp(day_of_testing == day);
            conf_day{day-1} = conf(day_of_testing == day);
        end

        %% Raw Metacognitive Measures (all trials combined)
        metas_raw(sub,:) = compute_all_measures(stim, resp, conf, nRatings);

        %% Metacognitive Bias Dependence
        % Two confidence re-codings from Xue et al. (2021)
        metas_confRecode(sub,1,:) = compute_all_measures(stim, resp, xue_recode(conf,1), nRatings-1);
        metas_confRecode(sub,2,:) = compute_all_measures(stim, resp, xue_recode(conf,2), nRatings-1);

        %% Across-Measure Correlations (odd vs. even trials)
        metas_oddEven(sub,1,:) = compute_all_measures(stim(1:2:end), resp(1:2:end), conf(1:2:end), nRatings);
        metas_oddEven(sub,2,:) = compute_all_measures(stim(2:2:end), resp(2:2:end), conf(2:2:end), nRatings);

        %% Split-Half Reliability (by day)
        for day = 1:6
            for bs = 1:length(binSize_SH)-1
                bin_size = 2 * binSize_SH(bs);
                for bin_num = 1:400/bin_size
                    filter_odds  = [false(1,bin_size*(bin_num-1)), repmat([true,false], 1,bin_size/2)];
                    filter_evens = [false(1,bin_size*(bin_num-1)), repmat([false,true], 1,bin_size/2)];
                    metas_splitHalf{bs}(sub,bin_num,day,1,:) = compute_all_measures( ...
                        stim_day{day}(filter_odds),  resp_day{day}(filter_odds),  conf_day{day}(filter_odds),  nRatings);
                    metas_splitHalf{bs}(sub,bin_num,day,2,:) = compute_all_measures( ...
                        stim_day{day}(filter_evens), resp_day{day}(filter_evens), conf_day{day}(filter_evens), nRatings);
                end
            end

            % 400-trial bin: spans two consecutive days
            if mod(day,2) == 1
                stim400 = [stim_day{day}; stim_day{day+1}(1:300)];
                resp400 = [resp_day{day}; resp_day{day+1}(1:300)];
                conf400 = [conf_day{day}; conf_day{day+1}(1:300)];
                metas_splitHalf{4}(sub,1,ceil(day/2),1,:) = compute_all_measures( ...
                    stim400(1:2:800), resp400(1:2:800), conf400(1:2:800), nRatings);
                metas_splitHalf{4}(sub,1,ceil(day/2),2,:) = compute_all_measures( ...
                    stim400(2:2:800), resp400(2:2:800), conf400(2:2:800), nRatings);
            end
        end

        %% Test-Retest Reliability (by day and bin)
        for day = 1:6
            for bs = 1:length(binSize_TR)
                bin_size = binSize_TR(bs);
                for bin_num = 1:400/bin_size
                    filt = [false(1,bin_size*(bin_num-1)), true(1,bin_size)];
                    metas_testRetest{bs}(sub,bin_num,day,:) = compute_all_measures( ...
                        stim_day{day}(filt), resp_day{day}(filt), conf_day{day}(filt), nRatings);
                end
            end
        end

        %% Precision (original + confidence-altered versions)
        for day = 1:6
            for bs = 1:length(binSize_TR)
                bin_size = binSize_TR(bs);
                for bin_num = 1:400/bin_size
                    filt = [false(1,bin_size*(bin_num-1)), true(1,bin_size)];
                    metas_precision{bs}(sub,bin_num,day,1,:) = metas_testRetest{bs}(sub,bin_num,day,:);
                    for alter = 1:length(prop_altered)
                        metas_precision{bs}(sub,bin_num,day,alter+1,:) = metasAlteredConf( ...
                            stim_day{day}(filt), resp_day{day}(filt), conf_day{day}(filt), ...
                            nRatings, prop_altered(alter));
                    end
                end
            end
        end
    end
end

%% Save or Load Results
results_path = fullfile(root_dir, 'Results', 'results_Haddara');

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
