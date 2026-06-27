%% Shekhar (2021) Dataset Analysis
% Replicates the analysis from Rahnev (2025), *Nature Communications*, 16(1), 701.
%
% Dataset: Shekhar & Rahnev (2021) — multi-day, multi-contrast visual task.
% Subjects completed 3 days with 3 contrast levels (easy/medium/hard),
% ~200 trials per contrast per day. Continuous confidence ratings (50–100)
% are discretized into 6 bins. Analyses cover difficulty dependence,
% metacognitive bias dependence, split-half, test-retest, and across-measure
% correlations — all stratified by contrast level.

%% Setup
clear; close all; clc

% Set to 1 to recompute from raw data; 0 to load saved results
recompute_measures = 0;

%% Parameters
num_contrasts = 3;                   % easy, medium, hard
nRatings      = 6;                   % discretized confidence bins
edges         = 50:50/nRatings:100;  % bin edges for confidence discretization
binSize_SH    = [50, 100, 200, 400]; % split-half bin sizes
binSize_TR    = [50, 100, 200];      % test-retest bin sizes (max 200 trials/contrast/day)
prop_altered  = [.02, .04, .06];     % confidence alteration levels for precision

%% Load Data and Helper Functions
root_dir = fileparts(fileparts(mfilename('fullpath')));
load(fullfile(root_dir, 'Preprocess', 'dataset_Shekhar_2021'));
addpath(genpath(fullfile(root_dir, 'helperFunctions')));

fprintf('Loaded %d subjects.\n', length(data));

%% Measure Labels
variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', '\DeltaConf', ...
    'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', '\DeltaConf-Ratio', ...
    'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', '\DeltaConf-Diff', ...
    'meta-noise', 'meta-uncertainty', 'd''', 'Criterion', 'Confidence'};

%% Compute All Measures (loop over subjects and contrasts)
if recompute_measures
    for sub = 1:length(data)
        fprintf('Processing subject %d / %d\n', sub, length(data));

        for contr = 1:num_contrasts
            % Extract trials for this contrast
            filter         = data{sub}.contrast == contr;
            stim           = data{sub}.stim(filter);
            resp           = data{sub}.resp(filter);
            conf_raw       = data{sub}.conf(filter);
            conf           = discretize(conf_raw, edges); % bin into nRatings levels
            day_of_testing = data{sub}.day(filter);

            % Segment by day (days 1–3)
            for d = 1:3
                stim_day{d} = stim(day_of_testing == d);
                resp_day{d} = resp(day_of_testing == d);
                conf_day{d} = conf(day_of_testing == d);
            end

            %% Raw Measures and Difficulty Dependence
            metas_raw(sub,contr,:)  = compute_all_measures(stim, resp, conf, nRatings);
            metas_diff(sub,contr,:) = compute_all_measures(stim, resp, conf, nRatings);

            %% Metacognitive Bias Dependence (per contrast)
            metas_confRecode(sub,contr,1,:) = compute_all_measures(stim, resp, xue_recode(conf,1), nRatings-1);
            metas_confRecode(sub,contr,2,:) = compute_all_measures(stim, resp, xue_recode(conf,2), nRatings-1);

            %% Across-Measure Correlations (odd vs. even trials, per contrast)
            metas_oddEven(sub,1,contr,:) = compute_all_measures(stim(1:2:end), resp(1:2:end), conf(1:2:end), nRatings);
            metas_oddEven(sub,2,contr,:) = compute_all_measures(stim(2:2:end), resp(2:2:end), conf(2:2:end), nRatings);

            %% Split-Half Reliability (by contrast, ignoring day)
            for bs = 1:length(binSize_SH)
                bin_size = 2 * binSize_SH(bs);
                for bin_num = 1:800/bin_size
                    filter_odds  = [false(1,bin_size*(bin_num-1)), repmat([true,false], 1,bin_size/2)];
                    filter_evens = [false(1,bin_size*(bin_num-1)), repmat([false,true], 1,bin_size/2)];
                    metas_splitHalf{bs}(sub,bin_num,contr,1,:) = compute_all_measures( ...
                        stim(filter_odds),  resp(filter_odds),  conf(filter_odds),  nRatings);
                    metas_splitHalf{bs}(sub,bin_num,contr,2,:) = compute_all_measures( ...
                        stim(filter_evens), resp(filter_evens), conf(filter_evens), nRatings);
                end
            end

            %% Test-Retest Reliability (by contrast and day)
            for d = 1:3
                for bs = 1:length(binSize_TR)
                    bin_size = binSize_TR(bs);
                    for bin_num = 1:200/bin_size
                        filt = [false(1,bin_size*(bin_num-1)), true(1,bin_size)];
                        metas_testRetest{bs}(sub,contr,bin_num,d,:) = compute_all_measures( ...
                            stim_day{d}(filt), resp_day{d}(filt), conf_day{d}(filt), nRatings);
                    end
                end
            end
        end
    end
end

%% Save or Load Results
results_path = fullfile(root_dir, 'Results', 'results_Shekhar');

if recompute_measures
    save(results_path, 'metas_*', 'variable_names');
    fprintf('Results saved to %s\n', results_path);
else
    load(results_path);
    fprintf('Results loaded from %s\n', results_path);
end

%% Summary: Average Measures by Contrast Level
fprintf('\n--- Group Means by Contrast Level ---\n');
contrast_labels = {'Low (easy)', 'Medium', 'High (hard)'};
for contr = 1:num_contrasts
    fprintf('\n  %s contrast:\n', contrast_labels{contr});
    for m = [1,6,16,17,18]  % key measures: meta-d', M-Ratio, meta-noise, meta-unc, d'
        fprintf('    %-20s : %.4f\n', variable_names{m}, nanmean(metas_diff(:,contr,m)));
    end
end
