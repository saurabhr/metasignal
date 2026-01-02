%analysis_Rouault2

clear
close all
clc

% Decide whether to compute M measures or just load saved values
recompute_measures = 0;

% Define important parameters
nRatings = 6;

% Load data and add helper functions
load('Preprocess/dataset_Rouault_2018_Expt2');
addpath(genpath(fullfile(pwd, 'helperFunctions')));

% Decide whether to recompute all measures and loop over all subjects
if recompute_measures
    for sub=1:length(data)
        sub
        filt_high = data{sub}.contrast>median(data{sub}.contrast);
        metas_diff(sub,1,:) = compute_all_measures(data{sub}.stim(filt_high==0), ...
            data{sub}.resp(filt_high==0), data{sub}.conf(filt_high==0), nRatings);
        metas_diff(sub,2,:) = compute_all_measures(data{sub}.stim(filt_high==1), ...
            data{sub}.resp(filt_high==1), data{sub}.conf(filt_high==1), nRatings);
    end
end


%% Load or save all meta scores
if recompute_measures
    variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', '\DeltaConf', ...
        'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', '\DeltaConf-Ratio', ...
        'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', '\DeltaConf-Diff', ...
        'meta-noise', 'meta-uncertainty', 'd''', 'Criterion', 'Confidence'};
    save Results/results_Rouault2 metas_* variable_names
else
    load Results/results_Rouault2
end