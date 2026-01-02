%analysis_Rouault1

clear
close all
clc

% Decide whether to compute M measures or just load saved values
recompute_measures = 0;

% Define important parameters
nRatings = 6;

% Load data and add helper functions
load('Preprocess/dataset_Rouault_2018_Expt1');
addpath(genpath(fullfile(pwd, 'helperFunctions')));

% Decide whether to recompute all measures and loop over all subjects
if recompute_measures
    for sub=1:length(data)
        sub
        
        % Determine basic variables
        stim = data{sub}.stim;
        resp = data{sub}.resp;
        conf = data{sub}.conf - 5;
        conf(conf < 1) = 1;
        
        % COMPUTE DEPENDENCE ON DIFFICULTY
        metas_diff(sub,1,:) = compute_all_measures(stim(data{sub}.contrast<=35), ...
            resp(data{sub}.contrast<=35), conf(data{sub}.contrast<=35), nRatings);
        metas_diff(sub,2,:) = compute_all_measures(stim(data{sub}.contrast>35), ...
            resp(data{sub}.contrast>35), conf(data{sub}.contrast>35), nRatings);
        
    end
end


%% Load or save all meta scores
if recompute_measures
    variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', '\DeltaConf', ...
        'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', '\DeltaConf-Ratio', ...
        'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', '\DeltaConf-Diff', ...
        'meta-noise', 'meta-uncertainty', 'd''', 'Criterion', 'Confidence'};
    save Results/results_Rouault1 metas_* variable_names
else
    load Results/results_Rouault1
end