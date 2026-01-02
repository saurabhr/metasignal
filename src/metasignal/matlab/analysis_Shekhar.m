%analysis_Shekhar

clear
close all
clc

% Decide whether to compute M measures or just load saved values
recompute_measures = 0;

% Define important parameters
num_contrasts = 3;
nRatings = 6;
edges = 50:50/nRatings:100;
binSize_SH = [50, 100, 200, 400]; %split-half bin sizes; ignore day
binSize_TR = [50, 100, 200]; %test-retest bin sizes; only use the first 200 trials of a given contrast each day

% Load data and add helper functions
load('Preprocess/dataset_Shekhar_2021');
addpath(genpath(fullfile(pwd, 'helperFunctions')));

% Decide whether to recompute all measures and loop over all subjects
if recompute_measures
    for sub=1:length(data)
        sub
        
        % Loop over contrasts
        for contr=1:num_contrasts
            
            % Determine basic variables
            filter = data{sub}.contrast == contr;
            stim = data{sub}.stim(filter);
            resp = data{sub}.resp(filter);
            conf_raw = data{sub}.conf(filter);
            conf = discretize(conf_raw, edges); %Discretize confidence into bins
            day_of_testing = data{sub}.day(filter);
            for contr=1:3
                stim_day{contr} = stim(day_of_testing==contr);
                resp_day{contr} = resp(day_of_testing==contr);
                conf_day{contr} = conf(day_of_testing==contr);
            end
            metas_raw(sub,contr,:) = compute_all_measures(stim, resp, conf, nRatings);
            
            %% COMPUTE DEPENDENCE ON DIFFICULTY (Compute values for each contrast separately)
            metas_diff(sub,contr,:) = compute_all_measures(stim, resp, conf, nRatings);
            
            %% COMPUTE DEPENDENCE ON METACOGNITIVE BIAS (Compute values for each contrast separately)
            % Compute meta measures for each of the two recodings from Xue et al. (2021)
            metas_confRecode(sub,contr,1,:) = compute_all_measures(stim, resp, xue_recode(conf,1), nRatings-1);
            metas_confRecode(sub,contr,2,:) = compute_all_measures(stim, resp, xue_recode(conf,2), nRatings-1);
            
            %% COMPUTE DATA FOR ACROSS-MEASURE CORRELATIONS
            % Compute each measure for odd and even trials
            metas_oddEven(sub,1,contr,:) = compute_all_measures(stim(1:2:end), resp(1:2:end), conf(1:2:end), nRatings);
            metas_oddEven(sub,2,contr,:) = compute_all_measures(stim(2:2:end), resp(2:2:end), conf(2:2:end), nRatings);
            
            %% COMPUTE SPLIT-HALF RELIABILITY (Separate by contrast but ignore day)
            for binSize_num=1:length(binSize_SH)
                size = 2*binSize_SH(binSize_num); %size of block from which the odd/even trials equal binSize_SH(binSize_num)
                for bin_num=1:800/size
                    filter_odds = [false(1,size*(bin_num-1)), repmat([true,false],1,size/2)];
                    metas_splitHalf{binSize_num}(sub,bin_num,contr,1,:) = compute_all_measures(stim(filter_odds), ...
                        resp(filter_odds), conf(filter_odds), nRatings);
                    filter_evens = [false(1,size*(bin_num-1)), repmat([false,true],1,size/2)];
                    metas_splitHalf{binSize_num}(sub,bin_num,contr,2,:) = compute_all_measures(stim(filter_evens), ...
                        resp(filter_evens), conf(filter_evens), nRatings);
                end
            end
            
            %% COMPUTE TEST-RETEST RELIABILITY (Separated by contrast and day)
            for contr=1:3
                for binSize_num=1:length(binSize_TR)
                    size = binSize_TR(binSize_num);
                    for bin_num=1:200/size
                        filt = [false(1,size*(bin_num-1)), true(1,size)];
                        metas_testRetest{binSize_num}(sub,contr,bin_num,contr,:) = ...
                            compute_all_measures(stim_day{contr}(filt),resp_day{contr}(filt),conf_day{contr}(filt),nRatings);
                    end
                end
            end
        end
    end
end


%% Load or save all meta scores
if recompute_measures
    variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', '\DeltaConf', ...
        'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', '\DeltaConf-Ratio', ...
        'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', '\DeltaConf-Diff', ...
        'meta-noise', 'meta-uncertainty', 'd''', 'Criterion', 'Confidence'};
    save Results/results_Shekhar metas_* variable_names
else
    load Results/results_Shekhar
end