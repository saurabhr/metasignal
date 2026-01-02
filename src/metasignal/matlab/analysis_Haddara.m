%analysis_Haddara

clear
close all
clc

% Decide whether to compute M measures or just load saved values
recompute_measures = 0;

% Define important parameters
nRatings = 4;
binSize_SH = [50, 100, 200, 400]; %split-half bin sizes; ignore day
binSize_TR = [50, 100, 200, 400]; %test-retest bin sizes; only use the first 400 trials each day
prop_altered = [.02, .04, .06]; %proportion of altered confidence ratings for precision analyses

% Load data and add helper functions
load('Preprocess/dataset_Haddara_2022_Expt2');
addpath(genpath(fullfile(pwd, 'helperFunctions')));

% Decide whether to recompute all measures and loop over all subjects
if recompute_measures
    for sub=1:length(data)
        sub
        
        % Determine basic variables
        stim = data{sub}.stim;
        resp = data{sub}.resp;
        conf = data{sub}.conf;
        day_of_testing = data{sub}.day;
        for day=2:7 %Look at days 2-7 only (recode them to 1-6)
            stim_day{day-1} = stim(day_of_testing==day);
            resp_day{day-1} = resp(day_of_testing==day);
            conf_day{day-1} = conf(day_of_testing==day);
        end
        metas_raw(sub,:) = compute_all_measures(stim, resp, conf, nRatings);
               
        %% COMPUTE DEPENDENCE ON METACOGNITIVE BIAS
        % Compute meta measures for each of the two recodings from Xue et al. (2021)
        metas_confRecode(sub,1,:) = compute_all_measures(stim, resp, xue_recode(conf,1), nRatings-1);
        metas_confRecode(sub,2,:) = compute_all_measures(stim, resp, xue_recode(conf,2), nRatings-1);
        
        %% COMPUTE DATA FOR ACROSS-MEASURE CORRELATIONS
        % Compute each measure for odd and even trials
        metas_oddEven(sub,1,:) = compute_all_measures(stim(1:2:end), resp(1:2:end), conf(1:2:end), nRatings);
        metas_oddEven(sub,2,:) = compute_all_measures(stim(2:2:end), resp(2:2:end), conf(2:2:end), nRatings);
        
        %% COMPUTE SPLIT-HALF RELIABILITY (Separate by day)
        for day=1:6
            for binSize_num=1:length(binSize_SH)-1 %deal with the 400-trial bin separately
                bin_size = 2*binSize_SH(binSize_num); %size of block from which the odd/even trials equal binSize_SH(binSize_num)
                for bin_num=1:400/bin_size
                    filter_odds = [false(1,bin_size*(bin_num-1)), repmat([true,false],1,bin_size/2)];
                    metas_splitHalf{binSize_num}(sub,bin_num,day,1,:) = compute_all_measures(stim_day{day}(filter_odds), ...
                        resp_day{day}(filter_odds), conf_day{day}(filter_odds), nRatings);
                    filter_evens = [false(1,bin_size*(bin_num-1)), repmat([false,true],1,bin_size/2)];
                    metas_splitHalf{binSize_num}(sub,bin_num,day,2,:) = compute_all_measures(stim_day{day}(filter_evens), ...
                        resp_day{day}(filter_evens), conf_day{day}(filter_evens), nRatings);
                end
            end
            
            % Deal with the 400-trial bin
            if mod(day,2)==1
                stim400 = [stim_day{day}; stim_day{day+1}(1:300)];
                resp400 = [resp_day{day}; resp_day{day+1}(1:300)];
                conf400 = [conf_day{day}; conf_day{day+1}(1:300)];
                metas_splitHalf{4}(sub,1,ceil(day/2),1,:) = compute_all_measures(stim400(1:2:800), ...
                    resp400(1:2:800), conf400(1:2:800), nRatings);
                metas_splitHalf{4}(sub,1,ceil(day/2),2,:) = compute_all_measures(stim400(2:2:800), ...
                    resp400(2:2:800), conf400(2:2:800), nRatings);
            end
        end
        
        %% COMPUTE TEST-RETEST RELIABILITY (Separated by day)
        for day=1:6
            for binSize_num=1:length(binSize_TR)
                bin_size = binSize_TR(binSize_num);
                for bin_num=1:400/bin_size
                    filt = [false(1,bin_size*(bin_num-1)), true(1,bin_size)];
                    metas_testRetest{binSize_num}(sub,bin_num,day,:) = ...
                        compute_all_measures(stim_day{day}(filt),resp_day{day}(filt),conf_day{day}(filt),nRatings);
                end
            end
        end
        
        %% COMPUTE PRECISION
        for day=1:6
            for binSize_num=1:length(binSize_TR)
                bin_size = binSize_TR(binSize_num);
                for bin_num=1:400/bin_size
                    filt = [false(1,bin_size*(bin_num-1)), true(1,bin_size)];
                    metas_precision{binSize_num}(sub,bin_num,day,1,:) = metas_testRetest{binSize_num}(sub,bin_num,day,:); %original
                    for alter=1:length(prop_altered)
                        metas_precision{binSize_num}(sub,bin_num,day,alter+1,:) = metasAlteredConf(stim_day{day}(filt), ...
                            resp_day{day}(filt), conf_day{day}(filt), nRatings, prop_altered(alter));
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
    save Results/results_Haddara metas_* variable_names
else
    load('Results/results_Haddara');
end