%analysis_Maniscalco

clear
close all
clc

% Decide whether to compute M measures or just load saved values
recompute_measures = 0;

% Define important parameters
nRatings = 4;
binSize_SH = [50, 100, 200, 400]; %split-half bin sizes
prop_altered = [.02, .04, .06]; %proportion of altered confidence ratings for precision analyses

% Load data and add helper functions
load('Preprocess/dataset_Maniscalco_2017_expt1');
addpath(genpath(fullfile(pwd, 'helperFunctions')));

% Decide whether to recompute all measures and loop over all subjects
if recompute_measures
    for sub=1:length(data)
        sub
        
        % Determine basic variables
        stim = data{sub}.stim;
        resp = data{sub}.resp;
        conf = data{sub}.conf;
        metas_raw(sub,:) = compute_all_measures(stim, resp, conf, nRatings);
        
        %% COMPUTE PRECISION
        for binSize_num=1:length(binSize_SH)
            bin_size = binSize_SH(binSize_num);
            for bin_num=1:floor(length(conf)/bin_size)
                filt = [false(1,bin_size*(bin_num-1)), true(1,bin_size)];
                metas_precision{binSize_num}(sub,bin_num,1,:) = compute_all_measures(stim(filt), ...
                    resp(filt), conf(filt), nRatings); %original
                for alter=1:length(prop_altered)
                    metas_precision{binSize_num}(sub,bin_num,alter+1,:) = metasAlteredConf(stim(filt), ...
                        resp(filt), conf(filt), nRatings, prop_altered(alter));
                end
            end
        end
        
        %% COMPUTE DEPENDENCE ON METACOGNITIVE BIAS
        % Compute meta measures for each of the two recodings from Xue et al. (2021)
        metas_confRecode(sub,1,:) = compute_all_measures(stim, resp, xue_recode(conf,1), nRatings-1);
        metas_confRecode(sub,2,:) = compute_all_measures(stim, resp, xue_recode(conf,2), nRatings-1);
        
        %% COMPUTE DATA FOR ACROSS-MEASURE CORRELATIONS
        % Compute each measure for odd and even trials
        metas_oddEven(sub,1,:) = compute_all_measures(stim(1:2:end), resp(1:2:end), conf(1:2:end), nRatings);
        metas_oddEven(sub,2,:) = compute_all_measures(stim(2:2:end), resp(2:2:end), conf(2:2:end), nRatings);
        
        %% COMPUTE SPLIT-HALF RELIABILITY
        for binSize_num=1:length(binSize_SH)
            bin_size = 2*binSize_SH(binSize_num); %size of block from which the # of odd/even trials equal binSize_SH(binSize_num)
            for bin_num=1:floor(length(conf)/bin_size)
                filter_odds = [false(1,bin_size*(bin_num-1)), repmat([true,false],1,bin_size/2)];
                metas_splitHalf{binSize_num}(sub,bin_num,1,:) = compute_all_measures(stim(filter_odds), ...
                    resp(filter_odds), conf(filter_odds), nRatings);
                filter_evens = [false(1,bin_size*(bin_num-1)), repmat([false,true],1,bin_size/2)];
                metas_splitHalf{binSize_num}(sub,bin_num,2,:) = compute_all_measures(stim(filter_evens), ...
                    resp(filter_evens), conf(filter_evens), nRatings);
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
    save Results/results_Maniscalco metas_* variable_names
else
    load Results/results_Maniscalco
end