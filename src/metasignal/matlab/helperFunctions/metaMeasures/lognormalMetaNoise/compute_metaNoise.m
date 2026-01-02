function [metaNoise, info] = compute_metaNoise(stim,resp,conf,nRatings)

%--------------------------------------------------------------------------
% This function generates model fits for the Lognormal meta noise model and
% outputs metaNoise parameter as a measure of metacognition

% The output field model contains the following information
% dprime - stimulus sensitivity (can vary between experimental conditions)
% c - response bias (same across all condition)
% confidence criteria - estimates from the fitting procedure
% logL - log likelihood computed from the best fitting parameters
% nParams - number of parameters in the model
%--------------------------------------------------------------------------

clear global
global metaNoise_tested criteria_for_tested_metaNoise lookupTable values

% Load the lookupTable
load lookupTable

%% Generate the dataCounts matrix (do not correct for zero counts)
% Set nR_S1 and nR_S1 to 0 
nR_stimS1 = zeros(1,2*nRatings);
nR_stimS2 = zeros(1,2*nRatings);

% Loop over the confidence ratings
for rating=1:nRatings
    % S1 responses
    nR_stimS1(nRatings-rating+1) = sum(stim==min(stim) & resp==min(stim) & conf==rating);
    nR_stimS2(nRatings-rating+1) = sum(stim==max(stim) & resp==min(stim) & conf==rating);
    
    % S2 responses
    nR_stimS1(nRatings+rating) = sum(stim==min(stim) & resp==max(stim) & conf==rating);
    nR_stimS2(nRatings+rating) = sum(stim==max(stim) & resp==max(stim) & conf==rating);
end
dataCounts = [nR_stimS1; nR_stimS2];


%% Compute d' and all criteria using logL
[dprime, c_init] = compute_SDTcriteria(stim, resp, conf, nRatings);
c_init(isnan(c_init)) = Inf; %Correct for rare cases where HR=1 and FAR=0
c = c_init;

% Loop over all criteria by the conf ratings that they separate
for crit=1:2*nRatings-1
    if c_init(crit)==-inf || c_init(crit)==inf
        range = -5:.01:5;
    else
        range = c_init(crit)-.5:.01:c_init(crit)+.5;
    end
    probR_S1_model = normcdf(range, -dprime/2, 1);
    probR_S2_model = normcdf(range, dprime/2, 1);
    probR_model = [probR_S1_model;probR_S2_model;1-probR_S1_model;1-probR_S2_model];
    probR_model(probR_model==0) = 10^-5;
    data = [sum(dataCounts(:,1:crit),2); sum(dataCounts(:,crit+1:end),2)];
    [~,idx] = min(-sum(log(probR_model).*data));
    c(crit) = range(idx);
end


%% Compute logL for metaNoise of 0
probR_S1_model = diff([0, normcdf(c, -dprime/2, 1), 1]);
probR_S2_model = diff([0, normcdf(c, dprime/2, 1), 1]);
probR_model = [probR_S1_model; probR_S2_model];
prob_decResp = [sum(probR_model(:,1:nRatings),2),sum(probR_model(:,nRatings+1:end),2)]; 
probR_model(probR_model==0) = 10^-5;
logL_mN0 = -sum(sum(log(probR_model).*dataCounts));

% Transform the computed criteria in the Lognormal's mu space
posCrit = c(nRatings+1:end)-c(nRatings);
negCrit = -(c(nRatings-1:-1:1)-c(nRatings)); %make positive
crit_values_all = [posCrit;negCrit];
crit_values_all(crit_values_all==0) = 10^-5; %remove zeros
crit_Lognorm = log(crit_values_all);
metaNoise_tested = 0;
criteria_for_tested_metaNoise{1} = crit_Lognorm;


%% Find best metaNoise value
f_metaNoise_input.dataCounts = dataCounts;
f_metaNoise_input.prob_decResp = prob_decResp;
f_metaNoise_input.dprime = dprime;
f_metaNoise_input.c = c;
lower_bound_info.x = 0;
lower_bound_info.logL = logL_mN0;
info_best_value = searchWithLowerBound(lower_bound_info, 'logL_func_metaNoise', f_metaNoise_input);
metaNoise = info_best_value.x;
logL = info_best_value.logL;


%% Save information about the model
info.dprime = dprime;
info.c = c;
info.logL = logL;