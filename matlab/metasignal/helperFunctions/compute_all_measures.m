function metas = compute_all_measures(stim, resp, conf, nRatings)

%% Remove NaN values from input
stim_noNaN = stim(~isnan(stim) & ~isnan(resp) & ~isnan(conf));
resp_noNaN = resp(~isnan(stim) & ~isnan(resp) & ~isnan(conf));
conf_noNaN = conf(~isnan(stim) & ~isnan(resp) & ~isnan(conf));
stim=stim_noNaN; resp=resp_noNaN; conf=conf_noNaN;

%% Make input into 0/1 (for meta-d' codes to work properly)
stim(stim==min(stim)) = 0; stim(stim==max(stim)) = 1; 
resp(resp==min(resp)) = 0; resp(resp==max(resp)) = 1; 


%% Compute basic quantities
[dprime, c] = compute_SDT_resp(stim, resp);
mean_conf = mean(conf);

%% Return NaN in cases of (1) perfect accuracy, (2) d' is exactly 0, or 
%% (3) a single confidence rating for all trials
if isequal(stim,resp) || dprime==0 || length(unique(conf))==1
    metas = NaN(1,20);
    return
end

%% Compute all measures
% meta-d', M-Ratio, M-Diff
output = type2_SDT_MLE(stim, resp, conf, nRatings, [], 1);
meta_d = output.meta_da; M_ratio = output.M_ratio; M_diff = output.M_diff;

% Turn M-Ratio measure into NaN if we need to divide by a negative or a very small value
if dprime < .2
    M_ratio = NaN;
end

% AUC2, AUC2-Ratio, AUC2-Diff
[AUC2, AUC2_ratio, AUC2_diff] = SDTtype2AUC(stim, resp, conf, nRatings);

% Gamma, Gamma-Ratio, Gamma-Diff
[gamma, gamma_ratio, gamma_diff] = SDTgamma(stim, resp, conf, nRatings);

% Phi, Phi-Ratio, Phi-Diff
[phi, phi_ratio, phi_diff] = SDTphi(stim, resp, conf, nRatings);

% DeltaConf, DeltaConf-Ratio, DeltaConf-Diff
[deltaConf, deltaConf_ratio, deltaConf_diff] = SDTdeltaConf(stim, resp, conf, nRatings);

% metaNoise (Lognormal model; Shekhar & Rahnev, 2021)
metaNoise = compute_metaNoise(stim, resp, conf, nRatings);

% metaUncertainty (Boundy-Singer et al)
metaUncertainty = compute_metaUncertainty(stim, resp, conf, nRatings);

%% Combine all measures into a single variable
metas = [meta_d, AUC2, gamma, phi, deltaConf, ...
    M_ratio, AUC2_ratio, gamma_ratio, phi_ratio, deltaConf_ratio, ...
    M_diff, AUC2_diff, gamma_diff, phi_diff, deltaConf_diff, ...
    metaNoise, metaUncertainty, dprime, c, mean_conf];