function output = logL_func_criteria(muConf, input)

%--------------------------------------------------------------------------
% Calculate log likelihood for a given criterion location based on observed
% and predicted proportions of trials on each side of the criterion.
%--------------------------------------------------------------------------

% Calculate response probabilities
p_high_conf_stimS1 = evaluateIntegral(input.mus(1), muConf, input.metaNoise); 
p_high_conf_stimS2 = evaluateIntegral(input.mus(2), muConf, input.metaNoise);
probR_model = [1-p_high_conf_stimS1,p_high_conf_stimS1;...
    1-p_high_conf_stimS2,p_high_conf_stimS2];
probR_model(probR_model==0) = 10^-5;

% Compute logL
output.logL = -sum(sum(log(probR_model).*input.dataCounts_binary));

% Save the proportions of high confidence
output.p_HC = [p_high_conf_stimS1; p_high_conf_stimS2];
output.x = muConf;