function info_best_value_metaNoise = logL_func_metaNoise(metaNoise, f_metaNoise_input)

%--------------------------------------------------------------------------
% Estimate best fitting confidence criterion locations for each criterion
% for given metaNoise by maximising the log likelihood.
%--------------------------------------------------------------------------

global metaNoise_tested criteria_for_tested_metaNoise

% Set parameters
crit_dist_no_search = .01; %if the criteria distance is less than that, don't search but just pick the middle

% Simplify input
dataCounts = f_metaNoise_input.dataCounts;
prob_decResp = f_metaNoise_input.prob_decResp;
dprime = f_metaNoise_input.dprime;
c = f_metaNoise_input.c;
nRatings = size(dataCounts,2)/2;

% Constrain the criteria: if this metaNoise value is between two tested
% ones, then use those to constrain the criteria; if not, use [-4,4]
if metaNoise < max(metaNoise_tested)
    higher_values = metaNoise_tested;
    higher_values(higher_values < metaNoise) = NaN;
    [~,idx_closest_above] = min(higher_values);
    lower_values = metaNoise_tested;
    lower_values(lower_values > metaNoise) = NaN;
    [~,idx_closest_below] = max(lower_values);
    crit_bounds{1} = criteria_for_tested_metaNoise{idx_closest_below};
    crit_bounds{2} = criteria_for_tested_metaNoise{idx_closest_above};
else
    mu_limits = [-6, 6]; %limits on the mu of the confidence criteria
end
mu_limits = [-6, 6]; 

% Set the means of the stimulus distributions for S1 and S2 so that the 
% decision criterion is at 0 (which simplifies the later formulas). The 
% SD (sigma_sens) of the stimulus distributions is assumed to be 1.
mu_s1 = -dprime/2 - c(nRatings); 
mu_s2 = dprime/2 - c(nRatings);
f_crit_input.metaNoise = metaNoise;


%% Loop over the positive and negative confidence criteria
for critSide=1:2 %1:positive criteria, 2:negative criteria
    
    % Determine the mus (flip the mus for negative criteria to flip the axis)
    if critSide==1
        f_crit_input.mus = [mu_s1, mu_s2];
    else
        f_crit_input.mus = [-mu_s1, -mu_s2];
    end
    
    % Loop over the criteria
    for crit=1:nRatings-1
        % Split the data counts into two bins
        if critSide==1
            f_crit_input.dataCounts_binary = [sum(dataCounts(:,1:nRatings+crit),2), ...
                sum(dataCounts(:,nRatings+crit+1:end),2)];
        else
            % For negative criteria flip the direction so that the second value represents the high conf for the chosen response
            f_crit_input.dataCounts_binary = [sum(dataCounts(:,nRatings-crit+1:end),2), ...
                sum(dataCounts(:,1:nRatings-crit),2)];
        end
        
        % Check if criterion bounds already exist
        if exist('crit_bounds', 'var')
            limits = sort([crit_bounds{1}(critSide,crit), crit_bounds{2}(critSide,crit)]);
            if diff(limits) < crit_dist_no_search
                info_best_value_crit{crit} = logL_func_criteria(mean(limits), f_crit_input);
            else
                info_best_value_crit{crit} = goldenSearch(limits, 'logL_func_criteria', f_crit_input);
            end
        else
            % For the first criterion do goldenSearch; for subsequent ones, do searchWithLowerBound
            if crit == 1
                limits = mu_limits;
                info_best_value_crit{crit} = goldenSearch(limits, 'logL_func_criteria', f_crit_input);
            else
                info_best_value_crit{crit} = searchWithLowerBound(info_best_value_crit{crit-1}, 'logL_func_criteria', f_crit_input);
            end
        end
        critValues(critSide,crit) = info_best_value_crit{crit}.x;
        p_HC{critSide}(:,crit) = info_best_value_crit{crit}.p_HC;
    end
end


%% Organize the data from each criterion into a probR_model variable and compute logL
% Organize data from positive confidence criteria
high_conf_cum = [prob_decResp(:,2), p_HC{1}, [0;0]];
probR_model_critPos = high_conf_cum(:,1:end-1)-high_conf_cum(:,2:end);

% Organize data from negative confidence criteria
high_conf_cum = [prob_decResp(:,1), p_HC{2}, [0;0]];
probR_model_critNeg = high_conf_cum(:,1:end-1)-high_conf_cum(:,2:end);

% Combine data for low and high conf criteria
probR_model = [flip(probR_model_critNeg,2), probR_model_critPos];
probR_model(probR_model==0) = 10^-5;
logL = -sum(sum(log(probR_model).*dataCounts));

% Return all data
info_best_value_metaNoise = f_metaNoise_input;
info_best_value_metaNoise.x = metaNoise;
info_best_value_metaNoise.logL = logL;
criteria_for_tested_metaNoise{end+1} = critValues;
metaNoise_tested(end+1) = metaNoise;