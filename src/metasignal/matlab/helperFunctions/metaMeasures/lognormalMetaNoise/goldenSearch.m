function info_best_value = goldenSearch(bounds, func_name, input, x1_info)

%-------------
% The function implements the golder ratio search.
%-------------

% Set parameters
tau = (sqrt(5)-1) / 2;
low_bound = bounds(1);
high_bound = bounds(2);
if strcmp(func_name, 'logL_func_metaNoise')
    max_diff_between_bounds = .02;
elseif strcmp(func_name, 'logL_func_criteria')
    max_diff_between_bounds = .1;
end

% Compute initial function values
if ~exist('x1_info', 'var')
    x1 = low_bound + (1-tau)*(high_bound-low_bound);
    x1_info = eval([func_name '(x1,input)']);
end
x2 = low_bound + tau*(high_bound-low_bound);
x2_info = eval([func_name '(x2,input)']);

% Save interim steps
values_tested = [x1_info.x, x2_info.x];
orig_bounds = [low_bound, high_bound];
func_values = [x1_info.logL, x2_info.logL];

% Perform the golden ratio search
while abs(high_bound-low_bound) > max_diff_between_bounds
    
    % In case the function gives lower value on the left, shift the upper
    % bound to the right value and re-define x1 and x2
    if x1_info.logL < x2_info.logL
        
        % Redefine the values
        high_bound = x2_info.x;
        x2_info = x1_info;
        x1 = low_bound+(1-tau)*(high_bound-low_bound);
        x1_info = eval([func_name '(x1,input)']);
        
        % Save interim steps
        values_tested(end+1) = x1;
        func_values(end+1) = x1_info.logL;
        
        % In case the function gives lower value on the right, shift the lower
        % bound to the left value and re-define x1 and x2
    else
        
        % Redefine the values
        low_bound = x1_info.x;
        x1_info = x2_info;
        x2 = low_bound+tau*(high_bound-low_bound);
        x2_info = eval([func_name '(x2,input)']);
        
        % Save interim steps
        values_tested(end+1) = x2;
        func_values(end+1) = x2_info.logL;
    end
end

% At the end of the search, determine the best answer among the values tested
if x1_info.logL < x2_info.logL
    info_best_value = x1_info;
else
    info_best_value = x2_info;
end

% % Display info
%values_tested
% [orig_bounds,values_tested,info_best_value.p_HC']
% func_values(3:end)