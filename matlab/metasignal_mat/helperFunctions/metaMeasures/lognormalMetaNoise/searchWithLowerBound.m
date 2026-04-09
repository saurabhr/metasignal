function info_best_value = searchWithLowerBound(lower_bound_info, func_name, input)

% Set parameters
initial_step = .5;
tau = (sqrt(5)-1) / 2;
x1_info = lower_bound_info;

% Compute the first step of the search away from the boundary
x1 = x1_info.x;
x2 = x1 + initial_step;
x2_info = eval([func_name '(x2,input)']);

% If the function value for the second point is higher, switch to
% goldenSearch within the [x1, x2] interval; else, continue to sample
if x1_info.logL < x2_info.logL
    info_best_value = goldenSearch([x1,x2], func_name, input);
    
else
    % If the function value for the second point is lower, continue to
    % sample points until a boundary is reached
    while x1_info.logL >= x2_info.logL

        % Decide when to stop if the sequence continues downward
        if strcmp(func_name, 'logL_func_criteria') && x2_info.x > 6
            % When going over the criteria, stop when muConf > 8
            info_best_value = x2_info;
            return
        elseif strcmp(func_name, 'logL_func_metaNoise') && x2_info.x > 4
            % When going over metaNoise, when we get to 4.7361, test 5
            info5 = eval([func_name '(5,input)']);
            
            % If the value continues to decrease at 5, return 5; if it
            % increases, then do goldenSearch on previous interval
            if x2_info.logL > info5.logL
                info_best_value = info5;
            else
                bounds = [x1_info.x, x2_info.x];
                info_best_value = goldenSearch(bounds, func_name, input);
            end
            return
        end
        
        % Make the new initial value the former x2 and repeat the process
        lower_bound_info = x1_info;
        x1_info = x2_info;
        x2 = (x1_info.x - lower_bound_info.x*tau) / (1-tau);
        x2_info = eval([func_name '(x2,input)']);
    end

    % Once we reach a point where x1_info.logL <= x2_info.logL, switch to a
    % goldenSearch with a known x1
    bounds = [lower_bound_info.x, x2_info.x];
    info_best_value = goldenSearch(bounds, func_name, input, x1_info);
end