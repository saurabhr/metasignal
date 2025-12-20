function metas = metasAlteredConf(stim, resp, conf, nRatings, prop_altered)

% Compute general parameters
num_trials = length(conf);
num_to_alter = round(num_trials * prop_altered);
conf_altered = conf;

% Loop over trials and alter conf ratings until desired alterations reached
num_altered = 0;
for trial=1:num_trials
    % If trial is correct and has conf > 1, decrease conf by 1
    if stim(trial)==resp(trial) && conf(trial)>1
        conf_altered(trial) = conf_altered(trial) - 1;
        num_altered = num_altered + 1;
        
    % If trial is incorrect and has conf less than maximum, increase conf by 1
    elseif stim(trial)~=resp(trial) && conf(trial)<nRatings
        conf_altered(trial) = conf_altered(trial) + 1;
        num_altered = num_altered + 1;
    end
    
    % Break loop as soon as enough trials were altered
    if num_altered == num_to_alter
        break;
    end
end

% Compute meta measures with the altered confidence
metas = compute_all_measures(stim, resp, conf_altered, nRatings);