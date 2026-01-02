function conf = xue_recode(conf, lowHighRecoding)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Recoding procedure developed by Xue, Shekhar, & Rahnev (2021) Consc Cogn.
% Input:
%   conf: confidence vector (should have at least 3 unique values and only
%   consist of integers)
%   lowHighRecoding: whether confidence should be recoded to lower values
%   (1) or higher values (2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check if there are at least 3+ unique confidence ratings and that only
% integer values are present. If not, the procedure can't work and thus 
% should return NaN. Ignore NaN values in the input.
if length(unique(conf(~isnan(conf)))) < 3 || any(conf(~isnan(conf)) ~= round(conf(~isnan(conf))))
    conf = NaN;
    warning('Xue recoding procedure returned NaN. Not enough unique values or non-integet confidence values');
    return
end

%% Do the recoding
if lowHighRecoding == 1 %remove lowest criterion (results in lower average confidence)
    conf = conf - 1;
    conf(conf==min(conf)) = min(conf) + 1;
elseif lowHighRecoding == 2 %remove highest criterion (results in higher average confidence)
    conf(conf==max(conf)) = max(conf) - 1;
end
