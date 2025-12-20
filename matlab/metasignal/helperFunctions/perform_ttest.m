function [p, t, df, Cohen_d, CI] = perform_ttest(data, test_description, display_result)

% Fix missing input
if ~exist('test_description', 'var')
    test_description = '';
end

if ~exist('display_result', 'var')
    display_result = 1;
end

% Perform one sample t-test
[~,p,~,stats] = ttest(data);
t = stats.tstat;
df = stats.df;
Cohen_d = t./sqrt(df+1);

% Compute 95% Confidence Interval (CI)
SEM = nanstd(data)/sqrt(sum(~isnan(data)));  % SE computer by ignoring NaN values
ts = tinv([0.025  0.975],length(data)-1);    % t cutoffs
CI = nanmean(data) + ts'.*SEM;

% Display results
if display_result==1
    fprintf([test_description ': t(' num2str(df) ') = ' num2str(t) ...
        ', p = ' num2str(p) ', Cohen''s d = ' num2str(Cohen_d) ...
        ', 95% CI = ' num2str(CI) '\n']);
end