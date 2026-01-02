function p_high_conf = evaluateIntegral(mu_Gauss, mu_Lognormal, metaNoise)

%--------------------------------------------------------------------------
% Numerically evaluate the double integral with Lognormal meta noise and 
% Gaussian evidence distributions to determine the proportion of high 
% confidence trials.
% Evidence distribution:
% - mu_Gauss - mean (SD = 1)
% Conf criterion distribution:
% - location parameter - mu_Lognormal
% - scale parameter - metaNoise
%--------------------------------------------------------------------------

% % Area of high confidence
% fun = @(y,x) exp(-(x-mu_Gauss).^2./2 - (log(y)-mu_Lognormal).^2./...
%     (2.*metaNoise.^2))./(2*pi*y*metaNoise);
% p_high_conf = integral2(fun, 0, Inf, @(y) y, Inf);

global lookupTable values

% Find the closest indexes
mu_index = findInterval(mu_Gauss, values.mus);
crit_index = findInterval(mu_Lognormal, values.crit);
metaNoise_index = findInterval(metaNoise, values.metaNoise);

% Compute the closest values and their distances from current value
for num_mu=1:length(mu_index)
    for num_crit=1:length(crit_index)
        for num_metaNoise=1:length(metaNoise_index)
            p(num_mu,num_crit,num_metaNoise) = lookupTable(mu_index(num_mu),...
                crit_index(num_crit), metaNoise_index(num_metaNoise));
            weight(num_mu,num_crit,num_metaNoise) = 1/sqrt((mu_index(num_mu)-mu_Gauss)^2 + ...
                (crit_index(num_crit)-mu_Lognormal)^2 + (metaNoise_index(num_metaNoise)-metaNoise)^2);
        end
    end
end

% Take a weighted mean
if length(mu_index)+length(crit_index)+length(metaNoise_index) == 3
    p_high_conf = p;
else
    p_high_conf = sum(sum(sum(p.*weight))) / sum(sum(sum(weight)));
end