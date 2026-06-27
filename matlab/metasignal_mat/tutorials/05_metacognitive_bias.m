%% Tutorial 5 — Metacognitive Bias
% Tests whether each measure is sensitive to confidence *bias* — a systematic
% tendency to use high or low confidence regardless of accuracy.
%
% Uses the Xue et al. (2021) recoding method (xue_recode.m):
%   Recode 1: shifts ratings toward lower confidence (remove lowest criterion)
%   Recode 2: shifts ratings toward higher confidence (remove highest criterion)
%
% A good measure should change as little as possible between the two recodings.
% Replicates Rahnev (2025) Supp Tables 6–8.

%% Setup
clear; close all; clc

root_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root_dir, 'helperFunctions')));

variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', 'DeltaConf', ...
    'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', 'DeltaConf-Ratio', ...
    'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', 'DeltaConf-Diff', ...
    'meta-noise', 'meta-uncertainty', 'd''', 'Criterion', 'Confidence'};

N_MEAS     = 20;
nRatings   = 4;
n_subjects = 25;
n_trials   = 300;
accuracy   = 0.78;

rng(0);

%% The Xue recoding function
% Demonstrate on a simple example first.
ex_conf = [1 2 3 4 4 3 2 1]';
rc1 = xue_recode(ex_conf, 1);
rc2 = xue_recode(ex_conf, 2);

fprintf('Original : '); fprintf('%d ', ex_conf); fprintf('\n');
fprintf('Recode 1 : '); fprintf('%d ', rc1);     fprintf('  (lower confidence bias)\n');
fprintf('Recode 2 : '); fprintf('%d ', rc2);     fprintf('  (higher confidence bias)\n\n');

%% Compute measures under both recodings
% After xue_recode, the confidence range collapses from 1-4 to 1-3,
% so nRatings_rc = nRatings - 1.

nRatings_rc = nRatings - 1;    % = 3 after recoding

bias = NaN(n_subjects, 2, N_MEAS);  % [sub, recode (1|2), measure]

for sub = 1:n_subjects
    stim = randi([0,1], n_trials, 1);
    resp = stim;
    flip = rand(n_trials,1) > accuracy;
    resp(flip) = 1 - resp(flip);
    correct = (stim == resp);
    conf = zeros(n_trials,1);
    conf( correct) = randi([3, nRatings], sum( correct), 1);
    conf(~correct) = randi([1,         2], sum(~correct), 1);

    for rt = 1:2
        conf_rc = xue_recode(conf, rt);
        if isnan(conf_rc)
            % xue_recode returns NaN scalar when conditions not met
            bias(sub, rt, :) = NaN;
        else
            valid = ~isnan(conf_rc);
            bias(sub, rt, :) = compute_all_measures( ...
                stim(valid), resp(valid), conf_rc(valid), nRatings_rc);
        end
    end
end

fprintf('Bias array size: %s\n', mat2str(size(bias)));

%% Test recode2 − recode1 against zero
% A large, significant difference means the measure is contaminated by bias.

SKIP = {'d''', 'Criterion'};   % purely Type-1 measures, always skip

delta = squeeze(bias(:, 2, :)) - squeeze(bias(:, 1, :));  % [sub × meas]

fprintf('\n%-22s  %8s  %10s  %9s  %4s\n', 'Measure', 't', 'p', 'd', 'sig');
fprintf('%s\n', repmat('-', 1, 58));

for m = 1:N_MEAS
    if any(strcmp(variable_names{m}, SKIP)), continue; end
    d = delta(:, m);
    d = d(~isnan(d));
    if length(d) < 2, continue; end
    [~, p, ~, stats] = ttest(d);
    t  = stats.tstat;
    cd = t / sqrt(length(d));
    if     p < 0.001; stars = '***';
    elseif p < 0.01;  stars = ' **';
    elseif p < 0.05;  stars = '  *';
    else;             stars = '   ';
    end
    fprintf('%-22s  %8.3f  %10.4f  %9.3f  %s\n', variable_names{m}, t, p, cd, stars);
end

%% Visualise bias effect
means = nanmean(delta, 1);            % [1 × N_MEAS]
n_ok  = sum(~isnan(delta), 1);
sems  = nanstd(delta, 0, 1) ./ sqrt(max(n_ok, 1));

% Compute p-values for colouring
ps = NaN(1, N_MEAS);
for m = 1:N_MEAS
    d = delta(:,m); d = d(~isnan(d));
    if length(d) >= 2
        [~, ps(m)] = ttest(d);
    end
end

figure('Color','w');
x = 1:N_MEAS;
bar_colors = zeros(N_MEAS, 3);
for m = 1:N_MEAS
    if ~isnan(ps(m)) && ps(m) < 0.05
        bar_colors(m,:) = [0.84 0.37 0.00];   % orange = significant
    else
        bar_colors(m,:) = [0.60 0.60 0.60];   % grey = not significant
    end
end

hold on;
for m = 1:N_MEAS
    bar(m, means(m), 'FaceColor', bar_colors(m,:), 'FaceAlpha', 0.85);
    plot([m m], [means(m)-sems(m), means(m)+sems(m)], 'k-', 'LineWidth', 1.2);
end
yline(0, '--k', 'LineWidth', 0.8);

set(gca, 'XTick', x, 'XTickLabel', variable_names, 'FontSize', 8);
xtickangle(45);
ylabel('Recode 2 − Recode 1 ± SEM', 'FontSize', 11);
title('Metacognitive Bias Sensitivity  (orange = significant p < 0.05)', 'FontSize', 12);
box off;
