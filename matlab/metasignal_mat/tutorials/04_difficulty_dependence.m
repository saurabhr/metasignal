%% Tutorial 4 — Difficulty Dependence
% Tests whether each measure changes as task difficulty changes.
% A well-calibrated metacognitive measure should be difficulty-independent.
% Replicates the analysis in Rahnev (2025) Supp Tables 3–5.
%
% Pipeline:
%   1. Simulate subjects with easy / medium / hard difficulty levels
%   2. Compute all 20 measures per difficulty level
%   3. Remove ±3 SD outliers (propagated across levels per subject)
%   4. One-sample t-test: easy − hard

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
difficulty = [0.65, 0.75, 0.85];   % hard, medium, easy  (P(correct))
n_levels   = length(difficulty);
n_subjects = 20;
n_per_level = 80;

rng(0);

%% Simulate subjects
% Each subject performs n_per_level trials at each difficulty level.
% Higher accuracy = easier task.

raw = NaN(n_subjects, n_levels, N_MEAS);

for sub = 1:n_subjects
    for lv = 1:n_levels
        acc  = difficulty(lv);
        stim = randi([0,1], n_per_level, 1);
        resp = stim;
        flip = rand(n_per_level,1) > acc;
        resp(flip) = 1 - resp(flip);
        correct = (stim == resp);
        conf = zeros(n_per_level,1);
        conf( correct) = randi([3, nRatings], sum( correct), 1);
        conf(~correct) = randi([1,         2], sum(~correct), 1);
        raw(sub, lv, :) = compute_all_measures(stim, resp, conf, nRatings);
    end
end

fprintf('Simulated %d subjects × %d difficulty levels\n', n_subjects, n_levels);
fprintf('Computed array: %s\n', mat2str(size(raw)));

%% Compute measures per difficulty level

%% 3-SD outlier removal
% Values beyond ±3 SD per measure × level are set to NaN.
% If ANY level is NaN for a given subject × measure, ALL levels are set NaN.

clean = raw;
for m = 1:N_MEAS
    for lv = 1:n_levels
        col  = clean(:, lv, m);
        mu   = nanmean(col);
        sd   = nanstd(col);
        if sd > 0
            clean((col < mu - 3*sd) | (col > mu + 3*sd), lv, m) = NaN;
        end
    end
    % propagate NaN across all levels
    bad = any(isnan(squeeze(clean(:,:,m))), 2);
    clean(bad, :, m) = NaN;
end

removed = sum(isnan(clean(:)) & ~isnan(raw(:)));
fprintf('Values set to NaN by 3-SD removal: %d\n', removed);

%% One-sample t-test: easy − hard
% Index 1 = hard (0.65), index 3 = easy (0.85)

delta = squeeze(clean(:, 3, :)) - squeeze(clean(:, 1, :));  % [sub × meas]

fprintf('\n%-22s  %8s  %10s  %8s  %4s\n', 'Measure', 't', 'p', 'd', 'sig');
fprintf('%s\n', repmat('-', 1, 58));

for m = 1:N_MEAS
    d = delta(:, m);
    d = d(~isnan(d));
    if length(d) < 2, continue; end
    [~, p, ~, stats] = ttest(d);
    t = stats.tstat;
    cd = t / sqrt(length(d));
    if     p < 0.001; stars = '***';
    elseif p < 0.01;  stars = ' **';
    elseif p < 0.05;  stars = '  *';
    else;             stars = '   ';
    end
    fprintf('%-22s  %8.3f  %10.4f  %8.3f  %s\n', variable_names{m}, t, p, cd, stars);
end

%% Visualise difficulty effect
figure('Color','w', 'DefaultAxesFontSize', 9);
level_labels = {'Hard', 'Med', 'Easy'};

for m = 1:N_MEAS
    ax = subplot(4, 5, m);
    col   = squeeze(clean(:, :, m));       % [sub × level]
    means = nanmean(col, 1);
    n_ok  = sum(~isnan(col), 1);
    sems  = nanstd(col, 0, 1) ./ sqrt(max(n_ok, 1));
    errorbar(1:n_levels, means, sems, '-o', 'Color', [0 0.45 0.70], ...
        'LineWidth', 1.2, 'MarkerSize', 4, 'CapSize', 4);
    set(ax, 'XTick', 1:n_levels, 'XTickLabel', level_labels);
    title(variable_names{m}, 'FontSize', 8, 'FontWeight', 'bold');
    yline(0, '--k', 'LineWidth', 0.5);
    box off;
end

% Add a figure-level title using a hidden axes (avoids needing suplabel)
ax_title = axes('Position',[0 0 1 1], 'Visible','off');
text(ax_title, 0.5, 1.01, 'Effect of Difficulty on 20 Metacognitive Measures', ...
    'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
    'FontSize', 12, 'FontWeight', 'bold', 'Units','normalized');
