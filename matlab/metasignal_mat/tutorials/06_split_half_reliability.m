%% Tutorial 6 — Split-Half Reliability
% Measures how consistently each metacognitive measure reproduces across two
% random halves of each participant's trial set.
%
% Pipeline:
%   1. Simulate subjects
%   2. Split each subject's trials into two random halves
%   3. Compute all 20 measures in each half
%   4. Correlate half-1 vs half-2 values across subjects → r_SH
%   5. Apply Spearman-Brown correction → r_SB = 2*r_SH / (1 + r_SH)
%   6. Bonus: corrupt confidence ratings and measure the drop in reliability

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
n_subjects = 30;
n_trials   = 400;   % enough for reliable split-half estimates
accuracy   = 0.78;

rng(0);

%% 1. Simulate subjects and compute split-half measures
half1 = NaN(n_subjects, N_MEAS);
half2 = NaN(n_subjects, N_MEAS);

for sub = 1:n_subjects
    rng(sub);
    stim = randi([0,1], n_trials, 1);
    resp = stim;
    flip = rand(n_trials,1) > accuracy;
    resp(flip) = 1 - resp(flip);
    correct = (stim == resp);
    conf = zeros(n_trials,1);
    conf( correct) = randi([3, nRatings], sum( correct), 1);
    conf(~correct) = randi([1,         2], sum(~correct), 1);

    % Random split
    perm = randperm(n_trials);
    h1   = perm(1:n_trials/2);
    h2   = perm(n_trials/2+1:end);

    half1(sub,:) = compute_all_measures(stim(h1), resp(h1), conf(h1), nRatings);
    half2(sub,:) = compute_all_measures(stim(h2), resp(h2), conf(h2), nRatings);
end

fprintf('Split-half arrays: %s each\n', mat2str(size(half1)));

%% 2. Spearman-Brown corrected reliability
r_sh = NaN(1, N_MEAS);
r_sb = NaN(1, N_MEAS);

for m = 1:N_MEAS
    x = half1(:,m);
    y = half2(:,m);
    valid = ~isnan(x) & ~isnan(y);
    if sum(valid) < 4, continue; end
    R = corrcoef(x(valid), y(valid));
    r_sh(m) = R(1,2);
    r_sb(m) = 2*r_sh(m) / (1 + r_sh(m));    % Spearman-Brown correction
end

fprintf('\n%-22s  %8s  %8s\n', 'Measure', 'r_SH', 'r_SB');
fprintf('%s\n', repmat('-', 1, 42));
for m = 1:N_MEAS
    if isnan(r_sh(m))
        fprintf('%-22s  %8s  %8s\n', variable_names{m}, 'NaN', 'NaN');
    else
        fprintf('%-22s  %8.3f  %8.3f\n', variable_names{m}, r_sh(m), r_sb(m));
    end
end

%% 3. Effect of confidence corruption on reliability
% Replace each subject's confidence with random integers.
% A measure that relies on confidence should become less reliable.

half1_corrupt = NaN(n_subjects, N_MEAS);
half2_corrupt = NaN(n_subjects, N_MEAS);

for sub = 1:n_subjects
    rng(sub);
    stim = randi([0,1], n_trials, 1);
    resp = stim;
    flip = rand(n_trials,1) > accuracy;
    resp(flip) = 1 - resp(flip);

    % Corrupt: random confidence unrelated to accuracy
    conf_corrupt = randi([1, nRatings], n_trials, 1);

    perm = randperm(n_trials);
    h1   = perm(1:n_trials/2);
    h2   = perm(n_trials/2+1:end);

    half1_corrupt(sub,:) = compute_all_measures(stim(h1), resp(h1), conf_corrupt(h1), nRatings);
    half2_corrupt(sub,:) = compute_all_measures(stim(h2), resp(h2), conf_corrupt(h2), nRatings);
end

r_sb_corrupt = NaN(1, N_MEAS);
for m = 1:N_MEAS
    x = half1_corrupt(:,m);
    y = half2_corrupt(:,m);
    valid = ~isnan(x) & ~isnan(y);
    if sum(valid) < 4, continue; end
    R = corrcoef(x(valid), y(valid));
    r_sh_c = R(1,2);
    r_sb_corrupt(m) = 2*r_sh_c / (1 + r_sh_c);
end

%% 4. Visualise reliability drop
fprintf('\n%-22s  %8s  %12s  %8s\n', 'Measure', 'r_SB', 'r_SB corrupt', 'Drop');
fprintf('%s\n', repmat('-', 1, 56));
for m = 1:N_MEAS
    drop = r_sb(m) - r_sb_corrupt(m);
    fprintf('%-22s  %8.3f  %12.3f  %8.3f\n', variable_names{m}, r_sb(m), r_sb_corrupt(m), drop);
end

figure('Color','w', 'Position', [100 100 900 350]);

subplot(1,2,1);
bar(r_sb, 'FaceColor', [0 0.45 0.70]);
hold on;
yline(0, '--k');
set(gca, 'XTick', 1:N_MEAS, 'XTickLabel', variable_names, 'FontSize', 8);
xtickangle(45);
ylim([-0.2 1.1]);
ylabel('Spearman-Brown r', 'FontSize', 11);
title('Reliability (normal confidence)', 'FontSize', 11);
box off;

subplot(1,2,2);
drop = r_sb - r_sb_corrupt;
bar(drop, 'FaceColor', [0.84 0.37 0.00]);
hold on;
yline(0, '--k');
set(gca, 'XTick', 1:N_MEAS, 'XTickLabel', variable_names, 'FontSize', 8);
xtickangle(45);
ylabel('Reliability drop (corrupted conf)', 'FontSize', 11);
title('Sensitivity to confidence quality', 'FontSize', 11);
box off;
