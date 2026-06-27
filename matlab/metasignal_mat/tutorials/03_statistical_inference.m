%% Tutorial 3 — Statistical Inference
% Bootstrap confidence intervals, permutation tests, and group-level
% summary using perform_ttest from the metasignal_mat helper functions.
%
% > Speed note: meta-d', meta-noise, and meta-uncertainty each require an
% > MLE optimisation (~1-3 s per subject).  This tutorial uses the four
% > fast non-MLE measures (d', AUC2, Gamma, Phi) so it runs in seconds.
% > The concepts transfer directly to any of the 20 measures.

%% Setup
clear; close all; clc

root_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root_dir, 'helperFunctions')));

nRatings   = 4;
n_trials   = 200;
n_subjects = 20;
accuracy   = 0.78;
rng(0);

FAST_LABELS = {"d'", 'AUC2', 'Gamma', 'Phi'};

%% 1. Simulate a group of n_subjects participants
participants = cell(n_subjects, 1);
for i = 1:n_subjects
    participants{i} = sim_participant(i, accuracy, n_trials, nRatings);
end
fprintf('Simulated %d participants\n', n_subjects);

%% 2. Group-level summary
individual = NaN(n_subjects, 4);
for i = 1:n_subjects
    pt = participants{i};
    individual(i,:) = fast_measures(pt.stim, pt.resp, pt.conf, nRatings);
end

fprintf('\n%-8s  %8s  %8s\n', 'Measure', 'Mean', 'SEM');
fprintf('%s\n', repmat('-', 1, 28));
for k = 1:4
    col  = individual(:,k);
    mu   = nanmean(col);
    sem  = nanstd(col,0) / sqrt(sum(~isnan(col)));
    fprintf('%-8s  %8.3f  %8.3f\n', FAST_LABELS{k}, mu, sem);
end

%% 3. Bootstrap confidence interval
% Resample trials with replacement and recompute the measure.
% The percentile interval of the resampled distribution is the CI.

pt1 = participants{1};
auc2_fn = @(s,r,c) SDTtype2AUC(s, r, c, nRatings);

ci_auc2 = bootstrap_ci(pt1.stim, pt1.resp, pt1.conf, auc2_fn, 2000, 0.95, 1);
fprintf('\nAUC2 95%% CI (participant 1): [%.3f, %.3f]\n', ci_auc2(1), ci_auc2(2));

fn_list = {@(s,r,c) compute_SDT_resp(s,r), ...
           @(s,r,c) SDTtype2AUC(s,r,c,nRatings), ...
           @(s,r,c) SDTgamma(s,r,c,nRatings), ...
           @(s,r,c) SDTphi(s,r,c,nRatings)};

fprintf('\n%-8s  %-25s\n', 'Measure', '95% CI');
fprintf('%s\n', repmat('-', 1, 36));
for k = 1:4
    ci = bootstrap_ci(pt1.stim, pt1.resp, pt1.conf, fn_list{k}, 2000, 0.95, k);
    fprintf('%-8s  [%.3f, %.3f]\n', FAST_LABELS{k}, ci(1), ci(2));
end

%% 4. Permutation test — two-condition comparison
% Pool all trials, shuffle condition labels n_perm times to build a null
% distribution of the difference in a measure between two conditions.

% Condition A: metacognitive (conf tracks accuracy)
ptA = sim_participant(1, 0.80, n_trials, nRatings);

% Condition B: same accuracy, random (non-metacognitive) confidence
rng(200);
stim_b  = randi([0,1], n_trials, 1);
resp_b  = stim_b;
flip_b  = rand(n_trials,1) > 0.80;
resp_b(flip_b) = 1 - resp_b(flip_b);
conf_b  = randi([1, nRatings], n_trials, 1);

[p_perm, obs_diff] = permutation_test(ptA.stim, ptA.resp, ptA.conf, ...
                                       stim_b, resp_b, conf_b, ...
                                       auc2_fn, 5000, 42);

fprintf('\nAUC2 observed difference (A - B): %.3f\n', obs_diff);
fprintf('Two-sided permutation p-value:    %.4f\n', p_perm);

%% 5. One-sample t-test across participants using perform_ttest
% perform_ttest wraps MATLAB's ttest and also returns Cohen's d and 95% CI.

auc2_vals = individual(:, 2);   % AUC2 for all 20 participants
delta_vs_chance = auc2_vals - 0.5;

fprintf('\nAUC2 vs chance (0.5):\n');
[p_t, t_stat, df, Cohen_d, CI] = perform_ttest(delta_vs_chance, 'AUC2 vs 0.5', 1);  %#ok<ASGLU>

fprintf('\n  Tip: call perform_ttest for any of the 20 measures, e.g.:\n');
fprintf('  perform_ttest(metas(:,1) - 0, ''meta-d'''' > 0'', 1)\n');

% =========================================================================
% Local helper functions  (must appear at the end of a MATLAB script file)
% =========================================================================

function pt = sim_participant(seed, acc, n, nR)
    saved = rng(seed);
    stim    = randi([0,1], n, 1);
    resp    = stim;
    flip    = rand(n,1) > acc;
    resp(flip) = 1 - resp(flip);
    correct = (stim == resp);
    conf    = zeros(n,1);
    conf( correct) = randi([3,  nR], sum( correct), 1);
    conf(~correct) = randi([1,   2], sum(~correct), 1);
    pt = struct('stim',stim,'resp',resp,'conf',conf);
    rng(saved);
end

function vals = fast_measures(stim, resp, conf, nRatings)
    try
        dp = compute_SDT_resp(stim, resp);
    catch
        vals = NaN(1,4); return;
    end
    auc2 = SDTtype2AUC(stim, resp, conf, nRatings);
    gam  = SDTgamma(stim, resp, conf, nRatings);
    phi  = SDTphi(stim, resp, conf, nRatings);
    vals = [dp, auc2, gam, phi];
end

function ci = bootstrap_ci(stim, resp, conf, measure_fn, n_boot, ci_level, seed)
    saved = rng(seed);
    n    = length(stim);
    vals = NaN(n_boot,1);
    for b = 1:n_boot
        idx = randi(n, n, 1);
        v   = measure_fn(stim(idx), resp(idx), conf(idx));
        if ~isnan(v), vals(b) = v; end
    end
    rng(saved);
    alpha = 1 - ci_level;
    ci = prctile(vals(~isnan(vals)), [100*alpha/2, 100*(1-alpha/2)]);
end

function [p_val, obs_diff] = permutation_test(stim_a, resp_a, conf_a, ...
                                               stim_b, resp_b, conf_b, ...
                                               measure_fn, n_perm, seed)
    saved    = rng(seed);
    obs_a    = measure_fn(stim_a, resp_a, conf_a);
    obs_b    = measure_fn(stim_b, resp_b, conf_b);
    obs_diff = obs_a - obs_b;

    all_stim = [stim_a; stim_b];
    all_resp = [resp_a; resp_b];
    all_conf = [conf_a; conf_b];
    n_a = length(stim_a);
    n   = length(all_stim);

    null = NaN(n_perm, 1);
    for t = 1:n_perm
        perm = randperm(n);
        ia   = perm(1:n_a);
        ib   = perm(n_a+1:end);
        d = measure_fn(all_stim(ia), all_resp(ia), all_conf(ia)) - ...
            measure_fn(all_stim(ib), all_resp(ib), all_conf(ib));
        if ~isnan(d), null(t) = d; end
    end
    rng(saved);
    null  = null(~isnan(null));
    if isempty(null), p_val = NaN; return; end
    p_val = mean(abs(null) >= abs(obs_diff));
end
