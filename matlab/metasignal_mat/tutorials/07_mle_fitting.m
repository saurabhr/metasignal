%% Tutorial 7 — MLE Fitting with type2_SDT_MLE
% Demonstrates the MLE fitting pipeline for meta-d' using type2_SDT_MLE,
% which is the MATLAB equivalent of metadpy's fit_meta_d_mle.
%
% Topics:
%   1. Fit meta-d' for a single participant
%   2. Understand the output structure
%   3. Group-level MLE fits and summary statistics
%   4. Apply to the rm dataset (real data, 2 conditions × 20 subjects)
%   5. Test within-subject differences across conditions

%% Setup
clear; close all; clc

root_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root_dir, 'helperFunctions')));

nRatings = 4;
rng(42);

%% 1. Fit meta-d' for a single participant
fprintf('=== Single-participant MLE fit ===\n');

n_trials = 400;
accuracy = 0.80;

stim = randi([0,1], n_trials, 1);
resp = stim;
flip = rand(n_trials,1) > accuracy;
resp(flip) = 1 - resp(flip);
correct = (stim == resp);
conf = zeros(n_trials,1);
conf( correct) = randi([3, nRatings], sum( correct), 1);
conf(~correct) = randi([1,         2], sum(~correct), 1);

% equalVariance=1 matches metadpy default
out = type2_SDT_MLE(stim, resp, conf, nRatings, [], 1);

fprintf('da         = %.4f   (type-1 sensitivity)\n', out.da);
fprintf('meta_da    = %.4f   (type-2 metacognitive sensitivity)\n', out.meta_da);
fprintf('M_ratio    = %.4f   (meta-d'' / d'')\n', out.M_ratio);
fprintf('M_diff     = %.4f   (meta-d'' - d'')\n', out.meta_da - out.da);
fprintf('logL       = %.2f\n', out.type2_fit.logL);

%% 2. Inspect the full output structure
fprintf('\nFull output fields:\n');
fields = fieldnames(out);
for f = 1:length(fields)
    val = out.(fields{f});
    if isscalar(val)
        fprintf('  %-20s  %.4f\n', fields{f}, val);
    else
        fprintf('  %-20s  [1×%d array]\n', fields{f}, numel(val));
    end
end

%% 3. Alternative: use compute_all_measures and extract index 1 (meta-d')
meas_vec = compute_all_measures(stim, resp, conf, nRatings);
fprintf('\ncompute_all_measures → meta-d'' (index 1): %.4f\n', meas_vec(1));
fprintf('type2_SDT_MLE       → meta_da           : %.4f\n', out.meta_da);

%% 4. Group-level MLE fits
fprintf('\n=== Group-level MLE fits ===\n');

n_subjects = 20;
group_da     = NaN(n_subjects, 1);
group_metada = NaN(n_subjects, 1);
group_mratio = NaN(n_subjects, 1);

for sub = 1:n_subjects
    rng(sub * 100);
    st = randi([0,1], n_trials, 1);
    rs   = st;
    flip = rand(n_trials,1) > accuracy;
    rs(flip) = 1 - rs(flip);
    cr = (st == rs);
    cn = zeros(n_trials,1);
    cn( cr) = randi([3, nRatings], sum( cr), 1);
    cn(~cr) = randi([1,         2], sum(~cr), 1);

    try
        o = type2_SDT_MLE(st, rs, cn, nRatings, [], 1);
        group_da(sub)     = o.da;
        group_metada(sub) = o.meta_da;
        group_mratio(sub) = o.M_ratio;
    catch
        % leave as NaN
    end
end

fprintf('%-12s  %8s ± %6s\n', 'Measure', 'Mean', 'SEM');
fprintf('%s\n', repmat('-', 1, 30));
measures_g = {group_da, group_metada, group_mratio};
labels_g   = {'da', 'meta_da', 'M_ratio'};
for k = 1:3
    v   = measures_g{k};
    v   = v(~isnan(v));
    mu  = mean(v);
    sem = std(v) / sqrt(length(v));
    fprintf('%-12s  %8.3f ± %6.3f\n', labels_g{k}, mu, sem);
end

%% 5. Test meta-d' > 0 and M-Ratio > 1 across participants
fprintf('\nOne-sample t-tests:\n');
v = group_metada(~isnan(group_metada));
[~, p_meta, ~, stats_meta] = ttest(v);
fprintf('  meta-d'' > 0: t(%d)=%.2f, p=%.4f\n', ...
    length(v)-1, stats_meta.tstat, p_meta);

v2 = group_mratio(~isnan(group_mratio)) - 1;
[~, p_mr, ~, stats_mr] = ttest(v2);
fprintf('  M-Ratio > 1: t(%d)=%.2f, p=%.4f\n', ...
    length(v2)-1, stats_mr.tstat, p_mr);

%% 6. Real-data analysis — rm dataset (two conditions × 20 subjects)
% rm.txt lives in matlab/metasignal_mat/datasets/ (or docs/tutorials/).
% Columns: Stimuli Responses Accuracy Confidence nTrial Subject Condition

rm_candidates = { ...
    fullfile(root_dir, '..', '..', 'docs', 'tutorials', 'rm.txt'), ...
    fullfile(root_dir, 'datasets', 'rm.txt') };

rm_path = '';
for c = 1:length(rm_candidates)
    if exist(rm_candidates{c}, 'file')
        rm_path = rm_candidates{c};
        break;
    end
end

if isempty(rm_path)
    fprintf('\n[SKIP] rm.txt not found; skipping real-data section.\n');
    fprintf('       Expected at: %s\n', rm_candidates{1});
    return;
end

fprintf('\n=== rm dataset: 2 conditions × 20 subjects ===\n');

T = readtable(rm_path, 'Delimiter', ' ');
T.Properties.VariableNames = {'Stimuli','Responses','Accuracy','Confidence',...
                               'nTrial','Subject','Condition'};

subjects   = unique(T.Subject);
conditions = unique(T.Condition);
n_sub      = length(subjects);
n_cond     = length(conditions);

rm_da     = NaN(n_sub, n_cond);
rm_metada = NaN(n_sub, n_cond);
rm_mratio = NaN(n_sub, n_cond);

for si = 1:n_sub
    for ci = 1:n_cond
        mask = (T.Subject == subjects(si)) & (T.Condition == conditions(ci));
        if sum(mask) < 10, continue; end
        st = T.Stimuli(mask);
        rs = T.Responses(mask);
        cn = T.Confidence(mask);
        nR = length(unique(cn));
        try
            o = type2_SDT_MLE(st, rs, cn, nR, [], 1);
            rm_da(si, ci)     = o.da;
            rm_metada(si, ci) = o.meta_da;
            rm_mratio(si, ci) = o.M_ratio;
        catch
        end
    end
end

% Condition difference in M-Ratio
delta_mratio = rm_mratio(:,2) - rm_mratio(:,1);
valid = ~isnan(delta_mratio);
[p_t, t_stat, df, Cohen_d, CI] = perform_ttest(delta_mratio(valid), ...
    'rm M-Ratio: cond1 vs cond0', 1);  %#ok<ASGLU>

% Per-condition summary
fprintf('\n%-10s  %8s ± SEM     %8s ± SEM     %8s ± SEM\n', ...
    'Condition', 'da', 'meta_da', 'M_ratio');
fprintf('%s\n', repmat('-', 1, 60));
for ci = 1:n_cond
    da_v  = rm_da(~isnan(rm_da(:,ci)), ci);
    md_v  = rm_metada(~isnan(rm_metada(:,ci)), ci);
    mr_v  = rm_mratio(~isnan(rm_mratio(:,ci)), ci);
    fprintf('%-10d  %8.3f±%.3f   %8.3f±%.3f   %8.3f±%.3f\n', ...
        conditions(ci), ...
        mean(da_v), std(da_v)/sqrt(length(da_v)), ...
        mean(md_v), std(md_v)/sqrt(length(md_v)), ...
        mean(mr_v), std(mr_v)/sqrt(length(mr_v)));
end
