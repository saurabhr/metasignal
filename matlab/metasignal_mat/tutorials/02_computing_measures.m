%% Tutorial 2 — Computing All 20 Measures
% Detailed walkthrough of each block of the 20-measure array, plus how to
% call individual measure functions directly.
%
% Measure order:
%   [1-5]   Absolute sensitivity: meta-d', AUC2, Gamma, Phi, DeltaConf
%   [6-10]  Efficiency ratios:   M-Ratio, AUC2-Ratio, Gamma-Ratio, Phi-Ratio, DeltaConf-Ratio
%   [11-15] Efficiency diffs:    M-Diff, AUC2-Diff, Gamma-Diff, Phi-Diff, DeltaConf-Diff
%   [16-17] Noise models:        meta-noise, meta-uncertainty
%   [18-20] Type-1 / raw:        d', Criterion, mean Confidence

%% Setup
clear; close all; clc

root_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root_dir, 'helperFunctions')));

variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', 'DeltaConf', ...
    'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', 'DeltaConf-Ratio', ...
    'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', 'DeltaConf-Diff', ...
    'meta-noise', 'meta-uncertainty', 'd''', 'Criterion', 'Confidence'};

rng(0);
n_trials = 400;
nRatings = 4;
accuracy = 0.78;

stim = randi([0,1], n_trials, 1);
resp = stim;
flip = rand(n_trials,1) > accuracy;
resp(flip) = 1 - resp(flip);
correct = (stim == resp);
conf = zeros(n_trials,1);
conf( correct) = randi([3, nRatings], sum( correct), 1);
conf(~correct) = randi([1,         2], sum(~correct), 1);

fprintf('Data ready: %d trials, nRatings=%d\n', n_trials, nRatings);

%% Block 1 — Absolute sensitivity (indices 1–5)
% These five measures ask: how well does confidence track accuracy?
% None of them are normalised by what an ideal observer would achieve.

meas = compute_all_measures(stim, resp, conf, nRatings);

labels = {'meta-d''', 'AUC2', 'Gamma', 'Phi', 'DeltaConf'};
fprintf('\nBlock 1 — Absolute sensitivity:\n');
for i = 1:5
    fprintf('  %-12s = %.4f\n', labels{i}, meas(i));
end

%% Block 2 & 3 — Efficiency ratios and differences (indices 6–15)
% Normalise observed metacognition by the expected performance of an ideal
% observer with the same d'. Removes spurious dependence on task difficulty.
% SDTexpectConf computes the expected nR_S1/nR_S2 under ideal SDT.

% AUC2 observed vs expected
[auc2_obs, auc2_ratio, auc2_diff] = SDTtype2AUC(stim, resp, conf, nRatings);

fprintf('\nBlock 2 & 3 — Efficiency measures (AUC2 example):\n');
fprintf('  AUC2 observed = %.4f\n', auc2_obs);
fprintf('  AUC2-Ratio    = %.4f  (obs / ideal)\n', auc2_ratio);
fprintf('  AUC2-Diff     = %.4f  (obs - ideal)\n', auc2_diff);

%% Individual measure functions
% You can also call each measure directly without going through
% compute_all_measures.

[gamma, gamma_ratio, gamma_diff]           = SDTgamma(stim, resp, conf, nRatings);
[phi,   phi_ratio,   phi_diff]             = SDTphi(stim, resp, conf, nRatings);
[dc,    dc_ratio,    dc_diff]              = SDTdeltaConf(stim, resp, conf, nRatings);

fprintf('\nIndividual functions:\n');
fprintf('  Gamma      = %.4f\n', gamma);
fprintf('  Phi        = %.4f\n', phi);
fprintf('  DeltaConf  = %.4f\n', dc);

% meta-d' via MLE (type2_SDT_MLE)
output  = type2_SDT_MLE(stim, resp, conf, nRatings, [], 1);
fprintf('  meta_da    = %.4f\n', output.meta_da);
fprintf('  M_ratio    = %.4f\n', output.M_ratio);

%% Block 4 — Meta-noise and meta-uncertainty (indices 16–17)
[metaNoise, ~]     = compute_metaNoise(stim, resp, conf, nRatings);
metaUncertainty    = compute_metaUncertainty(stim, resp, conf, nRatings);

fprintf('\nBlock 4 — Noise-based measures:\n');
fprintf('  meta-noise       = %.4f\n', metaNoise);
fprintf('  meta-uncertainty = %.4f\n', metaUncertainty);

%% Full 20-measure summary
fprintf('\n%s\n', repmat('-', 1, 42));
fprintf('%-6s  %-22s  %s\n', 'Index', 'Measure', 'Value');
fprintf('%s\n', repmat('-', 1, 42));
for i = 1:20
    if isnan(meas(i))
        fprintf('[%2d]   %-22s  NaN\n', i, variable_names{i});
    else
        fprintf('[%2d]   %-22s  %10.4f\n', i, variable_names{i}, meas(i));
    end
end
