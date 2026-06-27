%% Tutorial 1 — Getting Started
% Verifies your metasignal_mat setup and walks through the three input arrays
% every function expects: stim, resp, conf.
%
% After completing this tutorial you will know how to:
%   - Add the helper functions to your MATLAB path
%   - Build a simulated dataset in the correct format
%   - Compute Type-1 SDT parameters (d', c)
%   - Convert trial vectors to response-count arrays (nR_S1 / nR_S2)
%   - Compute the full 20-element measure vector

%% 1. Setup — add helpers to path
clear; close all; clc

root_dir = fileparts(fileparts(mfilename('fullpath')));  % .../metasignal_mat
addpath(genpath(fullfile(root_dir, 'helperFunctions')));

fprintf('Root directory: %s\n', root_dir);
fprintf('Helpers on path — compute_all_measures: %s\n', ...
    which('compute_all_measures'));

%% 2. Input format
% Every function in this toolbox expects three parallel row/column vectors:
%
%   stim  : 0 = S1 (noise), 1 = S2 (signal)
%   resp  : 0 = subject responded S1, 1 = responded S2
%   conf  : integer in 1 … nRatings (1 = lowest confidence)
%
% nRatings is the total number of confidence levels (typically 4 or 6).

%% 3. Build a minimal simulated dataset
rng(42);                         % reproducible seed
n_trials  = 300;
nRatings  = 4;
accuracy  = 0.80;                % P(correct)

stim = randi([0, 1], n_trials, 1);
resp = stim;
flip = rand(n_trials, 1) > accuracy;
resp(flip) = 1 - resp(flip);     % introduce errors

correct = (stim == resp);
conf = zeros(n_trials, 1);
conf( correct) = randi([3, nRatings], sum( correct), 1);
conf(~correct) = randi([1,         2], sum(~correct), 1);

fprintf('Trials   : %d\n', n_trials);
fprintf('Accuracy : %.1f%%\n', mean(correct)*100);
fprintf('Mean conf: %.2f\n', mean(conf));

%% 4. Type-1 SDT parameters
[dprime, c, ln_beta] = compute_SDT_resp(stim, resp);
fprintf('d-prime    = %.3f\n', dprime);
fprintf('criterion  = %.3f\n', c);
fprintf('ln(beta)   = %.3f\n', ln_beta);

%% 5. Convert trials to response-count arrays
% trials2counts packs stim/resp/conf into the nR_S1 / nR_S2 format used
% by fit_meta_d_MLE and the individual measure functions.
[nR_S1, nR_S2] = trials2counts(stim, resp, conf, nRatings);

fprintf('\nnR_S1 (length %d): ', length(nR_S1));
fprintf('%g ', nR_S1);
fprintf('\nnR_S2 (length %d): ', length(nR_S2));
fprintf('%g ', nR_S2);
fprintf('\n');

%% 6. Inspect the full 20-element output
variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', 'DeltaConf', ...
    'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', 'DeltaConf-Ratio', ...
    'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', 'DeltaConf-Diff', ...
    'meta-noise', 'meta-uncertainty', 'd''', 'Criterion', 'Confidence'};

meas = compute_all_measures(stim, resp, conf, nRatings);

fprintf('\n%s\n', repmat('-', 1, 40));
fprintf('%-5s  %-22s  %s\n', 'Index', 'Measure', 'Value');
fprintf('%s\n', repmat('-', 1, 40));
for i = 1:20
    if isnan(meas(i))
        fprintf('[%2d]   %-22s  NaN\n', i, variable_names{i});
    else
        fprintf('[%2d]   %-22s  %.4f\n', i, variable_names{i}, meas(i));
    end
end
