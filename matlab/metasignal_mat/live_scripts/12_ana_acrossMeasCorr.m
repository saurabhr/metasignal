%% Aggregate Analysis: Across-Measure Correlations
% Replicates Figure 11 from Rahnev (2025), *Nature Communications*, 16(1), 701.
%
% Computes the full 17×17 inter-measure correlation matrix across subjects,
% then visualizes it as a heat-map. Also reports average correlations within
% and between measure families:
%   Set 1 (absolute measures): meta-d', AUC2, Gamma, Phi, ΔConf  [cols 1-5]
%   Set 2 (relative measures): M-Ratio, AUC2-Ratio, ..., ΔConf-Diff [cols 6-15]
%   Measures 16-17: meta-noise, meta-uncertainty
%
% Note: meta-noise and meta-uncertainty are sign-flipped before correlating
% so that higher values consistently reflect better metacognition.
%
% Datasets: Haddara, Maniscalco, Shekhar
%
% Run AFTER: 01_analysis_Haddara.m, 02_analysis_Maniscalco.m, 05_analysis_Shekhar.m

%% Setup
clear; close all; clc

root_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root_dir, 'helperFunctions')));

%% Parameters
datasets   = {'results_Haddara', 'results_Maniscalco', 'results_Shekhar'};
dset_names = {'Haddara', 'Maniscalco', 'Shekhar'};

variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', '\DeltaConf', ...
    'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', '\DeltaConf-Ratio', ...
    'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', '\DeltaConf-Diff', ...
    'meta-noise', 'meta-uncertainty'};  % 17 meta measures only

num_meas = 17;

%% Compute Correlation Matrices
fprintf('Computing inter-measure correlations...\n');

for dset = 1:length(datasets)
    load(fullfile(root_dir, 'Results', datasets{dset}));

    % For Shekhar: average across the 3 contrast levels
    if dset < 3
        metas = metas_raw;
    else
        metas = squeeze(mean(metas_raw, 2));
    end

    % Sign-flip meta-noise and meta-uncertainty so higher = better
    metas(:,16:17) = -metas(:,16:17);

    % Full 17x17 correlation matrix
    [r{dset}, p{dset}] = corr(metas(:,1:num_meas), metas(:,1:num_meas), 'rows', 'complete');

    % Set diagonal to NaN for averaging
    r_nan = r{dset};
    for meas = 1:num_meas; r_nan(meas,meas) = NaN; end

    % Summary correlations (Fisher z-averaged)
    r_av17(dset)      = z2r(mean(nanmean(r2z(r_nan(1:num_meas, 1:num_meas)))));
    r_set1(dset)      = z2r(mean(nanmean(r2z(r_nan(1:5,   1:5)))));
    r_set2(dset)      = z2r(mean(nanmean(r2z(r_nan(6:15,  6:15)))));
    r_set1_set2(dset) = z2r(mean(nanmean(r2z(r_nan(1:5,   6:15)))));
    r_16(dset)        = z2r(mean(nanmean(r2z(r_nan(1:num_meas, 16)))));
    r_17(dset)        = z2r(mean(nanmean(r2z(r_nan(1:num_meas, 17)))));
    r_1617(dset)      = r_nan(16, 17);
end

%% Report Correlation Summary
fprintf('\n--- Average inter-measure correlations (r, Fisher z-averaged) ---\n');
fprintf('%-30s  %8s  %8s  %8s  %8s\n', 'Correlation', 'Haddara', 'Maniscalco', 'Shekhar', 'Average');
fprintf('%s\n', repmat('-',1,70));

r_av17      = [r_av17,      z2r(mean(r2z(r_av17)))]
r_set1      = [r_set1,      z2r(mean(r2z(r_set1)))]
r_set2      = [r_set2,      z2r(mean(r2z(r_set2)))]
r_set1_set2 = [r_set1_set2, z2r(mean(r2z(r_set1_set2)))]
r_16        = [r_16,        z2r(mean(r2z(r_16)))]
r_17        = [r_17,        z2r(mean(r2z(r_17)))]
r_1617      = [r_1617,      z2r(mean(r2z(r_1617)))]

rows = {'All 17 measures', 'Set 1 (absolute)', 'Set 2 (relative)', ...
    'Set1 vs Set2', 'meta-noise vs all', 'meta-uncertainty vs all', 'noise vs uncertainty'};
vals = [r_av17; r_set1; r_set2; r_set1_set2; r_16; r_17; r_1617];

for k = 1:size(vals,1)
    fprintf('%-30s  %8.3f  %8.3f  %8.3f  %8.3f\n', rows{k}, vals(k,1), vals(k,2), vals(k,3), vals(k,4));
end

%% Figure: Correlation Matrix Heat-Maps (Figure 11)
plot_corrTables(r, p, variable_names, dset_names);

%% Figure: Average Correlation Matrix
r_avg = zeros(num_meas);
for dset = 1:length(datasets)
    r_avg = r_avg + r2z(r{dset});
end
r_avg = z2r(r_avg / length(datasets));

% Build red-blue colormap inline (scripts cannot contain local functions)
n    = 64; half = n/2;
cmap_rb = [linspace(0,1,half)', linspace(0,1,half)', ones(half,1); ...
           ones(half,1), linspace(1,0,half)', linspace(1,0,half)'];

figure('Color','w', 'DefaultAxesFontSize',12);
imagesc(r_avg);
colormap(cmap_rb);
caxis([-1, 1]);
colorbar;
set(gca, 'XTick', 1:num_meas, 'XTickLabel', variable_names, ...
    'YTick', 1:num_meas, 'YTickLabel', variable_names);
xtickangle(45);
title('Average inter-measure correlation (all 3 datasets)', 'FontSize', 14);
axis square;
