%% Aggregate Analysis: Precision of Metacognitive Measures
% Replicates Figure 1 from Rahnev (2025), *Nature Communications*, 16(1), 701.
%
% Precision quantifies how sensitive each measure is to small random
% perturbations of the confidence data. A small proportion of trials have
% their confidence ratings randomly altered. The resulting change in each
% measure (normalized by its within-subject SD) is the precision index:
% *larger drop = less precise*.
%
% Datasets: Haddara (multi-day), Maniscalco (single-session)
% Key finding: meta-uncertainty has the highest precision; meta-noise the
% lowest among the 17 meta measures.
%
% Run AFTER: 01_analysis_Haddara.m, 02_analysis_Maniscalco.m

%% Setup
clear; close all; clc

root_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root_dir, 'helperFunctions')));

%% Parameters
datasets              = {'results_Haddara', 'results_Maniscalco'};
dset_names            = {'Haddara', 'Maniscalco'};
cutoff_for_outliers   = 4.5;  % ±4.5 SD (measure-specific)

variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', '\DeltaConf', ...
    'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', '\DeltaConf-Ratio', ...
    'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', '\DeltaConf-Diff', ...
    'meta-noise', 'meta-uncertainty', 'd''', 'Criterion', 'Confidence'};

num_meas = 17; % meta measures only (exclude d', c, conf)

%% Compute Precision Indices
fprintf('Computing precision indices...\n');

for dset = 1:length(datasets)
    load(fullfile(root_dir, 'Results', datasets{dset}));
    num_sub = size(metas_precision{1}, 1);

    for bs = 1:length(metas_precision)
        metas = metas_precision{bs};
        metas(abs(metas) > cutoff_for_outliers) = NaN;  % remove outliers

        % For Haddara: collapse day dimension so shape is [sub, bins, alter, meas]
        if dset == 1
            metas = reshape(metas, size(metas,1), [], size(metas,4), size(metas,5));
            metad_means(bs,:) = squeeze(nanmean(nanmean(nanmean( ...
                metas_precision{bs}(:,:,:,:,1), 1), 2), 3))';
        end

        % Flip meta-noise and meta-uncertainty: higher = better (more precise)
        metas(:,:,:,16:17) = -metas(:,:,:,16:17);

        % Within-subject SD across bins (using original, unaltered data)
        SD_sub_meas = squeeze(nanstd(metas(:,:,1,:), [], 2));

        % Compute drop in measure value per altered proportion, normalized by SD
        for alter = 2:size(metas,3)
            diff_sub_meas = squeeze(nanmean(metas(:,:,1,:) - metas(:,:,alter,:), 2));
            ratio_sub{dset}(bs, alter-1, :, :) = diff_sub_meas ./ repmat(mean(SD_sub_meas), num_sub, 1);
        end
    end

    % Average across all proportions altered and all bin sizes
    mean_ratio_sub{dset} = squeeze(mean(mean(ratio_sub{dset}, 2), 1));

    % Pairwise t-tests across measures
    for meas1 = 1:num_meas
        for meas2 = 1:num_meas
            [~, p{dset}(meas1,meas2)] = ttest( ...
                mean_ratio_sub{dset}(:,meas1), mean_ratio_sub{dset}(:,meas2));
        end
    end
end

%% Report Key Statistics
fprintf('\n--- Mean meta-d'' values by bin size (Haddara) ---\n');
mean_metad_values = mean(metad_means)

fprintf('\n--- Average precision per bin size (Haddara, meta measures 1-17) ---\n');
mean_precision_per_bin = [
    mean(mean(mean(ratio_sub{1}(1,:,:,1:num_meas)))), ...
    mean(mean(mean(ratio_sub{1}(2,:,:,1:num_meas)))), ...
    mean(mean(mean(ratio_sub{1}(3,:,:,1:num_meas)))), ...
    mean(mean(mean(ratio_sub{1}(4,:,:,1:num_meas))))]

fprintf('\n--- Meta-uncertainty precision (Haddara, Maniscalco) ---\n');
meta_uncertainty_precision_values = [mean(mean_ratio_sub{1}(:,17)), mean(mean_ratio_sub{2}(:,17))]

fprintf('\n--- Average precision for other measures (meas 1-16) ---\n');
average_of_other_measures_precision = [ ...
    mean(mean(mean_ratio_sub{1}(:,1:16))), mean(mean(mean_ratio_sub{2}(:,1:16)))]

fprintf('\n--- Ratio: meta-uncertainty vs. others ---\n');
meta_uncertainty_ratio = meta_uncertainty_precision_values ./ average_of_other_measures_precision

fprintf('\n--- Average precision across both datasets ---\n');
average_precision = (mean(mean_ratio_sub{1}(:,1:num_meas)) + mean(mean_ratio_sub{2}(:,1:num_meas))) / 2

p_Expts12_uncorrected = p{1} < .05 & p{2} < .05
p_Expt1_Bonferroni    = p{1} < .05 / num_meas / (num_meas-1) * 2
p_Expt2_Bonferroni    = p{2} < .05 / num_meas / (num_meas-1) * 2

%% Figure: Precision per % Altered — Haddara Dataset (Figure 1a)
figure('Color','w', 'DefaultAxesFontSize',12);

for meas = 1:num_meas
    ax = subplot(6,5,meas);
    plot(1:3, mean(ratio_sub{1}(:,:,:,meas), 3), 'o-');
    xlabel('% trials altered', 'FontSize', 14);
    if mod(meas,5) == 1; ylabel('Decrease in SD units', 'FontSize', 13); end
    xlim([.5, 3.5]);
    ylim([0, 1.8]);
    box off;
    set(ax, 'XTick', 1:3, 'XTickLabel', {'2%', '4%', '6%'});
    title(variable_names{meas}, 'FontSize', 13);
end
legend('bin=50', 'bin=100', 'bin=200', 'bin=400', 'FontSize', 12, 'Location', 'best');

%% Figure: Normalized Precision Comparison (Figure 1b)
ax = subplot(6,5,21:30);
normalized_precision = zeros(2, num_meas);

for dset = 1:2
    avg16 = mean(mean(mean_ratio_sub{dset}(:,1:16)));
    normalized_precision(dset,:) = mean(mean_ratio_sub{dset}(:,1:num_meas)) / avg16;
end

plot(1:num_meas, normalized_precision(1,:), 'm.', 'MarkerSize', 28); hold on;
plot(1:num_meas, normalized_precision(2,:), 'b.', 'MarkerSize', 28);
for meas = 1:num_meas
    plot([meas, meas], normalized_precision(:,meas), 'k-', 'LineWidth', 1);
end
plot([.5, num_meas+.5], [1,1], 'k-', 'LineWidth', 2);

xlabel('Measure', 'FontSize', 14);
ylabel('Normalized precision', 'FontSize', 14);
xlim([.5, num_meas+.5]);
box off;
set(ax, 'XTick', 1:num_meas, 'XTickLabel', variable_names(1:num_meas));
ax.XGrid = 'on'; ax.YGrid = 'on';
xtickangle(45);
legend('Haddara dataset', 'Maniscalco dataset', 'FontSize', 12, 'Location', 'best');
title('Average normalized precision', 'FontSize', 16);

[~, h] = suplabel('Validity and precision of each measure', 't');
set(h, 'FontSize', 18);
annotation('textbox', [0.02, 0.9, 0.1, 0.1], 'String', "\bf a", 'FontSize', 26, 'EdgeColor', 'none');
annotation('textbox', [0.02, 0.3, 0.1, 0.1], 'String', "\bf b", 'FontSize', 26, 'EdgeColor', 'none');

%% Supplementary Figure: Non-normalized Precision — Maniscalco Dataset
figure('Color','w', 'DefaultAxesFontSize',12);

for meas = 1:num_meas
    ax = subplot(5,5,meas);
    plot(1:3, mean(ratio_sub{2}(:,:,:,meas), 3), 'o-');
    xlabel('% trials altered', 'FontSize', 13);
    if mod(meas,5) == 1; ylabel('Decrease in SD units', 'FontSize', 12); end
    xlim([.5, 3.5]);
    ylim([0, 1.8]);
    box off;
    set(ax, 'XTick', 1:3, 'XTickLabel', {'2%', '4%', '6%'});
    title(variable_names{meas}, 'FontSize', 12);
end
legend('bin=50', 'bin=100', 'bin=200', 'bin=400', 'FontSize', 11, 'Location', 'best');

marker_colors = 'mb';
for dset = 1:2
    ax = subplot(5,5,21+(dset-1)*3:22+(dset-1)*3);
    means = mean(mean_ratio_sub{dset}(:,1:num_meas));
    sems  = std(mean_ratio_sub{dset}(:,1:num_meas)) / sqrt(size(mean_ratio_sub{dset},1)-1);
    plot([.5, num_meas+.5], repmat(mean(means(1:16)),1,2), 'k-'); hold on;
    errorbar(means, sems, ['o' marker_colors(dset)]);
    xlabel('Measure', 'FontSize', 13);
    ylabel('Decrease in SD units', 'FontSize', 12);
    xlim([.5, num_meas+.5]);
    ylim([0, 1]);
    box off;
    set(ax, 'XTick', 1:num_meas, 'XTickLabel', variable_names(1:num_meas));
    xtickangle(45);
    title(['Average precision — ' dset_names{dset}], 'FontSize', 13);
end
