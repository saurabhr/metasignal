%% Aggregate Analysis: Metacognitive Bias Dependence
% Replicates Figure 3 from Rahnev (2025), *Nature Communications*, 16(1), 701.
%
% This script examines whether metacognitive measures are contaminated by
% metacognitive bias (i.e., the tendency to use high vs. low confidence
% regardless of actual accuracy). Bias is induced by re-coding confidence
% ratings using the Xue et al. (2021) method: recode 1 shifts ratings toward
% low confidence; recode 2 shifts ratings toward high confidence.
%
% Datasets: Haddara, Maniscalco, Shekhar (3 datasets with bias manipulation)
%
% Run AFTER: 01_analysis_Haddara.m, 02_analysis_Maniscalco.m, 05_analysis_Shekhar.m

%% Setup
clear; close all; clc

root_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root_dir, 'helperFunctions')));

%% Datasets and Parameters
datasets   = {'results_Haddara', 'results_Maniscalco', 'results_Shekhar'};
dset_names = {'Haddara', 'Maniscalco', 'Shekhar'};
xvals      = {1:2, 4:5, 7:8};  % x-positions for each dataset in shared plot
colors     = good_colors_for_plotting(3);

variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', '\DeltaConf', ...
    'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', '\DeltaConf-Ratio', ...
    'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', '\DeltaConf-Diff', ...
    'meta-noise', 'meta-uncertainty', 'd''', 'Criterion', 'Confidence'};

%% Load Data and Compute Bias Effects
fprintf('Loading datasets...\n');

for dset = 1:length(datasets)
    load(fullfile(root_dir, 'Results', datasets{dset}));

    % For Shekhar: average across the 3 contrast levels
    if dset == 3
        metas{dset}     = squeeze(mean(metas_confRecode, 2));
        raw_metas{dset} = squeeze(mean(metas_diff, 2));
    else
        metas{dset}     = metas_confRecode;
        raw_metas{dset} = metas_raw;
    end

    % Paired t-test: recode 2 (high bias) vs. recode 1 (low bias)
    for meas = 1:20
        delta = squeeze(metas{dset}(:,2,meas) - metas{dset}(:,1,meas));
        [p(dset,meas), t(dset,meas), df(dset,meas), Cohen_d(dset,meas), CI(dset,meas,:)] = ...
            perform_ttest(delta, [], 0);
    end

    % How much do recoded scores differ from original scores?
    diff_recoded_minus_raw{dset} = squeeze(mean(metas{dset}(:,:,1:17), 2)) - raw_metas{dset}(:,1:17);
    prop_recoded_higher_per_dataset(dset) = mean(sum(diff_recoded_minus_raw{dset} > 0) / ...
        size(diff_recoded_minus_raw{dset}, 1));
end

%% Report Summary Statistics
fprintf('\n--- Proportion of subjects where recoded > raw score (per dataset) ---\n');
for dset = 1:length(datasets)
    fprintf('  %s: %.3f\n', dset_names{dset}, prop_recoded_higher_per_dataset(dset));
end

fprintf('\n--- Average Cohen''s d across measures ---\n');
averaga_Cohen_d = mean(Cohen_d)

t
p
Cohen_d

%% Figure: All Measures — Low vs. High Metacognitive Bias (Figure 3a)
figure('Color','w', 'DefaultAxesFontSize',14);

for meas = 1:20
    ax = subplot(6,5,meas);
    p_text = [];

    for dset = 1:length(datasets)
        num_sub = size(metas{dset}, 1);

        % Group means
        h(dset) = plot(xvals{dset}, nanmean(metas{dset}(:,:,meas)), ...
            'Color', colors{dset}, 'LineWidth', 2);
        hold on

        % SEM bars
        for recode = 1:2
            mu  = nanmean(metas{dset}(:,recode,meas));
            sem = nanstd(metas{dset}(:,recode,meas)) / sqrt(num_sub);
            plot([xvals{dset}(recode), xvals{dset}(recode)], [mu - sem, mu + sem], ...
                'k', 'LineWidth', 2);
        end

        % Significance annotation
        if     p(dset,meas) < .001; p_text = [p_text, sprintf('\\color[rgb]{%s} ***', num2str(colors{dset}))];
        elseif p(dset,meas) < .01;  p_text = [p_text, sprintf('\\color[rgb]{%s}  **', num2str(colors{dset}))];
        elseif p(dset,meas) < .05;  p_text = [p_text, sprintf('\\color[rgb]{%s}   *', num2str(colors{dset}))];
        else;                        p_text = [p_text, sprintf('\\color[rgb]{%s}  ns', num2str(colors{dset}))];
        end
    end

    xlim([.5, 8.5]);
    delta = diff(ax.YLim);
    ylim([ax.YLim(1) - delta/5, ax.YLim(2) + delta/5]);
    box off;
    set(ax, 'XTick', [1:2, 4:5, 7:8], 'XTickLabel', {'low','high','low','high','low','high'});
    xtickangle(45);
    if meas >= 16; xlabel('Confidence recode', 'FontSize', 14); end
    title({variable_names{meas}, p_text}, 'FontSize', 12, 'interpreter', 'tex');
end

legend(h, dset_names, 'FontSize', 14);
[~, h2] = suplabel('Dependence on metacognitive bias', 't');
set(h2, 'FontSize', 18);
annotation('textbox', [0.02, 0.9, 0.1, 0.1], 'String', "\bf a", 'FontSize', 28, 'EdgeColor', 'none');

%% Figure: Effect Size Bar Plot (Figure 3b)
num_meas = 17;
ax = subplot(6,5,21:30);

for meas = 1:num_meas
    for dset = 1:3
        bar((meas-1)*4 + dset, Cohen_d(dset,meas), 'FaceColor', colors{dset});
        hold on
    end
end

xlabel('Measure', 'FontSize', 14);
ylabel('Effect size (Cohen''s d)', 'FontSize', 14);
xlim([.5, 4*num_meas + .5]);
set(ax, 'XTick', 2:4:4*num_meas, 'XTickLabel', variable_names(1:num_meas));
xtickangle(45);
legend(dset_names, 'FontSize', 12);
title('Effect sizes for metacognitive bias dependence', 'FontSize', 16);
annotation('textbox', [0.02, 0.3, 0.1, 0.1], 'String', "\bf b", 'FontSize', 28, 'EdgeColor', 'none');

%% Supplementary Figure: Recoded vs. Raw Scores
figure('Color','w', 'DefaultAxesFontSize',14);

for meas = 1:20
    ax = subplot(4,5,meas);
    p_text = [];

    for dset = 1:length(datasets)
        num_sub = size(metas{dset}, 1);

        % Raw score (horizontal reference line)
        plot(xvals{dset}, repmat(mean(raw_metas{dset}(:,meas)), 1, 2), 'k-', 'LineWidth', 3);
        hold on

        % Recoded scores with SEM
        h(dset) = plot(xvals{dset}, nanmean(metas{dset}(:,:,meas)), ...
            'Color', colors{dset}, 'LineWidth', 2);
        for recode = 1:2
            mu  = nanmean(metas{dset}(:,recode,meas));
            sem = nanstd(metas{dset}(:,recode,meas)) / sqrt(num_sub);
            plot([xvals{dset}(recode), xvals{dset}(recode)], [mu - sem, mu + sem], ...
                'k', 'LineWidth', 2);
        end

        if     p(dset,meas) < .001; p_text = [p_text, sprintf('\\color[rgb]{%s} ***', num2str(colors{dset}))];
        elseif p(dset,meas) < .01;  p_text = [p_text, sprintf('\\color[rgb]{%s}  **', num2str(colors{dset}))];
        elseif p(dset,meas) < .05;  p_text = [p_text, sprintf('\\color[rgb]{%s}   *', num2str(colors{dset}))];
        else;                        p_text = [p_text, sprintf('\\color[rgb]{%s}  ns', num2str(colors{dset}))];
        end
    end

    xlim([.5, 8.5]);
    box off;
    set(ax, 'XTick', [1:2, 4:5, 7:8], 'XTickLabel', {'low','high','low','high','low','high'});
    xtickangle(45);
    title({variable_names{meas}, p_text}, 'FontSize', 12, 'interpreter', 'tex');
end

legend(h, dset_names, 'FontSize', 14);
[~, h2] = suplabel('Dependence on metacognitive bias (with raw scores)', 't');
set(h2, 'FontSize', 18);
