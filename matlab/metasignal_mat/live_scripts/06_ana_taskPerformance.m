%% Aggregate Analysis: Task Performance (Difficulty Dependence)
% Replicates Figure 2 from Rahnev (2025), *Nature Communications*, 16(1), 701.
%
% Loads difficulty-split results from three datasets (Shekhar, Rouault1,
% Rouault2) and tests how each metacognitive measure changes as task
% difficulty increases. Effect sizes (Cohen's d) are reported and plotted.
%
% Run AFTER: 03_analysis_Rouault1.m, 04_analysis_Rouault2.m, 05_analysis_Shekhar.m

%% Setup
clear; close all; clc

root_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root_dir, 'helperFunctions')));

%% Datasets
datasets = {'results_Shekhar', 'results_Rouault1', 'results_Rouault2'};

variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', '\DeltaConf', ...
    'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', '\DeltaConf-Ratio', ...
    'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', '\DeltaConf-Diff', ...
    'meta-noise', 'meta-uncertainty', 'd''', 'Criterion', 'Confidence'};

fprintf(['Measures: 1:meta-d'', 2:AUC2, 3:gamma, 4:phi, 5:deltaConf\n' ...
    '6:M_ratio, 7:AUC2_ratio, 8:gamma_ratio, 9:phi_ratio, 10:deltaConf_ratio\n' ...
    '11:M_diff, 12:AUC2_diff, 13:gamma_diff, 14:phi_diff, 15:deltaConf_diff\n' ...
    '16:meta-noise, 17:meta-uncertainty, 18:d'', 19:c, 20:conf\n\n']);

%% Load and Clean Data (±3 SD outlier removal per measure per difficulty level)
for dset = 1:length(datasets)
    load(fullfile(root_dir, 'Results', datasets{dset}));

    for meas = 1:20
        for difficulty = 1:size(metas_diff,2)
            cutoff_min = mean(metas_diff(:,difficulty,meas)) - 3*std(metas_diff(:,difficulty,meas));
            cutoff_max = mean(metas_diff(:,difficulty,meas)) + 3*std(metas_diff(:,difficulty,meas));
            metas_diff(metas_diff(:,difficulty,meas)<cutoff_min | ...
                metas_diff(:,difficulty,meas)>cutoff_max, difficulty, meas) = NaN;
        end

        % If NaN at any difficulty, NaN all difficulty levels for that subject
        metas_diff(isnan(sum(metas_diff(:,:,meas),2)), :, meas) = NaN;

        % Effect size: highest vs. lowest difficulty
        [p(dset,meas), t(dset,meas), df(dset,meas), Cohen_d(dset,meas), CI(dset,meas,:)] = ...
            perform_ttest(metas_diff(:,end,meas) - metas_diff(:,1,meas), '', 0);
    end

    metas_all{dset} = metas_diff;
end

%% Report Stats
av_Cohen_d = mean(Cohen_d)

%% Create a figure for the paper (Figure 2)
% First half of the figure plots each measure as a function of d'
figure('Color','w', 'DefaultAxesFontSize',14);
colors = good_colors_for_plotting(3);
num_meas = 17;

for meas = 1:20
    ax = subplot(6,5,meas);
    p_text = [];

    for dset = 1:length(datasets)
        metas   = metas_all{dset};
        num_sub = size(metas,1);

        % Plot means
        h(dset) = plot(nanmean(metas(:,:,18)), nanmean(metas(:,:,meas)), ...
            'Color', colors{dset}, 'LineWidth', 2); % d' is measure 18
        hold on

        % Plot SEM (vertical and horizontal error bars)
        for difficulty = 1:size(metas,2)
            % Vertical error bars (for the measure on y-axis)
            plot([nanmean(metas(:,difficulty,18)), nanmean(metas(:,difficulty,18))], ...
                [nanmean(metas(:,difficulty,meas)) - nanstd(metas(:,difficulty,meas))/sqrt(num_sub), ...
                 nanmean(metas(:,difficulty,meas)) + nanstd(metas(:,difficulty,meas))/sqrt(num_sub)], ...
                'Color', colors{dset}, 'LineWidth', 2);

            % Horizontal error bars (for d' on x-axis)
            plot([nanmean(metas(:,difficulty,18)) - nanstd(metas(:,difficulty,18))/sqrt(num_sub), ...
                  nanmean(metas(:,difficulty,18)) + nanstd(metas(:,difficulty,18))/sqrt(num_sub)], ...
                [nanmean(metas(:,difficulty,meas)), nanmean(metas(:,difficulty,meas))], ...
                'Color', colors{dset}, 'LineWidth', 2);
        end

        % Compute local p-value and add to title annotation
        [~,p] = ttest(metas(:,end,meas) - metas(:,1,meas));
        p_text = [p_text, ['\color[rgb]{' num2str(colors{dset}) '}']];
        if p < .001
            p_text = [p_text, ' *** '];
        elseif p < .01
            p_text = [p_text, ' **  '];
        elseif p < .05
            p_text = [p_text, '  *  '];
        else
            p_text = [p_text, ' ns  '];
        end
    end

    xlabel('d''', 'FontSize', 14);
    xlim([0, 3]);
    box off;
    title({variable_names{meas}, p_text}, 'FontSize', 14, 'interpreter', 'tex');
end

% Second half: effect size bar plot
ax = subplot(6,5,21:30);
for meas = 1:num_meas
    for dset = 1:3
        bar((meas-1)*4 + dset, Cohen_d(dset,meas), 'FaceColor', colors{dset});
        hold on
    end
end
xlabel('Measure', 'FontSize', 16);
ylabel('Effect size (Cohen''s d)', 'FontSize', 16);
xlim([.5, 4*num_meas + .5]);
ylim([-1, 7]);
box off;
set(ax, 'XTick', 2:4:4*num_meas, 'YTick', -1:1:7);
set(gca, 'XTickLabel', variable_names);
xtickangle(45);
legend('Shekhar', 'Rouault1', 'Rouault2', 'FontSize', 16);
title('Effect sizes for dependence on task performance', 'FontSize', 18);

% Super label + panel letters
legend(h, 'Shekhar', 'Rouault1', 'Rouault2', 'FontSize', 16);
[~,h] = suplabel('Dependence on task performance', 't');
set(h, 'FontSize', 18);
annotation('textbox', [0.02, 0.9, 0.1, 0.1], 'String', "\bf a", 'FontSize', 30, 'EdgeColor', 'none');
annotation('textbox', [0.02, 0.3, 0.1, 0.1], 'String', "\bf b", 'FontSize', 30, 'EdgeColor', 'none');

%% Plot each measure as a function of difficulty level (Supp Fig 2)
figure('Color','w', 'DefaultAxesFontSize',14);
xvals  = {1:3, 5:6, 8:9};
colors = good_colors_for_plotting(3);

for meas = 1:20
    ax = subplot(4,5,meas);
    p_text = [];

    for dset = 1:length(datasets)
        metas   = metas_all{dset};
        num_sub = size(metas,1);

        h(dset) = plot(xvals{dset}, nanmean(metas(:,:,meas)), ...
            'Color', colors{dset}, 'LineWidth', 2);
        hold on

        for difficulty = 1:length(xvals{dset})
            plot([xvals{dset}(difficulty), xvals{dset}(difficulty)], ...
                [nanmean(metas(:,difficulty,meas)) - nanstd(metas(:,difficulty,meas))/sqrt(num_sub), ...
                 nanmean(metas(:,difficulty,meas)) + nanstd(metas(:,difficulty,meas))/sqrt(num_sub)], ...
                'k', 'LineWidth', 2);
        end

        [~,p] = ttest(metas(:,end,meas) - metas(:,1,meas));
        p_text = [p_text, ['\color[rgb]{' num2str(colors{dset}) '}']];
        if p < .001
            p_text = [p_text, ' *** '];
        elseif p < .01
            p_text = [p_text, ' **  '];
        elseif p < .05
            p_text = [p_text, '  *  '];
        else
            p_text = [p_text, ' ns  '];
        end
    end

    xlabel('Difficulty', 'FontSize', 14);
    xlim([.5, 9.5]);
    box off;
    set(ax, 'XTick', [1:3, 5:6, 8:9]);
    set(gca, 'XTickLabel', {'1','2','3','1','2','1','2'});
    title({variable_names{meas}, p_text}, 'FontSize', 14, 'interpreter', 'tex');
end

legend(h, 'Shekhar', 'Rouault1', 'Rouault2', 'FontSize', 16);
[~,h2] = suplabel('Dependence on difficulty level', 't');
set(h2, 'FontSize', 20);
