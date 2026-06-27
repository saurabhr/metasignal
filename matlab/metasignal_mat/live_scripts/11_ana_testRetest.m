%% Aggregate Analysis: Test-Retest Reliability
% Replicates the test-retest reliability analysis from Rahnev (2025),
% *Nature Communications*, 16(1), 701.
%
% Test-retest reliability: each of 6 testing days in the Haddara dataset
% provides an independent measure estimate. All C(6,2) = 15 day-pair
% correlations are computed (both Pearson r and ICC), z-transformed,
% averaged, then back-transformed.
%
% ICC type: A-1 (absolute agreement, one-way random model).
% Dataset: Haddara (the only dataset with repeated sessions across days).
% Bin sizes: 50, 100, 200, 400 trials per day.
%
% Run AFTER: 01_analysis_Haddara.m

%% Setup
clear; close all; clc

root_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root_dir, 'helperFunctions')));

%% Parameters
dataset    = 'results_Haddara';
binSize_TR = [50, 100, 200, 400];

variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', '\DeltaConf', ...
    'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', '\DeltaConf-Ratio', ...
    'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', '\DeltaConf-Diff', ...
    'meta-noise', 'meta-uncertainty', 'd''', 'Criterion', 'Confidence'};

fprintf(['Measure order: 1:meta-d'', 2:AUC2, 3:gamma, 4:phi, 5:deltaConf\n' ...
    '               6:M_ratio ... 10:deltaConf_ratio\n' ...
    '               11:M_diff  ... 15:deltaConf_diff\n' ...
    '               16:meta-noise, 17:meta-uncertainty, 18:d'', 19:c, 20:conf\n\n']);

%% Load Data
load(fullfile(root_dir, 'Results', dataset));
metas = metas_testRetest;  % {binSize_num}(sub, bin_num, day, meas)

%% Outlier Removal (±3 SD per bin/day/measure)
fprintf('Removing outliers...\n');
for bs = 1:length(metas)
    for bin_num = 1:size(metas{bs},2)
        for day = 1:size(metas{bs},3)
            for meas = 1:size(metas{bs},4)
                mu  = mean(metas{bs}(:,bin_num,day,meas));
                sd  = std( metas{bs}(:,bin_num,day,meas));
                lo  = mu - 3*sd;
                hi  = mu + 3*sd;
                metas{bs}(metas{bs}(:,bin_num,day,meas)<lo | ...
                    metas{bs}(:,bin_num,day,meas)>hi, bin_num, day, meas) = NaN;
            end
        end
    end
end

%% Compute ICC and Pearson r for All Day Pairs
fprintf('Computing all day-pair correlations...\n');

for bs = 1:length(metas)
    for meas = 1:length(variable_names)
        n_bins = size(metas{bs}, 2);
        z      = NaN(n_bins, 5, 5);
        z_icc  = NaN(n_bins, 5, 5);

        for bin_num = 1:n_bins
            for day1 = 1:5
                for day2 = day1+1:6
                    m1 = metas{bs}(:,bin_num,day1,meas);
                    m2 = metas{bs}(:,bin_num,day2,meas);

                    % ICC (absolute agreement, one-way random)
                    data_matrix = [m1, m2];
                    data_matrix(any(isnan(data_matrix),2),:) = [];
                    if size(data_matrix,1) > 2
                        z_icc(bin_num,day1,day2-1) = r2z(ICC(data_matrix, 'A-1'));
                    end

                    % Pearson r
                    z(bin_num,day1,day2-1) = r2z(corr(m1, m2, 'rows', 'complete'));
                end
            end
        end

        z_iccTR(bs, meas) = nanmean(z_icc(:));
        z_TR(bs, meas)    = nanmean(z(:));
    end
end

%% Transform to r and Flip So Largest Bin is First
iccTR = flip(z2r(z_iccTR), 1)
rTR   = flip(z2r(z_TR),    1)

%% Report Average Reliability
fprintf('\n--- Average test-retest reliability (meta measures 1-17) ---\n');
fprintf('Metric       | 400 trials | 200 trials | 100 trials |  50 trials\n');
fprintf('%s\n', repmat('-',1,60));
fprintf('ICC (avg)    | %10.3f | %10.3f | %10.3f | %10.3f\n', mean(iccTR(:,1:17),2));
fprintf('Pearson r    | %10.3f | %10.3f | %10.3f | %10.3f\n', mean(rTR(:,1:17),2));

%% Figure: Test-Retest Reliability per Measure
figure('Color','w', 'DefaultAxesFontSize',14);

plot_vars  = {iccTR, rTR};
panel_titles = {'Test-retest reliability (ICC)', 'Test-retest reliability (Pearson r)'};
ylabels    = {'ICC', 'r-value'};

for mt = 1:2
    ax = subplot(2,1,mt);
    plot(plot_vars{mt}', '.', 'MarkerSize', 32);
    xlim([.5, length(variable_names)+.5]);
    ylim([0, 1]);
    ax.YTick  = 0:.1:1;
    ax.YGrid  = 'on';
    ax.XTick  = 1:length(variable_names);
    ax.XGrid  = 'on';
    box off;
    set(gca, 'XTickLabel', variable_names);
    xtickangle(45);
    ylabel(ylabels{mt}, 'FontSize', 18);
    title(panel_titles{mt}, 'FontSize', 18);
    if mt == 2; xlabel('Measure', 'FontSize', 18); end
    legend({'bin=400','bin=200','bin=100','bin=50'}, 'FontSize', 14, 'Location', 'best');
end
