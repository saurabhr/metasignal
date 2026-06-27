%% Aggregate Analysis: Response Bias Dependence (Locke Dataset)
% Replicates Figure 4 from Rahnev (2025), *Nature Communications*, 16(1), 701.
%
% Dataset: Locke et al. (2020) — 7 conditions varying response bias via
% prior probability and reward asymmetry. Conditions are re-ordered for
% plotting so that the x-axis goes from most negative to most positive bias:
%   Orig → Re-ordered: 6→1, 2→2, 4→3, 1→4, 5→5, 7→6, 3→7
%
% Analysis:
%   1. Repeated-measures ANOVA (condition as fixed, subject as random effect)
%   2. Correlation between each measure and absolute criterion |c|
%
% Run AFTER: preprocessing steps that generate dataset_Locke_2020.mat

%% Setup
clear; close all; clc

root_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root_dir, 'helperFunctions')));

%% Parameters
nRatings       = 2;  % binary confidence ratings in this dataset
num_conditions = 7;
recompute_measures = 0;

%% Load Data
load(fullfile(root_dir, 'Preprocess', 'dataset_Locke_2020'));
fprintf('Loaded %d subjects.\n', length(data));

%% Measure Labels
variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', '\DeltaConf', ...
    'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', '\DeltaConf-Ratio', ...
    'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', '\DeltaConf-Diff', ...
    'meta-noise', 'meta-uncertainty', 'd''', 'Criterion', 'Confidence'};

%% Compute Measures Per Condition
if recompute_measures
    for sub = 1:length(data)
        fprintf('Processing subject %d / %d\n', sub, length(data));
        for cond = 1:num_conditions
            filt = data{sub}.condition == cond;
            metas_bias(sub,cond,:) = compute_all_measures( ...
                data{sub}.stim(filt), data{sub}.resp(filt), data{sub}.conf(filt), nRatings);
        end
    end
end

%% Save or Load Results
results_path = fullfile(root_dir, 'Results', 'results_Locke');

if recompute_measures
    save(results_path, 'metas_*', 'variable_names');
    fprintf('Results saved to %s\n', results_path);
else
    load(results_path);
    fprintf('Results loaded from %s\n', results_path);
end

fprintf(['Measures: 1:meta-d'', 2:AUC2, 3:gamma, 4:phi, 5:deltaConf\n' ...
    '6:M_ratio, 7:AUC2_ratio, 8:gamma_ratio, 9:phi_ratio, 10:deltaConf_ratio\n' ...
    '11:M_diff, 12:AUC2_diff, 13:gamma_diff, 14:phi_diff, 15:deltaConf_diff\n' ...
    '16:meta-noise, 17:meta-uncertainty, 18:d'', 19:c, 20:conf\n\n']);

%% Analyze Dependence on Response Bias
fprintf('--------------- DEPENDENCE ON RESPONSE BIAS -----------------\n');

% Re-order the 7 conditions from most negative to most positive response bias
metas = [metas_bias(:,6,:), ...
    metas_bias(:,2,:), ...
    metas_bias(:,4,:), ...
    metas_bias(:,1,:), ...
    metas_bias(:,5,:), ...
    metas_bias(:,7,:), ...
    metas_bias(:,3,:)];

%% Repeated-Measures ANOVA (condition = fixed effect, subject = random effect)
num_sub  = size(metas,1);
num_cond = size(metas,2);
num_meas = size(metas,3);

x_cond    = reshape(repmat(1:num_cond, num_sub, 1), [], 1);
x_subject = repmat((1:num_sub)', num_cond, 1);
x2 = {x_cond, x_subject};

for meas = 1:num_meas
    [p_anova(meas,:), tbl] = anovan(reshape(metas(:,:,meas),[],1), x2, ...
        'random', 2, 'display', 'off');
    Fval(meas)              = cell2mat(tbl(2,6));
    partial_eta_squared(meas) = cell2mat(tbl(2,2)) / ...
        (cell2mat(tbl(2,2)) + cell2mat(tbl(4,2)));
end

p_anova = p_anova(:,1)'

%% Correlation: Measures vs. Absolute Criterion |c|
% For each subject × condition, correlate |c| with each measure
for meas = 1:num_meas
    % corr returns [num_sub × num_sub]; eye(num_sub) keeps within-subject diagonal
    r_eye = corr(abs(metas_bias(:,:,19))', metas_bias(:,:,meas)') .* eye(num_sub);
    r_eye(r_eye == 0) = NaN;
    r(meas,:)        = nansum(r_eye);
    r_average(meas)  = z2r(mean(r2z(r(meas,:))));
end

r_average

%% Create a figure for the paper (Figure 4)
% First half: each measure as a function of re-ordered condition
figure('Color','w', 'DefaultAxesFontSize',14);

for meas = 1:num_meas
    ax = subplot(6,5,meas);

    % Plot means
    plot(1:num_cond, nanmean(metas(:,:,meas)), 'r');
    hold on

    % Plot SEM
    for cond = 1:num_cond
        plot([cond, cond], ...
            [nanmean(metas(:,cond,meas)) - nanstd(metas(:,cond,meas))/sqrt(num_sub), ...
             nanmean(metas(:,cond,meas)) + nanstd(metas(:,cond,meas))/sqrt(num_sub)], ...
            'k', 'LineWidth', 2);
    end

    if meas >= 16; xlabel('Condition', 'FontSize', 16); end
    xlim([.5, num_cond + .5]);
    box off;
    set(ax, 'XTick', 1:num_cond);

    % Title with significance annotation (only *** vs ns at p < .001)
    if p_anova(meas) < .001
        p_text = '\color{red}***';
    else
        p_text = '\color{red}ns';
    end
    title({variable_names{meas}, p_text}, 'FontSize', 14);
end

% Second half: bar chart of correlations with |c| (Figure 4b)
ax = subplot(6,5,21:30);
colors = good_colors_for_plotting(1);
bar(1:17, r_average(1:17), 'FaceColor', colors);
hold on

for meas = 1:17
    % SEM using Fisher z-transform
    plot([meas, meas], ...
        [r_average(meas) - z2r(std(r2z(r(meas,:))))/sqrt(num_sub), ...
         r_average(meas) + z2r(std(r2z(r(meas,:))))/sqrt(num_sub)], 'k');
    plot(meas + .2, r(meas,:), '.k', 'MarkerSize', 8);
end

xlabel('Measure', 'FontSize', 16);
ylabel('Correlation coefficient (r-value)', 'FontSize', 16);
xlim([.5, 17.5]);
box off;
set(ax, 'XTick', 1:17);
set(gca, 'XTickLabel', variable_names);
xtickangle(45);
title('Correlation with absolute response bias', 'FontSize', 20);

% Super label + panel letters
[~,h2] = suplabel('Dependence on response bias', 't');
set(h2, 'FontSize', 20);
annotation('textbox', [0.02, 0.9, 0.1, 0.1], 'String', "\bf a", 'FontSize', 30, 'EdgeColor', 'none');
annotation('textbox', [0.02, 0.3, 0.1, 0.1], 'String', "\bf b", 'FontSize', 30, 'EdgeColor', 'none');
