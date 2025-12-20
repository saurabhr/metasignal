%ana_respBias

%------------------
% Analysis of the Locke dataset that contained 7 conditions with different
% levels of induced response bias.
%------------------

clear
close all
clc

% Conditions
% 1: Prior = .50, Reward = 3:3
% 2: Prior = .75, Reward = 3:3
% 3: Prior = .25, Reward = 3:3
% 4: Prior = .50, Reward = 4:2
% 5: Prior = .50, Reward = 2:4
% 6: Prior = .75, Reward = 2:4
% 7: Prior = .25, Reward = 4:2

% Decide whether to compute M measures or just load saved values
recompute_measures = 0;

% Define important parameters
nRatings = 2;
num_conditions = 7;

% Load data and add helper functions
load('Preprocess/dataset_Locke_2020');
addpath(genpath(fullfile(pwd, 'helperFunctions')));

% If recomputing all measures, loop over all subjects and conditions
if recompute_measures
    for sub=1:length(data)
        sub
        for cond=1:num_conditions
            filter = data{sub}.condition == cond;
            metas_bias(sub,cond,:) = compute_all_measures(data{sub}.stim(filter), ...
                data{sub}.resp(filter), data{sub}.conf(filter), nRatings);
        end
    end
end


%% Load or save MetaNoise and M results as needed
if recompute_measures
    variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', '\DeltaConf', ...
        'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', '\DeltaConf-Ratio', ...
        'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', '\DeltaConf-Diff', ...
        'meta-noise', 'meta-uncertainty', 'd''', 'Criterion', 'Confidence'};
    save Results/results_Locke metas_* variable_names
else
    load Results/results_Locke
end

% Display the variable order for reference
fprintf(['Measures: 1:meta-d'', 2:AUC2, 3:gamma, 4:phi, 5:deltaConf\n' ...
    '6:M_ratio, 7:AUC2_ratio, 8:gamma_ratio, 9:phi_ratio, 10:deltaConf_ratio\n' ...
    '11:M_diff, 12:AUC2_diff, 13:gamma_diff, 14:phi_diff, 15:deltaConf_diff\n' ...
    '16:meta-noise, 17:meta-uncertainty, 18:d'', 19:c, 20:conf\n\n']);


%% ANALYZE DEPENDENCE ON RESPONSE BIAS
fprintf('--------------- DEPENDENCE ON RESPONSE BIAS -----------------\n');
% Re-order the 7 conditions
metas = [metas_bias(:,6,:), ...
    metas_bias(:,2,:), ...
    metas_bias(:,4,:), ...
    metas_bias(:,1,:), ...
    metas_bias(:,5,:), ...
    metas_bias(:,7,:), ...
    metas_bias(:,3,:)];

% Perform repeated measures ANOVA
num_sub = size(metas,1);
num_cond = size(metas,2);
num_meas = size(metas,3);
x_cond = reshape(repmat(1:num_cond,num_sub,1),[],1);
x_subject = repmat([1:num_sub]',num_cond,1);
x2 = {x_cond,x_subject}; %the main effect of the first factor (the 1st p value)
for meas=1:num_meas
    [p_anova(meas,:),tbl] = anovan(reshape(metas(:,:,meas),[],1),x2,'random',2,'display','off');
    Fval(meas) = cell2mat(tbl(2,6));
    partial_eta_squared(meas) = cell2mat(tbl(2,2)) / (cell2mat(tbl(2,2)) + cell2mat(tbl(4,2)));
end
p_anova = p_anova(:,1)'

% Correlate measures of metacognition with absolute value of the criterion (|c|)
for meas=1:num_meas
    r_eye = corr(abs(metas_bias(:,:,19))', metas_bias(:,:,meas)').*eye(10);
    r_eye(r_eye==0) = NaN; %remove 0's
    r(meas,:) = nansum(r_eye);
    r_average(meas) = z2r(mean(r2z(r(meas,:))));
end
r_average


%% Create table with ANOVA results (Supplementary Table 9)
%writematrix([Fval',p_anova', partial_eta_squared'], 'SuppTable9.xlsx');


%% Create a figure for the paper (Figure 4)
% First half of the figure plots each measure as a function of condition
figure('Color','w', 'DefaultAxesFontSize',14);

% Loop over all measures
for meas=1:num_meas
    % Create a new subplot
    ax = subplot(6,5,meas);
    
    % Plot means
    plot(1:num_cond, nanmean(metas(:,:,meas)), 'r');
    hold on
    
    % Plot SEM
    for cond=1:num_cond
        plot([cond,cond], [nanmean(metas(:,cond,meas))-nanstd(metas(:,cond,meas))/sqrt(num_sub), ...
            nanmean(metas(:,cond,meas))+nanstd(metas(:,cond,meas))/sqrt(num_sub)], 'k', 'LineWidth',2);
    end
    
    % Add details to the plot
    if meas>=16;   xlabel('Condition', 'FontSize',16);   end
    xlim([.5, num_cond+.5]);
    box off;
    set(ax,'XTick', 1:num_cond);
    
    % Add title
    if p_anova(meas) < .001
        p_text = '\color{red}***';
    else
        p_text = '\color{red}ns';
    end
    title({variable_names{meas}, p_text}, 'FontSize',14);
end


% Second half of the figure plots effect sizes (r-values)
ax = subplot(6,5,21:30);
colors = good_colors_for_plotting(1);
bar(1:17, r_average(1:17), 'FaceColor', colors);
hold on

% Plot individual data and SEM
for meas=1:17
    plot([meas,meas], [r_average(meas)-z2r(std(r2z(r(meas,:))))/sqrt(10), ...
        r_average(meas)+z2r(std(r2z(r(meas,:))))/sqrt(10)], 'k');
    plot(meas+.2, r(meas,:), '.k', 'MarkerSize', 8);
end
xlabel('Measure', 'FontSize',16);
ylabel('Correlation coefficient (r-value)', 'FontSize', 16);
xlim([.5, 17.5]);
box off;
set(ax,'XTick', 1:17);
set(gca,'XTickLabel',variable_names);
xtickangle(45);
title('Correlation with absolute response bias', 'FontSize', 20);

% Create super label + add panel letters
[~,h2] = suplabel('Dependence on response bias', 't');
set(h2, 'FontSize', 20);
annotation('textbox', [0.02, 0.9, 0.1, 0.1], 'String', "\bf a", 'FontSize', 30, 'EdgeColor', 'none')
annotation('textbox', [0.02, 0.3, 0.1, 0.1], 'String', "\bf b", 'FontSize', 30, 'EdgeColor', 'none')