%ana_taskPerformance

clear
close all
clc

% Add helper functions
addpath(genpath(fullfile(pwd, 'helperFunctions')));

% Datasets
datasets = {'results_Shekhar', 'results_Rouault1', 'results_Rouault2'};

% Display the variable order for reference
fprintf(['Measures: 1:meta-d'', 2:AUC2, 3:gamma, 4:phi, 5:deltaConf\n' ...
    '6:M_ratio, 7:AUC2_ratio, 8:gamma_ratio, 9:phi_ratio, 10:deltaConf_ratio\n' ...
    '11:M_diff, 12:AUC2_diff, 13:gamma_diff, 14:phi_diff, 15:deltaConf_diff\n' ...
    '16:meta-noise, 17:meta-uncertainty, 18:d'', 19:c, 20:conf\n\n']);

% Clean the data by setting extreme values (+/-3SD) to NaN
for dset=1:length(datasets)
    load(fullfile('Results', datasets{dset}));
    for meas=1:20
        for difficulty=1:size(metas_diff,2)
            cutoff_min = mean(metas_diff(:,difficulty,meas))-3*std(metas_diff(:,difficulty,meas));
            cutoff_max = mean(metas_diff(:,difficulty,meas))+3*std(metas_diff(:,difficulty,meas));
            metas_diff(metas_diff(:,difficulty,meas)<cutoff_min | metas_diff(:,difficulty,meas)>cutoff_max, difficulty, meas) = NaN;
        end
        
        %If an NaN value exists for any difficulty level, make the remaining difficulty levels also NaN
        metas_diff(isnan(sum(metas_diff(:,:,meas),2)),:,meas) = NaN;
        
        % Compute effect sizes of highest vs. lowest difficulty value
        [p(dset,meas), t(dset,meas), df(dset,meas), Cohen_d(dset,meas), CI(dset,meas,:)] = ...
            perform_ttest(metas_diff(:,end,meas) - metas_diff(:,1,meas), '', 0);
    end
    metas_all{dset} = metas_diff;
end


%% Report stats
av_Cohen_d = mean(Cohen_d)


%% Create tables with t-test results (Supplementary Tables 3-5)
% for dset=1:3
%     writematrix([t(dset,:)', df(dset,:)', p(dset,:)', Cohen_d(dset,:)', ...
%         CI(dset,:,1)', CI(dset,:,2)'], ['SuppTable' num2str(2+dset) '.xlsx']);
% end


%% Create a figure for the paper (Figure 2)
% First half of the figure plots each measure as a function of d'

% Open a new figure
figure('Color','w', 'DefaultAxesFontSize',14);
colors = good_colors_for_plotting(3);
num_meas = 17;

% Loop over all measures
for meas=1:20
    % Create relevant subplot
    ax = subplot(6,5,meas);
    
    % Load over the datasets
    p_text = [];
    for dset=1:length(datasets)
        metas = metas_all{dset};
        num_sub = size(metas,1);
        
        % Plot means
        h(dset) = plot(nanmean(metas(:,:,18)), nanmean(metas(:,:,meas)), 'Color', colors{dset}, 'LineWidth',2); %d' is measure 18
        hold on
        
        % Plot SEM
        for difficulty=1:size(metas,2)
            % Horizontal error bars
            plot([nanmean(metas(:,difficulty,18)),nanmean(metas(:,difficulty,18))], ...
                [nanmean(metas(:,difficulty,meas))-nanstd(metas(:,difficulty,meas))/sqrt(num_sub), ...
                nanmean(metas(:,difficulty,meas))+nanstd(metas(:,difficulty,meas))/sqrt(num_sub)], 'Color', colors{dset}, 'LineWidth',2);
            
            % Vertical error bars
            plot([nanmean(metas(:,difficulty,18))-nanstd(metas(:,difficulty,18))/sqrt(num_sub),...
                nanmean(metas(:,difficulty,18))+nanstd(metas(:,difficulty,18))/sqrt(num_sub)], ...
                [nanmean(metas(:,difficulty,meas)), nanmean(metas(:,difficulty,meas))], 'Color', colors{dset}, 'LineWidth',2);
        end
        
        % Compute p value
        [~,p] = ttest(metas(:,end,meas)-metas(:,1,meas));
        
        % Add to p_text
        p_text = [p_text, ['\color[rgb]{' num2str(colors{dset}) '}']];
        if p<.001
            p_text = [p_text, ' *** '];
        elseif p<.01
            p_text = [p_text, ' **  '];
        elseif p<.05
            p_text = [p_text, '  *  '];
        else
            p_text = [p_text, ' ns  '];
        end
    end
    
    % Add details to the plot
    xlabel('d''', 'FontSize',14)
    xlim([0,3]);
    box off;
    title({variable_names{meas}, p_text}, 'FontSize',14, 'interpreter', 'tex');
end


% Second half of the figure plots effect sizes (Cohen's d)
ax = subplot(6,5,21:30);
for meas=1:num_meas
    for dset=1:3
        bar((meas-1)*4+dset, Cohen_d(dset,meas), 'FaceColor', colors{dset});
        hold on
    end
end
xlabel('Measure', 'FontSize',16);
ylabel('Effect size (Cohen''s d)', 'FontSize', 16);
xlim([.5, 4*num_meas+.5]);
ylim([-1,7]);
box off;
set(ax,'XTick', 2:4:4*num_meas);
set(ax,'YTick', -1:1:7);
set(gca,'XTickLabel',variable_names);
xtickangle(45);
legend('Shekhar', 'Rouault1', 'Rouault2', 'FontSize', 16);
title('Effect sizes for dependence on task performance', 'FontSize', 18);

% Overall title of the figure + add panel letters
legend(h, 'Shekhar', 'Rouault1', 'Rouault2', 'FontSize', 16);
[~,h] = suplabel('Dependence on task performance' ,'t');
set(h, 'FontSize', 18);
annotation('textbox', [0.02, 0.9, 0.1, 0.1], 'String', "\bf a", 'FontSize', 30, 'EdgeColor', 'none')
annotation('textbox', [0.02, 0.3, 0.1, 0.1], 'String', "\bf b", 'FontSize', 30, 'EdgeColor', 'none')


%% Plot each measure as a function of task difficulty (Supp Fig 2)
% Open a new figure
figure('Color','w', 'DefaultAxesFontSize',14);
xvals = {1:3, 5:6, 8:9};
colors = good_colors_for_plotting(3);

% Loop over all measures
for meas=1:20
    % Create relevant subplot
    ax = subplot(4,5,meas);
    
    % Load over the datasets
    p_text = [];
    for dset=1:length(datasets)
        metas = metas_all{dset};
        num_sub = size(metas,1);
        
        % Plot means
        h(dset) = plot(xvals{dset}, nanmean(metas(:,:,meas)), 'Color', colors{dset}, 'LineWidth',2);
        hold on
        
        % Plot SEM
        for difficulty=1:length(xvals{dset})
            plot([xvals{dset}(difficulty),xvals{dset}(difficulty)], [nanmean(metas(:,difficulty,meas))-nanstd(metas(:,difficulty,meas))/sqrt(num_sub), ...
                nanmean(metas(:,difficulty,meas))+nanstd(metas(:,difficulty,meas))/sqrt(num_sub)], 'k', 'LineWidth',2);
        end
        
        % Compute p value
        [~,p] = ttest(metas(:,end,meas)-metas(:,1,meas));
        
        % Add to p_text
        p_text = [p_text, ['\color[rgb]{' num2str(colors{dset}) '}']];
        if p<.001
            p_text = [p_text, ' *** '];
        elseif p<.01
            p_text = [p_text, ' **  '];
        elseif p<.05
            p_text = [p_text, '  *  '];
        else
            p_text = [p_text, ' ns  '];
        end
    end
    
    % Add details to the plot
    xlabel('Difficulty', 'FontSize',14)
    xlim([.5, 9.5]);
    box off;
    set(ax,'XTick', [1:3, 5:6, 8:9]);
    set(gca,'XTickLabel',{'1','2','3','1','2','1','2'});
    title({variable_names{meas}, p_text}, 'FontSize',14, 'interpreter', 'tex');
end
legend(h, 'Shekhar', 'Rouault1', 'Rouault2', 'FontSize', 16);

% Create super label
[~,h2] = suplabel('Dependence on difficulty level' ,'t');
set(h2, 'FontSize', 20);