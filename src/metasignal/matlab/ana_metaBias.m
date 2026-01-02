%ana_metaBias

clear
close all
clc

% Add helper functions
addpath(genpath(fullfile(pwd, 'helperFunctions')));

% Datasets
datasets = {'results_Haddara', 'results_Maniscalco', 'results_Shekhar'};
xvals = {1:2, 4:5, 7:8};
colors = good_colors_for_plotting(3);

% Display the variable order for reference
fprintf(['Measures: 1:meta-d'', 2:AUC2, 3:gamma, 4:phi, 5:deltaConf\n' ...
    '6:M_ratio, 7:AUC2_ratio, 8:gamma_ratio, 9:phi_ratio, 10:deltaConf_ratio\n' ...
    '11:M_diff, 12:AUC2_diff, 13:gamma_diff, 14:phi_diff, 15:deltaConf_diff\n' ...
    '16:meta-noise, 17:meta-uncertainty, 18:d'', 19:c, 20:conf\n\n']);

% Load each dataset
for dset=1:length(datasets)
    load(fullfile('Results', datasets{dset}));
    if dset==3 % For Shekhar, average across contrasts
        metas{dset} = squeeze(mean(metas_confRecode,2));
        raw_metas{dset} = squeeze(mean(metas_diff,2));
    else
        metas{dset} = metas_confRecode;
        raw_metas{dset} = metas_raw;
    end

    for meas=1:20
        [p(dset,meas), t(dset,meas), df(dset,meas), Cohen_d(dset,meas), CI(dset,meas,:)] = ...
            perform_ttest(squeeze(metas{dset}(:,2,meas)-metas{dset}(:,1,meas)), [], 0);
    end
    
    % Compare average recoded meta scores to raw scores before recoding
    diff_recoded_minus_raw{dset} = squeeze(mean(metas{dset}(:,:,1:17),2)) - raw_metas{dset}(:,1:17);
    prop_recoded_higher_per_dataset(dset) = mean(sum(diff_recoded_minus_raw{dset}>0)/size(diff_recoded_minus_raw{dset},1));
end
prop_recoded_higher_per_dataset


%% Display t and p values
t
p
Cohen_d
averaga_Cohen_d = mean(Cohen_d)


%% Create tables with t-test results (Supplementary Tables 6-8)
% range = [1:17,20]; %only create results for meta measures and conf (d' and c are undefined)
% for dset=1:3
%     writematrix([t(dset,range)', df(dset,range)', p(dset,range)', Cohen_d(dset,range)', ...
%         CI(dset,range,1)', CI(dset,range,2)'], ['SuppTable' num2str(5+dset) '.xlsx']);
% end


%% Create a figure for the paper (Figure 3)
% First half of the figure plots each measure as a function of meta bias (low/high)

% Open a new figure
figure('Color','w', 'DefaultAxesFontSize',14);

% Loop over all measures
for meas=1:20
    % Create relevant subplot
    ax = subplot(6,5,meas);
    
    % Load over the datasets
    p_text = [];
    for dset=1:length(datasets)
        num_sub = size(metas{dset},1);
        
        % Plot means
        h(dset) = plot(xvals{dset}, nanmean(metas{dset}(:,:,meas)), 'Color', colors{dset}, 'LineWidth',2);
        hold on
        
        % Plot SEM
        for recode=1:2
            plot([xvals{dset}(recode),xvals{dset}(recode)], [nanmean(metas{dset}(:,recode,meas))-nanstd(metas{dset}(:,recode,meas))/sqrt(num_sub), ...
                nanmean(metas{dset}(:,recode,meas))+nanstd(metas{dset}(:,recode,meas))/sqrt(num_sub)], 'k', 'LineWidth',2);
        end
        
        % Add to p_text
        p_text = [p_text, ['\color[rgb]{' num2str(colors{dset}) '}']];
        if p(dset,meas) < .001
            p_text = [p_text, ' *** '];
        elseif p(dset,meas) < .01
            p_text = [p_text, ' **  '];
        elseif p(dset,meas) < .05
            p_text = [p_text, '  *  '];
        else
            p_text = [p_text, ' ns  '];
        end
    end
    
    % Add details to the plot
    if meas>=16;   xlabel('Confidence recode', 'FontSize',16);  end
    xlim([.5, 8.5]);
    delta = diff(ax.YLim);
    ylim([ax.YLim(1)-delta/5, ax.YLim(2)+delta/5]);
    box off;
    set(ax,'XTick', [1:2, 4:5, 7:8]);
    set(gca,'XTickLabel',{'low','high','low','high','low','high'});
    xtickangle(45);
    title({variable_names{meas}, p_text}, 'FontSize',14, 'interpreter', 'tex');
end
legend(h, 'Haddara', 'Maniscalco', 'Shekhar', 'FontSize', 16);


% Second half of the figure plots effect sizes (Cohen's d)
ax = subplot(6,5,21:30);
num_meas = 17;
colors = good_colors_for_plotting(3);
for meas=1:num_meas
    for dset=1:3
        bar((meas-1)*4+dset, Cohen_d(dset,meas), 'FaceColor', colors{dset});
        hold on
    end
end
xlabel('Measure', 'FontSize',16);
ylabel('Effect size (Cohen''s d)', 'FontSize', 16);
xlim([.5, 4*num_meas+.5]);
% for ytick=-1:.5:2
%     yline(ytick,'--');
% end
box off;
set(ax,'XTick', 2:4:4*num_meas);
set(ax,'YTick', -1:.5:2);
set(gca,'XTickLabel',variable_names);
xtickangle(45);
legend('Haddara', 'Maniscalco', 'Shekhar', 'FontSize', 16);
title('Effect sizes for dependence on metacognitive bias', 'FontSize', 20);

% Create super label + add panel letters
[~,h2] = suplabel('Dependence on metacognitive bias' ,'t');
set(h2, 'FontSize', 20);
annotation('textbox', [0.02, 0.9, 0.1, 0.1], 'String', "\bf a", 'FontSize', 30, 'EdgeColor', 'none')
annotation('textbox', [0.02, 0.3, 0.1, 0.1], 'String', "\bf b", 'FontSize', 30, 'EdgeColor', 'none')


%% Create a figure for the paper (Supp Fig 3)
% Same as Figure 3a but also shows scores without using Xue et al method

% Open a new figure
figure('Color','w', 'DefaultAxesFontSize',14);

% Loop over all measures
for meas=1:20
    % Create relevant subplot
    ax = subplot(4,5,meas);
    
    % Load over the datasets
    p_text = [];
    for dset=1:length(datasets)
        num_sub = size(metas{dset},1);
        
        % Plot raw meta values before recoding
        h(dset) = plot(xvals{dset}, repmat(mean(raw_metas{dset}(:,meas)),1,2), 'k-', 'LineWidth',3);
        hold on
        
        % Plot means
        h(dset) = plot(xvals{dset}, nanmean(metas{dset}(:,:,meas)), 'Color', colors{dset}, 'LineWidth',2);
        hold on
        
        % Plot SEM
        for recode=1:2
            plot([xvals{dset}(recode),xvals{dset}(recode)], [nanmean(metas{dset}(:,recode,meas))-nanstd(metas{dset}(:,recode,meas))/sqrt(num_sub), ...
                nanmean(metas{dset}(:,recode,meas))+nanstd(metas{dset}(:,recode,meas))/sqrt(num_sub)], 'k', 'LineWidth',2);
        end
        
        % Add to p_text
        p_text = [p_text, ['\color[rgb]{' num2str(colors{dset}) '}']];
        if p(dset,meas) < .001
            p_text = [p_text, ' *** '];
        elseif p(dset,meas) < .01
            p_text = [p_text, ' **  '];
        elseif p(dset,meas) < .05
            p_text = [p_text, '  *  '];
        else
            p_text = [p_text, ' ns  '];
        end
    end
    
    % Add details to the plot
    if meas>=16;   xlabel('Confidence recode', 'FontSize',16);  end
    xlim([.5, 8.5]);
    delta = diff(ax.YLim);
    ylim([ax.YLim(1)-delta/5, ax.YLim(2)+delta/5]);
    box off;
    set(ax,'XTick', [1:2, 4:5, 7:8]);
    set(gca,'XTickLabel',{'low','high','low','high','low','high'});
    xtickangle(45);
    title({variable_names{meas}, p_text}, 'FontSize',14, 'interpreter', 'tex');
end
legend(h, 'Haddara', 'Maniscalco', 'Shekhar', 'FontSize', 16);

% Create super label
[~,h2] = suplabel('Dependence on metacognitive bias' ,'t');
set(h2, 'FontSize', 20);