%ana_precision

clear
close all
clc

% Add helper functions
addpath(genpath(fullfile(pwd, 'helperFunctions')));

% Set cutoff for removing outliers
cutoff_for_outliers = 4.5;

% Datasets
datasets = {'results_Haddara', 'results_Maniscalco'};

% Display the variable order for reference
fprintf(['Measures: 1:meta-d'', 2:AUC2, 3:gamma, 4:phi, 5:deltaConf\n' ...
    '6:M_ratio, 7:AUC2_ratio, 8:gamma_ratio, 9:phi_ratio, 10:deltaConf_ratio\n' ...
    '11:M_diff, 12:AUC2_diff, 13:gamma_diff, 14:phi_diff, 15:deltaConf_diff\n' ...
    '16:meta-noise, 17:meta-uncertainty, 18:d'', 19:c, 20:conf\n\n']);

% Load datasets and compute r values
for dset=1:length(datasets)
    load(fullfile('Results', datasets{dset}));
    num_sub = size(metas_precision{1},1);
    for binSize_num=1:length(metas_precision)
        metas = metas_precision{binSize_num};
        metas(abs(metas)>cutoff_for_outliers) = NaN; %Remove outliers
        if dset==1 %Combine data from all days for Haddara but not for Maniscalco (only one day there)
            metas = reshape(metas, size(metas,1), [], size(metas,4), size(metas,5));
            metad_means(binSize_num,:) = squeeze(nanmean(nanmean(nanmean(metas_precision{binSize_num}(:,:,:,:,1),1),2),3))';
        end
        metas(:,:,:,16:17) = -metas(:,:,:,16:17); %Flip meta-noise and meta-uncertainty
        SD_sub_meas = squeeze(nanstd(metas(:,:,1,:),[],2));
        
        % Compute the drop in meta accuracy in SD units. To compute the SD
        % units, use data from all subjects because doing this on a
        % single-subject basis is too noisy due to little data.
        for bin_num=2:size(metas,3)
            diff_sub_meas = squeeze(nanmean(metas(:,:,1,:)-metas(:,:,bin_num,:),2));
            ratio_sub{dset}(binSize_num,bin_num-1,:,:) = diff_sub_meas./repmat(mean(SD_sub_meas),num_sub,1);
        end
    end
    
    % Compute the average precision across all proportions of trials altered and bin sizes
    mean_ratio_sub{dset} = squeeze(mean(mean(ratio_sub{dset},2),1));
    
    % Compute all pairwise t-tests
    for meas1=1:17
        for meas2=1:17
            [~,p{dset}(meas1,meas2)] = ttest(mean_ratio_sub{dset}(:,meas1), mean_ratio_sub{dset}(:,meas2));
        end
    end
end

%% Perform analyses
mean_metad_values = mean(metad_means)
mean_precision_per_bin = [mean(mean(mean(ratio_sub{1}(1,:,:,1:17)))), mean(mean(mean(ratio_sub{1}(2,:,:,1:17)))), ...
    mean(mean(mean(ratio_sub{1}(3,:,:,1:17)))), mean(mean(mean(ratio_sub{1}(4,:,:,1:17))))]
meta_uncertainty_precision_values = [mean(mean_ratio_sub{1}(:,17)), mean(mean_ratio_sub{2}(:,17))]
average_of_other_measures_precision = [mean(mean(mean_ratio_sub{1}(:,1:16))), mean(mean(mean_ratio_sub{2}(:,1:16)))]
p_Expts12_uncorrected = p{1}<.05 & p{2}<.05
p_Expt1_Bonferroni = p{1}<.05/17/16*2 %control for 17*16/2 multiple comparisons
p_Expt2_Bonferroni = p{2}<.05/17/16*2 %control for 17*16/2 multiple comparisons
meta_uncertainty_ratio = meta_uncertainty_precision_values./average_of_other_measures_precision
z_scores = [normalize(mean(mean_ratio_sub{1}(:,1:17))); normalize(mean(mean_ratio_sub{2}(:,1:17)))]
average_precision = (mean(mean_ratio_sub{1}(:,1:17))+mean(mean_ratio_sub{2}(:,1:17)))/2


%% Create a figure for the paper (Figure 1)
% First half of the figure plots each measure as a function of % trials
% altered in the Haddara dataset
figure('Color','w', 'DefaultAxesFontSize',12);

% Loop over all measures
for meas=1:17
    % Create a new subplot
    ax = subplot(6,5,meas);
    
    % Plot means
    plot(1:3, mean(ratio_sub{1}(:,:,:,meas),3), 'o-');
    
    % Add details to the subplot
    xlabel('% trials altered', 'FontSize',16);
    if mod(meas,5)==1
        ylabel('Decrease in SD units', 'FontSize', 16);
    end
    xlim([.5, 3.5]);
    ylim([0, 1.8]);
    box off;
    set(ax,'XTick', 1:3);
    set(gca,'XTickLabel',{'2%', '4%', '6%'});
    title(variable_names{meas}, 'FontSize', 16);
end
legend('bin=50', 'bin=100', 'bin=200', 'bin=400', 'FontSize', 16);

% Second half of the figure plots normalized precision for the Haddara and
% Maniscalco datasets
ax = subplot(6,5,21:30);
for dset=1:2
    average16 = mean(mean(mean_ratio_sub{dset}(:,1:16)));
    normalized_precision(dset,:) = mean(mean_ratio_sub{dset}(:,1:17)) / average16;
end
plot(1:17, normalized_precision(1,:), 'm.', 'MarkerSize',30);
hold on
plot(1:17, normalized_precision(2,:), 'b.', 'MarkerSize',30);
for meas=1:17
    plot([meas, meas], normalized_precision(:,meas), 'k-', 'LineWidth', 1);
end
xlabel('Measure', 'FontSize',16);
ylabel('Normalized precision', 'FontSize', 16);
xlim([.5, 17.5]);
plot([.5, 17.5], [1,1], 'k-', 'LineWidth', 2);
box off;
set(ax,'XTick', 1:17);
set(gca,'XTickLabel',variable_names);
ax.XGrid = 'on';
ax.YGrid = 'on';
xtickangle(45);
legend('Haddara dataset', 'Maniscalco dataset', 'FontSize', 16);
title('Average normalized precision', 'FontSize', 20);

% Create super label + add panel letters
[~,h] = suplabel('Validity and precision of each measure' ,'t');
set(h, 'FontSize', 20);
annotation('textbox', [0.02, 0.9, 0.1, 0.1], 'String', "\bf a", 'FontSize', 30, 'EdgeColor', 'none')
annotation('textbox', [0.02, 0.3, 0.1, 0.1], 'String', "\bf b", 'FontSize', 30, 'EdgeColor', 'none')


%% Create a figure for Maniscalco dataset (Supplementary Fig 1)
% First half of the figure plots each measure as a function of % trials
% altered in the Maniscalco dataset
figure('Color','w', 'DefaultAxesFontSize',12);
num_meas = 17;

% Loop over all measures
for meas=1:num_meas
    % Create a new subplot
    ax = subplot(5,5,meas);
    
    % Plot means
    plot(1:3, mean(ratio_sub{2}(:,:,:,meas),3), 'o-');
    
    % Add details to the subplot
    xlabel('% trials altered', 'FontSize',16);
    if mod(meas,5)==1
        ylabel('Decrease in SD units', 'FontSize', 16);
    end
    xlim([.5, 3.5]);
    ylim([0, 1.8]);
    box off;
    set(ax,'XTick', 1:3);
    set(gca,'XTickLabel',{'2%', '4%', '6%'});
    title(variable_names{meas}, 'FontSize', 16);
end
legend('bin=50', 'bin=100', 'bin=200', 'bin=400', 'FontSize', 16);

% Second half of the figure plots non-normalized precision for the Haddara 
% and Maniscalco datasets
dset_names = {'Haddara', 'Maniscalco'};
colors = 'mb'; %magenta and blue
for dset=1:2
    ax = subplot(5,5,21+(dset-1)*3:22+(dset-1)*3);
    plot([.5, num_meas+.5], repmat(mean(mean(mean_ratio_sub{dset}(:,1:16))), 1,2), 'k-');
    hold on
    means = mean(mean_ratio_sub{dset}(:,1:17));
    sem = std(mean_ratio_sub{dset}(:,1:17))/sqrt(size(mean_ratio_sub{dset},1)-1);
    errorbar(means, sem, ['o' colors(dset)]);
    xlabel('Measure', 'FontSize',16);
    ylabel('Decrease in SD units', 'FontSize', 16);
    xlim([.5, num_meas+.5]);
    ylim([0, 1]);
    box off;
    set(ax,'XTick', 1:num_meas);
    set(gca,'XTickLabel',variable_names);
    xtickangle(45);
    title(['Average precision in ' dset_names{dset} ' dataset'], 'FontSize', 16);
end

% Add panel letters
annotation('textbox', [0.02, 0.9, 0.1, 0.1], 'String', "\bf a", 'FontSize', 30, 'EdgeColor', 'none')
annotation('textbox', [0.02, 0.3, 0.1, 0.1], 'String', "\bf b", 'FontSize', 30, 'EdgeColor', 'none')