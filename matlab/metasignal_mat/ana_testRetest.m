%ana_testRetest

clear
close all
clc

% Add helper functions
addpath(genpath(fullfile(pwd, 'helperFunctions')));

% Datasets
dataset = 'results_Haddara';
binSize_TR = [50, 100, 200, 400];

% Display the variable order for reference
fprintf(['Measures: 1:meta-d'', 2:AUC2, 3:gamma, 4:phi, 5:deltaConf\n' ...
    '6:M_ratio, 7:AUC2_ratio, 8:gamma_ratio, 9:phi_ratio, 10:deltaConf_ratio\n' ...
    '11:M_diff, 12:AUC2_diff, 13:gamma_diff, 14:phi_diff, 15:deltaConf_diff\n' ...
    '16:meta-noise, 17:meta-uncertainty, 18:d'', 19:c, 20:conf\n\n']);

% Go over the different datasets
load(fullfile('Results', dataset));

%metas_testRetest{binSize_num}(sub,bin_num,day,:)
metas = metas_testRetest;

% Clean the data by setting extreme values (+/-3SD) to NaN
for binSize_num=1:length(metas)
    for bin_num=1:size(metas{binSize_num},2)
        for day=1:size(metas{binSize_num},3)
            for meas=1:size(metas{binSize_num},4)
                cutoff_min = mean(metas{binSize_num}(:,bin_num,day,meas)) - ...
                    3*std(metas{binSize_num}(:,bin_num,day,meas));
                cutoff_max = mean(metas{binSize_num}(:,bin_num,day,meas)) + ...
                    3*std(metas{binSize_num}(:,bin_num,day,meas));
                metas{binSize_num}(metas{binSize_num}(:,bin_num,day,meas)<cutoff_min | ...
                    metas{binSize_num}(:,bin_num,day,meas)>cutoff_max, bin_num, day, meas) = NaN;
            end
        end
    end
end

% Correlate all pairs of days
for binSize_num=1:length(metas)
    bin_size = binSize_TR(binSize_num);
    for meas=1:length(variable_names)
        z = NaN(400/bin_size,5,5); %initialize variable
        z_icc = NaN(400/bin_size,5,5); %initialize variable
        for bin_num=1:size(metas{binSize_num},2)
            for day1=1:5
                for day2=day1+1:6
                    m1 = metas{binSize_num}(:,bin_num,day1,meas);
                    m2 = metas{binSize_num}(:,bin_num,day2,meas);
                    
                    % Compute ICC correlation (z-transformed)
                    data_matrix = [m1,m2];
                    data_matrix(any(isnan(data_matrix),2),:)=[]; %remove rows with any NaN values
                    z_icc(bin_num,day1,day2-1) = r2z(ICC(data_matrix, 'A-1'));
                    
                    % Compute Pearson correlation (z-transformed)
                    z(bin_num,day1,day2-1) = r2z(corr(m1, m2, 'rows','complete'));
                end
            end
        end
        z_iccTR(binSize_num,meas) = nanmean(z_icc(:));
        z_TR(binSize_num,meas) = nanmean(z(:));
    end
end

%% Transform the z-values to r-values and flip order so 400 bin is first
iccTR = flip(z2r(z_iccTR),1)
rTR = flip(z2r(z_TR),1)


%% Plot figure
figure('Color','w', 'DefaultAxesFontSize',14);
var_toPlot = {iccTR, rTR};
for measure_type=1:2
    ax = subplot(2,1,measure_type);
    plot(var_toPlot{measure_type}', '.', 'MarkerSize',35)
    xlim([.5, length(variable_names)+.5]);
    ylim([0,1]);
    ax.YTick = 0:.1:1;
    ax.YGrid = 'on';
    box off;
    ax.XTick = 1:length(variable_names);
    ax.XGrid = 'on';
    set(gca,'XTickLabel',variable_names);
    xtickangle(45);
    
    if measure_type==1
        ylabel('ICC', 'FontSize', 20);
        title('Test-retest reliability (ICC)', 'FontSize', 20);
    else
        ylabel('r-value', 'FontSize', 20);
        title('Test-retest reliability (Pearson correlation)', 'FontSize', 20);
        xlabel('Measure', 'FontSize', 20);
    end
    legend('bin=400', 'bin=200', 'bin=100', 'bin=50', 'FontSize', 16);
end