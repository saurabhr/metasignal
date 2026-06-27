%% Aggregate Analysis: Split-Half Reliability
% Replicates the split-half reliability analysis from Rahnev (2025),
% *Nature Communications*, 16(1), 701.
%
% Split-half reliability: odd vs. even trials within a bin are correlated.
% z-transformed Pearson r is averaged across bins/days/contrasts, then
% back-transformed. Datasets: Haddara, Shekhar (5D arrays with extra
% day/contrast dim), Maniscalco (4D, no day/contrast dim).
%
% Run AFTER: 01_analysis_Haddara.m, 02_analysis_Maniscalco.m, 05_analysis_Shekhar.m

%% Setup
clear; close all; clc

root_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root_dir, 'helperFunctions')));

%% Parameters
datasets   = {'results_Haddara', 'results_Shekhar', 'results_Maniscalco'};
binSize_TR = [50, 100, 200, 400];

variable_names = {'meta-d''', 'AUC2', 'Gamma', 'Phi', '\DeltaConf', ...
    'M-Ratio', 'AUC2-Ratio', 'Gamma-Ratio', 'Phi-Ratio', '\DeltaConf-Ratio', ...
    'M-Diff', 'AUC2-Diff', 'Gamma-Diff', 'Phi-Diff', '\DeltaConf-Diff', ...
    'meta-noise', 'meta-uncertainty', 'd''', 'Criterion', 'Confidence'};

fprintf(['Measures: 1:meta-d'', 2:AUC2, 3:gamma, 4:phi, 5:deltaConf\n' ...
    '6:M_ratio, 7:AUC2_ratio, 8:gamma_ratio, 9:phi_ratio, 10:deltaConf_ratio\n' ...
    '11:M_diff, 12:AUC2_diff, 13:gamma_diff, 14:phi_diff, 15:deltaConf_diff\n' ...
    '16:meta-noise, 17:meta-uncertainty, 18:d'', 19:c, 20:conf\n\n']);

%% Compute Split-Half Correlations
for dset = 1:length(datasets)
    load(fullfile(root_dir, 'Results', datasets{dset}));

    for binSize_num = 1:length(metas_splitHalf)

        % Haddara (dset=1) and Shekhar (dset=2): 5D array with day/contrast dim
        if dset <= 2
            for meas = 1:size(metas_splitHalf{binSize_num},5)
                for bin_num = 1:size(metas_splitHalf{binSize_num},2)
                    for day = 1:size(metas_splitHalf{binSize_num},3)
                        zSH_binDayorContr(bin_num,day) = r2z(corr( ...
                            metas_splitHalf{binSize_num}(:,bin_num,day,1,meas), ...
                            metas_splitHalf{binSize_num}(:,bin_num,day,2,meas), ...
                            'rows','complete'));
                    end
                end
                r(dset,binSize_num,meas) = z2r(mean(zSH_binDayorContr(:)));
            end

        % Maniscalco: 4D array (no day/contrast dim); inner loop handles all bin sizes
        else
            for binSize_num = 1:length(metas_splitHalf)
                for meas = 1:size(metas_splitHalf{binSize_num},4)
                    for bin_num = 1:size(metas_splitHalf{binSize_num},2)
                        zSH_bin(bin_num) = r2z(corr( ...
                            metas_splitHalf{binSize_num}(:,bin_num,1,meas), ...
                            metas_splitHalf{binSize_num}(:,bin_num,2,meas), ...
                            'rows','complete'));
                    end
                    r(dset,binSize_num,meas) = z2r(mean(zSH_bin));
                end
            end
        end
    end
end

%% Plot Results
figure('Color','w', 'DefaultAxesFontSize',13);

for meas = 1:20
    ax = subplot(4,5,meas);

    % plot(r(:,:,meas), '*-'): x = datasets (1:3), lines = bin sizes (4 lines)
    plot(r(:,:,meas), '*-');

    if meas >= 16; xlabel('Dataset', 'FontSize', 16); end
    ylabel('Pearson r', 'FontSize', 16);
    xlim([.5, length(datasets)+.5]);
    ylim([0, 1]);
    ax.YTick = 0:.1:1;
    ax.YGrid = 'on';
    box off;
    set(ax, 'XTick', 1:3);
    set(gca, 'XTickLabel', {'Hadda','Shekh','Manis'});
    title(variable_names{meas}, 'FontSize', 16, 'interpreter', 'tex');
end

legend('bin=50', 'bin=100', 'bin=200', 'bin=400', 'FontSize', 16);
[~,h2] = suplabel('Split-half reliability', 't');
set(h2, 'FontSize', 20);

%% Summary Statistics
mean_r_50trials         = z2r(squeeze(mean(r2z(r(:,1,:)))))'
mean_r_17meas_50trials  = z2r(mean(r2z(mean_r_50trials(1:17))))

mean_r_100trials        = z2r(squeeze(mean(r2z(r(:,2,:)))))'
mean_r_17meas_100trials = z2r(mean(r2z(mean_r_100trials(1:17))))

mean_r_200trials        = z2r(squeeze(mean(r2z(r(:,3,:)))))'
mean_r_17meas_200trials = z2r(mean(r2z(mean_r_200trials(1:17))))

mean_r_400trials        = z2r(squeeze(mean(r2z(r(:,4,:)))))'
mean_r_17meas_400trials = z2r(mean(r2z(mean_r_400trials(1:17))))
