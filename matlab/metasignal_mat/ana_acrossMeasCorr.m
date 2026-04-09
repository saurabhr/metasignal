%ana_acrossMeasCorr

clear
close all
clc

% Add helper functions
addpath(genpath(fullfile(pwd, 'helperFunctions')));

% Datasets
datasets = {'results_Haddara', 'results_Maniscalco', 'results_Shekhar'};

% Display the variable order for reference
fprintf(['Measures: 1:meta-d'', 2:AUC2, 3:gamma, 4:phi, 5:deltaConf\n' ...
    '6:M_ratio, 7:AUC2_ratio, 8:gamma_ratio, 9:phi_ratio, 10:deltaConf_ratio\n' ...
    '11:M_diff, 12:AUC2_diff, 13:gamma_diff, 14:phi_diff, 15:deltaConf_diff\n' ...
    '16:meta-noise, 17:meta-uncertainty, 18:d'', 19:c, 20:conf\n\n']);

% Load datasets and compute r values
for dset=1:length(datasets)
    load(fullfile('Results', datasets{dset}));
    if dset < 3
        metas = metas_raw;
    else %for Shekhar, average across the 3 contrasts
        metas = squeeze(mean(metas_raw,2));
    end
    
    % Reverse meta-noise and meta-uncertainty
    metas(:,16:17) = -metas(:,16:17);
    
    % Compute r values
    [r{dset},p{dset}] = corr(metas(1:17,1:17), metas(1:17,1:17), 'rows','complete');
    
    % Make diagonal values equal to NaN for ease of computation
    r_nan = r{dset};
    for meas=1:size(r{dset},1)
        r_nan(meas,meas) = NaN;
    end
    % Compute average correlation
    r_av17(dset) = z2r(mean(nanmean(z2r(r_nan(1:17,1:17)))));
    r_set1(dset) = z2r(mean(nanmean(z2r(r_nan(1:5,1:5)))));
    r_set2(dset) = z2r(mean(nanmean(z2r(r_nan(6:15,6:15)))));
    r_set1_set2(dset) = z2r(mean(nanmean(z2r(r_nan(1:5,6:15)))));
    r_16(dset) = z2r(mean(nanmean(z2r(r_nan(1:17,16)))));
    r_17(dset) = z2r(mean(nanmean(z2r(r_nan(1:17,17)))));
    r_1617(dset) = r_nan(16,17);
end

%% Display correlation values (each dataset, followed by the mean)
r_av17 = [r_av17, z2r(mean(r2z(r_av17)))] %each dataset + average
r_set1 = [r_set1, z2r(mean(r2z(r_set1)))] %each dataset + average
r_set2 = [r_set2, z2r(mean(r2z(r_set2)))] %each dataset + average
r_set1_set2 = [r_set1_set2, z2r(mean(r2z(r_set1_set2)))] %each dataset + average
r_16 = [r_16, z2r(mean(r2z(r_16)))] %each dataset + average
r_17 = [r_17, z2r(mean(r2z(r_17)))] %each dataset + average
r_1617 = [r_1617, z2r(mean(r2z(r_1617)))] %each dataset + average

%% Create correlation matrix plot (Figure 11)
plot_corrTables(r, p, variable_names, {'Haddara', 'Maniscalco', 'Shekhar'})