%step1_importDataToMatlab

clear
clc

% Select datasets
dataset_names = {'Haddara_2022_Expt2','Locke_2020','Maniscalco_2017_expt1',...
    'Rouault_2018_Expt1','Rouault_2018_Expt2','Shekhar_2021'};

% Load each file and save data in .mat file
for dataset_file_num=1:length(dataset_names)
    
    % Load the dataset
    file_name = fullfile(pwd, 'orig_csv_files', ['data_' dataset_names{dataset_file_num} '.csv']);
    table = readtable(file_name);
    
    % Extract needed fields
    clear data
    data.subj_idx = table.Subj_idx;
    data.stim = table.Stimulus;
    data.resp = table.Response;
    data.conf = table.Confidence;
    
    % Show the head of the table
    dataset_names{dataset_file_num}
    head(table)
    
    %% Save the data from each dataset
    % For Locke_2020, save the condition and remove training trials
    if strcmp(dataset_names{dataset_file_num}, 'Locke_2020')
        training = table.Training;
        data.subj_idx = data.subj_idx(training==0);
        data.stim = data.stim(training==0);
        data.resp = data.resp(training==0);
        data.conf = data.conf(training==0) + 1; %change confidence from 0/1 to 1/2
        data.condition = table.Condition(training==0);
        data.condition_names = {'1: Prior = .50, Reward = 3:3', '2: Prior = .75, Reward = 3:3',...
            '3: Prior = .25, Reward = 3:3', '4: Prior = .50, Reward = 4:2', ...
            '5: Prior = .50, Reward = 2:4', '6: Prior = .75, Reward = 2:4', '7: Prior = .25, Reward = 4:2'};
        
        % For Shekhar_2021, save the contrast
    elseif strcmp(dataset_names{dataset_file_num}, 'Shekhar_2021')
        data.contrast = table.Contrast;
        data.day = repmat([ones(800,1); 2*ones(1000,1); 3*ones(1000,1)], 20, 1);
        
        % For Rouault_2018_Expt1/2, simply save the data and add info about
        % DotDiff (the difficulty of the task -> lower means more difficult)
    elseif strcmp(dataset_names{dataset_file_num}(1:7), 'Rouault')
        data.contrast = table.DotDiff; %larger dotDiff makes the task easier
        
        % For Haddara_2022_Expt2, save the day of testing
    elseif strcmp(dataset_names{dataset_file_num}, 'Haddara_2022_Expt2')
        data.day = table.Day;
    end
    
    % Save the data
    %save(['raw_matlab_files/dataset_' dataset_names{dataset_file_num}], 'data');
end