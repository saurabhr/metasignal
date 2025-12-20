%step2_preprocessData

clear
clc

% Figure out the dataset names
files = dir('raw_matlab_files/dataset*.mat');

% Subject-level criteria for exclusion
num_excluded_subjects = zeros(1,length(files));
acc_thresholds = [.6, .95]; %minimum/maximum allowed accuracy
max_proportion_same_respConf = .85; %max proportion for either same response or same confidence rating

% Loop over all datasets
for dataset_num=1:length(files)
    
    % Load the dataset
    load(['raw_matlab_files/' files(dataset_num).name]);
    subject_number_idx = unique(data.subj_idx);
    
    % Display details about the dataset
    dataset_num
    files(dataset_num).name
    num_sub = length(subject_number_idx)
    num_trials_per_subj = length(data.subj_idx)/num_sub
    total_trials = 0;
    
    % Loop over all subjects in the dataset
    sub = 0;
    clear exclusion data_clean
    for subject=1:num_sub
        
        % Organize the data
        stim = data.stim(data.subj_idx==subject_number_idx(subject));
        resp = data.resp(data.subj_idx==subject_number_idx(subject));
        conf = data.conf(data.subj_idx==subject_number_idx(subject));
        correct = (stim==resp) + 0; %integers rather than boolians
        if isfield(data,'contrast')
            contrast = data.contrast(data.subj_idx==subject_number_idx(subject));
        end
        if isfield(data,'condition')
            condition = data.condition(data.subj_idx==subject_number_idx(subject));
        end
        if isfield(data,'day')
            day = data.day(data.subj_idx==subject_number_idx(subject));
        end
        
        % Variables to display the various kinds of exclusion reasons
        exclusion(subject, :) = [mean(correct) < acc_thresholds(1), ...
            mean(correct) > acc_thresholds(2), ...
            sum(resp==mode(resp)) > length(stim)*max_proportion_same_respConf, ...
            sum(conf==mode(conf)) > length(stim)*max_proportion_same_respConf];
        
        % Decide whether to exclude the subject
        if mean(correct(correct>=0)) < acc_thresholds(1) || mean(correct(correct>=0)) > acc_thresholds(2) || ...
                sum(resp==mode(resp)) > length(stim)*max_proportion_same_respConf || ...
                sum(conf==mode(conf)) > length(stim)*max_proportion_same_respConf
            num_excluded_subjects(dataset_num) = num_excluded_subjects(dataset_num) + 1;
        else
            % Update the subject number
            sub = sub + 1;
            data_clean{sub}.stim = stim;
            data_clean{sub}.resp = resp;
            data_clean{sub}.conf = conf;
            if isfield(data,'contrast')
                data_clean{sub}.contrast = contrast;
            end
            if isfield(data,'condition')
                data_clean{sub}.condition = condition;
            end
            if isfield(data,'day')
                data_clean{sub}.day = day;
            end
            total_trials = total_trials + length(stim);
        end
    end
    
    % Save data from the current dataset
    data = data_clean;
    %save(files(dataset_num).name, 'data');
    
    % Check how many subjects are excluded
    exclusion_reasons = 'low acc, high acc, identical responses, identical confidence'
    total_excluded = sum(exclusion)
    percent_excluded(dataset_num) = num_excluded_subjects(dataset_num)/num_sub*100;
    number_good_subjects(dataset_num) = length(data);
    trials_per_subj(dataset_num) = total_trials/length(data);
end

num_excluded_subjects
percent_excluded
number_good_subjects
trials_per_subj