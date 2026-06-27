%% Run All MATLAB Tutorials
% Sequentially executes tutorials 01–07 in this directory.
% Each tutorial calls clear/close/clc, so they run independently.
%
% Run from the MATLAB command window or via the Live Editor Run button.

%% Setup
root_dir  = fileparts(mfilename('fullpath'));  % …/tutorials/
scripts   = { ...
    '01_getting_started.m', ...
    '02_computing_measures.m', ...
    '03_statistical_inference.m', ...
    '04_difficulty_dependence.m', ...
    '05_metacognitive_bias.m', ...
    '06_split_half_reliability.m', ...
    '07_mle_fitting.m' };

n_passed = 0;
n_failed = 0;
errors   = {};

%% Execute each tutorial
for k = 1:length(scripts)
    script_path = fullfile(root_dir, scripts{k});
    header = repmat('=', 1, 60);
    fprintf('\n%s\n', header);
    fprintf('Running: %s\n', scripts{k});
    fprintf('%s\n', header);
    try
        run(script_path);
        fprintf('\n[OK] %s\n', scripts{k});
        n_passed = n_passed + 1;
    catch ME
        fprintf('\n[FAILED] %s\n', scripts{k});
        fprintf('  Error: %s\n', ME.message);
        errors{end+1} = sprintf('%s: %s', scripts{k}, ME.message); %#ok<AGROW>
        n_failed = n_failed + 1;
    end
    close all;
end

%% Summary
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('Summary: %d passed, %d failed\n', n_passed, n_failed);
if ~isempty(errors)
    fprintf('\nFailed scripts:\n');
    for e = 1:length(errors)
        fprintf('  %s\n', errors{e});
    end
end
