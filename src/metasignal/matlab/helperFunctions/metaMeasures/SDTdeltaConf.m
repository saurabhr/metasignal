function [delta_conf, delta_conf_ratio, delta_conf_diff] = SDTdeltaConf(stim, resp, conf, nRatings)

% Compute data
SDTexpectData = SDTexpectConf(stim, resp, conf, nRatings);

% Actual
delta_conf = compute_deltaConf(SDTexpectData.nR_S1_actual, SDTexpectData.nR_S2_actual, nRatings);

% Expected
conf_diff_expected = compute_deltaConf(SDTexpectData.nR_S1_SDTexpect, SDTexpectData.nR_S2_SDTexpect, nRatings);

% Compute final values
delta_conf_ratio = delta_conf/conf_diff_expected;
delta_conf_diff = delta_conf-conf_diff_expected;

end

% Nested function to perform the computations
function delta_conf = compute_deltaConf(nR_S1, nR_S2, nRatings)

multiplier = [nRatings:-1:1, 1:nRatings];
correct_cells = [nR_S1(1:nRatings), nR_S2(nRatings+1:2*nRatings)];
incorrect_cells = [nR_S2(1:nRatings), nR_S1(nRatings+1:2*nRatings)];
mean_conf_correct = sum(correct_cells.*multiplier) / sum(correct_cells);
mean_conf_incorrect = sum(incorrect_cells.*multiplier) / sum(incorrect_cells);
delta_conf = mean_conf_correct - mean_conf_incorrect;

end