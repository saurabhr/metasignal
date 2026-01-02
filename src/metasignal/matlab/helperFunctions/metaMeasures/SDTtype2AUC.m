function [type2AUC, type2AUC_ratio, type2AUC_diff] = SDTtype2AUC(stim, resp, conf, nRatings)

% Compute data
SDTexpectData = SDTexpectConf(stim, resp, conf, nRatings);

% Actual
type2AUC = compute_Type2AUC(SDTexpectData.nR_S1_actual, SDTexpectData.nR_S2_actual, nRatings);

% Expected
type2AUC_expected = compute_Type2AUC(SDTexpectData.nR_S1_SDTexpect, SDTexpectData.nR_S2_SDTexpect, nRatings);

% Compute final values
type2AUC_ratio = type2AUC/type2AUC_expected;
type2AUC_diff = type2AUC-type2AUC_expected;

end

% Nested function to perform the computations
function type2AUC = compute_Type2AUC(nR_S1, nR_S2, nRatings)

% Compute counts of confidence ratings for correct and incorrect responses
counts_correct = nR_S2(nRatings+1:end) + nR_S1(nRatings:-1:1);
counts_incorrect = nR_S1(nRatings+1:end) + nR_S2(nRatings:-1:1);

% Compute Type-2 HR and FAR
HR2 = cumsum(counts_correct, 'reverse') / sum(counts_correct);
FAR2 = cumsum(counts_incorrect, 'reverse') / sum(counts_incorrect);

% Transform HR2 and FAR2 by inverting the order and adding 0 to both
HR2 = [0, HR2(end:-1:1)];
FAR2 = [0, FAR2(end:-1:1)];

% Compute type2AUC by summing the area under each segment
type2AUC = 0;
for segment=1:nRatings
    type2AUC = type2AUC + (FAR2(segment+1)-FAR2(segment)) * (HR2(segment+1)+HR2(segment)) / 2;
end

end