function [phi, phi_ratio, phi_diff] = SDTphi(stim, resp, conf, nRatings)

% Compute data
SDTexpectData = SDTexpectConf(stim, resp, conf, nRatings);

% Actual
phi = compute_phi(SDTexpectData.nR_S1_actual, SDTexpectData.nR_S2_actual, nRatings);

% Expected
phi_expected = compute_phi(SDTexpectData.nR_S1_SDTexpect, SDTexpectData.nR_S2_SDTexpect, nRatings);

% Compute final values
phi_ratio = phi/phi_expected;
phi_diff = phi-phi_expected;

end

% Nested function to perform the computations
function phi = compute_phi(nR_S1, nR_S2, nRatings)

multiplier = [nRatings:-1:1, 1:nRatings];
correct_trials_S1 = [ones(1,nRatings), zeros(1,nRatings)];
correct_trials_S2 = [zeros(1,nRatings), ones(1,nRatings)];

correct_cells = [nR_S1(1:nRatings), nR_S2(nRatings+1:2*nRatings)];
incorrect_cells = [nR_S2(1:nRatings), nR_S1(nRatings+1:2*nRatings)];

av_acc = sum(correct_cells) / (sum(correct_cells)+sum(incorrect_cells));
av_conf = sum((correct_cells+incorrect_cells).*multiplier) / (sum(correct_cells)+sum(incorrect_cells));
numerator = sum((multiplier-av_conf).*(correct_trials_S1-av_acc).*nR_S1 + (multiplier-av_conf).*(correct_trials_S2-av_acc).*nR_S2);
denominator = sqrt(sum((multiplier-av_conf).^2.*(nR_S1+nR_S2)) * sum((correct_trials_S1-av_acc).^2.*(nR_S1+flip(nR_S2))));
phi = numerator/denominator;

end