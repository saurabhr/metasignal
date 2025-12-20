function [SDTexpectData, dprime, c] = SDTexpectConf(stim, resp, conf, nRatings)

%--------------------------------------------------------------------------
% The function computes d' and all criterion locations according to 
% standard SDT with equal variance assumption. It then generates optimal
% confidence responses based on SDT assumptions.
%
% Inputs:
% stim: vector of 2 values (lower one is S1, higher one is S2)
% resp: vector with the same values as the stimulus
% conf: vector with confidence values (should be integers 1-N)
% nRatings: number of confidence ratings used
%--------------------------------------------------------------------------

% Set nR_S1 and nR_S1 to 0 
nR_S1 = zeros(1,2*nRatings);
nR_S2 = zeros(1,2*nRatings);

% Loop over the confidence ratings
for rating=1:nRatings
    % S1 responses
    nR_S1(nRatings-rating+1) = sum(stim==min(stim) & resp==min(stim) & conf==rating);
    nR_S2(nRatings-rating+1) = sum(stim==max(stim) & resp==min(stim) & conf==rating);
    
    % S2 responses
    nR_S1(nRatings+rating) = sum(stim==min(stim) & resp==max(stim) & conf==rating);
    nR_S2(nRatings+rating) = sum(stim==max(stim) & resp==max(stim) & conf==rating);
end

% Correct for empty cells
if any(nR_S1==0) || any(nR_S2==0)
    nR_S1_corrected = nR_S1 + 1/(2*nRatings);
    nR_S2_corrected = nR_S2 + 1/(2*nRatings);
else
    nR_S1_corrected = nR_S1;
    nR_S2_corrected = nR_S2;
end

% Compute Hit Rate (HR) and False Alarm Rate (FAR)
HR = flip(cumsum(flip(nR_S2_corrected(2:end)))) / sum(nR_S2_corrected);
FAR = flip(cumsum(flip(nR_S1_corrected(2:end)))) / sum(nR_S1_corrected);

% Compute d' and all criterion locations
dprime = norminv(HR(nRatings)) - norminv(FAR(nRatings));
c = -.5 * (norminv(HR) + norminv(FAR));

% Compute expected FAR and HR rate (according to optimal SDT)
SDTexpectData.FAR = 1-normcdf(c,-dprime/2,1);
SDTexpectData.HR = 1-normcdf(c,dprime/2,1);

% Compute expected proportions for nR_S1 and nR_S2
SDTexpectData.nR_S1_SDTexpect = flip(diff([0, flip(SDTexpectData.FAR), 1]));
SDTexpectData.nR_S2_SDTexpect = flip(diff([0, flip(SDTexpectData.HR), 1]));

% Save the actual proportions for nR_S1 and nR_S2
SDTexpectData.nR_S1_actual = nR_S1;
SDTexpectData.nR_S2_actual = nR_S2;