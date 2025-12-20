function [dprime, c] = compute_SDTcriteria(stimulus, response, confidence, nRatings)

%--------------------------------------------------------------------------
% The function computes d' and all criterion locations according to 
% standard SDT with equal variance assumption.
%
% Inputs:
% stimulus: vector of 2 values (lower one is noise, higher one is stimulus)
% response: vector with the same values as the stimulus
% confidence: vector with confidence values
% nRatings: number of confidence ratings used
%--------------------------------------------------------------------------

% Confidence criteria on the left of the decision criterion
for roc_point=1:nRatings-1
    HR(roc_point) = sum(stimulus==max(stimulus) & (response==max(stimulus) | confidence<=nRatings-roc_point)) ...
        / sum(stimulus==max(stimulus));
    FAR(roc_point) = sum(stimulus==min(stimulus) & (response==max(stimulus) | confidence<=nRatings-roc_point)) ...
        / sum(stimulus==min(stimulus));
end

% Decision criterion and confidence criteria to the right of it
for roc_point=nRatings:2*nRatings-1
    HR(roc_point) = sum(stimulus==max(stimulus) & response==max(stimulus) & confidence>roc_point-nRatings) ...
        / sum(stimulus==max(stimulus));
    FAR(roc_point) = sum(stimulus==min(stimulus) & response==max(stimulus) & confidence>roc_point-nRatings) ...
        / sum(stimulus==min(stimulus));
end
        
% Compute d' and c
dprime = norminv(HR(nRatings)) - norminv(FAR(nRatings));
c = -.5 * (norminv(HR) + norminv(FAR));