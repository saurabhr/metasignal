function [metaUncertainty] = compute_metaUncertainty(stim, resp, conf, nRatings)
%% compute_metaUncertainty returns an estimate of meta-uncertainty given a vector of stimuli, perceptual reponses, and confidence ratings

nConfCrit       = nRatings - 1; % number of confidence criteria
stimValue       = unique(stim);
respOptions     = reshape(repmat(unique(resp)',nRatings,1),nRatings*numel(stimValue),1);
confOptions     = [nRatings:-1:1, 1:nRatings]; %[flipud(unique(conf))' unique(conf)'];

% Reorganize responses in a choice matrix:
nChoice         = zeros(nRatings*2,numel(stimValue));

for iS=1:numel(stimValue) % loop through each stimulus
    stim_ind = stim==stimValue(iS); % find when that stimulus was presented
    for iR=1:length(respOptions) % sum # of each response type 
      nChoice(iR,iS) = sum(resp(stim_ind)==respOptions(iR) & conf(stim_ind)==confOptions(iR));
    end
end

% Set bounds for fitting:
LB(1,1)     = 0;                        UB(1,1)        = 10;              % Stimulus sensitivity
LB(2,1)     = -3;                       UB(2,1)        = 3;               % Stimulus criterion
LB(3,1)     = 0.01;                     UB(3,1)        = 5;                % Meta uncertainty
LB(4:4+nConfCrit-1,1) = 0;              UB(4:4+nConfCrit-1,1)        = 5; % Confidence criteria

% Fit simulated data:
options     = optimset('Display', 'off', 'Maxiter', 10^5, 'MaxFuneval', 10^5);
obFun       = @(paramVec) giveNLL(paramVec, stimValue, nChoice);
startVec    = [ 1 0 0.2 sort(2*rand(1,nConfCrit))]; %[  Stim senstivity, stim criterion, meta-uncertainty, confidence criterion]; note - froze guess rate because guess rate estimation will be poor for only two stimulus strengths & code will runn faster
paramEst    = fmincon(obFun, startVec, [], [], [], [], LB, UB, [], options);

metaUncertainty = paramEst(3); % return only metaUncertainty
end

function [NLL] = giveNLL(paramVec, stimValue, nChoice)
choiceLlh = getLlhChoice(stimValue, paramVec);
NLL       = -sum(sum(nChoice.*log(choiceLlh)));
end

function [choiceLlh] = getLlhChoice(stimValue, modelParams)

% Decode function arguments
stimVal     = stimValue;                   % The different stimulus conditions in units of stimulus magnitude (e.g., orientation in degrees)
noiseSens   = 1;                           % If the sensory noise is set to 1, then distributions of decision variable and confidence variable can be compared directly
guessRate   = .00001;                      % The fraction of guesses
stimSens    = modelParams(1);              % Stimulus sensitvity parameter, higher values produce a steeper psychometric function, strictly positive
stimCrit    = modelParams(2);              % The sensory decision criterion in units of stimulus magnitude (e.g., orientation in degrees)
noiseMeta   = modelParams(3);              % Meta-uncertainty: the second stage noise parameter, only affects confidence judgments, strictly positive
confCrit    = cumsum(modelParams(4:end));  % The confidence criteria, unitless

% Set calculation precision
sampleRate = 100;             % Higher values produce slower, more precise estimates. Precision saturates after ~25

%% Compute model prediction
% Step 0 - rescale sensory representation by sensitivity parameter
sensMean = stimVal*stimSens;
sensCrit = stimCrit*stimSens;

for iC = 1:numel(stimVal)
    
        
    %% Compute llh of each response alternative
    % Step 1 - sample decision variable denominator in steps of constant cumulative density
    muLogN    = log((noiseSens.^2)./sqrt(noiseMeta.^2 + noiseSens.^2));
    sigmaLogN = sqrt(log((noiseMeta.^2)./(noiseSens.^2) + 1));
    dv_Den_x  = logninv(linspace(.5/sampleRate, 1-(.5/sampleRate), sampleRate), muLogN, sigmaLogN);
    
    % Step 2 - compute choice distribution under each scaled sensory distribution
    % Crucial property: linear transformation of normal variable is itself normal variable
    % Trick: we take inverse of denominator to work with products instead of ratios
    mu    = (1./dv_Den_x').*(sensMean(iC) - sensCrit);                                                     
    sigma = (1./dv_Den_x').*noiseSens;
    x     = sort([-confCrit, 0, confCrit]);
    P     = normcdf(repmat(x, [sampleRate 1]), repmat(mu, [1 numel(x)]), repmat(sigma, [1 numel(x)]));
    
    % Step 3 - average across all scaled sensory distributions to get likelihood functions
    ratio_dist_p  = mean(P);   
    
    for iX = 1:numel(x)+1
        if iX == 1
            llhC{iX}(iC) = (guessRate/(numel(x)+1)) + (1 - guessRate)*ratio_dist_p(1);
        elseif (iX > 1 && iX <= numel(x))
            llhC{iX}(iC) = (guessRate/(numel(x)+1)) + (1 - guessRate)*(ratio_dist_p(iX) - ratio_dist_p(iX-1));
        elseif iX == (numel(x)+1)
            llhC{iX}(iC) = (guessRate/(numel(x)+1)) + (1 - guessRate)*(1 - ratio_dist_p(numel(x)));
        end
    end
end
choiceLlh = cell2mat(llhC');
end

