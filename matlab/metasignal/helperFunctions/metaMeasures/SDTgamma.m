function [gamma, gamma_ratio, gamma_diff] = SDTgamma(stim, resp, conf, nRatings)

% Compute data
SDTexpectData = SDTexpectConf(stim, resp, conf, nRatings);

% Actual
conf_counts = nR_to_correctIncorrect(SDTexpectData.nR_S1_actual, SDTexpectData.nR_S2_actual, nRatings);
gamma = compute_gamma(conf_counts, nRatings);

% Expected
conf_counts = nR_to_correctIncorrect(SDTexpectData.nR_S1_SDTexpect, SDTexpectData.nR_S2_SDTexpect, nRatings);
gamma_expected = compute_gamma(conf_counts, nRatings);

% Compute final values
gamma_ratio = gamma/gamma_expected;
gamma_diff = gamma-gamma_expected;

end


%% Nested functions
% Transform nR_S1/2 into correct/incorrect confidence counts
function conf_counts = nR_to_correctIncorrect(nR_S1, nR_S2, nRatings)

counts_incorrect = nR_S1(nRatings+1:end) + nR_S2(nRatings:-1:1);
counts_correct = nR_S2(nRatings+1:end) + nR_S1(nRatings:-1:1);
conf_counts = [counts_incorrect; counts_correct];

end

% Compute gamma correlation
function gamma = compute_gamma(conf_counts, nRatings)

[R,C] = ndgrid(1:2,1:nRatings);
concordances = zeros(2,nRatings);
discordances = zeros(2,nRatings);
for i=1:2
    for j=1:nRatings
        concordances(i,j) =  sum(conf_counts(R < i & C < j)) + sum(conf_counts(R > i & C > j));
        discordances(i,j) =  sum(conf_counts(R < i & C > j)) + sum(conf_counts(R > i & C < j));
    end
end
C = sum(sum(conf_counts.*concordances))/2; %total of concordances
D = sum(sum(conf_counts.*discordances))/2; %total of discordances
gamma = (C - D)/(C + D); %Goodman-Kruskal's gamma statistic

end