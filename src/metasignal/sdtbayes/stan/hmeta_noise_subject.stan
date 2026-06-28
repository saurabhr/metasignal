/**
 * hmeta_noise_subject.stan  —  Single-subject Bayesian meta-noise estimation
 *
 * Non-hierarchical version of hmeta_noise.stan for single-participant fitting.
 *
 * Model
 * -----
 *   sigma_meta ~ log-Normal(0, 1)   [prior centred at σ_meta ≈ 1]
 *   meta_d     = d1 / sqrt(1 + σ_meta²)
 *   Mratio     = 1  / sqrt(1 + σ_meta²)   (generated quantity)
 *
 * Priors
 * ------
 *   d1          ~ Normal(0, sqrt(2))
 *   c1          ~ Normal(0, 1/sqrt(2))
 *   log_sigma_meta ~ Normal(0, 1)
 *
 * Data format: 1-D array of length nratings * 4:
 *   [CR_nR .. CR_1 | FA_1 .. FA_nR | M_nR .. M_1 | H_1 .. H_nR]
 */

data {
    int<lower=2> nratings;
    array[nratings * 4] int counts;
    real<lower=0> Tol;
}

parameters {
    real d1;
    real c1;
    real log_sigma_meta;

    vector<lower=0>[nratings - 1] cS1_offsets;
    vector<lower=0>[nratings - 1] cS2_offsets;
}

transformed parameters {
    real sigma_meta = exp(log_sigma_meta);
    real meta_d     = d1 / sqrt(1.0 + square(sigma_meta));

    vector[nratings - 1] cS1_sorted = sort_asc(cS1_offsets);
    vector[nratings - 1] cS2_sorted = sort_asc(cS2_offsets);
    vector[nratings - 1] cS1;
    vector[nratings - 1] cS2;
    for (k in 1:(nratings - 1)) {
        cS1[k] = c1 - cS1_sorted[nratings - k];
        cS2[k] = c1 + cS2_sorted[k];
    }
}

model {
    d1             ~ normal(0, sqrt(2.0));
    c1             ~ normal(0, inv_sqrt(2.0));
    log_sigma_meta ~ normal(0, 1);

    cS1_offsets ~ normal(0, inv_sqrt(2.0));
    cS2_offsets ~ normal(0, inv_sqrt(2.0));

    real S1mu = -meta_d / 2.0;
    real S2mu =  meta_d / 2.0;

    real C_area_rS1 = fmax(Phi(c1 - S1mu),       Tol);
    real I_area_rS1 = fmax(Phi(c1 - S2mu),       Tol);
    real C_area_rS2 = fmax(1.0 - Phi(c1 - S2mu), Tol);
    real I_area_rS2 = fmax(1.0 - Phi(c1 - S1mu), Tol);

    vector[nratings] prCR;
    vector[nratings] prFA;
    vector[nratings] prM;
    vector[nratings] prH;

    prCR[1] = fmax(Phi(cS1[1] - S1mu) / C_area_rS1, Tol);
    for (k in 1:(nratings - 2))
        prCR[k + 1] = fmax(
            (Phi(cS1[k + 1] - S1mu) - Phi(cS1[k] - S1mu)) / C_area_rS1, Tol);
    prCR[nratings] = fmax(
        (Phi(c1 - S1mu) - Phi(cS1[nratings - 1] - S1mu)) / C_area_rS1, Tol);

    prFA[1] = fmax(
        ((1.0 - Phi(c1 - S1mu)) - (1.0 - Phi(cS2[1] - S1mu))) / I_area_rS2, Tol);
    for (k in 1:(nratings - 2))
        prFA[k + 1] = fmax(
            ((1.0 - Phi(cS2[k] - S1mu)) - (1.0 - Phi(cS2[k + 1] - S1mu)))
            / I_area_rS2, Tol);
    prFA[nratings] = fmax(
        (1.0 - Phi(cS2[nratings - 1] - S1mu)) / I_area_rS2, Tol);

    prM[1] = fmax(Phi(cS1[1] - S2mu) / I_area_rS1, Tol);
    for (k in 1:(nratings - 2))
        prM[k + 1] = fmax(
            (Phi(cS1[k + 1] - S2mu) - Phi(cS1[k] - S2mu)) / I_area_rS1, Tol);
    prM[nratings] = fmax(
        (Phi(c1 - S2mu) - Phi(cS1[nratings - 1] - S2mu)) / I_area_rS1, Tol);

    prH[1] = fmax(
        ((1.0 - Phi(c1 - S2mu)) - (1.0 - Phi(cS2[1] - S2mu))) / C_area_rS2, Tol);
    for (k in 1:(nratings - 2))
        prH[k + 1] = fmax(
            ((1.0 - Phi(cS2[k] - S2mu)) - (1.0 - Phi(cS2[k + 1] - S2mu)))
            / C_area_rS2, Tol);
    prH[nratings] = fmax(
        (1.0 - Phi(cS2[nratings - 1] - S2mu)) / C_area_rS2, Tol);

    target += multinomial_lpmf(counts[1:nratings]                       | prCR / sum(prCR));
    target += multinomial_lpmf(counts[(nratings + 1):(2 * nratings)]     | prFA / sum(prFA));
    target += multinomial_lpmf(counts[(2 * nratings + 1):(3 * nratings)] | prM  / sum(prM));
    target += multinomial_lpmf(counts[(3 * nratings + 1):(4 * nratings)] | prH  / sum(prH));
}

generated quantities {
    real Mratio = 1.0 / sqrt(1.0 + square(sigma_meta));
}
