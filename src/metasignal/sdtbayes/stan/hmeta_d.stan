/**
 * hmeta_d.stan  —  Hierarchical Bayesian meta-d' (HMeta-d)
 *
 * A faithful Stan port of the Fleming (2017) JAGS model:
 *
 *   Fleming, S. M. (2017). HMeta-d: hierarchical Bayesian estimation of
 *   metacognitive efficiency from confidence ratings.
 *   Neuroscience of Consciousness, 2017(1), nix007.
 *   https://doi.org/10.1093/nc/nix007
 *
 * Extended here with an optional covariate design matrix X_cov so that the
 * same file handles both the plain group model (p_cov = 0) and a regression
 * model (p_cov > 0).  Covariates should be mean-centred before passing so
 * that alpha_logMratio is interpretable as the group mean at the covariate
 * mean.
 *
 * Count matrix format (hmetad_counts)
 * ------------------------------------
 * For each participant s, counts are arranged as four consecutive blocks of
 * nratings integers in the order:
 *
 *   [CR_1 .. CR_nR | FA_1 .. FA_nR | M_1 .. M_nR | H_1 .. H_nR]
 *
 * where within CR and M blocks ratings run from highest confidence (index 1)
 * to lowest, and within FA and H blocks from lowest to highest, matching the
 * nR_S1 / nR_S2 layout produced by metasignal.stdpy.trials_to_counts.
 *
 * Key posterior quantities
 * ------------------------
 *   alpha_logMratio   : group mean log M-ratio  (at covariate mean when p_cov>0)
 *   beta_logMratio    : covariate slopes on log M-ratio  (length p_cov)
 *   sigma_logMratio   : between-subject SD on log M-ratio scale
 *   Mratio[s]         : per-subject M-ratio  (exp scale, transformed parameter)
 *   meta_d[s]         : per-subject meta-d'  (= Mratio[s] * d1[s])
 *   d1[s], c1[s]      : per-subject Type-1 d' and criterion
 *   mu_d1, mu_c1      : group mean Type-1 parameters
 *   group_Mratio      : exp(alpha_logMratio) — group mean M-ratio (generated qty)
 */

data {
    int<lower=1> nsubj;                             // number of participants
    int<lower=2> nratings;                          // number of confidence levels
    array[nsubj, nratings * 4] int hmetad_counts;  // count matrix
    real<lower=0> Tol;                              // floor for multinomial probs (e.g. 1e-7)

    // Covariate design matrix (mean-centred).  Pass p_cov=0 and an empty
    // matrix when there are no covariates.
    int<lower=0> p_cov;
    matrix[nsubj, p_cov] X_cov;
}

parameters {
    // ── Type-1 hierarchical parameters ──────────────────────────────────────
    real mu_d1;
    real<lower=0> sigma_d1;
    vector[nsubj] d1_z;

    real mu_c1;
    real<lower=0> sigma_c1;
    vector[nsubj] c1_z;

    // ── Log M-ratio regression parameters ───────────────────────────────────
    real alpha_logMratio;           // intercept (group mean log M-ratio)
    vector[p_cov] beta_logMratio;  // covariate slopes (empty when p_cov = 0)
    real<lower=0> sigma_logMratio;
    vector[nsubj] logMratio_z;

    // ── Type-2 criterion hyperparameters ────────────────────────────────────
    real<lower=0> mu_c2;
    real<lower=0> sigma_c2;

    // ── Per-subject Type-2 criteria ──────────────────────────────────────────
    // cS1_raw[s]: nratings-1 ordered criteria BELOW c1[s]  (ascending)
    // cS2_raw[s]: nratings-1 ordered criteria ABOVE c1[s]  (ascending)
    array[nsubj] ordered[nratings - 1] cS1_raw;
    array[nsubj] ordered[nratings - 1] cS2_raw;
}

transformed parameters {
    vector[nsubj] d1;
    vector[nsubj] c1;
    vector[nsubj] Mratio;
    vector[nsubj] meta_d;

    d1 = mu_d1 + sigma_d1 * d1_z;
    c1 = mu_c1 + sigma_c1 * c1_z;

    for (s in 1:nsubj) {
        // Linear predictor on log M-ratio:
        //   alpha + X[s] * beta  (dot_product is 0 when p_cov = 0)
        real eta_s = alpha_logMratio + dot_product(beta_logMratio, X_cov[s]');
        Mratio[s] = exp(eta_s + sigma_logMratio * logMratio_z[s]);
        meta_d[s] = Mratio[s] * d1[s];
    }
}

model {
    // ── Priors ───────────────────────────────────────────────────────────────
    mu_d1           ~ normal(1, 2);
    sigma_d1        ~ exponential(1);
    d1_z            ~ normal(0, 1);

    mu_c1           ~ normal(0, 1);
    sigma_c1        ~ exponential(1);
    c1_z            ~ normal(0, 1);

    alpha_logMratio ~ normal(0, 1);
    beta_logMratio  ~ normal(0, 1);   // no-op when p_cov = 0
    sigma_logMratio ~ exponential(1);
    logMratio_z     ~ normal(0, 1);

    mu_c2           ~ normal(1, 1);
    sigma_c2        ~ exponential(1);

    // ── Per-subject likelihood ───────────────────────────────────────────────
    for (s in 1:nsubj) {
        // Type-2 criteria priors (soft truncation around c1[s])
        cS1_raw[s] ~ normal(c1[s] - mu_c2, sigma_c2);
        cS2_raw[s] ~ normal(c1[s] + mu_c2, sigma_c2);

        // SDT signal means (equal-variance assumption)
        real S1mu = -meta_d[s] / 2.0;
        real S2mu =  meta_d[s] / 2.0;

        // Normalising areas: probability mass on the correct side of c1
        real C_area_rS1 = fmax(Phi(c1[s] - S1mu),        Tol);  // CR area
        real I_area_rS1 = fmax(Phi(c1[s] - S2mu),        Tol);  // Miss area
        real C_area_rS2 = fmax(1.0 - Phi(c1[s] - S2mu),  Tol);  // Hit area
        real I_area_rS2 = fmax(1.0 - Phi(c1[s] - S1mu),  Tol);  // FA area

        vector[nratings] prCR;
        vector[nratings] prFA;
        vector[nratings] prM;
        vector[nratings] prH;

        // Correct rejections  (S1 trial → S1 resp, sorted by cS1, high conf first)
        prCR[1] = fmax(Phi(cS1_raw[s, 1] - S1mu) / C_area_rS1, Tol);
        for (k in 1:(nratings - 2)) {
            prCR[k + 1] = fmax(
                (Phi(cS1_raw[s, k + 1] - S1mu) - Phi(cS1_raw[s, k] - S1mu))
                / C_area_rS1, Tol);
        }
        prCR[nratings] = fmax(
            (Phi(c1[s] - S1mu) - Phi(cS1_raw[s, nratings - 1] - S1mu))
            / C_area_rS1, Tol);

        // False alarms  (S1 trial → S2 resp, sorted by cS2, low conf first)
        prFA[1] = fmax(
            ((1.0 - Phi(c1[s] - S1mu)) - (1.0 - Phi(cS2_raw[s, 1] - S1mu)))
            / I_area_rS2, Tol);
        for (k in 1:(nratings - 2)) {
            prFA[k + 1] = fmax(
                ((1.0 - Phi(cS2_raw[s, k] - S1mu)) - (1.0 - Phi(cS2_raw[s, k + 1] - S1mu)))
                / I_area_rS2, Tol);
        }
        prFA[nratings] = fmax(
            (1.0 - Phi(cS2_raw[s, nratings - 1] - S1mu)) / I_area_rS2, Tol);

        // Misses  (S2 trial → S1 resp, sorted by cS1, high conf first)
        prM[1] = fmax(Phi(cS1_raw[s, 1] - S2mu) / I_area_rS1, Tol);
        for (k in 1:(nratings - 2)) {
            prM[k + 1] = fmax(
                (Phi(cS1_raw[s, k + 1] - S2mu) - Phi(cS1_raw[s, k] - S2mu))
                / I_area_rS1, Tol);
        }
        prM[nratings] = fmax(
            (Phi(c1[s] - S2mu) - Phi(cS1_raw[s, nratings - 1] - S2mu))
            / I_area_rS1, Tol);

        // Hits  (S2 trial → S2 resp, sorted by cS2, low conf first)
        prH[1] = fmax(
            ((1.0 - Phi(c1[s] - S2mu)) - (1.0 - Phi(cS2_raw[s, 1] - S2mu)))
            / C_area_rS2, Tol);
        for (k in 1:(nratings - 2)) {
            prH[k + 1] = fmax(
                ((1.0 - Phi(cS2_raw[s, k] - S2mu)) - (1.0 - Phi(cS2_raw[s, k + 1] - S2mu)))
                / C_area_rS2, Tol);
        }
        prH[nratings] = fmax(
            (1.0 - Phi(cS2_raw[s, nratings - 1] - S2mu)) / C_area_rS2, Tol);

        // Multinomial likelihoods (re-normalised to exact simplex to absorb
        // rounding errors from the Tol floor)
        target += multinomial_lpmf(
            hmetad_counts[s, 1:nratings]                   | prCR / sum(prCR));
        target += multinomial_lpmf(
            hmetad_counts[s, (nratings + 1):(2 * nratings)] | prFA / sum(prFA));
        target += multinomial_lpmf(
            hmetad_counts[s, (2 * nratings + 1):(3 * nratings)] | prM / sum(prM));
        target += multinomial_lpmf(
            hmetad_counts[s, (3 * nratings + 1):(4 * nratings)] | prH / sum(prH));
    }
}

generated quantities {
    // Group mean M-ratio on the natural scale (at covariate mean when p_cov > 0)
    real group_Mratio = exp(alpha_logMratio);
    // Alias so code that expects mu_logMratio still works
    real mu_logMratio = alpha_logMratio;
}
