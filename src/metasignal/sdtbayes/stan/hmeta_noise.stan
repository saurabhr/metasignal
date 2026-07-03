/**
 * hmeta_noise.stan  —  Hierarchical Bayesian meta-noise estimation
 *
 * Ports the metacognitive noise parameterisation described in:
 *
 *   Guggenmos, M. (2022). Reverse engineering of metacognition.
 *   eLife, 11, e75420. https://doi.org/10.7554/eLife.75420
 *
 * and the earlier theoretical treatment in:
 *
 *   Maniscalco, B., & Lau, H. (2014). Signal detection theory analysis
 *   of type 1 and type 2 psychophysical discriminability.
 *   In S. M. Fleming & C. D. Frith (Eds.), The Cognitive Neuroscience
 *   of Metacognition (pp. 117–139). Springer.
 *
 * into a hierarchical Bayesian Stan model analogous to hmeta_d.stan.
 *
 * Model
 * -----
 * The metacognitive noise model assumes that the participant's internal
 * confidence signal is a corrupted copy of the Type-1 decision variable:
 *
 *   X_meta[t] = X_type1[t] + ε[t],   ε ~ Normal(0, σ_meta²)
 *
 * Under equal-variance SDT this gives an effective metacognitive d':
 *
 *   meta_d[s] = d1[s] / sqrt(1 + σ_meta[s]²)
 *
 * and an implied M-ratio:
 *
 *   Mratio[s] = 1 / sqrt(1 + σ_meta[s]²)
 *
 * When σ_meta → 0  the confidence signal is noise-free → Mratio → 1.
 * When σ_meta → ∞  confidence is pure noise         → Mratio → 0.
 *
 * The multinomial SDT likelihood is identical to hmeta_d.stan; only the
 * transformed parameters block differs (σ_meta drives meta_d instead of
 * M-ratio being modelled directly).
 *
 * Parameterisation
 * ----------------
 * σ_meta is modelled on the log scale so that the group hyperprior is
 * symmetric and unconstrained:
 *
 *   log_sigma_meta[s] = mu_logSigmaMeta
 *                       + X_cov[s] · beta_logSigmaMeta
 *                       + sigma_logSigmaMeta * logSigmaMeta_z[s]
 *   sigma_meta[s]     = exp(log_sigma_meta[s])
 *
 * Covariates (optional, mean-centred) are passed via X_cov exactly as in
 * hmeta_d.stan.  Set p_cov = 0 and pass an empty matrix for a plain
 * group-level model.
 *
 * Key posterior quantities
 * ------------------------
 *   mu_logSigmaMeta    : group mean log meta-noise
 *   beta_logSigmaMeta  : covariate slopes on log meta-noise (length p_cov)
 *   sigma_logSigmaMeta : between-subject SD on log meta-noise scale
 *   sigma_meta[s]      : per-subject metacognitive noise (transformed parameter)
 *   meta_d[s]          : per-subject meta-d'  (= d1[s] / sqrt(1 + σ_meta[s]²))
 *   d1[s], c1[s]       : per-subject Type-1 d' and criterion
 *   group_sigma_meta   : exp(mu_logSigmaMeta) — group mean meta-noise (gen qty)
 *   Mratio[s]          : implied M-ratio per subject (generated quantity)
 */

data {
    int<lower=1> nsubj;
    int<lower=2> nratings;
    array[nsubj, nratings * 4] int hmetad_counts;
    real<lower=0> Tol;

    int<lower=0> p_cov;
    matrix[nsubj, p_cov] X_cov;   // mean-centred covariate design matrix
}

transformed data {
    // Per-subject Type-1 totals; feeds the Type-1 binomial likelihood below.
    array[nsubj] int CR_total;
    array[nsubj] int FA_total;
    array[nsubj] int M_total;
    array[nsubj] int H_total;
    array[nsubj] int N_total;
    array[nsubj] int S_total;
    for (s in 1:nsubj) {
        CR_total[s] = sum(hmetad_counts[s, 1:nratings]);
        FA_total[s] = sum(hmetad_counts[s, (nratings + 1):(2 * nratings)]);
        M_total[s]  = sum(hmetad_counts[s, (2 * nratings + 1):(3 * nratings)]);
        H_total[s]  = sum(hmetad_counts[s, (3 * nratings + 1):(4 * nratings)]);
        N_total[s]  = CR_total[s] + FA_total[s];
        S_total[s]  = M_total[s] + H_total[s];
    }
}

parameters {
    // ── Type-1 hierarchical parameters ──────────────────────────────────────
    real mu_d1;
    real<lower=0> sigma_d1;
    vector[nsubj] d1_z;

    real mu_c1;
    real<lower=0> sigma_c1;
    vector[nsubj] c1_z;

    // ── Log meta-noise regression parameters ────────────────────────────────
    real mu_logSigmaMeta;           // group mean log σ_meta
    vector[p_cov] beta_logSigmaMeta; // covariate slopes (empty when p_cov = 0)
    real<lower=0> sigma_logSigmaMeta;
    vector[nsubj] logSigmaMeta_z;

    // ── Type-2 criterion hyperparameters ────────────────────────────────────
    real<lower=0> mu_c2;
    real<lower=0> sigma_c2;

    // ── Per-subject Type-2 criteria ──────────────────────────────────────────
    array[nsubj] ordered[nratings - 1] cS1_raw;
    array[nsubj] ordered[nratings - 1] cS2_raw;
}

transformed parameters {
    vector[nsubj] d1;
    vector[nsubj] c1;
    vector[nsubj] sigma_meta;
    vector[nsubj] meta_d;

    d1 = mu_d1 + sigma_d1 * d1_z;
    c1 = mu_c1 + sigma_c1 * c1_z;

    for (s in 1:nsubj) {
        real eta_s = mu_logSigmaMeta
                     + dot_product(beta_logSigmaMeta, X_cov[s]')
                     + sigma_logSigmaMeta * logSigmaMeta_z[s];
        sigma_meta[s] = exp(eta_s);
        // meta_d' = d' / sqrt(1 + σ_meta²)  [Guggenmos 2022, eq. 4]
        meta_d[s] = d1[s] / sqrt(1.0 + square(sigma_meta[s]));
    }
}

model {
    // ── Priors ───────────────────────────────────────────────────────────────
    mu_d1              ~ normal(1, 2);
    sigma_d1           ~ exponential(1);
    d1_z               ~ normal(0, 1);

    mu_c1              ~ normal(0, 1);
    sigma_c1           ~ exponential(1);
    c1_z               ~ normal(0, 1);

    // Prior on log σ_meta: Normal(0, 1) → σ_meta has log-normal prior centred
    // near 1 (moderate metacognitive noise).  Wide enough to cover 0.1–10.
    mu_logSigmaMeta    ~ normal(0, 1);
    beta_logSigmaMeta  ~ normal(0, 1);
    sigma_logSigmaMeta ~ exponential(1);
    logSigmaMeta_z     ~ normal(0, 1);

    mu_c2              ~ normal(1, 1);
    sigma_c2           ~ exponential(1);

    // ── Per-subject likelihood ───────────────────────────────────────────────
    // Identical to hmeta_d.stan: meta_d enters through S1mu/S2mu
    for (s in 1:nsubj) {
        target += binomial_lpmf(H_total[s]  | S_total[s], Phi(d1[s] / 2.0 - c1[s]));
        target += binomial_lpmf(FA_total[s] | N_total[s], Phi(-d1[s] / 2.0 - c1[s]));

        cS1_raw[s] ~ normal(c1[s] - mu_c2, sigma_c2);
        cS2_raw[s] ~ normal(c1[s] + mu_c2, sigma_c2);

        real S1mu = -meta_d[s] / 2.0;
        real S2mu =  meta_d[s] / 2.0;

        real C_area_rS1 = fmax(Phi(c1[s] - S1mu),       Tol);
        real I_area_rS1 = fmax(Phi(c1[s] - S2mu),       Tol);
        real C_area_rS2 = fmax(1.0 - Phi(c1[s] - S2mu), Tol);
        real I_area_rS2 = fmax(1.0 - Phi(c1[s] - S1mu), Tol);

        vector[nratings] prCR;
        vector[nratings] prFA;
        vector[nratings] prM;
        vector[nratings] prH;

        prCR[1] = fmax(Phi(cS1_raw[s, 1] - S1mu) / C_area_rS1, Tol);
        for (k in 1:(nratings - 2)) {
            prCR[k + 1] = fmax(
                (Phi(cS1_raw[s, k + 1] - S1mu) - Phi(cS1_raw[s, k] - S1mu))
                / C_area_rS1, Tol);
        }
        prCR[nratings] = fmax(
            (Phi(c1[s] - S1mu) - Phi(cS1_raw[s, nratings - 1] - S1mu))
            / C_area_rS1, Tol);

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

        prM[1] = fmax(Phi(cS1_raw[s, 1] - S2mu) / I_area_rS1, Tol);
        for (k in 1:(nratings - 2)) {
            prM[k + 1] = fmax(
                (Phi(cS1_raw[s, k + 1] - S2mu) - Phi(cS1_raw[s, k] - S2mu))
                / I_area_rS1, Tol);
        }
        prM[nratings] = fmax(
            (Phi(c1[s] - S2mu) - Phi(cS1_raw[s, nratings - 1] - S2mu))
            / I_area_rS1, Tol);

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

        target += multinomial_lpmf(
            hmetad_counts[s, 1:nratings]                       | prCR / sum(prCR));
        target += multinomial_lpmf(
            hmetad_counts[s, (nratings + 1):(2 * nratings)]     | prFA / sum(prFA));
        target += multinomial_lpmf(
            hmetad_counts[s, (2 * nratings + 1):(3 * nratings)] | prM  / sum(prM));
        target += multinomial_lpmf(
            hmetad_counts[s, (3 * nratings + 1):(4 * nratings)] | prH  / sum(prH));
    }
}

generated quantities {
    // Group mean meta-noise on the natural scale (at covariate mean)
    real group_sigma_meta = exp(mu_logSigmaMeta);

    // Implied M-ratio per subject: Mratio = 1 / sqrt(1 + σ_meta²)
    // Allows direct comparison with hmeta_d.stan posteriors
    vector[nsubj] Mratio;
    // Implied group mean M-ratio
    real group_Mratio;

    for (s in 1:nsubj) {
        Mratio[s] = 1.0 / sqrt(1.0 + square(sigma_meta[s]));
    }
    group_Mratio = 1.0 / sqrt(1.0 + square(group_sigma_meta));
}
