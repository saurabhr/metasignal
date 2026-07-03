/**
 * hmeta_d.stan  —  Hierarchical Bayesian meta-d' (HMeta-d)
 *
 * A Stan port of the Fleming (2017) JAGS model:
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
 * Numerical stability
 * -------------------
 * The Type-2 multinomial likelihood is computed entirely in log space using
 * normal_lcdf/normal_lccdf + log_diff_exp, then passed to
 * multinomial_logit_lpmf (which normalises internally via a numerically
 * stable softmax).  This avoids ever forming Phi() differences on the
 * natural scale, which underflow to exactly 0 once meta_d grows large
 * enough to push both endpoints near 0 or 1 — the underflow previously
 * produced a flat-gradient region (via a Tol floor) that trapped the
 * sampler and let d1[s]/c1[s] diverge to +-inf during warmup.
 *
 * Type-2 criteria are constructed as cumulative sums of positive offsets
 * (metadpy/hmetad-style), which guarantees cS1[s] < c1[s] < cS2[s] by
 * construction — no soft prior or truncation needed, so criteria can never
 * cross c1[s] and produce an invalid (negative) log-probability bin.
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
    real<lower=0> Tol;                              // unused; kept for API compatibility

    // Covariate design matrix (mean-centred).  Pass p_cov=0 and an empty
    // matrix when there are no covariates.
    int<lower=0> p_cov;
    matrix[nsubj, p_cov] X_cov;
}

transformed data {
    // Per-subject Type-1 totals (counts layout: [CR | FA | M | H], each block
    // nratings long).  Feeds the Type-1 binomial likelihood below, which ties
    // d1/c1 directly to the observed hit/FA totals — without it, d1/c1 are
    // only weakly constrained by the per-response-type Type-2 multinomial.
    array[nsubj] int CR_total;
    array[nsubj] int FA_total;
    array[nsubj] int M_total;
    array[nsubj] int H_total;
    array[nsubj] int N_total;  // total S1-stimulus trials
    array[nsubj] int S_total;  // total S2-stimulus trials
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

    // ── Log M-ratio regression parameters ───────────────────────────────────
    real alpha_logMratio;           // intercept (group mean log M-ratio)
    vector[p_cov] beta_logMratio;  // covariate slopes (empty when p_cov = 0)
    real<lower=0> sigma_logMratio;
    vector[nsubj] logMratio_z;

    // ── Type-2 criterion hyperparameters ────────────────────────────────────
    real<lower=0> mu_c2;
    real<lower=0> sigma_c2;

    // ── Per-subject Type-2 criteria offsets (guarantee cS1 < c1 < cS2) ──────
    // positive_ordered removes the permutation symmetry that a plain
    // vector<lower=0> + runtime sort_asc() would leave in the raw
    // components (label-switching: any permutation gives the same sorted
    // result and hence identical likelihood, so the unordered components
    // are non-identified and r_hat blows up for them specifically).
    array[nsubj] positive_ordered[nratings - 1] cS1_offsets;
    array[nsubj] positive_ordered[nratings - 1] cS2_offsets;
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
        // Type-1 binomial likelihood: ties d1[s], c1[s] to observed hit/FA
        // totals (mirrors metadpy's subject_level_pymc.py Binomial nodes).
        target += binomial_lpmf(H_total[s]  | S_total[s], Phi(d1[s] / 2.0 - c1[s]));
        target += binomial_lpmf(FA_total[s] | N_total[s], Phi(-d1[s] / 2.0 - c1[s]));

        // Type-2 criteria offset priors (positive_ordered, so cS1/cS2 never
        // cross c1 and the components are identified — no runtime sort needed)
        cS1_offsets[s] ~ normal(mu_c2, sigma_c2);
        cS2_offsets[s] ~ normal(mu_c2, sigma_c2);

        // Place around c1[s]:
        //   cS1[k] = c1 - cS1_offsets[nratings-k]  → ascending, all < c1
        //   cS2[k] = c1 + cS2_offsets[k]            → ascending, all > c1
        vector[nratings - 1] cS1;
        vector[nratings - 1] cS2;
        for (k in 1:(nratings - 1)) {
            cS1[k] = c1[s] - cS1_offsets[s][nratings - k];
            cS2[k] = c1[s] + cS2_offsets[s][k];
        }

        // SDT signal means (equal-variance assumption)
        real S1mu = -meta_d[s] / 2.0;
        real S2mu =  meta_d[s] / 2.0;

        // ── Log-space bin probabilities (unnormalised; multinomial_logit_lpmf
        //    normalises internally via a numerically stable softmax) ────────
        vector[nratings] log_prCR;
        vector[nratings] log_prFA;
        vector[nratings] log_prM;
        vector[nratings] log_prH;

        // Correct rejections (S1 trial → S1 resp, cS1 ascending, high conf first)
        log_prCR[1] = normal_lcdf(cS1[1] | S1mu, 1);
        for (k in 1:(nratings - 2))
            log_prCR[k + 1] = log_diff_exp(
                normal_lcdf(cS1[k + 1] | S1mu, 1), normal_lcdf(cS1[k] | S1mu, 1));
        log_prCR[nratings] = log_diff_exp(
            normal_lcdf(c1[s] | S1mu, 1), normal_lcdf(cS1[nratings - 1] | S1mu, 1));

        // False alarms (S1 trial → S2 resp, cS2 ascending, low conf first)
        log_prFA[1] = log_diff_exp(
            normal_lccdf(c1[s] | S1mu, 1), normal_lccdf(cS2[1] | S1mu, 1));
        for (k in 1:(nratings - 2))
            log_prFA[k + 1] = log_diff_exp(
                normal_lccdf(cS2[k] | S1mu, 1), normal_lccdf(cS2[k + 1] | S1mu, 1));
        log_prFA[nratings] = normal_lccdf(cS2[nratings - 1] | S1mu, 1);

        // Misses (S2 trial → S1 resp, cS1 ascending, high conf first)
        log_prM[1] = normal_lcdf(cS1[1] | S2mu, 1);
        for (k in 1:(nratings - 2))
            log_prM[k + 1] = log_diff_exp(
                normal_lcdf(cS1[k + 1] | S2mu, 1), normal_lcdf(cS1[k] | S2mu, 1));
        log_prM[nratings] = log_diff_exp(
            normal_lcdf(c1[s] | S2mu, 1), normal_lcdf(cS1[nratings - 1] | S2mu, 1));

        // Hits (S2 trial → S2 resp, cS2 ascending, low conf first)
        log_prH[1] = log_diff_exp(
            normal_lccdf(c1[s] | S2mu, 1), normal_lccdf(cS2[1] | S2mu, 1));
        for (k in 1:(nratings - 2))
            log_prH[k + 1] = log_diff_exp(
                normal_lccdf(cS2[k] | S2mu, 1), normal_lccdf(cS2[k + 1] | S2mu, 1));
        log_prH[nratings] = normal_lccdf(cS2[nratings - 1] | S2mu, 1);

        target += multinomial_logit_lpmf(hmetad_counts[s, 1:nratings]                       | log_prCR);
        target += multinomial_logit_lpmf(hmetad_counts[s, (nratings + 1):(2 * nratings)]     | log_prFA);
        target += multinomial_logit_lpmf(hmetad_counts[s, (2 * nratings + 1):(3 * nratings)] | log_prM);
        target += multinomial_logit_lpmf(hmetad_counts[s, (3 * nratings + 1):(4 * nratings)] | log_prH);
    }
}

generated quantities {
    // Group mean M-ratio on the natural scale (at covariate mean when p_cov > 0)
    real group_Mratio = exp(alpha_logMratio);
    // Alias so code that expects mu_logMratio still works
    real mu_logMratio = alpha_logMratio;
}
