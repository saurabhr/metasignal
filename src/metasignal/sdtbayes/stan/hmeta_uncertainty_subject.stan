/**
 * hmeta_uncertainty_subject.stan  —  Single-subject Bayesian CASANDRE
 *
 * Non-hierarchical version of hmeta_uncertainty.stan for single-participant
 * fitting.  Identical GH-quadrature likelihood; fixed priors instead of
 * hierarchical hyperpriors.
 *
 * Priors
 * ------
 *   d1        ~ Normal(0, sqrt(2))
 *   c1        ~ Normal(0, 1/sqrt(2))
 *   log_phi   ~ Normal(0, 1)        [φ ≈ 1 at centre, 95% in (0.14, 7.4)]
 *   log_theta ~ Normal(0, 1)        [shared confidence thresholds]
 *
 * Data format: 1-D array of length nratings * 4:
 *   [CR_nR .. CR_1 | FA_1 .. FA_nR | M_nR .. M_1 | H_1 .. H_nR]
 *
 * Key posterior quantities
 * ------------------------
 *   phi       : meta-uncertainty parameter
 *   log_theta : confidence thresholds on log-reliability scale
 *   d1, c1    : Type-1 sensitivity and criterion
 */

data {
    int<lower=2> nratings;
    array[nratings * 4] int counts;
    real<lower=0> Tol;
    real<lower=0> eps;
    real<lower=0> delta;

    int<lower=1> n_gh;
    vector[n_gh] gh_nodes;
    vector[n_gh] gh_weights;
}

parameters {
    real d1;
    real c1;
    real log_phi;

    ordered[nratings - 1] log_theta;
}

transformed parameters {
    real phi = exp(log_phi);
}

model {
    d1      ~ normal(0, sqrt(2.0));
    c1      ~ normal(0, inv_sqrt(2.0));
    log_phi ~ normal(0, 1);
    log_theta ~ normal(0, 1);

    real sqrt2       = sqrt(2.0);
    real inv_sqrt_pi = inv_sqrt(pi());

    real mu_S1 = -d1 / 2.0;
    real mu_S2 =  d1 / 2.0;

    vector[nratings] prCR = rep_vector(Tol, nratings);
    vector[nratings] prFA = rep_vector(Tol, nratings);
    vector[nratings] prM  = rep_vector(Tol, nratings);
    vector[nratings] prH  = rep_vector(Tol, nratings);

    for (i in 1:n_gh) {
        real w = gh_weights[i] * inv_sqrt_pi;

        real x_S1     = mu_S1 + sqrt2 * gh_nodes[i];
        real log_r_S1 = 0.5 * log(square(x_S1) + square(eps));

        real x_S2     = mu_S2 + sqrt2 * gh_nodes[i];
        real log_r_S2 = 0.5 * log(square(x_S2) + square(eps));

        real w_cr_S1 = Phi((c1 - x_S1) / delta);
        real w_fa_S1 = 1.0 - w_cr_S1;
        real w_cr_S2 = Phi((c1 - x_S2) / delta);
        real w_fa_S2 = 1.0 - w_cr_S2;

        for (k in 1:nratings) {
            real p_lo_S1 = (k == 1)        ? 0.0 : Phi((log_theta[k-1] - log_r_S1) / phi);
            real p_hi_S1 = (k == nratings) ? 1.0 : Phi((log_theta[k]   - log_r_S1) / phi);
            real p_lo_S2 = (k == 1)        ? 0.0 : Phi((log_theta[k-1] - log_r_S2) / phi);
            real p_hi_S2 = (k == nratings) ? 1.0 : Phi((log_theta[k]   - log_r_S2) / phi);

            real p_k_S1 = fmax(p_hi_S1 - p_lo_S1, Tol);
            real p_k_S2 = fmax(p_hi_S2 - p_lo_S2, Tol);

            prCR[nratings + 1 - k] += w * w_cr_S1 * p_k_S1;
            prFA[k]                += w * w_fa_S1 * p_k_S1;
            prM[nratings + 1 - k]  += w * w_cr_S2 * p_k_S2;
            prH[k]                 += w * w_fa_S2 * p_k_S2;
        }
    }

    target += multinomial_lpmf(counts[1:nratings]                       | prCR / sum(prCR));
    target += multinomial_lpmf(counts[(nratings + 1):(2 * nratings)]     | prFA / sum(prFA));
    target += multinomial_lpmf(counts[(2 * nratings + 1):(3 * nratings)] | prM  / sum(prM));
    target += multinomial_lpmf(counts[(3 * nratings + 1):(4 * nratings)] | prH  / sum(prH));
}

generated quantities {
    // Implied M-ratio (not directly modelled, but useful for comparison)
    // Use the relationship: Mratio = 1/sqrt(1 + sigma_meta^2) is not valid
    // here; we leave Mratio undefined for CASANDRE (phi is not M-ratio).
    real group_phi = phi;   // alias for consistency with group model naming
}
