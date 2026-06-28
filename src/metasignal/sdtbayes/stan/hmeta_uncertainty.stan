/**
 * hmeta_uncertainty.stan  —  Hierarchical Bayesian meta-uncertainty (CASANDRE)
 *
 * Hierarchical Bayesian port of the CASANDRE model:
 *
 *   Boundy-Singer, Z. M., Ziemba, C. M., & Goris, R. L. T. (2023).
 *   Confidence reflects a noisy decision reliability estimate.
 *   Nature Human Behaviour, 7, 142–154.
 *   https://doi.org/10.1038/s41562-022-01464-x
 *
 * Original Stan implementation: CDN-Lab/CAStanDRE (perceptual variant).
 * This file adapts CASANDRE to the binary detection + confidence rating
 * format used throughout metasignal (same count matrix as hmeta_d.stan).
 *
 * Model
 * -----
 * Confidence reflects the observer's estimate of their decision reliability.
 * Reliability is proxied by the absolute decision variable |x|.  The observer
 * cannot read |x| perfectly: their reliability estimate is corrupted by
 * log-normal noise parameterised by meta-uncertainty φ:
 *
 *   Reliability estimate:  log(R_obs) ~ Normal(log(|x|), φ²)
 *   Confidence rating k  iff  θ_{k-1} < R_obs ≤ θ_k
 *
 * So:
 *   P(conf = k | x, φ) = Φ((log θ_k   - log|x|) / φ)
 *                       - Φ((log θ_{k-1} - log|x|) / φ)
 *
 * where θ_0 = 0 (i.e. log θ_0 = −∞) and θ_K = ∞.
 *
 * The decision variable x is latent.  We marginalise over it using
 * Gauss-Hermite (GH) quadrature passed in as data (n_gh nodes):
 *
 *   P(conf = k, CR | S1) ≈ (1/√π) Σ_i w_i · I(x_i < c1)
 *                           · P(conf = k | x_i, φ)
 *
 *   x_i = μ_S1 + √2 · h_i,   μ_S1 = −d1/2
 *
 * φ is modelled hierarchically on the log scale:
 *
 *   log φ[s] = μ_logPhi + X_cov[s] · β_logPhi + σ_logPhi · z[s]
 *
 * Confidence thresholds θ (log scale) are shared across participants.
 *
 * Key posterior quantities
 * ------------------------
 *   μ_logPhi           : group mean log φ
 *   β_logPhi           : covariate slopes on log φ  (length p_cov)
 *   σ_logPhi           : between-subject SD on log φ scale
 *   phi[s]             : per-subject meta-uncertainty (transformed parameter)
 *   log_theta[1..K-1]  : shared confidence thresholds on log-reliability scale
 *   group_phi          : exp(μ_logPhi)  (generated quantity)
 *   d1[s], c1[s]       : per-subject Type-1 parameters
 *
 * Count matrix format (hmetad_counts)
 * ------------------------------------
 * Identical to hmeta_d.stan:
 *   [CR_nR … CR_1 | FA_1 … FA_nR | M_nR … M_1 | H_1 … H_nR]
 * i.e. CR and M blocks run from highest confidence (index 1) to lowest;
 * FA and H blocks run from lowest to highest.
 *
 * Gauss-Hermite quadrature data
 * ------------------------------
 * gh_nodes  : raw GH nodes h_i for ∫f(t)exp(−t²)dt  (length n_gh)
 * gh_weights: raw GH weights w_i                      (length n_gh)
 * The effective weight in normal-distribution form is w_i / √π.
 * Use numpy: nodes, weights = np.polynomial.hermite.hermgauss(n_gh)
 * Recommended n_gh ≥ 20 for nratings ≤ 6.
 */

data {
    int<lower=1> nsubj;
    int<lower=2> nratings;
    array[nsubj, nratings * 4] int hmetad_counts;
    real<lower=0> Tol;

    // Smooth absolute value regulariser: reliability = sqrt(x² + eps²) instead
    // of |x|.  Eliminates the 1/x gradient singularity at x = 0.
    // Default 0.05 (grad bounded by 1/(2*0.05)=10); reduce toward 0.01 for
    // higher fidelity at the cost of steeper gradients near x = 0.
    real<lower=0> eps;

    // Covariate design matrix (mean-centred; set p_cov=0 for plain model)
    int<lower=0> p_cov;
    matrix[nsubj, p_cov] X_cov;

    // Gauss-Hermite quadrature for marginalising over decision variable x
    int<lower=1> n_gh;
    vector[n_gh] gh_nodes;    // raw GH nodes (for exp(−t²) kernel)
    vector[n_gh] gh_weights;  // raw GH weights

    // Softness of the CR/FA boundary (controls how sharply x=c1 splits
    // contributions between CR and FA via a sigmoid approximation).
    // Default 0.1 works well; reduce toward 0.01 for higher accuracy at
    // the cost of steeper gradients near c1.
    real<lower=0> delta;
}

parameters {
    // ── Type-1 hierarchical parameters ──────────────────────────────────────
    real mu_d1;
    real<lower=0> sigma_d1;
    vector[nsubj] d1_z;

    real mu_c1;
    real<lower=0> sigma_c1;
    vector[nsubj] c1_z;

    // ── Log meta-uncertainty (φ) regression parameters ──────────────────────
    real mu_logPhi;
    vector[p_cov] beta_logPhi;
    real<lower=0> sigma_logPhi;
    vector[nsubj] logPhi_z;

    // ── Shared confidence thresholds on log-reliability scale ────────────────
    // log_theta[k] separates confidence rating k from k+1  (k=1..nratings-1)
    // log_theta[1] < log_theta[2] < ... < log_theta[nratings-1]
    ordered[nratings - 1] log_theta;
}

transformed parameters {
    vector[nsubj] d1 = mu_d1 + sigma_d1 * d1_z;
    vector[nsubj] c1 = mu_c1 + sigma_c1 * c1_z;
    vector[nsubj] phi;

    for (s in 1:nsubj) {
        real eta_s = mu_logPhi
                     + dot_product(beta_logPhi, X_cov[s]')
                     + sigma_logPhi * logPhi_z[s];
        phi[s] = exp(eta_s);
    }
}

model {
    real sqrt2       = sqrt(2.0);
    real inv_sqrt_pi = inv_sqrt(pi());

    // ── Priors ───────────────────────────────────────────────────────────────
    mu_d1      ~ normal(1, 2);
    sigma_d1   ~ exponential(1);
    d1_z       ~ normal(0, 1);

    mu_c1      ~ normal(0, 1);
    sigma_c1   ~ exponential(1);
    c1_z       ~ normal(0, 1);

    // φ prior: log-normal centred near 1 (moderate meta-uncertainty).
    // log φ ~ Normal(0, 1) → 95% of φ in [0.14, 7.4].
    mu_logPhi    ~ normal(0, 1);
    beta_logPhi  ~ normal(0, 1);
    sigma_logPhi ~ exponential(1);
    logPhi_z     ~ normal(0, 1);

    // Thresholds: centred near 0 on the log-reliability scale.
    // For d' ~ 1, |x| ~ 0.5–1.5 → log|x| ~ −0.7 to 0.4.
    log_theta ~ normal(0, 1);

    // ── Per-subject GH-quadrature likelihood ─────────────────────────────────
    for (s in 1:nsubj) {
        real d  = d1[s];
        real c  = c1[s];
        real p  = phi[s];

        // Signal means: S1 stim → μ = −d/2; S2 stim → μ = +d/2
        real mu_S1 = -d / 2.0;
        real mu_S2 =  d / 2.0;

        // Initialise probability vectors with Tol floor (avoids zero simplex)
        vector[nratings] prCR = rep_vector(Tol, nratings);
        vector[nratings] prFA = rep_vector(Tol, nratings);
        vector[nratings] prM  = rep_vector(Tol, nratings);
        vector[nratings] prH  = rep_vector(Tol, nratings);

        for (i in 1:n_gh) {
            real w = gh_weights[i] * inv_sqrt_pi;  // effective normal weight

            // ── S1 trial node ─────────────────────────────────────────────
            real x_S1     = mu_S1 + sqrt2 * gh_nodes[i];
            real log_r_S1 = 0.5 * log(square(x_S1) + square(eps));

            // ── S2 trial node ─────────────────────────────────────────────
            real x_S2     = mu_S2 + sqrt2 * gh_nodes[i];
            real log_r_S2 = 0.5 * log(square(x_S2) + square(eps));

            // Soft CR/FA split: Phi((c-x)/delta) ≈ I(x < c) as delta → 0.
            // Using Phi (smooth sigmoid) instead of a hard if-statement keeps
            // the log-probability differentiable in c1 for Stan's autodiff.
            real w_cr_S1 = Phi((c - x_S1) / delta);
            real w_fa_S1 = 1.0 - w_cr_S1;
            real w_cr_S2 = Phi((c - x_S2) / delta);
            real w_fa_S2 = 1.0 - w_cr_S2;

            for (k in 1:nratings) {
                // Confidence bin k (k=1 lowest, k=nratings highest).
                // Avoid passing ±∞/p to Phi — the autodiff gradient
                // phi_normal(±∞) * (∓∞/p²) = 0 * ±∞ = NaN.
                // Instead, set the boundary CDFs directly to 0 or 1.
                real p_lo_S1 = (k == 1)        ? 0.0 : Phi((log_theta[k-1] - log_r_S1) / p);
                real p_hi_S1 = (k == nratings) ? 1.0 : Phi((log_theta[k]   - log_r_S1) / p);
                real p_lo_S2 = (k == 1)        ? 0.0 : Phi((log_theta[k-1] - log_r_S2) / p);
                real p_hi_S2 = (k == nratings) ? 1.0 : Phi((log_theta[k]   - log_r_S2) / p);

                real p_k_S1 = fmax(p_hi_S1 - p_lo_S1, Tol);
                real p_k_S2 = fmax(p_hi_S2 - p_lo_S2, Tol);

                // Count-matrix ordering:
                //   prCR[1] = highest-confidence CR  →  map k → nratings+1−k
                //   prFA[1] = lowest-confidence FA   →  map k → k  (no reversal)
                //   prM[1]  = highest-confidence M   →  map k → nratings+1−k
                //   prH[1]  = lowest-confidence H    →  map k → k  (no reversal)
                prCR[nratings + 1 - k] += w * w_cr_S1 * p_k_S1;
                prFA[k]                += w * w_fa_S1 * p_k_S1;
                prM[nratings + 1 - k]  += w * w_cr_S2 * p_k_S2;
                prH[k]                 += w * w_fa_S2 * p_k_S2;
            }
        }

        // Multinomial log-likelihoods (re-normalise to absorb Tol floor)
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
    // Group mean meta-uncertainty on the natural scale (at covariate mean)
    real group_phi = exp(mu_logPhi);
}
