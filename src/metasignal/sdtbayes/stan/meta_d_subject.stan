/**
 * meta_d_subject.stan  —  Single-subject Bayesian meta-d' (mratio)
 *
 * Exact Stan port of the metadpy hmetad() PyMC model (Fleming 2017):
 *
 *   Fleming, S. M. (2017). HMeta-d: hierarchical Bayesian estimation of
 *   metacognitive efficiency from confidence ratings.
 *   Neuroscience of Consciousness, 2017(1), nix007.
 *
 * Priors
 * ------
 *   d1        ~ Normal(0, sqrt(2))
 *   c1        ~ Normal(0, 1/sqrt(2))
 *   meta_d    ~ Normal(d1, 1/sqrt(2))   [centred on d1, as in metadpy]
 *
 * Type-2 criteria (metadpy-matching parameterization)
 * ---------------------------------------------------
 * metadpy samples nratings-1 positive HalfNormal(0, 1/sqrt(2)) offsets for
 * each side, sorts them, then places them around c1:
 *
 *   cS1_actual = c1 - sort_desc(cS1_offsets)   [nR-1 criteria below c1]
 *   cS2_actual = c1 + sort_asc(cS2_offsets)    [nR-1 criteria above c1]
 *
 * Both give ascending vectors of actual threshold positions, as required by
 * the multinomial SDT likelihood.
 *
 * Data format: 1-D array of length nratings * 4
 *   [CR_nR .. CR_1 | FA_1 .. FA_nR | M_nR .. M_1 | H_1 .. H_nR]
 *
 * Type-1 binomial likelihood
 * --------------------------
 * metadpy additionally ties d1/c1 directly to the total hit and false-alarm
 * counts via a Binomial likelihood (subject_level_pymc.py):
 *
 *   h = Phi(d1/2 - c1);   H  ~ Binomial(S, h)
 *   f = Phi(-d1/2 - c1);  FA ~ Binomial(N, f)
 *
 * where S = total S2-stimulus trials, N = total S1-stimulus trials.  Without
 * this term d1/c1 are only weakly constrained by the Type-2 multinomial
 * (which is normalised per response type and so discards the total
 * hit/FA-rate information) and posteriors come out far wider than metadpy's.
 *
 * Key posterior quantities
 * ------------------------
 *   d1      : Type-1 sensitivity (d')
 *   c1      : Type-1 criterion
 *   meta_d  : Metacognitive sensitivity (meta-d')
 *   Mratio  : meta_d / d1  (generated quantity)
 */

data {
    int<lower=2> nratings;
    array[nratings * 4] int counts;
    real<lower=0> Tol;
}

transformed data {
    // Type-1 totals: counts layout is [CR | FA | M | H], each block nratings long.
    int CR_total = sum(counts[1:nratings]);
    int FA_total = sum(counts[(nratings + 1):(2 * nratings)]);
    int M_total  = sum(counts[(2 * nratings + 1):(3 * nratings)]);
    int H_total  = sum(counts[(3 * nratings + 1):(4 * nratings)]);
    int N_total  = CR_total + FA_total;  // total S1-stimulus trials
    int S_total  = M_total + H_total;    // total S2-stimulus trials
}

parameters {
    real d1;
    real c1;
    real meta_d;

    // Positive HalfNormal offsets from c1 — exact metadpy parameterisation.
    // positive_ordered (not vector<lower=0> + runtime sort_asc) removes the
    // permutation symmetry that would otherwise leave the raw components
    // non-identified: any permutation of an unordered vector gives the same
    // sorted result and hence identical likelihood, causing severe
    // label-switching (r_hat blows up for the raw offsets even though every
    // downstream quantity that depends only on the sorted values is fine).
    positive_ordered[nratings - 1] cS1_offsets;
    positive_ordered[nratings - 1] cS2_offsets;
}

transformed parameters {
    // Actual criterion positions (both ascending):
    //   cS1[k] = c1 - cS1_offsets[nratings - k]  → [c1-max, .., c1-min]
    //   cS2[k] = c1 + cS2_offsets[k]              → [c1+min, .., c1+max]
    vector[nratings - 1] cS1;
    vector[nratings - 1] cS2;
    for (k in 1:(nratings - 1)) {
        cS1[k] = c1 - cS1_offsets[nratings - k];
        cS2[k] = c1 + cS2_offsets[k];
    }
}

model {
    // ── Priors (exact metadpy defaults) ──────────────────────────────────
    d1     ~ normal(0, sqrt(2.0));
    c1     ~ normal(0, inv_sqrt(2.0));
    meta_d ~ normal(d1, inv_sqrt(2.0));

    // lower=0 constraint makes these half-normal, matching metadpy's
    // pm.HalfNormal("cS1", sigma=1/sqrt(2), shape=nratings-1)
    cS1_offsets ~ normal(0, inv_sqrt(2.0));
    cS2_offsets ~ normal(0, inv_sqrt(2.0));

    // ── Type-1 binomial likelihood (ties d1, c1 to observed hit/FA totals) ─
    target += binomial_lpmf(H_total  | S_total, Phi(d1 / 2.0 - c1));
    target += binomial_lpmf(FA_total | N_total, Phi(-d1 / 2.0 - c1));

    // ── Multinomial SDT likelihood ────────────────────────────────────────
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

    // Correct rejections — cS1[1] < .. < cS1[nR-1] < c1, high conf first
    prCR[1] = fmax(Phi(cS1[1] - S1mu) / C_area_rS1, Tol);
    for (k in 1:(nratings - 2))
        prCR[k + 1] = fmax(
            (Phi(cS1[k + 1] - S1mu) - Phi(cS1[k] - S1mu)) / C_area_rS1, Tol);
    prCR[nratings] = fmax(
        (Phi(c1 - S1mu) - Phi(cS1[nratings - 1] - S1mu)) / C_area_rS1, Tol);

    // False alarms — c1 < cS2[1] < .. < cS2[nR-1], low conf first
    prFA[1] = fmax(
        ((1.0 - Phi(c1 - S1mu)) - (1.0 - Phi(cS2[1] - S1mu))) / I_area_rS2, Tol);
    for (k in 1:(nratings - 2))
        prFA[k + 1] = fmax(
            ((1.0 - Phi(cS2[k] - S1mu)) - (1.0 - Phi(cS2[k + 1] - S1mu)))
            / I_area_rS2, Tol);
    prFA[nratings] = fmax(
        (1.0 - Phi(cS2[nratings - 1] - S1mu)) / I_area_rS2, Tol);

    // Misses — same S1 criteria, S2 stimulus
    prM[1] = fmax(Phi(cS1[1] - S2mu) / I_area_rS1, Tol);
    for (k in 1:(nratings - 2))
        prM[k + 1] = fmax(
            (Phi(cS1[k + 1] - S2mu) - Phi(cS1[k] - S2mu)) / I_area_rS1, Tol);
    prM[nratings] = fmax(
        (Phi(c1 - S2mu) - Phi(cS1[nratings - 1] - S2mu)) / I_area_rS1, Tol);

    // Hits — same S2 criteria, S2 stimulus
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
    real Mratio = (abs(d1) > 1e-6) ? meta_d / d1 : 0.0;
}
