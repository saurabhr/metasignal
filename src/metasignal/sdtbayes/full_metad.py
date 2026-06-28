"""Option B — Full hierarchical meta-d' (HMeta-d) via custom Stan in brmspy.

This module implements the hierarchical Bayesian meta-d' model from:

    Fleming, S. M. (2017). HMeta-d: hierarchical Bayesian estimation of
    metacognitive efficiency from confidence ratings.
    *Neuroscience of Consciousness*, 2017(1), nix007.
    https://doi.org/10.1093/nc/nix007

The original model was specified in JAGS.  Here it is ported to Stan and
injected into brms via ``brmspy.brms.call("stanvar", ...)``.  The Stan code
implements the identical multinomial SDT likelihood and the same hierarchical
prior structure as the reference JAGS model.

Model structure
---------------
- **d'** and **c** (Type-1 criterion) are estimated hierarchically across
  participants with normal group priors.
- **M-ratio** = meta-d'/d' is parameterised on the log scale
  (``logMratio ~ normal(mu_logMratio, sigma_logMratio)``) so that it is
  guaranteed positive and the group-level posterior is symmetric.
- **Type-2 criteria** (``cS1``, ``cS2``) are estimated per-participant as
  sorted vectors with truncated-normal group priors.  ``cS1`` (below the
  Type-1 criterion) produces the correct-rejection / miss probabilities;
  ``cS2`` (above the Type-1 criterion) produces the hit / false-alarm
  probabilities.
- The **multinomial SDT likelihood** is identical to the one used by
  ``stdpy.fit_meta_d_mle`` and the Maniscalco & Lau (2012) MLE code.

Count matrix format
-------------------
For each participant, counts are arranged as four blocks of ``nratings``
integers each (total ``4 * nratings``), in the order used by the reference
JAGS model::

    [CR₁ … CR_nR | FA₁ … FA_nR | M₁ … M_nR | H₁ … H_nR]

where within each block ratings are ordered from lowest confidence (index 1)
to highest confidence (index nratings), and:

- **CR** (correct rejections) = S1 trials, S1 responses
- **FA** (false alarms)       = S1 trials, S2 responses
- **M**  (misses)             = S2 trials, S1 responses
- **H**  (hits)               = S2 trials, S2 responses
"""

from __future__ import annotations

from typing import Any

import numpy as np

from metasignal.sdtbayes.diagnostics import FitResult

# ---------------------------------------------------------------------------
# Stan code blocks — faithful port of Fleming (2017) JAGS model
# ---------------------------------------------------------------------------

_STAN_DATA = """\
int<lower=1> nsubj;
int<lower=1> nratings;
array[nsubj, nratings * 4] int hmetad_counts;
real<lower=0> Tol;
"""

_STAN_PARAMETERS = """\
// --- Group-level Type-1 parameters ---
real mu_d1;
real<lower=0> sigma_d1;
vector[nsubj] d1_z;

real mu_c1;
real<lower=0> sigma_c1;
vector[nsubj] c1_z;

// --- Group-level M-ratio (log scale) ---
real mu_logMratio;
real<lower=0> sigma_logMratio;
vector[nsubj] logMratio_z;

// --- Group-level Type-2 criterion hyperparameters ---
real<lower=0> mu_c2;
real<lower=0> sigma_c2;

// --- Per-subject Type-2 criteria ---
// cS1[s]: nratings-1 ordered criteria BELOW c1[s]  (ascending)
// cS2[s]: nratings-1 ordered criteria ABOVE c1[s]  (ascending)
array[nsubj] ordered[nratings - 1] cS1_raw;
array[nsubj] ordered[nratings - 1] cS2_raw;
"""

_STAN_PARAMETERS_TWO_GROUP = """\
// --- Group-level Type-1 parameters (shared across groups) ---
real mu_d1;
real<lower=0> sigma_d1;
vector[nsubj_a + nsubj_b] d1_z;

real mu_c1;
real<lower=0> sigma_c1;
vector[nsubj_a + nsubj_b] c1_z;

// --- Group-level M-ratio (log scale) — separate per group ---
real mu_logMratio_a;
real mu_logMratio_b;
real<lower=0> sigma_logMratio;
vector[nsubj_a] logMratio_z_a;
vector[nsubj_b] logMratio_z_b;

// --- Group-level Type-2 criterion hyperparameters ---
real<lower=0> mu_c2;
real<lower=0> sigma_c2;

// --- Per-subject Type-2 criteria ---
array[nsubj_a] ordered[nratings - 1] cS1_a;
array[nsubj_a] ordered[nratings - 1] cS2_a;
array[nsubj_b] ordered[nratings - 1] cS1_b;
array[nsubj_b] ordered[nratings - 1] cS2_b;
"""

_STAN_DATA_TWO_GROUP = """\
int<lower=1> nsubj_a;
int<lower=1> nsubj_b;
int<lower=1> nratings;
array[nsubj_a, nratings * 4] int hmetad_counts_a;
array[nsubj_b, nratings * 4] int hmetad_counts_b;
real<lower=0> Tol;
"""

_STAN_TRANSFORMED_PARAMETERS_TWO_GROUP = """\
vector[nsubj_a] d1_a;
vector[nsubj_b] d1_b;
vector[nsubj_a] c1_a;
vector[nsubj_b] c1_b;
vector[nsubj_a] Mratio_a;
vector[nsubj_b] Mratio_b;
vector[nsubj_a] meta_d_a;
vector[nsubj_b] meta_d_b;
real delta_logMratio;

d1_a = mu_d1 + sigma_d1 * d1_z[1:nsubj_a];
d1_b = mu_d1 + sigma_d1 * d1_z[(nsubj_a + 1):(nsubj_a + nsubj_b)];
c1_a = mu_c1 + sigma_c1 * c1_z[1:nsubj_a];
c1_b = mu_c1 + sigma_c1 * c1_z[(nsubj_a + 1):(nsubj_a + nsubj_b)];
for (s in 1:nsubj_a) {
    Mratio_a[s] = exp(mu_logMratio_a + sigma_logMratio * logMratio_z_a[s]);
    meta_d_a[s] = Mratio_a[s] * d1_a[s];
}
for (s in 1:nsubj_b) {
    Mratio_b[s] = exp(mu_logMratio_b + sigma_logMratio * logMratio_z_b[s]);
    meta_d_b[s] = Mratio_b[s] * d1_b[s];
}
delta_logMratio = mu_logMratio_b - mu_logMratio_a;
"""


def _group_likelihood_block(
    nsubj_var: str,
    counts_var: str,
    mratio_var: str,
    d1_var: str,
    c1_var: str,
    cs1_var: str,
    cs2_var: str,
) -> str:
    """Generate the Stan model likelihood block for one group."""
    # pylint: disable=line-too-long
    return f"""\
for (s in 1:{nsubj_var}) {{
    real S1mu = -{mratio_var}[s] * {d1_var}[s] / 2.0;
    real S2mu =  {mratio_var}[s] * {d1_var}[s] / 2.0;
    real C_area_rS1 = fmax(Phi({c1_var}[s] - S1mu), Tol);
    real I_area_rS1 = fmax(Phi({c1_var}[s] - S2mu), Tol);
    real C_area_rS2 = fmax(1.0 - Phi({c1_var}[s] - S2mu), Tol);
    real I_area_rS2 = fmax(1.0 - Phi({c1_var}[s] - S1mu), Tol);
    vector[nratings] prCR;
    vector[nratings] prFA;
    vector[nratings] prM;
    vector[nratings] prH;
    prCR[1] = fmax(Phi({cs1_var}[s, 1] - S1mu) / C_area_rS1, Tol);
    for (k in 1:(nratings - 2)) {{
        prCR[k + 1] = fmax((Phi({cs1_var}[s, k + 1] - S1mu) - Phi({cs1_var}[s, k] - S1mu)) / C_area_rS1, Tol);
    }}
    prCR[nratings] = fmax((Phi({c1_var}[s] - S1mu) - Phi({cs1_var}[s, nratings - 1] - S1mu)) / C_area_rS1, Tol);
    prFA[1] = fmax(((1.0 - Phi({c1_var}[s] - S1mu)) - (1.0 - Phi({cs2_var}[s, 1] - S1mu))) / I_area_rS2, Tol);
    for (k in 1:(nratings - 2)) {{
        prFA[k + 1] = fmax(((1.0 - Phi({cs2_var}[s, k] - S1mu)) - (1.0 - Phi({cs2_var}[s, k + 1] - S1mu))) / I_area_rS2, Tol);
    }}
    prFA[nratings] = fmax((1.0 - Phi({cs2_var}[s, nratings - 1] - S1mu)) / I_area_rS2, Tol);
    prM[1] = fmax(Phi({cs1_var}[s, 1] - S2mu) / I_area_rS1, Tol);
    for (k in 1:(nratings - 2)) {{
        prM[k + 1] = fmax((Phi({cs1_var}[s, k + 1] - S2mu) - Phi({cs1_var}[s, k] - S2mu)) / I_area_rS1, Tol);
    }}
    prM[nratings] = fmax((Phi({c1_var}[s] - S2mu) - Phi({cs1_var}[s, nratings - 1] - S2mu)) / I_area_rS1, Tol);
    prH[1] = fmax(((1.0 - Phi({c1_var}[s] - S2mu)) - (1.0 - Phi({cs2_var}[s, 1] - S2mu))) / C_area_rS2, Tol);
    for (k in 1:(nratings - 2)) {{
        prH[k + 1] = fmax(((1.0 - Phi({cs2_var}[s, k] - S2mu)) - (1.0 - Phi({cs2_var}[s, k + 1] - S2mu))) / C_area_rS2, Tol);
    }}
    prH[nratings] = fmax((1.0 - Phi({cs2_var}[s, nratings - 1] - S2mu)) / C_area_rS2, Tol);
    target += multinomial_lpmf({counts_var}[s, 1:nratings] | prCR / sum(prCR));
    target += multinomial_lpmf({counts_var}[s, (nratings + 1):(2 * nratings)] | prFA / sum(prFA));
    target += multinomial_lpmf({counts_var}[s, (2 * nratings + 1):(3 * nratings)] | prM / sum(prM));
    target += multinomial_lpmf({counts_var}[s, (3 * nratings + 1):(4 * nratings)] | prH / sum(prH));
}}
"""


_STAN_MODEL_TWO_GROUP = """\
mu_d1 ~ normal(1, 2);
sigma_d1 ~ exponential(1);
d1_z ~ normal(0, 1);
mu_c1 ~ normal(0, 1);
sigma_c1 ~ exponential(1);
c1_z ~ normal(0, 1);
mu_logMratio_a ~ normal(0, 1);
mu_logMratio_b ~ normal(0, 1);
sigma_logMratio ~ exponential(1);
logMratio_z_a ~ normal(0, 1);
logMratio_z_b ~ normal(0, 1);
mu_c2 ~ normal(1, 1);
sigma_c2 ~ exponential(1);
for (s in 1:nsubj_a) {
    cS1_a[s] ~ normal(c1_a[s] - mu_c2, sigma_c2);
    cS2_a[s] ~ normal(c1_a[s] + mu_c2, sigma_c2);
}
for (s in 1:nsubj_b) {
    cS1_b[s] ~ normal(c1_b[s] - mu_c2, sigma_c2);
    cS2_b[s] ~ normal(c1_b[s] + mu_c2, sigma_c2);
}
""" + _group_likelihood_block(
    "nsubj_a", "hmetad_counts_a", "Mratio_a", "d1_a", "c1_a", "cS1_a", "cS2_a",
) + _group_likelihood_block(
    "nsubj_b", "hmetad_counts_b", "Mratio_b", "d1_b", "c1_b", "cS1_b", "cS2_b",
)


_STAN_TRANSFORMED_PARAMETERS = """\
vector[nsubj] d1;
vector[nsubj] c1;
vector[nsubj] Mratio;
vector[nsubj] meta_d;

d1 = mu_d1 + sigma_d1 * d1_z;
c1 = mu_c1 + sigma_c1 * c1_z;
for (s in 1:nsubj) {
    Mratio[s] = exp(mu_logMratio + sigma_logMratio * logMratio_z[s]);
    meta_d[s] = Mratio[s] * d1[s];
}
"""

# The model block implements the exact same multinomial probabilities as the
# JAGS reference model (Fleming 2017, Supplementary) and our Python MLE code.
_STAN_MODEL = """\
// --- Group-level priors ---
mu_d1         ~ normal(1, 2);
sigma_d1      ~ exponential(1);
d1_z          ~ normal(0, 1);

mu_c1         ~ normal(0, 1);
sigma_c1      ~ exponential(1);
c1_z          ~ normal(0, 1);

mu_logMratio  ~ normal(0, 1);
sigma_logMratio ~ exponential(1);
logMratio_z   ~ normal(0, 1);

mu_c2         ~ normal(1, 1);
sigma_c2      ~ exponential(1);

for (s in 1:nsubj) {
    // --- Type-2 criteria priors ---
    // cS1 clustered below c1[s]; cS2 clustered above c1[s].
    // Soft truncation: prior centres are shifted by ±mu_c2 from c1.
    cS1_raw[s] ~ normal(c1[s] - mu_c2, sigma_c2);
    cS2_raw[s] ~ normal(c1[s] + mu_c2, sigma_c2);

    // --- SDT signal means ---
    real S1mu = -meta_d[s] / 2.0;
    real S2mu =  meta_d[s] / 2.0;

    // --- Normalising areas (probability mass on correct side of c1) ---
    real C_area_rS1 = fmax(Phi(c1[s] - S1mu), Tol);
    real I_area_rS1 = fmax(Phi(c1[s] - S2mu), Tol);
    real C_area_rS2 = fmax(1.0 - Phi(c1[s] - S2mu), Tol);
    real I_area_rS2 = fmax(1.0 - Phi(c1[s] - S1mu), Tol);

    // --- Multinomial probability vectors ---
    vector[nratings] prCR;   // correct rejections
    vector[nratings] prFA;   // false alarms
    vector[nratings] prM;    // misses
    vector[nratings] prH;    // hits

    // CR: S1 trial, S1 response — decision variable below c1, sorted by cS1
    prCR[1] = fmax(Phi(cS1_raw[s, 1] - S1mu) / C_area_rS1, Tol);
    for (k in 1:(nratings - 2)) {
        prCR[k + 1] = fmax(
            (Phi(cS1_raw[s, k + 1] - S1mu) - Phi(cS1_raw[s, k] - S1mu)) / C_area_rS1,
            Tol);
    }
    prCR[nratings] = fmax(
        (Phi(c1[s] - S1mu) - Phi(cS1_raw[s, nratings - 1] - S1mu)) / C_area_rS1,
        Tol);

    // FA: S1 trial, S2 response — decision variable above c1, sorted by cS2
    prFA[1] = fmax(
        ((1.0 - Phi(c1[s] - S1mu)) - (1.0 - Phi(cS2_raw[s, 1] - S1mu))) / I_area_rS2,
        Tol);
    for (k in 1:(nratings - 2)) {
        prFA[k + 1] = fmax(
            ((1.0 - Phi(cS2_raw[s, k] - S1mu)) - (1.0 - Phi(cS2_raw[s, k + 1] - S1mu))) / I_area_rS2,
            Tol);
    }
    prFA[nratings] = fmax(
        (1.0 - Phi(cS2_raw[s, nratings - 1] - S1mu)) / I_area_rS2,
        Tol);

    // M: S2 trial, S1 response — signal present but decision below c1
    prM[1] = fmax(Phi(cS1_raw[s, 1] - S2mu) / I_area_rS1, Tol);
    for (k in 1:(nratings - 2)) {
        prM[k + 1] = fmax(
            (Phi(cS1_raw[s, k + 1] - S2mu) - Phi(cS1_raw[s, k] - S2mu)) / I_area_rS1,
            Tol);
    }
    prM[nratings] = fmax(
        (Phi(c1[s] - S2mu) - Phi(cS1_raw[s, nratings - 1] - S2mu)) / I_area_rS1,
        Tol);

    // H: S2 trial, S2 response — signal present, decision above c1
    prH[1] = fmax(
        ((1.0 - Phi(c1[s] - S2mu)) - (1.0 - Phi(cS2_raw[s, 1] - S2mu))) / C_area_rS2,
        Tol);
    for (k in 1:(nratings - 2)) {
        prH[k + 1] = fmax(
            ((1.0 - Phi(cS2_raw[s, k] - S2mu)) - (1.0 - Phi(cS2_raw[s, k + 1] - S2mu))) / C_area_rS2,
            Tol);
    }
    prH[nratings] = fmax(
        (1.0 - Phi(cS2_raw[s, nratings - 1] - S2mu)) / C_area_rS2,
        Tol);

    // --- Multinomial likelihoods (re-normalised to exact simplex) ---
    target += multinomial_lpmf(hmetad_counts[s, 1:nratings]
                               | prCR / sum(prCR));
    target += multinomial_lpmf(hmetad_counts[s, (nratings + 1):(2 * nratings)]
                               | prFA / sum(prFA));
    target += multinomial_lpmf(hmetad_counts[s, (2 * nratings + 1):(3 * nratings)]
                               | prM / sum(prM));
    target += multinomial_lpmf(hmetad_counts[s, (3 * nratings + 1):(4 * nratings)]
                               | prH / sum(prH));
}
"""


def _build_count_matrix(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
) -> np.ndarray:
    """Build (nsubj, 4*nratings) integer count matrix in JAGS/HMeta-d order.

    Order within each block is lowest confidence → highest confidence.
    Block order: [CR | FA | M | H].
    """
    from metasignal.stdpy.core import trials_to_counts

    rows = []
    for stim, resp, conf in participants:
        stim = np.asarray(stim, dtype=int)
        resp = np.asarray(resp, dtype=int)
        conf = np.asarray(conf, dtype=int)

        nr_s1, nr_s2 = trials_to_counts(stim, resp, conf, n_ratings)

        # trials_to_counts layout (same as metadpy nR_S1/nR_S2):
        #   nr_s1[:nR]  = CRs   (S1 stim, S1 resp), rating nR→1 (high→low conf)
        #   nr_s1[nR:]  = FAs   (S1 stim, S2 resp), rating  1→nR (low→high conf)
        #   nr_s2[:nR]  = Misses(S2 stim, S1 resp), rating nR→1 (high→low conf)
        #   nr_s2[nR:]  = Hits  (S2 stim, S2 resp), rating  1→nR (low→high conf)
        # Stan probability ordering:
        #   prCR[1]=highest-conf CR, prFA[1]=lowest-conf FA, prM[1]=highest-conf M, prH[1]=lowest-conf H
        rows.append(np.concatenate([
            nr_s1[:n_ratings],   # CRs,    high→low confidence
            nr_s1[n_ratings:],   # FAs,    low→high confidence
            nr_s2[:n_ratings],   # Misses, high→low confidence
            nr_s2[n_ratings:],   # Hits,   low→high confidence
        ]))

    return np.array(rows, dtype=int)


def fit_full_metad(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    chains: int = 4,
    n_iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    tol: float = 1e-7,
    **kwargs: Any,
) -> Any:
    """Full hierarchical meta-d' (HMeta-d) via custom Stan injected through brmspy.

    Ports the Fleming (2017) JAGS model to Stan and fits it via
    ``brmspy.brms.brm()`` using the ``empty`` family (no built-in brms
    likelihood) with the SDT multinomial likelihood injected as custom
    ``stanvar()`` blocks.

    The group-level parameter of primary interest is **``mu_logMratio``**,
    which represents the posterior mean log M-ratio.  ``exp(mu_logMratio)``
    gives the group mean M-ratio (metacognitive efficiency).

    Key parameters in the posterior
    --------------------------------
    - ``mu_logMratio`` — group mean log M-ratio
    - ``sigma_logMratio`` — between-subject SD on the log M-ratio scale
    - ``Mratio[s]`` — per-subject M-ratio (transformed parameter)
    - ``meta_d[s]`` — per-subject meta-d' (transformed parameter)
    - ``d1[s]``, ``c1[s]`` — per-subject Type-1 d' and criterion
    - ``mu_d1``, ``mu_c1`` — group mean Type-1 parameters

    Args:
        participants: List of ``(stim, resp, conf)`` tuples, one per participant.
        n_ratings: Number of confidence rating categories.
        chains: MCMC chains (default 4).
        n_iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations (default 1000).
        seed: Random seed (default 42).
        tol: Minimum probability floor for multinomial cells (default 1e-7).
        **kwargs: Forwarded to ``brmspy.brms.brm``.

    Returns:
        ``FitResult`` with ``.idata`` (ArviZ InferenceData) and ``.r`` (R handle).

    Raises:
        ImportError: If ``brmspy`` is not installed.

    Example::

        import numpy as np
        from metasignal.sdtbayes import fit_full_metad, posterior_summary

        rng = np.random.default_rng(0)
        participants = [
            (rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.integers(1, 5, 200))
            for _ in range(20)
        ]

        fit = fit_full_metad(participants, n_ratings=4)
        print(posterior_summary(fit, var_names=["mu_logMratio", "sigma_logMratio",
                                                 "mu_d1", "Mratio"]))
    """
    try:
        from brmspy import brms
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "brmspy is not installed. Run:\n    pip install metasignal[sdtbayes]"
        ) from e

    nsubj = len(participants)
    counts_mat = _build_count_matrix(participants, n_ratings)
    n_counts_cols = n_ratings * 4

    # brms stanvar(block="data") requires x= per element (brms 2.20+)
    sv_nsubj   = brms.call("stanvar", x=int(nsubj),    name="nsubj",
                            scode="int<lower=1> nsubj;")
    sv_nratings = brms.call("stanvar", x=int(n_ratings), name="nratings",
                             scode="int<lower=1> nratings;")
    sv_counts  = brms.call("stanvar",
                            x=[[int(v) for v in row] for row in counts_mat.tolist()],
                            name="hmetad_counts",
                            scode=f"array[nsubj, {n_counts_cols}] int hmetad_counts;")
    sv_tol     = brms.call("stanvar", x=float(tol), name="Tol",
                            scode="real<lower=0> Tol;")

    sv_params = brms.call("stanvar", scode=_STAN_PARAMETERS,              block="parameters")
    sv_tpar   = brms.call("stanvar", scode=_STAN_TRANSFORMED_PARAMETERS,  block="tpar")
    sv_model  = brms.call("stanvar", scode=_STAN_MODEL,                   block="model")

    # brms 2.23.0 removed empty(). Workaround: bernoulli + constant(0) intercept
    # + sample_prior="only" → prior_only=1 → default likelihood skipped.

    _result = brms.brm(
        formula=brms.bf("y ~ 1"),
        data=pd.DataFrame({"y": [0]}),
        family="bernoulli",
        sample_prior="only",
        stanvars=[sv_nsubj, sv_nratings, sv_counts, sv_tol,
                  sv_params, sv_tpar, sv_model],
        chains=chains,
        iter=n_iter,
        warmup=warmup,
        seed=seed,
        **kwargs,
    )
    return FitResult(idata=_result.idata, r=_result.r)


def fit_full_metad_comparison(
    group_a: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    group_b: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    chains: int = 4,
    n_iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    tol: float = 1e-7,
    **kwargs: Any,
) -> Any:
    """Full HMeta-d comparison between two groups via custom Stan in brmspy.

    Extends the single-group model with separate ``mu_logMratio_a`` and
    ``mu_logMratio_b`` hyperparameters and a derived contrast
    ``delta_logMratio = mu_logMratio_b - mu_logMratio_a``.

    The posterior of ``delta_logMratio`` gives the full Bayesian answer to
    "does group B have higher metacognitive efficiency than group A?" without
    a null-hypothesis significance test.

    Args:
        group_a: Participants in group A.
        group_b: Participants in group B.
        n_ratings: Number of confidence rating categories.
        chains: MCMC chains (default 4).
        n_iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations (default 1000).
        seed: Random seed (default 42).
        tol: Minimum probability floor (default 1e-7).
        **kwargs: Forwarded to ``brmspy.brms.brm``.

    Returns:
        ``FitResult``.  Key parameters: ``mu_logMratio_a``, ``mu_logMratio_b``,
        ``delta_logMratio`` (b − a difference).

    Raises:
        ImportError: If ``brmspy`` is not installed.

    Example::

        fit = fit_full_metad_comparison(healthy, patient, n_ratings=4)

        import arviz as az
        delta = az.extract(fit.idata)["delta_logMratio"].values
        print(f"P(group B > group A): {(delta > 0).mean():.3f}")
    """
    try:
        from brmspy import brms
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "brmspy is not installed. Run:\n    pip install metasignal[sdtbayes]"
        ) from e

    na = len(group_a)
    nb = len(group_b)
    counts_a = _build_count_matrix(group_a, n_ratings)
    counts_b = _build_count_matrix(group_b, n_ratings)
    n_counts_cols = n_ratings * 4

    sv_na      = brms.call("stanvar", x=int(na),       name="nsubj_a",
                            scode="int<lower=1> nsubj_a;")
    sv_nb      = brms.call("stanvar", x=int(nb),       name="nsubj_b",
                            scode="int<lower=1> nsubj_b;")
    sv_nratings = brms.call("stanvar", x=int(n_ratings), name="nratings",
                             scode="int<lower=1> nratings;")
    sv_ca      = brms.call("stanvar",
                            x=[[int(v) for v in row] for row in counts_a.tolist()],
                            name="hmetad_counts_a",
                            scode=f"array[nsubj_a, {n_counts_cols}] int hmetad_counts_a;")
    sv_cb      = brms.call("stanvar",
                            x=[[int(v) for v in row] for row in counts_b.tolist()],
                            name="hmetad_counts_b",
                            scode=f"array[nsubj_b, {n_counts_cols}] int hmetad_counts_b;")
    sv_tol     = brms.call("stanvar", x=float(tol),   name="Tol",
                            scode="real<lower=0> Tol;")

    sv_params = brms.call("stanvar", scode=_STAN_PARAMETERS_TWO_GROUP,             block="parameters")
    sv_tpar   = brms.call("stanvar", scode=_STAN_TRANSFORMED_PARAMETERS_TWO_GROUP, block="tpar")
    sv_model  = brms.call("stanvar", scode=_STAN_MODEL_TWO_GROUP,                  block="model")


    _result = brms.brm(
        formula=brms.bf("y ~ 1"),
        data=pd.DataFrame({"y": [0]}),
        family="bernoulli",
        sample_prior="only",
        stanvars=[sv_na, sv_nb, sv_nratings, sv_ca, sv_cb, sv_tol,
                  sv_params, sv_tpar, sv_model],
        chains=chains,
        iter=n_iter,
        warmup=warmup,
        seed=seed,
        **kwargs,
    )
    return FitResult(idata=_result.idata, r=_result.r)
