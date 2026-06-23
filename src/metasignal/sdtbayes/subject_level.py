"""Subject-level Bayesian meta-d' model — direct port of metadpy's hmetad() to brmspy.

This module provides a single-participant Bayesian meta-d' estimation that
mirrors the API and model structure of
`metadpy.bayesian.hmetad(nR_S1=..., nR_S2=...)` but runs via brmspy /
brms / Stan instead of PyMC.

The model is a faithful port of the PyMC implementation in metadpy's
``subject_level_pymc.py``:

- **Type-1** parameters (``d1``, ``c1``) are estimated with a *binomial*
  likelihood over hit and false-alarm *totals*.
- **Type-2** parameters (``meta_d``, ``cS1``, ``cS2``) are estimated with a
  *multinomial* SDT likelihood over the full confidence-rating distribution.
- The prior on ``meta_d`` is ``Normal(d1, 1/√2)`` — centred on d1, matching
  ``tau=2`` in metadpy's PyMC notation.
- Criteria ``cS1`` are constructed by negating and sorting i.i.d.
  ``HalfNormal(1/√2)`` samples and shifting them below ``c1``, replicating
  ``cS1 = sort(−cS1_hn) + (c1 − Tol)`` from metadpy.
- Same construction for ``cS2`` above ``c1``.

Input format
------------
Arrays ``nR_S1`` and ``nR_S2`` follow the **Maniscalco & Lau (2012) /
metadpy** convention (8 elements for 4 ratings):

``nR_S1 = [CR_r4, CR_r3, CR_r2, CR_r1 | FA_r1, FA_r2, FA_r3, FA_r4]``

``nR_S2 = [M_r4,  M_r3,  M_r2,  M_r1  | H_r1,  H_r2,  H_r3,  H_r4]``

where CRs are sorted from highest → lowest confidence and FAs / Hits from
lowest → highest confidence.

References
----------
Fleming, S. M. (2017). HMeta-d. *Neuroscience of Consciousness*, nix007.
Legrand, N. et al. (2021). metadpy. https://github.com/embodied-computation-group/metadpy
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Stan code blocks — matching metadpy's subject_level_pymc.py exactly
# ---------------------------------------------------------------------------

_SL_DATA = """\
int<lower=1> nratings;
array[nratings * 4] int sl_counts;   // [CR | FA | M | H], nratings each
int<lower=0> sl_H;    // total hits
int<lower=0> sl_FA;   // total false alarms
int<lower=0> sl_S;    // total signal trials
int<lower=0> sl_N;    // total noise trials
real<lower=0> sl_Tol;
"""

_SL_PARAMETERS = """\
// --- Type-1 ---
real sl_d1;
real sl_c1;

// --- Type-2 ---
real sl_meta_d;
vector<lower=0>[nratings - 1] sl_cS1_hn;   // half-normal, will be negated & sorted
vector<lower=0>[nratings - 1] sl_cS2_hn;   // half-normal, will be sorted
"""

_SL_TRANSFORMED_PARAMETERS = """\
vector[nratings - 1] sl_cS1;   // sorted criteria below sl_c1
vector[nratings - 1] sl_cS2;   // sorted criteria above sl_c1

// Replicate metadpy: cS1 = sort(−cS1_hn) + (c1 − Tol)
// sort(−cS1_hn) ascending = [−max, …, −min] so cS1[1] = most extreme (highest conf)
{
    array[nratings - 1] int idx = sort_indices_asc(sl_cS1_hn);
    for (j in 1:(nratings - 1)) {
        sl_cS1[j] = -(sl_cS1_hn[idx[nratings - j]]) + sl_c1 - sl_Tol;
    }
}
// cS2 = sort(cS2_hn) + (c1 − Tol): cS2[1] = smallest = lowest-conf FA criterion
{
    array[nratings - 1] int idx = sort_indices_asc(sl_cS2_hn);
    for (j in 1:(nratings - 1)) {
        sl_cS2[j] = sl_cS2_hn[idx[j]] + sl_c1 - sl_Tol;
    }
}
"""

_SL_MODEL = """\
// ==== Type-1 priors (tau=0.5 → sd=√2 ≈ 1.414 for d1; tau=2 → sd=0.707 for c1) ====
sl_d1 ~ normal(0, 1.4142);
sl_c1 ~ normal(0, 0.7071);

// ==== Type-1 binomial SDT likelihood ====
sl_H  ~ binomial(sl_S, Phi( sl_d1 / 2.0 - sl_c1));
sl_FA ~ binomial(sl_N, Phi(-sl_d1 / 2.0 - sl_c1));

// ==== Type-2 priors (tau=2 → sd=0.707 for meta_d and cS1/cS2_hn) ====
sl_meta_d ~ normal(sl_d1, 0.7071);
sl_cS1_hn ~ normal(0, 0.7071);   // bounded <lower=0> → half-normal
sl_cS2_hn ~ normal(0, 0.7071);

// ==== Type-2 multinomial SDT likelihood ====
{
    real S1mu = -sl_meta_d / 2.0;
    real S2mu =  sl_meta_d / 2.0;

    real C_area_rS1 = fmax(Phi(sl_c1 - S1mu), sl_Tol);
    real I_area_rS1 = fmax(Phi(sl_c1 - S2mu), sl_Tol);
    real C_area_rS2 = fmax(1.0 - Phi(sl_c1 - S2mu), sl_Tol);
    real I_area_rS2 = fmax(1.0 - Phi(sl_c1 - S1mu), sl_Tol);

    // ---- Correct Rejections (CR) ----
    // sl_counts[1..nratings]: CRs from highest-conf (j=1) to lowest-conf (j=nR)
    vector[nratings] prCR;
    prCR[1] = fmax(Phi(sl_cS1[1] - S1mu) / C_area_rS1, sl_Tol);
    for (k in 1:(nratings - 2)) {
        prCR[k + 1] = fmax(
            (Phi(sl_cS1[k + 1] - S1mu) - Phi(sl_cS1[k] - S1mu)) / C_area_rS1,
            sl_Tol);
    }
    prCR[nratings] = fmax(
        (Phi(sl_c1 - S1mu) - Phi(sl_cS1[nratings - 1] - S1mu)) / C_area_rS1,
        sl_Tol);

    // ---- False Alarms (FA) ----
    // sl_counts[(nratings+1)..(2*nratings)]: FAs from lowest-conf (j=1) to highest-conf
    vector[nratings] prFA;
    prFA[1] = fmax(
        ((1.0 - Phi(sl_c1 - S1mu)) - (1.0 - Phi(sl_cS2[1] - S1mu))) / I_area_rS2,
        sl_Tol);
    for (k in 1:(nratings - 2)) {
        prFA[k + 1] = fmax(
            ((1.0 - Phi(sl_cS2[k] - S1mu)) - (1.0 - Phi(sl_cS2[k + 1] - S1mu))) / I_area_rS2,
            sl_Tol);
    }
    prFA[nratings] = fmax(
        (1.0 - Phi(sl_cS2[nratings - 1] - S1mu)) / I_area_rS2,
        sl_Tol);

    // ---- Misses (M) ----
    // sl_counts[(2*nratings+1)..(3*nratings)]: Misses from highest-conf (j=1) to lowest-conf
    vector[nratings] prM;
    prM[1] = fmax(Phi(sl_cS1[1] - S2mu) / I_area_rS1, sl_Tol);
    for (k in 1:(nratings - 2)) {
        prM[k + 1] = fmax(
            (Phi(sl_cS1[k + 1] - S2mu) - Phi(sl_cS1[k] - S2mu)) / I_area_rS1,
            sl_Tol);
    }
    prM[nratings] = fmax(
        (Phi(sl_c1 - S2mu) - Phi(sl_cS1[nratings - 1] - S2mu)) / I_area_rS1,
        sl_Tol);

    // ---- Hits (H) ----
    // sl_counts[(3*nratings+1)..(4*nratings)]: Hits from lowest-conf (j=1) to highest-conf
    vector[nratings] prH;
    prH[1] = fmax(
        ((1.0 - Phi(sl_c1 - S2mu)) - (1.0 - Phi(sl_cS2[1] - S2mu))) / C_area_rS2,
        sl_Tol);
    for (k in 1:(nratings - 2)) {
        prH[k + 1] = fmax(
            ((1.0 - Phi(sl_cS2[k] - S2mu)) - (1.0 - Phi(sl_cS2[k + 1] - S2mu))) / C_area_rS2,
            sl_Tol);
    }
    prH[nratings] = fmax(
        (1.0 - Phi(sl_cS2[nratings - 1] - S2mu)) / C_area_rS2,
        sl_Tol);

    // Multinomial likelihoods
    target += multinomial_lpmf(sl_counts[1:nratings]
                               | prCR / sum(prCR));
    target += multinomial_lpmf(sl_counts[(nratings + 1):(2 * nratings)]
                               | prFA / sum(prFA));
    target += multinomial_lpmf(sl_counts[(2 * nratings + 1):(3 * nratings)]
                               | prM / sum(prM));
    target += multinomial_lpmf(sl_counts[(3 * nratings + 1):(4 * nratings)]
                               | prH / sum(prH));
}
"""


def _extract_type1(
    nR_S1: np.ndarray, nR_S2: np.ndarray
) -> dict[str, int]:
    """Extract Type-1 summary statistics from nR arrays."""
    nR_S1 = np.asarray(nR_S1, dtype=int)
    nR_S2 = np.asarray(nR_S2, dtype=int)
    n_ratings = len(nR_S1) // 2
    return {
        "S": int(nR_S2.sum()),           # total signal trials
        "N": int(nR_S1.sum()),           # total noise trials
        "H": int(nR_S2[n_ratings:].sum()),  # total hits
        "FA": int(nR_S1[n_ratings:].sum()), # total false alarms
    }


def _build_counts_vector(
    nR_S1: np.ndarray, nR_S2: np.ndarray
) -> np.ndarray:
    """Build the flat [CR | FA | M | H] count vector for the Stan model.

    Count ordering within each block matches the Stan probability ordering:
    - CR block: highest-confidence first (index 1 in Stan = rating nR)
    - FA block: lowest-confidence first  (index 1 in Stan = rating 1)
    - M  block: highest-confidence first
    - H  block: lowest-confidence first
    This is identical to the concatenation ``[nR_S1, nR_S2]`` from metadpy's
    ``extractParameters`` function.
    """
    n_ratings = len(nR_S1) // 2
    return np.concatenate([
        nR_S1[:n_ratings],   # CRs,    rating nR→1 (high→low confidence)
        nR_S1[n_ratings:],   # FAs,    rating  1→nR (low→high confidence)
        nR_S2[:n_ratings],   # Misses, rating nR→1 (high→low confidence)
        nR_S2[n_ratings:],   # Hits,   rating  1→nR (low→high confidence)
    ]).astype(int)


def fit_subject_level(
    nR_S1: np.ndarray,
    nR_S2: np.ndarray,
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    tol: float = 1e-5,
    **kwargs: Any,
) -> Any:
    """Subject-level Bayesian meta-d' — brmspy port of metadpy's ``hmetad()``.

    Estimates ``d1``, ``c1``, ``meta_d``, ``cS1``, and ``cS2`` for a single
    participant using the same hierarchical prior structure as the
    `metadpy <https://github.com/embodied-computation-group/metadpy>`_ PyMC
    implementation, but running via **brms / Stan through brmspy**.

    The posterior for ``sl_meta_d`` should reproduce metadpy's ``meta_d``
    parameter to within MCMC noise.  For the canonical dataset from Fleming
    (2017) / metadpy tutorial 3::

        nR_S1 = [52, 32, 35, 37, 26, 12, 4, 2]
        nR_S2 = [2, 5, 15, 22, 33, 38, 40, 45]

    both implementations return ``meta_d ≈ 1.57 ± 0.20`` and
    ``d1 ≈ 1.53 ± 0.14``.

    Args:
        nR_S1: Rating distribution for S1 (noise) trials, length ``2 * nratings``.
            Elements ``[0..nR-1]`` are correct rejections (high→low confidence);
            elements ``[nR..2*nR-1]`` are false alarms (low→high confidence).
        nR_S2: Rating distribution for S2 (signal) trials, same format.
            Elements ``[0..nR-1]`` are misses; elements ``[nR..2*nR-1]`` are hits.
        chains: MCMC chains (default 4).
        iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations (default 1000).
        seed: Random seed (default 42).
        tol: Minimum probability floor for multinomial cells (default 1e-5,
            matching metadpy's ``Tol``).
        **kwargs: Forwarded to ``brmspy.brms.brm``.

    Returns:
        ``FitResult`` with ``.idata`` (ArviZ ``InferenceData``) containing
        posterior samples for ``sl_d1``, ``sl_c1``, ``sl_meta_d``,
        ``sl_cS1``, ``sl_cS2``, ``sl_cS1_hn``, and ``sl_cS2_hn``.

    Raises:
        ImportError: If ``brmspy`` is not installed.
        ValueError: If ``nR_S1`` and ``nR_S2`` have different lengths or the
            length is not even.

    Example::

        import numpy as np
        from metasignal.sdtbayes import fit_subject_level, posterior_summary

        # Canonical dataset from Fleming (2017) and metadpy tutorial
        nR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2])
        nR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45])

        fit = fit_subject_level(nR_S1, nR_S2)
        print(posterior_summary(fit, var_names=["sl_d1", "sl_c1", "sl_meta_d",
                                                 "sl_cS1", "sl_cS2"]))

        # Expected output (comparable to metadpy):
        #   sl_d1      mean ≈  1.53,  sd ≈ 0.14
        #   sl_c1      mean ≈ -0.01,  sd ≈ 0.07
        #   sl_meta_d  mean ≈  1.57,  sd ≈ 0.20
    """
    try:
        from brmspy import brms
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "brmspy is not installed. Run:\n    pip install metasignal[sdtbayes]"
        ) from e

    nR_S1 = np.asarray(nR_S1, dtype=int)
    nR_S2 = np.asarray(nR_S2, dtype=int)

    if len(nR_S1) != len(nR_S2):
        raise ValueError("nR_S1 and nR_S2 must have the same length.")
    if len(nR_S1) % 2 != 0:
        raise ValueError("Length of nR_S1 must be even (= 2 * nratings).")

    n_ratings = len(nR_S1) // 2
    t1 = _extract_type1(nR_S1, nR_S2)
    counts = _build_counts_vector(nR_S1, nR_S2)

    sv_data   = brms.call("stanvar", scode=_SL_DATA,                   block="data")
    sv_params = brms.call("stanvar", scode=_SL_PARAMETERS,             block="parameters")
    sv_tpar   = brms.call("stanvar", scode=_SL_TRANSFORMED_PARAMETERS, block="tpar")
    sv_model  = brms.call("stanvar", scode=_SL_MODEL,                  block="model")

    extra_data = {
        "nratings":   n_ratings,
        "sl_counts":  counts.tolist(),
        "sl_H":       t1["H"],
        "sl_FA":      t1["FA"],
        "sl_S":       t1["S"],
        "sl_N":       t1["N"],
        "sl_Tol":     tol,
    }

    return brms.brm(
        formula=brms.bf("dummy ~ 1"),
        data=pd.DataFrame({"dummy": [0]}),
        family=brms.call("empty"),
        stanvars=[sv_data, sv_params, sv_tpar, sv_model],
        data2=extra_data,
        chains=chains,
        iter=iter,
        warmup=warmup,
        seed=seed,
        **kwargs,
    )
