"""Subject-level Bayesian meta-d' model — matches metadpy's hmetad() via cmdstanpy.

This module provides a single-participant Bayesian meta-d' estimation that
mirrors the API and model structure of
`metadpy.bayesian.hmetad(nR_S1=..., nR_S2=...)`.  :func:`fit_subject_level`
delegates to :func:`metasignal.sdtbayes.fit_meta_formula`
(``parameterization="mratio"``, ``backend="stan"``), which runs the
Fleming (2017) meta-d' model via **cmdstanpy** with a numerically stable
log-space likelihood and hard-by-construction ordered Type-2 criteria.

This module previously injected custom Stan code into brms via ``brmspy``;
that path is no longer used because the brmspy/rpy2 bridge cannot reliably
convert a list of multiple ``stanvar()`` objects into R (see
:func:`metasignal.sdtbayes.fit_full_metad` docstring for details).  The two
helper functions below (``_extract_type1``, ``_build_counts_vector``) are
retained as pure-Python utilities with their own test coverage.

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

from metasignal.sdtbayes.diagnostics import FitResult



def _extract_type1(
    nR_S1: np.ndarray, nR_S2: np.ndarray  # pylint: disable=invalid-name
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
    nR_S1: np.ndarray, nR_S2: np.ndarray  # pylint: disable=invalid-name
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


def fit_subject_level(  # pylint: disable=invalid-name
    nR_S1: np.ndarray,
    nR_S2: np.ndarray,
    chains: int = 4,
    n_iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    tol: float = 1e-5,
    **kwargs: Any,
) -> Any:
    """Subject-level Bayesian meta-d' — matches metadpy's ``hmetad()``.

    Estimates ``d1``, ``c1``, and ``meta_d`` for a single participant using
    the same fixed-prior model as metadpy's PyMC ``hmetad()`` implementation,
    but running via **cmdstanpy** (delegates to
    :func:`metasignal.sdtbayes.fit_meta_formula` with
    ``parameterization="mratio"``).

    This function previously injected custom Stan code into brms via
    ``brmspy``; that path is no longer used because the brmspy/rpy2 bridge
    cannot reliably convert a list of multiple ``stanvar()`` objects into R
    (see :func:`metasignal.sdtbayes.fit_full_metad` docstring for details).

    The posterior for ``meta_d`` should reproduce metadpy's ``meta_d``
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
        n_iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations (default 1000).
        seed: Random seed (default 42).
        tol: Minimum probability floor for multinomial cells (default 1e-5,
            matching metadpy's ``Tol``).
        **kwargs: Forwarded to :func:`fit_meta_formula`.

    Returns:
        ``FitResult`` with ``.idata`` (ArviZ ``InferenceData``) containing
        posterior samples for ``d1``, ``c1``, ``meta_d``, ``Mratio``.

    Raises:
        ImportError: If ``cmdstanpy`` is not installed.
        ValueError: If ``nR_S1`` and ``nR_S2`` have different lengths or the
            length is not even.

    Example::

        import numpy as np
        from metasignal.sdtbayes import fit_subject_level, posterior_summary

        # Canonical dataset from Fleming (2017) and metadpy tutorial
        nR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2])
        nR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45])

        fit = fit_subject_level(nR_S1, nR_S2)
        print(posterior_summary(fit, var_names=["d1", "c1", "meta_d"]))

        # Expected output (comparable to metadpy):
        #   d1      mean ≈  1.53,  sd ≈ 0.14
        #   c1      mean ≈ -0.01,  sd ≈ 0.07
        #   meta_d  mean ≈  1.57,  sd ≈ 0.20
    """
    nR_S1 = np.asarray(nR_S1, dtype=int)
    nR_S2 = np.asarray(nR_S2, dtype=int)

    if len(nR_S1) != len(nR_S2):
        raise ValueError("nR_S1 and nR_S2 must have the same length.")
    if len(nR_S1) % 2 != 0:
        raise ValueError("Length of nR_S1 must be even (= 2 * nratings).")

    import pandas as pd

    from metasignal.sdtbayes.formula import fit_meta_formula

    n_ratings = len(nR_S1) // 2

    # Expand counts back to trial-level (stim, resp, conf) arrays — exact
    # inverse of metasignal.stdpy.core.trials_to_counts.
    stims: list[int] = []
    resps: list[int] = []
    confs: list[int] = []
    for idx in range(n_ratings):
        conf = n_ratings - idx  # high confidence at idx=0
        n = int(nR_S1[idx])         # CR: S1 stim, S1 resp
        stims += [0] * n; resps += [0] * n; confs += [conf] * n
        n = int(nR_S2[idx])         # Miss: S2 stim, S1 resp
        stims += [1] * n; resps += [0] * n; confs += [conf] * n
    for idx in range(n_ratings):
        conf = idx + 1               # low confidence at idx=0
        n = int(nR_S1[n_ratings + idx])  # FA: S1 stim, S2 resp
        stims += [0] * n; resps += [1] * n; confs += [conf] * n
        n = int(nR_S2[n_ratings + idx])  # Hit: S2 stim, S2 resp
        stims += [1] * n; resps += [1] * n; confs += [conf] * n

    participants = [(np.array(stims), np.array(resps), np.array(confs))]
    pred_df = pd.DataFrame({"participant": [0]})
    return fit_meta_formula(
        participants=participants,
        n_ratings=n_ratings,
        formula="~ 1",
        data=pred_df,
        parameterization="mratio",
        backend="stan",
        chains=chains,
        n_iter=n_iter,
        warmup=warmup,
        seed=seed,
        tol=tol,
        **kwargs,
    )
