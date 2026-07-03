"""Gaussian mixture model on log M-ratio.

Two-stage approach: per-participant log M-ratio estimated via MLE (Stage 1),
then modelled as a K-component Gaussian mixture via brms (Stage 2).

The mixture model is suited for datasets where participants may belong to
distinct metacognitive subpopulations — for example a clinical sample in
which some individuals have near-zero metacognitive sensitivity and others
have intact monitoring.  Each component has its own mean and SD; the mixing
weight ``theta1`` is the estimated proportion of participants in component 1.

Key posterior parameters
------------------------
- ``mu1_logmratio``, ``mu2_logmratio`` — component-specific mean log M-ratio.
  Convert to M-ratio: ``exp(mu1_logmratio)``.
- ``sigma1_logmratio``, ``sigma2_logmratio`` — within-component spread.
- ``theta1`` — mixing weight (proportion of participants in component 1).
  ``theta2 = 1 − theta1``.

Label-switching note
--------------------
Gaussian mixture posteriors are invariant to relabelling of components.
After fitting, label the components by their posterior mean: the component
with the smaller mean is the "low metacognition" group.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from metasignal.sdtbayes.diagnostics import FitResult
from metasignal.sdtbayes.two_stage import _compute_participant_estimates

_BRMSPY_MSG = "brmspy is not installed. Run:\n    pip install metasignal[sdtbayes]"


def fit_mixture_group(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    n_components: int = 2,
    chains: int = 4,
    n_iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    **kwargs: Any,
) -> FitResult:
    """Gaussian mixture model on log M-ratio for a single group.

    Uses MLE to compute per-participant log M-ratio (Stage 1), then fits a
    K-component Gaussian mixture via brms's ``mixture()`` family (Stage 2).

    Args:
        participants: List of ``(stim, resp, conf)`` tuples, one per participant.
        n_ratings: Number of confidence rating categories.
        n_components: Number of Gaussian mixture components (default 2).
        chains: MCMC chains (default 4).
        n_iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations (default 1000).
        seed: Random seed (default 42).
        **kwargs: Forwarded to ``brmspy.brms.brm``.

    Returns:
        ``FitResult``.  Key parameters (for ``n_components=2``):
        ``mu1_logmratio``, ``mu2_logmratio`` (component means),
        ``sigma1_logmratio``, ``sigma2_logmratio`` (component SDs),
        ``theta1`` (mixing weight for component 1).

    Raises:
        RuntimeError: Always — see Notes.
        ValueError: If fewer than ``n_components × 3`` participants have valid
            MLE estimates.

    Notes:
        This function's brmspy path is currently **non-functional**: the
        mixture family object returned by ``brms.call("mixture", ...)`` loses
        its required R S3 class (``mixfamily``) when it round-trips through
        Python back into ``brm(family=...)``, causing ``validate_family()``
        to reject it with "Argument 'family' is invalid".  This is the same
        category of upstream brmspy/rpy2 limitation documented in
        :func:`~metasignal.sdtbayes.fit_full_metad` (there for ``stanvar()``
        objects; here for the family object) — any special R object with a
        non-trivial S3 class appears to lose that class across the Python
        round-trip, and no workaround was found.

        There is currently no cmdstanpy-backed replacement for Gaussian
        mixture modelling of log M-ratio.  If you suspect distinct
        metacognitive subpopulations, consider fitting
        :func:`~metasignal.sdtbayes.fit_full_metad` separately to
        data-driven subgroups instead.

    Example::

        import numpy as np
        import arviz as az
        from metasignal.sdtbayes import fit_mixture_group

        rng = np.random.default_rng(0)
        participants = [
            (rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.integers(1, 5, 200))
            for _ in range(40)
        ]
        fit = fit_mixture_group(participants, n_ratings=4, n_components=2)

        post = az.extract(fit.idata)
        mu1 = post["mu1_logmratio"].values.mean()
        mu2 = post["mu2_logmratio"].values.mean()
        theta1 = post["theta1"].values.mean()

        # Label by magnitude — component 1 is the higher-metacognition group
        low, high = sorted([mu1, mu2])
        print(f"Low-metacognition component  M-ratio ≈ {np.exp(low):.2f}")
        print(f"High-metacognition component M-ratio ≈ {np.exp(high):.2f}")
        print(f"Mixing weight theta1 ≈ {theta1:.2f}")
    """
    mle_df = _compute_participant_estimates(participants, n_ratings)
    valid = mle_df.dropna(subset=["log_m_ratio"])

    min_n = n_components * 3
    if len(valid) < min_n:
        msg = (
            f"Only {len(valid)} participants have valid estimates — "
            f"need at least {min_n} for a {n_components}-component mixture."
        )
        raise ValueError(msg)

    raise RuntimeError(
        "fit_mixture_group is currently unavailable: the brms mixture family "
        "object loses its required R S3 class when round-tripped through brmspy, "
        "causing brm() to reject it with \"Argument 'family' is invalid\". This is "
        "an upstream brmspy/rpy2 limitation with no known workaround. There is no "
        "cmdstanpy-backed replacement for Gaussian mixture modelling of log "
        "M-ratio currently."
    )
    return FitResult(idata=_result.idata, r=_result.r)
