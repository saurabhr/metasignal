"""Variational inference wrappers for the full HMeta-d models.

Provides fast approximate posteriors using Stan's built-in variational
inference algorithms.  Typical runtime is seconds vs. minutes for HMC/NUTS,
at the cost of an approximate (rather than asymptotically exact) posterior.

Supported algorithms
--------------------
``"pathfinder"`` (default)
    Stan's Pathfinder algorithm.  Quasi-Newton optimisation traces the log-
    posterior landscape and fits a Gaussian approximation along the path.
    Substantially more accurate than classical VI while remaining very fast.
    Recommended for pilot analyses and large datasets.

``"meanfield"``
    Mean-field ADVI.  Fastest but assumes all parameters are independent in
    the posterior — underestimates correlations.

``"fullrank"``
    Full-rank ADVI.  Captures correlations but is slower than meanfield and
    can be numerically unstable for high-dimensional models.

Caveats
-------
- R-hat and ESS convergence diagnostics do **not** apply to VI output.
- For final publication results, confirm key conclusions with MCMC
  (``fit_full_metad`` or ``fit_robust_metad``).
- ``chains`` and ``warmup`` are ignored by Stan for VI; this wrapper sets
  ``chains=1, warmup=0`` automatically.

References
----------
Zhang, L. et al. (2022). Pathfinder: Parallel quasi-Newton variational
inference. *Journal of Machine Learning Research*, 23(1).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from metasignal.sdtbayes.diagnostics import FitResult

_Algorithm = Literal["pathfinder", "meanfield", "fullrank"]


def fit_full_metad_vi(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    algorithm: _Algorithm = "pathfinder",
    n_iter: int = 1000,
    seed: int = 42,
    **kwargs: Any,
) -> FitResult:
    """Full HMeta-d (Fleming 2017) via variational inference.

    Equivalent to :func:`~metasignal.sdtbayes.fit_full_metad` but uses VI
    instead of HMC/NUTS.  The same Stan model and priors are used; only the
    inference algorithm changes.

    Args:
        participants: List of ``(stim, resp, conf)`` tuples, one per participant.
        n_ratings: Number of confidence rating categories.
        algorithm: VI algorithm — ``"pathfinder"`` (default), ``"meanfield"``,
            or ``"fullrank"``.
        n_iter: VI iterations (default 1000).  For Pathfinder this is the number
            of draws from the approximation; for ADVI it is gradient steps.
        seed: Random seed (default 42).
        **kwargs: Forwarded to ``brmspy.brms.brm`` (e.g. ``tol=1e-7``).

    Returns:
        ``FitResult``.  ``.idata`` contains a single "chain" of approximate
        posterior draws.  Key parameters are identical to ``fit_full_metad``:
        ``mu_logMratio``, ``sigma_logMratio``, ``Mratio``, ``meta_d``.

    Raises:
        NotImplementedError: Always — see Notes.

    Notes:
        :func:`~metasignal.sdtbayes.fit_full_metad` now runs via **cmdstanpy**
        (see its docstring for why the brmspy path was replaced).
        ``cmdstanpy.CmdStanModel`` exposes VI through dedicated methods
        (``.pathfinder()``, ``.variational()``), not through an ``algorithm=``
        keyword on ``.sample()``, so this wrapper's approach of forwarding
        ``algorithm=`` no longer applies and would silently be rejected or
        misinterpreted by the sampling call.  Genuine cmdstanpy-backed VI
        support has not been implemented.  Use
        :func:`~metasignal.sdtbayes.fit_full_metad` (full MCMC) instead — with
        the numerically stable log-space likelihood it now uses, typical
        group models run in well under a minute.

    Example::

        import numpy as np
        from metasignal.sdtbayes import fit_full_metad_vi

        rng = np.random.default_rng(0)
        participants = [
            (rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.integers(1, 5, 200))
            for _ in range(20)
        ]
        # Approximate posterior in seconds
        fit = fit_full_metad_vi(participants, n_ratings=4)
        print(fit.posterior_summary(var_names=["mu_logMratio", "sigma_logMratio"]))
    """
    raise NotImplementedError(
        "fit_full_metad_vi is currently unavailable: fit_full_metad now runs via "
        "cmdstanpy, which exposes variational inference through dedicated methods "
        "(.pathfinder(), .variational()) rather than an algorithm= kwarg on "
        ".sample(). Use fit_full_metad(...) for full MCMC instead — it is fast "
        "enough for most use cases with the current numerically stable likelihood."
    )


def fit_robust_metad_vi(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    algorithm: _Algorithm = "pathfinder",
    n_iter: int = 1000,
    seed: int = 42,
    **kwargs: Any,
) -> FitResult:
    """Robust HMeta-d (Student-t hyperprior) via variational inference.

    Combines the outlier-robustness of :func:`~metasignal.sdtbayes.fit_robust_metad`
    with the speed of variational inference.  Useful for quick sensitivity
    checks: if ``nu_logMratio`` is large under VI, the robust and standard
    models are likely to agree; if it is small (< 10), full MCMC is warranted.

    Args:
        participants: List of ``(stim, resp, conf)`` tuples, one per participant.
        n_ratings: Number of confidence rating categories.
        algorithm: VI algorithm — ``"pathfinder"`` (default), ``"meanfield"``,
            or ``"fullrank"``.
        n_iter: VI iterations (default 1000).
        seed: Random seed (default 42).
        **kwargs: Forwarded to ``brmspy.brms.brm``.

    Returns:
        ``FitResult`` with approximate posterior.  Key parameters:
        ``mu_logMratio``, ``sigma_logMratio``, ``nu_logMratio``.

    Raises:
        NotImplementedError: Always — see Notes.

    Notes:
        Unavailable for two independent reasons: (1) the same cmdstanpy/VI
        method mismatch documented in :func:`fit_full_metad_vi`, and (2) the
        underlying :func:`~metasignal.sdtbayes.fit_robust_metad` itself is
        blocked by an upstream brmspy stanvar-injection limitation (see its
        docstring).

    Example::

        fit = fit_robust_metad_vi(participants, n_ratings=4)
        import arviz as az
        nu = az.extract(fit.idata)["nu_logMratio"].values
        print(f"nu_logMratio ≈ {nu.mean():.1f}")
    """
    raise NotImplementedError(
        "fit_robust_metad_vi is currently unavailable: fit_robust_metad is blocked "
        "by an upstream brmspy stanvar-injection limitation, and cmdstanpy VI "
        "requires dedicated methods (.pathfinder(), .variational()) rather than an "
        "algorithm= kwarg on .sample(). Use fit_full_metad(...) for the standard "
        "(non-robust) model via full MCMC instead."
    )
