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
    iter: int = 1000,
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
        iter: VI iterations (default 1000).  For Pathfinder this is the number
            of draws from the approximation; for ADVI it is gradient steps.
        seed: Random seed (default 42).
        **kwargs: Forwarded to ``brmspy.brms.brm`` (e.g. ``tol=1e-7``).

    Returns:
        ``FitResult``.  ``.idata`` contains a single "chain" of approximate
        posterior draws.  Key parameters are identical to ``fit_full_metad``:
        ``mu_logMratio``, ``sigma_logMratio``, ``Mratio``, ``meta_d``.

    Raises:
        ImportError: If ``brmspy`` is not installed.

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
    from metasignal.sdtbayes.full_metad import fit_full_metad

    return fit_full_metad(
        participants,
        n_ratings,
        chains=1,
        iter=iter,
        warmup=0,
        seed=seed,
        algorithm=algorithm,
        **kwargs,
    )


def fit_robust_metad_vi(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    algorithm: _Algorithm = "pathfinder",
    iter: int = 1000,
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
        iter: VI iterations (default 1000).
        seed: Random seed (default 42).
        **kwargs: Forwarded to ``brmspy.brms.brm``.

    Returns:
        ``FitResult`` with approximate posterior.  Key parameters:
        ``mu_logMratio``, ``sigma_logMratio``, ``nu_logMratio``.

    Raises:
        ImportError: If ``brmspy`` is not installed.

    Example::

        fit = fit_robust_metad_vi(participants, n_ratings=4)
        import arviz as az
        nu = az.extract(fit.idata)["nu_logMratio"].values
        print(f"nu_logMratio ≈ {nu.mean():.1f}")
    """
    from metasignal.sdtbayes.robust import fit_robust_metad

    return fit_robust_metad(
        participants,
        n_ratings,
        chains=1,
        iter=iter,
        warmup=0,
        seed=seed,
        algorithm=algorithm,
        **kwargs,
    )
