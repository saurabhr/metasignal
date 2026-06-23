"""ArviZ-based posterior analysis and diagnostics for metasignal Bayesian fits.

All functions accept a ``FitResult`` returned by
:func:`~metasignal.sdtbayes.fit_hierarchical_metad` or
:func:`~metasignal.sdtbayes.fit_group_comparison` and delegate to ArviZ
for computation and plotting.
"""

from __future__ import annotations

from typing import Any, Sequence


def posterior_summary(
    fit: Any,
    var_names: Sequence[str] | None = None,
    *,
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    """Posterior summary table for all (or selected) parameters.

    Returns mean, standard deviation, 94% HDI bounds, Monte Carlo standard
    error, R-hat, and bulk/tail effective sample sizes — matching the
    conventions used in the brms ``summary()`` output.

    Args:
        fit: ``FitResult`` from :func:`fit_hierarchical_metad` or
            :func:`fit_group_comparison`.
        var_names: Parameter names to include. ``None`` returns all parameters.
        hdi_prob: Highest density interval probability mass (default 0.94).

    Returns:
        ``pandas.DataFrame`` with one row per parameter.

    Example::

        summary = posterior_summary(fit)
        print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat"]])
    """
    try:
        import arviz as az
    except ImportError as e:
        raise ImportError(
            "arviz is not installed. Run:\n    pip install metasignal[sdtbayes]"
        ) from e

    import pandas as pd  # noqa: F401  (az.summary returns a DataFrame; ensure pandas is present)
    return az.summary(fit.idata, var_names=var_names, hdi_prob=hdi_prob)


def plot_trace(
    fit: Any,
    var_names: Sequence[str] | None = None,
    **kwargs: Any,
) -> Any:
    """MCMC trace and rank plots for convergence diagnostics.

    Displays the sampled parameter values over iterations (trace) and their
    rank-normalised values (rank plot).  Healthy chains show well-mixed
    traces and uniform rank distributions.

    Args:
        fit: ``FitResult`` from a hierarchical model fit.
        var_names: Parameters to plot. ``None`` plots all parameters
            (can be slow for large models).
        **kwargs: Forwarded to ``arviz.plot_trace``.

    Returns:
        Array of ``matplotlib.axes.Axes``.

    Example::

        plot_trace(fit, var_names=["b_correct", "sd_participant__correct"])
    """
    try:
        import arviz as az
    except ImportError as e:
        raise ImportError(
            "arviz is not installed. Run:\n    pip install metasignal[sdtbayes]"
        ) from e

    return az.plot_trace(fit.idata, var_names=var_names, **kwargs)


def plot_posterior(
    fit: Any,
    var_names: Sequence[str] | None = None,
    *,
    hdi_prob: float = 0.94,
    ref_val: float | None = None,
    **kwargs: Any,
) -> Any:
    """Marginal posterior density plots with HDI and optional reference value.

    Args:
        fit: ``FitResult`` from a hierarchical model fit.
        var_names: Parameters to plot. Defaults to population-level effects.
        hdi_prob: HDI probability mass to shade (default 0.94).
        ref_val: Optional vertical reference line (e.g. 0 for a null hypothesis).
        **kwargs: Forwarded to ``arviz.plot_posterior``.

    Returns:
        Array of ``matplotlib.axes.Axes``.

    Example::

        # For group comparison — is b_correct:group1 credibly different from 0?
        plot_posterior(fit, var_names=["b_correct:group1"], ref_val=0)
    """
    try:
        import arviz as az
    except ImportError as e:
        raise ImportError(
            "arviz is not installed. Run:\n    pip install metasignal[sdtbayes]"
        ) from e

    kw: dict[str, Any] = {"hdi_prob": hdi_prob}
    if ref_val is not None:
        kw["ref_val"] = ref_val
    kw.update(kwargs)
    return az.plot_posterior(fit.idata, var_names=var_names, **kw)


def plot_forest(
    fit: Any,
    var_names: Sequence[str] | None = None,
    *,
    hdi_prob: float = 0.94,
    r_hat: bool = True,
    ess: bool = True,
    **kwargs: Any,
) -> Any:
    """Forest plot of parameter estimates with HDI intervals.

    Useful for visualising participant-level random effects alongside
    the group-level (population) estimate in a single panel.

    Args:
        fit: ``FitResult`` from a hierarchical model fit.
        var_names: Parameters to include. Defaults to all.
        hdi_prob: HDI probability mass (default 0.94).
        r_hat: Show R-hat values alongside estimates (default True).
        ess: Show effective sample size alongside estimates (default True).
        **kwargs: Forwarded to ``arviz.plot_forest``.

    Returns:
        Array of ``matplotlib.axes.Axes``.

    Example::

        # Show group-level effect and all participant random slopes
        plot_forest(fit, var_names=["b_correct", "r_participant"])
    """
    try:
        import arviz as az
    except ImportError as e:
        raise ImportError(
            "arviz is not installed. Run:\n    pip install metasignal[sdtbayes]"
        ) from e

    return az.plot_forest(
        fit.idata,
        var_names=var_names,
        hdi_prob=hdi_prob,
        r_hat=r_hat,
        ess=ess,
        **kwargs,
    )


def convergence_diagnostics(fit: Any) -> pd.DataFrame:
    """Return R-hat and effective sample size (ESS) for all parameters.

    R-hat values above 1.01 indicate poor convergence; ESS values below
    ~400 indicate insufficient sampling.  Both thresholds follow the
    recommendations of Vehtari et al. (2021).

    Args:
        fit: ``FitResult`` from a hierarchical model fit.

    Returns:
        ``pandas.DataFrame`` with columns ``r_hat``, ``ess_bulk``,
        ``ess_tail``, indexed by parameter name.  Parameters exceeding
        the R-hat threshold are flagged in a ``converged`` column.

    Example::

        diag = convergence_diagnostics(fit)
        print(diag[~diag["converged"]])   # show any non-converged parameters
    """
    try:
        import arviz as az
    except ImportError as e:
        raise ImportError(
            "arviz is not installed. Run:\n    pip install metasignal[sdtbayes]"
        ) from e

    import pandas as pd  # noqa: F401  (az.summary returns a DataFrame; ensure pandas is present)
    summary = az.summary(fit.idata)
    diag = summary[["r_hat", "ess_bulk", "ess_tail"]].copy()
    diag["converged"] = diag["r_hat"] <= 1.01
    return diag
