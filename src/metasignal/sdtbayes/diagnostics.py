"""FitResult and ArviZ-based diagnostics — shared vertical layer for all sdtbayes approaches.

All four estimation approaches (hierarchical, two_stage, full_metad, subject_level)
return a :class:`FitResult`.  Diagnostic methods are defined here once and
available on every fit object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class FitResult:
    """Container returned by every sdtbayes fitting function.

    Wraps the brmspy/Stan fit with ArviZ diagnostics as methods, so that
    every approach exposes the same interface::

        fit = fit_hierarchical_metad(participants, n_ratings=4)
        fit.posterior_summary()
        fit.plot_trace(var_names=["b_correct"])
        fit.convergence_diagnostics()

    Attributes:
        idata: ArviZ ``InferenceData`` with posterior samples, log-likelihood,
            and (if requested) prior predictive.
        r: Lightweight R object handle for downstream brms calls (e.g.
            ``brms.hypothesis(fit.r, "b_correct > 0")``).
    """

    idata: Any
    r: Any

    # ------------------------------------------------------------------
    # Diagnostics — one implementation, shared by all four approaches
    # ------------------------------------------------------------------

    def posterior_summary(
        self,
        var_names: Sequence[str] | None = None,
        *,
        hdi_prob: float = 0.94,
    ) -> Any:
        """Posterior summary table (mean, SD, HDI, R-hat, ESS).

        Args:
            var_names: Parameter names to include. ``None`` returns all.
            hdi_prob: Highest density interval probability mass (default 0.94).

        Returns:
            ``pandas.DataFrame`` with one row per parameter.

        Example::

            summary = fit.posterior_summary()
            print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat"]])
        """
        az = _require_arviz()
        return az.summary(self.idata, var_names=var_names, hdi_prob=hdi_prob)

    def convergence_diagnostics(self) -> Any:
        """R-hat and ESS for all parameters, with a ``converged`` flag.

        R-hat > 1.01 or ESS < 400 indicates sampling problems.

        Returns:
            ``pandas.DataFrame`` with columns ``r_hat``, ``ess_bulk``,
            ``ess_tail``, ``converged``.

        Example::

            diag = fit.convergence_diagnostics()
            print(diag[~diag["converged"]])   # non-converged parameters
        """
        az = _require_arviz()
        summary = az.summary(self.idata)
        diag = summary[["r_hat", "ess_bulk", "ess_tail"]].copy()
        diag["converged"] = (
            (diag["r_hat"] <= 1.01)
            & (diag["ess_bulk"] >= 400)
            & (diag["ess_tail"] >= 400)
        )
        return diag

    def plot_trace(
        self,
        var_names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """MCMC trace and rank plots for convergence inspection.

        Args:
            var_names: Parameters to plot. ``None`` plots all (can be slow).
            **kwargs: Forwarded to ``arviz.plot_trace``.

        Returns:
            Array of ``matplotlib.axes.Axes``.

        Example::

            fit.plot_trace(var_names=["b_correct", "sd_participant__correct"])
        """
        az = _require_arviz()
        return az.plot_trace(self.idata, var_names=var_names, **kwargs)

    def plot_posterior(
        self,
        var_names: Sequence[str] | None = None,
        *,
        hdi_prob: float = 0.94,
        ref_val: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Marginal posterior density plots with HDI.

        Args:
            var_names: Parameters to plot.
            hdi_prob: HDI probability mass to shade (default 0.94).
            ref_val: Optional reference line (e.g. ``0`` for a null).
            **kwargs: Forwarded to ``arviz.plot_posterior``.

        Returns:
            Array of ``matplotlib.axes.Axes``.

        Example::

            fit.plot_posterior(var_names=["b_correct:group1"], ref_val=0)
        """
        az = _require_arviz()
        kw: dict[str, Any] = {"hdi_prob": hdi_prob}
        if ref_val is not None:
            kw["ref_val"] = ref_val
        kw.update(kwargs)
        return az.plot_posterior(self.idata, var_names=var_names, **kw)

    def plot_forest(
        self,
        var_names: Sequence[str] | None = None,
        *,
        hdi_prob: float = 0.94,
        r_hat: bool = True,
        ess: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Forest plot of parameter estimates with HDI intervals.

        Args:
            var_names: Parameters to include. Defaults to all.
            hdi_prob: HDI probability mass (default 0.94).
            r_hat: Show R-hat values (default True).
            ess: Show effective sample size (default True).
            **kwargs: Forwarded to ``arviz.plot_forest``.

        Returns:
            Array of ``matplotlib.axes.Axes``.

        Example::

            fit.plot_forest(var_names=["b_correct", "r_participant"])
        """
        az = _require_arviz()
        return az.plot_forest(
            self.idata,
            var_names=var_names,
            hdi_prob=hdi_prob,
            r_hat=r_hat,
            ess=ess,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions (thin wrappers — kept for backwards compat)
# ---------------------------------------------------------------------------

def posterior_summary(
    fit: FitResult,
    var_names: Sequence[str] | None = None,
    *,
    hdi_prob: float = 0.94,
) -> Any:
    """Module-level alias for :meth:`FitResult.posterior_summary`."""
    return fit.posterior_summary(var_names=var_names, hdi_prob=hdi_prob)


def convergence_diagnostics(fit: FitResult) -> Any:
    """Module-level alias for :meth:`FitResult.convergence_diagnostics`."""
    return fit.convergence_diagnostics()


def plot_trace(
    fit: FitResult,
    var_names: Sequence[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Module-level alias for :meth:`FitResult.plot_trace`."""
    return fit.plot_trace(var_names=var_names, **kwargs)


def plot_posterior(
    fit: FitResult,
    var_names: Sequence[str] | None = None,
    *,
    hdi_prob: float = 0.94,
    ref_val: float | None = None,
    **kwargs: Any,
) -> Any:
    """Module-level alias for :meth:`FitResult.plot_posterior`."""
    return fit.plot_posterior(var_names=var_names, hdi_prob=hdi_prob, ref_val=ref_val, **kwargs)


def plot_forest(
    fit: FitResult,
    var_names: Sequence[str] | None = None,
    *,
    hdi_prob: float = 0.94,
    r_hat: bool = True,
    ess: bool = True,
    **kwargs: Any,
) -> Any:
    """Module-level alias for :meth:`FitResult.plot_forest`."""
    return fit.plot_forest(var_names=var_names, hdi_prob=hdi_prob, r_hat=r_hat, ess=ess, **kwargs)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _require_arviz() -> Any:
    try:
        import arviz as az
        return az
    except ImportError as e:
        raise ImportError(
            "arviz is not installed. Run:\n    pip install metasignal[sdtbayes]"
        ) from e
