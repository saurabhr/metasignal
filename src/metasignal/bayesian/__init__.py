"""Bayesian metacognition models using brms (via brmspy) and ArviZ.

This sub-package fits hierarchical Bayesian models over the SDT measures
computed by ``metasignal.stdpy``.  It requires an optional install:

    pip install metasignal[bayesian]

and a one-time brms/Stan runtime setup::

    from metasignal.bayesian import setup_runtime
    setup_runtime()   # downloads and installs brms + Stan (~5 min, once only)

Public API
----------
- ``setup_runtime``       — install the brms/Stan runtime (one-time)
- ``fit_hierarchical_metad`` — hierarchical Bayesian meta-d' across participants
- ``fit_group_comparison``   — Bayesian test for metacognition differences between two groups
- ``posterior_summary``      — ArviZ summary table (mean, SD, HDI, R-hat, ESS)
- ``plot_trace``             — MCMC trace and rank plots
- ``plot_posterior``         — marginal posterior densities
- ``plot_forest``            — forest plot of group and participant effects
- ``convergence_diagnostics``— R-hat and ESS dataframe for all parameters
"""

from metasignal.bayesian.hierarchical import fit_hierarchical_metad, fit_group_comparison
from metasignal.bayesian.diagnostics import (
    posterior_summary,
    plot_trace,
    plot_posterior,
    plot_forest,
    convergence_diagnostics,
)
from metasignal.bayesian._runtime import setup_runtime

__all__ = [
    "setup_runtime",
    "fit_hierarchical_metad",
    "fit_group_comparison",
    "posterior_summary",
    "plot_trace",
    "plot_posterior",
    "plot_forest",
    "convergence_diagnostics",
]
