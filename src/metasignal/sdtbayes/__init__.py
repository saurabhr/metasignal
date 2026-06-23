"""Bayesian metacognition models using brms (via brmspy) and ArviZ.

This sub-package fits hierarchical Bayesian models over the SDT measures
computed by ``metasignal.stdpy``.  It requires an optional install:

    pip install metasignal[sdtbayes]

and a one-time brms/Stan runtime setup::

    from metasignal.sdtbayes import setup_runtime
    setup_runtime()   # downloads and installs brms + Stan (~5 min, once only)

Four modelling approaches are provided, ordered by fidelity and complexity:

Approach 1 — Ordered logistic (``hierarchical``)
    Trial-level model.  Confidence ratings are the outcome in a cumulative
    logistic regression; meta-d' is captured by the ``correct`` predictor.
    Fast, extensible with brms syntax, but the mapping to meta-d' is
    approximate (logistic scale, not Gaussian-SDT scale).

Approach 2 — Two-stage Bayesian (``two_stage``)
    Stage 1: per-participant MLE meta-d' via ``stdpy``.
    Stage 2: hierarchical Bayesian model over log M-ratio using brms.
    Very fast; does not propagate Stage-1 uncertainty but adequate for
    typical sample sizes (20–50 participants).

Approach 3 — Full HMeta-d (``full_metad``)
    Single-stage fully Bayesian port of Fleming (2017).  Custom Stan code
    injected into brms via ``stanvar()``.  Uses the identical SDT
    multinomial likelihood as the MLE code; group-level posterior over
    log M-ratio with per-subject Type-2 criteria.  Slowest, most accurate.

Approach 4 — Subject-level Bayesian (``subject_level``)
    Single-participant model matching metadpy's ``hmetad(nR_S1, nR_S2)``
    API and prior structure but running via brmspy/Stan instead of PyMC.
    Estimates d1, c1, meta_d and full Type-2 criteria from a single
    participant's rating distribution.

Public API
----------
- ``setup_runtime``              — install brms/Stan runtime (one-time)
- ``fit_hierarchical_metad``     — ordered logistic hierarchical meta-d'
- ``fit_group_comparison``       — ordered logistic two-group comparison
- ``fit_two_stage_group``        — two-stage Bayesian group-level M-ratio
- ``fit_two_stage_comparison``   — two-stage Bayesian group comparison
- ``fit_full_metad``             — full HMeta-d (Fleming 2017 Stan port)
- ``fit_full_metad_comparison``  — full HMeta-d two-group comparison
- ``fit_subject_level``          — subject-level Bayesian (metadpy API, Stan backend)
- ``posterior_summary``          — ArviZ summary table (mean, SD, HDI, R-hat, ESS)
- ``plot_trace``                 — MCMC trace and rank plots
- ``plot_posterior``             — marginal posterior densities
- ``plot_forest``                — forest plot of group and participant effects
- ``convergence_diagnostics``    — R-hat and ESS dataframe for all parameters
"""

from metasignal.sdtbayes.hierarchical import fit_hierarchical_metad, fit_group_comparison
from metasignal.sdtbayes.two_stage import fit_two_stage_group, fit_two_stage_comparison
from metasignal.sdtbayes.full_metad import fit_full_metad, fit_full_metad_comparison
from metasignal.sdtbayes.subject_level import fit_subject_level
from metasignal.sdtbayes.diagnostics import (
    FitResult,
    posterior_summary,
    plot_trace,
    plot_posterior,
    plot_forest,
    convergence_diagnostics,
)
from metasignal.sdtbayes._runtime import setup_runtime

__all__ = [
    "setup_runtime",
    "FitResult",
    # Approach 1 — ordered logistic
    "fit_hierarchical_metad",
    "fit_group_comparison",
    # Approach 2 — two-stage Bayesian
    "fit_two_stage_group",
    "fit_two_stage_comparison",
    # Approach 3 — full HMeta-d
    "fit_full_metad",
    "fit_full_metad_comparison",
    # Approach 4 — subject-level (metadpy API, Stan backend)
    "fit_subject_level",
    # Diagnostics and posteriors
    "posterior_summary",
    "plot_trace",
    "plot_posterior",
    "plot_forest",
    "convergence_diagnostics",
]
