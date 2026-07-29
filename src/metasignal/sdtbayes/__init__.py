"""Bayesian metacognition models via cmdstanpy and brms (through brmspy).

This sub-package fits hierarchical Bayesian models over the SDT measures
computed by ``metasignal.stdpy``.  It requires an optional install:

    pip install "metasignal[sdtbayes] @ git+https://github.com/saurabhr/metasignal.git"

Two backends are in use, depending on the approach:

- **cmdstanpy** (recommended, no R required) — used by
  :func:`fit_meta_formula` and the approaches redirected to it below.  Install
  CmdStan once with ``python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"``.
- **brms via brmspy** (requires R) — used by the plain-regression approaches
  below, which need a one-time runtime setup::

    from metasignal.sdtbayes import setup_runtime
    setup_runtime()   # downloads and installs brms + Stan (~5 min, once only)

Working approaches
-------------------
Approach 1 — Ordered logistic (``hierarchical``) — *brms*
    Trial-level model.  Confidence ratings are the outcome in a cumulative
    logistic regression; meta-d' is captured by the ``correct`` predictor.
    Supports crossed item random effects via the ``items`` argument.

Approach 2 — Two-stage Bayesian (``two_stage``) — *brms*
    Stage 1: per-participant MLE meta-d' via ``stdpy``.
    Stage 2: hierarchical Bayesian model over log M-ratio using brms.
    Very fast; does not propagate Stage-1 uncertainty.

Approach 3 — Full HMeta-d (``full_metad``) — *cmdstanpy*
    Single-stage fully Bayesian port of Fleming (2017), delegating to
    :func:`fit_meta_formula`.  Numerically stable log-space likelihood,
    ``positive_ordered`` Type-2 criteria, Type-1 binomial likelihood.

Approach 4 — Subject-level Bayesian (``subject_level``) — *cmdstanpy*
    Single-participant model matching metadpy's ``hmetad()`` API.  Reproduces
    metadpy's canonical-dataset reference values to within MCMC noise.

Approach 7 — Beta regression on AUC2 (``beta_auc``) — *brms*
    Non-parametric alternative to meta-d'.  Models Type-2 AUC directly
    using a Beta likelihood; no Gaussian SDT assumption.

Approach 10 — Bayesian meta-regression (``meta_regression``) — *brms + cmdstanpy*
    Regresses log M-ratio on participant-level covariates (age, scores, etc.)
    via either the two-stage (brms) or full hierarchical (cmdstanpy) path.

Approach 12 — Within-subject condition comparison (``within_subject``) — *brms*
    Paired model for within-subject designs: same participants tested in two
    conditions.  Participant random intercepts absorb stable between-subject
    differences; ``b_condition1`` is the within-person condition effect.

Approaches 5, 6, 8, 9, and 11 (robust HMeta-d, variational inference,
Gaussian mixture, bivariate hierarchical, and state-space models) were
removed — each was permanently blocked by an upstream brmspy/rpy2 limitation
with no working implementation. See :doc:`roadmap </roadmap>` for status.

Public API
----------
Runtime
    ``setup_runtime`` (only needed for the brms-backed approaches above)

Approach 1 — ordered logistic
    ``fit_hierarchical_metad``, ``fit_group_comparison``

Approach 2 — two-stage
    ``fit_two_stage_group``, ``fit_two_stage_comparison``

Approach 3 — full HMeta-d
    ``fit_full_metad``, ``fit_full_metad_comparison``

Approach 4 — subject-level
    ``fit_subject_level``

Approach 7 — Beta AUC
    ``fit_beta_auc_group``, ``fit_beta_auc_comparison``

Approach 10 — meta-regression
    ``fit_two_stage_regression``, ``fit_full_metad_regression``

Approach 12 — within-subject comparison
    ``fit_within_subject_comparison``

Formula interface
    ``fit_meta_formula`` — the cmdstanpy entry point used by Approaches 3, 4,
    and 10's full path.  Supports ``parameterization="mratio"|"meta_noise"|
    "casandre"``.

Diagnostics
    ``FitResult``, ``posterior_summary``, ``plot_trace``, ``plot_posterior``,
    ``plot_forest``, ``convergence_diagnostics``
"""

from metasignal.sdtbayes.hierarchical import fit_hierarchical_metad, fit_group_comparison
from metasignal.sdtbayes.two_stage import fit_two_stage_group, fit_two_stage_comparison
from metasignal.sdtbayes.full_metad import fit_full_metad, fit_full_metad_comparison
from metasignal.sdtbayes.subject_level import fit_subject_level
from metasignal.sdtbayes.beta_auc import fit_beta_auc_group, fit_beta_auc_comparison
from metasignal.sdtbayes.meta_regression import (
    fit_two_stage_regression,
    fit_full_metad_regression,
)
from metasignal.sdtbayes.within_subject import (
    fit_within_subject_comparison,
)
from metasignal.sdtbayes.formula import fit_meta_formula
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
    # Approach 1 — ordered logistic (+ crossed item effects via items=)
    "fit_hierarchical_metad",
    "fit_group_comparison",
    # Approach 2 — two-stage Bayesian
    "fit_two_stage_group",
    "fit_two_stage_comparison",
    # Approach 3 — full HMeta-d (Fleming 2017)
    "fit_full_metad",
    "fit_full_metad_comparison",
    # Approach 4 — subject-level (metadpy API, Stan backend)
    "fit_subject_level",
    # Approach 7 — Beta regression on Type-2 AUC
    "fit_beta_auc_group",
    "fit_beta_auc_comparison",
    # Approach 10 — Bayesian meta-regression (two paths)
    "fit_two_stage_regression",
    "fit_full_metad_regression",
    # Approach 12 — within-subject paired condition comparison
    "fit_within_subject_comparison",
    # Formula interface (Stan or brms backend)
    "fit_meta_formula",
    # Diagnostics
    "posterior_summary",
    "plot_trace",
    "plot_posterior",
    "plot_forest",
    "convergence_diagnostics",
]
