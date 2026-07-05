"""Robust hierarchical meta-d' with Student-t hyperprior on log M-ratio.

Identical to the full HMeta-d model (``full_metad.py``) except that the
per-subject log M-ratio deviations follow a Student-t distribution instead
of a standard normal.  This downweights participants whose MLE estimates are
far from the group mean, protecting ``mu_logMratio`` against outliers.

The degrees-of-freedom parameter ``nu_logMratio`` is estimated from the data
with a ``Gamma(2, 0.1)`` prior.  When ``nu_logMratio`` is large (≥ 30) the
posterior converges to the Gaussian model; small values (2–5) indicate heavy
tails and signal the presence of outlier participants.

Key posterior parameters
------------------------
- ``mu_logMratio`` — robust group mean log M-ratio (same interpretation as
  ``full_metad``; ``exp(mu_logMratio)`` = group M-ratio).
- ``sigma_logMratio`` — scale of between-subject variability.
- ``nu_logMratio`` — estimated degrees of freedom (check this: values < 10
  mean outliers are influencing the model non-trivially).
- ``delta_logMratio`` (comparison only) — group B − group A contrast.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from metasignal.sdtbayes.diagnostics import FitResult
from metasignal.sdtbayes.full_metad import (
    _STAN_DATA,
    _STAN_DATA_TWO_GROUP,
    _STAN_TRANSFORMED_PARAMETERS,
    _STAN_TRANSFORMED_PARAMETERS_TWO_GROUP,
    _build_count_matrix,
    _group_likelihood_block,
)

_BRMSPY_MSG = "brmspy is not installed. Run:\n    pip install metasignal[sdtbayes]"

# ---------------------------------------------------------------------------
# Robust Stan blocks (single group)
# Re-uses _STAN_DATA and _STAN_TRANSFORMED_PARAMETERS from full_metad.
# Only the parameters block and model block change.
# ---------------------------------------------------------------------------

_ROBUST_PARAMETERS = """\
// --- Group-level Type-1 parameters ---
real mu_d1;
real<lower=0> sigma_d1;
vector[nsubj] d1_z;

real mu_c1;
real<lower=0> sigma_c1;
vector[nsubj] c1_z;

// --- Group-level M-ratio (log scale, Student-t non-centered) ---
real mu_logMratio;
real<lower=0> sigma_logMratio;
real<lower=1> nu_logMratio;    // degrees of freedom; >1 ensures finite mean
vector[nsubj] logMratio_z;     // non-centred auxiliary (t-distributed)

// --- Group-level Type-2 criterion hyperparameters ---
real<lower=0> mu_c2;
real<lower=0> sigma_c2;

// --- Per-subject Type-2 criteria ---
array[nsubj] ordered[nratings - 1] cS1_raw;
array[nsubj] ordered[nratings - 1] cS2_raw;
"""

_ROBUST_MODEL = (
    """\
mu_d1           ~ normal(1, 2);
sigma_d1        ~ exponential(1);
d1_z            ~ normal(0, 1);

mu_c1           ~ normal(0, 1);
sigma_c1        ~ exponential(1);
c1_z            ~ normal(0, 1);

mu_logMratio    ~ normal(0, 1);
sigma_logMratio ~ exponential(1);
nu_logMratio    ~ gamma(2, 0.1);
logMratio_z     ~ student_t(nu_logMratio, 0, 1);

mu_c2           ~ normal(1, 1);
sigma_c2        ~ exponential(1);

for (s in 1:nsubj) {
    cS1_raw[s] ~ normal(c1[s] - mu_c2, sigma_c2);
    cS2_raw[s] ~ normal(c1[s] + mu_c2, sigma_c2);
}
"""
    + _group_likelihood_block(
        "nsubj", "hmetad_counts", "Mratio", "d1", "c1", "cS1_raw", "cS2_raw"
    )
)

# ---------------------------------------------------------------------------
# Robust Stan blocks (two groups)
# ---------------------------------------------------------------------------

_ROBUST_PARAMETERS_TWO_GROUP = """\
// --- Type-1 parameters (shared across groups) ---
real mu_d1;
real<lower=0> sigma_d1;
vector[nsubj_a + nsubj_b] d1_z;

real mu_c1;
real<lower=0> sigma_c1;
vector[nsubj_a + nsubj_b] c1_z;

// --- Group-level M-ratio (log scale, Student-t) — separate per group ---
real mu_logMratio_a;
real mu_logMratio_b;
real<lower=0> sigma_logMratio;
real<lower=1> nu_logMratio;       // shared df across groups
vector[nsubj_a] logMratio_z_a;
vector[nsubj_b] logMratio_z_b;

// --- Type-2 criterion hyperparameters ---
real<lower=0> mu_c2;
real<lower=0> sigma_c2;

// --- Per-subject Type-2 criteria ---
array[nsubj_a] ordered[nratings - 1] cS1_a;
array[nsubj_a] ordered[nratings - 1] cS2_a;
array[nsubj_b] ordered[nratings - 1] cS1_b;
array[nsubj_b] ordered[nratings - 1] cS2_b;
"""

_ROBUST_MODEL_TWO_GROUP = (
    """\
mu_d1           ~ normal(1, 2);
sigma_d1        ~ exponential(1);
d1_z            ~ normal(0, 1);

mu_c1           ~ normal(0, 1);
sigma_c1        ~ exponential(1);
c1_z            ~ normal(0, 1);

mu_logMratio_a  ~ normal(0, 1);
mu_logMratio_b  ~ normal(0, 1);
sigma_logMratio ~ exponential(1);
nu_logMratio    ~ gamma(2, 0.1);
logMratio_z_a   ~ student_t(nu_logMratio, 0, 1);
logMratio_z_b   ~ student_t(nu_logMratio, 0, 1);

mu_c2           ~ normal(1, 1);
sigma_c2        ~ exponential(1);

for (s in 1:nsubj_a) {
    cS1_a[s] ~ normal(c1_a[s] - mu_c2, sigma_c2);
    cS2_a[s] ~ normal(c1_a[s] + mu_c2, sigma_c2);
}
for (s in 1:nsubj_b) {
    cS1_b[s] ~ normal(c1_b[s] - mu_c2, sigma_c2);
    cS2_b[s] ~ normal(c1_b[s] + mu_c2, sigma_c2);
}
"""
    + _group_likelihood_block(
        "nsubj_a", "hmetad_counts_a", "Mratio_a", "d1_a", "c1_a", "cS1_a", "cS2_a"
    )
    + _group_likelihood_block(
        "nsubj_b", "hmetad_counts_b", "Mratio_b", "d1_b", "c1_b", "cS1_b", "cS2_b"
    )
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_robust_metad(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    chains: int = 4,
    n_iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    tol: float = 1e-7,
    **kwargs: Any,
) -> FitResult:
    """Robust hierarchical meta-d' with Student-t hyperprior on log M-ratio.

    Drop-in replacement for :func:`~metasignal.sdtbayes.fit_full_metad` that
    is more resistant to participants with extreme metacognitive efficiency
    estimates.  The degrees-of-freedom parameter ``nu_logMratio`` is estimated
    from the data: small values (2–5) indicate heavy-tailed between-subject
    variability; large values (≥ 30) indicate the robust and standard models
    agree.

    Key posterior parameters
    ------------------------
    - ``mu_logMratio`` — robust group mean log M-ratio.
    - ``sigma_logMratio`` — scale of between-subject variability.
    - ``nu_logMratio`` — estimated degrees of freedom.
    - ``Mratio[s]``, ``meta_d[s]`` — per-subject estimates.

    Args:
        participants: List of ``(stim, resp, conf)`` tuples, one per participant.
        n_ratings: Number of confidence rating categories.
        chains: MCMC chains (default 4).
        n_iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations (default 1000).
        seed: Random seed (default 42).
        tol: Minimum probability floor for multinomial cells (default 1e-7).
        **kwargs: Forwarded to ``brmspy.brms.brm``.

    Returns:
        ``FitResult`` with ``.idata`` and ``.r``.

    Raises:
        NotImplementedError: Always — see Notes.

    Notes:
        This function's brmspy stanvar-injection path is currently
        **non-functional**: the brmspy/rpy2 bridge cannot reliably convert a
        list of multiple ``stanvar()`` objects into R.  A ``block="data"``
        stanvar (carrying a real data value via ``x=``) round-trips to Python
        as a ``dict``, while a ``block != "data"`` stanvar (code-only, e.g.
        ``block="parameters"``) round-trips as a ``list`` — mixing the two
        shapes in one Python list breaks rpy2's homogeneous-type dispatch, and
        no combination method (``c()``, ``+``, dummy data values) was found
        to fix this from calling code.  This is an upstream brmspy limitation.

        If your use case doesn't need the Student-t robustness to outlier
        participants, use :func:`~metasignal.sdtbayes.fit_full_metad` instead,
        which delegates to the working cmdstanpy backend
        (:func:`~metasignal.sdtbayes.fit_meta_formula`).
    """
    raise NotImplementedError(
        "fit_robust_metad is currently unavailable: its brmspy stanvar-injection "
        "path is blocked by an upstream brmspy/rpy2 conversion limitation (a list "
        "mixing data-carrying and code-only stanvar() objects cannot be converted "
        "to R). Use fit_full_metad(...) instead for the standard (non-robust, "
        "Gaussian hyperprior) HMeta-d model, which runs via the working cmdstanpy "
        "backend."
    )


def fit_robust_metad_comparison(
    group_a: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    group_b: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    chains: int = 4,
    n_iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    tol: float = 1e-7,
    **kwargs: Any,
) -> FitResult:
    """Robust HMeta-d comparison between two groups.

    Extends :func:`fit_robust_metad` with separate ``mu_logMratio_a`` and
    ``mu_logMratio_b`` hyperparameters and a shared ``nu_logMratio`` for both
    groups.  The derived quantity ``delta_logMratio = mu_logMratio_b −
    mu_logMratio_a`` carries the full posterior for the group difference.

    Args:
        group_a: Participants in group A.
        group_b: Participants in group B.
        n_ratings: Number of confidence rating categories.
        chains: MCMC chains (default 4).
        n_iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations (default 1000).
        seed: Random seed (default 42).
        tol: Minimum probability floor (default 1e-7).
        **kwargs: Forwarded to ``brmspy.brms.brm``.

    Returns:
        ``FitResult``.  Key parameters: ``mu_logMratio_a``, ``mu_logMratio_b``,
        ``delta_logMratio``, ``nu_logMratio``.

    Raises:
        NotImplementedError: Always — see Notes.

    Notes:
        This function's brmspy stanvar-injection path is currently
        **non-functional** for the same reason documented in
        :func:`fit_robust_metad` — an upstream brmspy/rpy2 conversion
        limitation.  Use :func:`~metasignal.sdtbayes.fit_full_metad_comparison`
        instead for the standard (non-robust) between-groups comparison.

    Example::

        fit = fit_robust_metad_comparison(healthy, patient, n_ratings=4)
        import arviz as az
        delta = az.extract(fit.idata)["delta_logMratio"].values
        print(f"P(patient < healthy): {(delta < 0).mean():.3f}")
    """
    raise NotImplementedError(
        "fit_robust_metad_comparison is currently unavailable: its brmspy "
        "stanvar-injection path is blocked by an upstream brmspy/rpy2 conversion "
        "limitation (a list mixing data-carrying and code-only stanvar() objects "
        "cannot be converted to R). Use fit_full_metad_comparison(...) instead for "
        "the standard (non-robust, Gaussian hyperprior) between-groups comparison, "
        "which runs via the working cmdstanpy backend."
    )
