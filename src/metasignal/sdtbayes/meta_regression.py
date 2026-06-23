"""Bayesian meta-regression: covariates on log M-ratio.

Provides two paths for regressing log M-ratio on participant-level covariates
(e.g. age, clinical scores, task difficulty):

Two-stage path  (:func:`fit_two_stage_regression`)
    Stage 1 — MLE log M-ratio per participant.
    Stage 2 — brms regression of log M-ratio on covariates.
    Fast (seconds); does not propagate Stage-1 uncertainty.

Full hierarchical path  (:func:`fit_full_metad_regression`)
    Single-stage Stan model.  Covariates enter as a design matrix
    ``X`` that predicts the per-participant expected log M-ratio::

        logMratio[s] ~ Normal(alpha + X[s] @ beta, sigma_logMratio)

    Propagates estimation uncertainty through the SDT likelihood.
    Uses the same multinomial SDT likelihood as :func:`fit_full_metad`.

Covariate format
----------------
- ``fit_two_stage_regression``: ``covariates`` is a :class:`pandas.DataFrame`
  with a ``"participant"`` column (integers 0 … N−1) and one or more predictor
  columns.  Supply a custom ``formula`` string to control the brms model.
- ``fit_full_metad_regression``: ``covariates`` is a :class:`numpy.ndarray`
  of shape ``(N,)`` (single covariate) or ``(N, p)`` (multiple covariates).
  Covariates are mean-centred automatically before fitting so that the
  intercept ``alpha_logMratio`` represents the expected log M-ratio at the
  covariate mean.

Key posterior parameters
------------------------
Two-stage: whatever the brms formula produces (e.g. ``b_Intercept``,
``b_age``, ``b_score``).

Full hierarchical:

- ``alpha_logMratio`` — intercept (log M-ratio at covariate mean).
- ``beta_logMratio`` — slope vector (one coefficient per covariate column).
- ``sigma_logMratio`` — residual between-subject SD after accounting for
  covariates.
- ``Mratio[s]``, ``meta_d[s]`` — covariate-adjusted per-subject estimates.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from metasignal.sdtbayes.diagnostics import FitResult
from metasignal.sdtbayes.full_metad import (
    _STAN_DATA,
    _STAN_TRANSFORMED_PARAMETERS,
    _build_count_matrix,
    _group_likelihood_block,
)
from metasignal.sdtbayes.two_stage import _compute_participant_estimates

_BRMSPY_MSG = "brmspy is not installed. Run:\n    pip install metasignal[sdtbayes]"


# ---------------------------------------------------------------------------
# Path 1 — Two-stage
# ---------------------------------------------------------------------------

def fit_two_stage_regression(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    covariates: "pd.DataFrame",
    formula: str | None = None,
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    **kwargs: Any,
) -> tuple[FitResult, "pd.DataFrame"]:
    """Two-stage Bayesian meta-regression of log M-ratio on covariates.

    Stage 1 computes per-participant MLE estimates; Stage 2 fits a brms
    regression of log M-ratio on the supplied covariates.

    Args:
        participants: List of ``(stim, resp, conf)`` tuples, one per participant.
            Participant indices must align with the ``"participant"`` column in
            ``covariates`` (0-based integers).
        n_ratings: Number of confidence rating categories.
        covariates: :class:`pandas.DataFrame` with a ``"participant"`` column
            (integer) and one or more numeric predictor columns.  Passed
            directly to brms after merging with MLE estimates.
        formula: brms formula string.  Default: ``"log_m_ratio ~ col1 + col2"``
            using all non-``participant`` columns in ``covariates``.
        chains: MCMC chains (default 4).
        iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations (default 1000).
        seed: Random seed (default 42).
        **kwargs: Forwarded to ``brmspy.brms.brm``.

    Returns:
        Tuple of:

        - ``FitResult`` with Stage-2 posterior.
        - ``pd.DataFrame`` with per-participant MLE estimates and covariates.

    Raises:
        ImportError: If ``brmspy`` is not installed.
        ValueError: If ``"participant"`` column is missing from ``covariates``
            or fewer than 3 participants have valid log M-ratio estimates.

    Example::

        import numpy as np
        import pandas as pd
        from metasignal.sdtbayes import fit_two_stage_regression

        rng = np.random.default_rng(0)
        N = 30
        participants = [
            (rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.integers(1, 5, 200))
            for _ in range(N)
        ]
        covs = pd.DataFrame({
            "participant": range(N),
            "age": rng.normal(35, 10, N),
            "score": rng.normal(50, 15, N),
        })
        fit, mle_df = fit_two_stage_regression(participants, n_ratings=4, covariates=covs)
        print(fit.posterior_summary(var_names=["b_Intercept", "b_age", "b_score"]))
    """
    try:
        from brmspy import brms
        import pandas as pd
    except ImportError as e:
        raise ImportError(_BRMSPY_MSG) from e

    if "participant" not in covariates.columns:
        msg = "'covariates' must contain a 'participant' column."
        raise ValueError(msg)

    mle_df = _compute_participant_estimates(participants, n_ratings)
    merged = mle_df.merge(covariates, on="participant", how="left")
    valid = merged.dropna(subset=["log_m_ratio"])

    if len(valid) < 3:
        msg = f"Only {len(valid)} participants have valid estimates — need at least 3."
        raise ValueError(msg)

    if formula is None:
        pred_cols = [c for c in covariates.columns if c != "participant"]
        formula = "log_m_ratio ~ " + " + ".join(pred_cols)

    priors = [
        brms.prior("normal(0, 1)", class_="Intercept"),
        brms.prior("normal(0, 1)", class_="b"),
        brms.prior("exponential(1)", class_="sigma"),
    ]

    _result = brms.brm(
        formula=brms.bf(formula),
        data=valid,
        family="student",
        priors=priors,
        chains=chains,
        iter=iter,
        warmup=warmup,
        seed=seed,
        **kwargs,
    )
    return FitResult(idata=_result.idata, r=_result.r), merged


# ---------------------------------------------------------------------------
# Path 2 — Full hierarchical with Stan design matrix
# ---------------------------------------------------------------------------

def _build_regression_stan_blocks(p_cov: int) -> tuple[str, str, str, str]:
    """Return (data, parameters, tpar, model) Stan blocks for regression model.

    The design matrix ``X_cov`` (nsubj × p_cov, mean-centred) replaces the
    scalar ``mu_logMratio`` with ``alpha_logMratio + X_cov[s] * beta_logMratio``.
    """
    stan_data = _STAN_DATA + f"""\
int<lower=1> p_cov;
matrix[nsubj, p_cov] X_cov;   // covariate design matrix (mean-centred)
"""

    stan_parameters = """\
// --- Group-level Type-1 parameters ---
real mu_d1;
real<lower=0> sigma_d1;
vector[nsubj] d1_z;

real mu_c1;
real<lower=0> sigma_c1;
vector[nsubj] c1_z;

// --- M-ratio regression (log scale) ---
real alpha_logMratio;           // intercept (at covariate mean)
vector[p_cov] beta_logMratio;   // slopes
real<lower=0> sigma_logMratio;  // residual between-subject SD
vector[nsubj] logMratio_z;

// --- Group-level Type-2 criterion hyperparameters ---
real<lower=0> mu_c2;
real<lower=0> sigma_c2;

// --- Per-subject Type-2 criteria ---
array[nsubj] ordered[nratings - 1] cS1_raw;
array[nsubj] ordered[nratings - 1] cS2_raw;
"""

    stan_tpar = """\
vector[nsubj] d1;
vector[nsubj] c1;
vector[nsubj] Mratio;
vector[nsubj] meta_d;

d1 = mu_d1 + sigma_d1 * d1_z;
c1 = mu_c1 + sigma_c1 * c1_z;
for (s in 1:nsubj) {
    real eta_s = alpha_logMratio + dot_product(beta_logMratio, X_cov[s]');
    Mratio[s] = exp(eta_s + sigma_logMratio * logMratio_z[s]);
    meta_d[s] = Mratio[s] * d1[s];
}
"""

    stan_model = (
        """\
mu_d1             ~ normal(1, 2);
sigma_d1          ~ exponential(1);
d1_z              ~ normal(0, 1);

mu_c1             ~ normal(0, 1);
sigma_c1          ~ exponential(1);
c1_z              ~ normal(0, 1);

alpha_logMratio   ~ normal(0, 1);
beta_logMratio    ~ normal(0, 1);
sigma_logMratio   ~ exponential(1);
logMratio_z       ~ normal(0, 1);

mu_c2             ~ normal(1, 1);
sigma_c2          ~ exponential(1);

for (s in 1:nsubj) {
    cS1_raw[s] ~ normal(c1[s] - mu_c2, sigma_c2);
    cS2_raw[s] ~ normal(c1[s] + mu_c2, sigma_c2);
}
"""
        + _group_likelihood_block(
            "nsubj", "hmetad_counts", "Mratio", "d1", "c1", "cS1_raw", "cS2_raw"
        )
    )

    return stan_data, stan_parameters, stan_tpar, stan_model


def fit_full_metad_regression(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    covariates: "np.ndarray",
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    tol: float = 1e-7,
    **kwargs: Any,
) -> FitResult:
    """Full hierarchical meta-d' with covariate regression on log M-ratio.

    A single-stage fully Bayesian model that extends :func:`fit_full_metad`
    with participant-level covariates.  Covariates are incorporated into the
    SDT likelihood directly, so estimation uncertainty propagates through the
    regression.

    The model replaces the scalar group hyperprior with a linear predictor::

        logMratio[s] ~ Normal(alpha + X[s] @ beta, sigma_logMratio)

    where ``alpha`` is the expected log M-ratio at the covariate mean and
    ``beta`` is the vector of slopes.  Covariates are mean-centred
    automatically before fitting.

    Args:
        participants: List of ``(stim, resp, conf)`` tuples, one per participant.
        n_ratings: Number of confidence rating categories.
        covariates: :class:`numpy.ndarray` of shape ``(N,)`` for a single
            continuous covariate, or ``(N, p)`` for multiple covariates.
            Must have the same length as ``participants``.  Covariates are
            mean-centred automatically.
        chains: MCMC chains (default 4).
        iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations (default 1000).
        seed: Random seed (default 42).
        tol: Minimum probability floor for multinomial cells (default 1e-7).
        **kwargs: Forwarded to ``brmspy.brms.brm``.

    Returns:
        ``FitResult``.  Key parameters:

        - ``alpha_logMratio`` — intercept (log M-ratio at covariate mean).
        - ``beta_logMratio`` — slope(s), one per covariate column.
        - ``sigma_logMratio`` — residual between-subject SD.
        - ``Mratio[s]``, ``meta_d[s]`` — covariate-adjusted per-subject estimates.

    Raises:
        ImportError: If ``brmspy`` is not installed.
        ValueError: If ``covariates`` length does not match ``participants``.

    Example::

        import numpy as np
        from metasignal.sdtbayes import fit_full_metad_regression

        rng = np.random.default_rng(0)
        N = 25
        participants = [
            (rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.integers(1, 5, 200))
            for _ in range(N)
        ]
        age = rng.normal(35, 10, N)   # single continuous covariate

        fit = fit_full_metad_regression(participants, n_ratings=4, covariates=age)
        print(fit.posterior_summary(var_names=["alpha_logMratio", "beta_logMratio",
                                                "sigma_logMratio"]))
    """
    try:
        from brmspy import brms
        import pandas as pd
    except ImportError as e:
        raise ImportError(_BRMSPY_MSG) from e

    X = np.atleast_2d(np.asarray(covariates, dtype=float))
    if X.shape[0] == 1 and X.shape[1] == len(participants):
        X = X.T
    if X.shape[0] != len(participants):
        msg = (
            f"covariates has {X.shape[0]} rows but participants has "
            f"{len(participants)} entries."
        )
        raise ValueError(msg)

    # Mean-centre covariates so alpha is interpretable at the covariate mean
    X = X - X.mean(axis=0)
    p_cov = X.shape[1]

    nsubj = len(participants)
    counts_mat = _build_count_matrix(participants, n_ratings)
    stan_data_str, stan_params_str, stan_tpar_str, stan_model_str = (
        _build_regression_stan_blocks(p_cov)
    )

    sv_data   = brms.call("stanvar", scode=stan_data_str,   block="data")
    sv_params = brms.call("stanvar", scode=stan_params_str, block="parameters")
    sv_tpar   = brms.call("stanvar", scode=stan_tpar_str,   block="tpar")
    sv_model  = brms.call("stanvar", scode=stan_model_str,  block="model")

    dummy_df = pd.DataFrame({"dummy": [0]})
    extra_data = {
        "nsubj":         nsubj,
        "nratings":      n_ratings,
        "hmetad_counts": counts_mat.tolist(),
        "Tol":           tol,
        "p_cov":         p_cov,
        "X_cov":         X.tolist(),
    }

    _result = brms.brm(
        formula=brms.bf("dummy ~ 1"),
        data=dummy_df,
        family=brms.call("empty"),
        stanvars=[sv_data, sv_params, sv_tpar, sv_model],
        data2=extra_data,
        chains=chains,
        iter=iter,
        warmup=warmup,
        seed=seed,
        **kwargs,
    )
    return FitResult(idata=_result.idata, r=_result.r)
