"""Formula interface for hierarchical Bayesian metacognition models.

Provides :func:`fit_meta_formula` — a brms-style entry point that accepts a
Wilkinson formula string, a participant-level DataFrame, and a choice of
generative model (parameterization), then dispatches to one of two backends.

Generative models (``parameterization``)
-----------------------------------------
``"mratio"`` *(default)*
    Fleming (2017) HMeta-d.  Group parameter: **log M-ratio**
    (meta_d / d').  Stan file: :file:`stan/hmeta_d.stan`.

``"meta_noise"``
    Maniscalco & Lau (2014) / Guggenmos (2022).  Group parameter:
    **log σ_meta** (additive noise on the metacognitive signal).
    Relationship to M-ratio: M-ratio = 1 / sqrt(1 + σ_meta²).
    Stan file: :file:`stan/hmeta_noise.stan`.

``"casandre"``
    Boundy-Singer, Ziemba & Goris (2023).  Group parameter: **log φ**
    (meta-uncertainty — log-normal noise on the reliability estimate
    |x|).  Uses Gauss-Hermite quadrature to marginalise over the latent
    decision variable.  Stan file: :file:`stan/hmeta_uncertainty.stan`.

Backends (``backend``)
-----------------------
``"stan"`` *(default)*
    Compiles the chosen Stan model via cmdstanpy and runs NUTS.  Does not
    require R.  Supports fixed-effect predictors on the group-level
    metacognitive parameter via a mean-centred design matrix (patsy or
    manual dummy coding).

``"brms"``
    Two-stage approach: Stage 1 computes per-participant MLE log M-ratio;
    Stage 2 passes the formula to ``brmspy.brms.brm``.  Requires R, brms,
    and brmspy.  Supports any brms formula including random effects such as
    ``(1 | participant)``.  Only valid with ``parameterization="mratio"``.

Formula syntax
--------------
The left-hand side is ignored (the DV is determined by the parameterization).
Examples::

    "~ sensory_cond * perform_cond + vviq_c"
    "~ sensory_cond + perform_cond + vviq_c + img_excess_c"
    "~ condition + (1 | group)"   # brms backend only

Public API
----------
``fit_meta_formula``
"""

from __future__ import annotations

import os
import pathlib
import warnings
from typing import Any

import numpy as np

from metasignal.sdtbayes.diagnostics import FitResult
from metasignal.sdtbayes.full_metad import _build_count_matrix
from metasignal.sdtbayes.two_stage import _compute_participant_estimates

_STAN_DIR = pathlib.Path(__file__).parent / "stan"

_STAN_FILES = {
    "mratio":      _STAN_DIR / "hmeta_d.stan",
    "meta_noise":  _STAN_DIR / "hmeta_noise.stan",
    "casandre":    _STAN_DIR / "hmeta_uncertainty.stan",
}

# Single-subject variants: fixed priors, no hierarchical layer.
# Used automatically when len(participants) == 1.
_STAN_FILES_SUBJECT = {
    "mratio":      _STAN_DIR / "meta_d_subject.stan",
    "meta_noise":  _STAN_DIR / "hmeta_noise_subject.stan",
    "casandre":    _STAN_DIR / "hmeta_uncertainty_subject.stan",
}

_VALID_PARAMETERIZATIONS = tuple(_STAN_FILES)

_BRMSPY_MSG  = (
    "brmspy is not installed. Run:\n"
    '    pip install "metasignal[sdtbayes] @ git+https://github.com/saurabhr/metasignal.git"'
)
_CMDSTAN_MSG = (
    "cmdstanpy is not installed. Run:\n    pip install cmdstanpy\n"
    "Then set up CmdStan:  import cmdstanpy; cmdstanpy.install_cmdstan()"
)


# ---------------------------------------------------------------------------
# Gauss-Hermite quadrature (used by the casandre model)
# ---------------------------------------------------------------------------

def _gh_quadrature(n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Return (nodes, weights) for n-point Gauss-Hermite quadrature.

    These are the raw nodes/weights for ∫f(t)exp(−t²)dt.
    To approximate ∫f(x)·Normal(x; μ, 1)dx use:
        x_i = μ + √2·h_i,  effective_weight = w_i / √π
    """
    return np.polynomial.hermite.hermgauss(n)


# ---------------------------------------------------------------------------
# Design-matrix builder (patsy or manual fallback)
# ---------------------------------------------------------------------------

def _build_design_matrix(formula_rhs: str, data: "pd.DataFrame") -> np.ndarray:
    """Return mean-centred design matrix (nsubj × p) from a formula RHS string."""
    try:
        import patsy  # type: ignore[import]
        _, X = patsy.dmatrices(
            "_y_ ~ " + formula_rhs,
            {"_y_": np.zeros(len(data)), **dict(data.items())},
            return_type="matrix",
        )
        col_names = X.design_info.column_names
        keep = [i for i, n in enumerate(col_names) if n != "Intercept"]
        X_arr = np.asarray(X)[:, keep].astype(float)
    except ImportError:
        warnings.warn(
            "patsy is not installed — falling back to manual dummy coding. "
            "Install patsy for full formula support: pip install patsy",
            ImportWarning,
            stacklevel=4,
        )
        X_arr = _manual_design_matrix(formula_rhs, data)

    return X_arr - X_arr.mean(axis=0)


def _manual_design_matrix(formula_rhs: str, data: "pd.DataFrame") -> np.ndarray:
    """Minimal formula parser: handles + and * for categorical/continuous vars."""
    terms = [t.strip() for t in formula_rhs.split("+") if "|" not in t]
    cols: list[np.ndarray] = []
    for term in terms:
        if "*" in term:
            parts = [p.strip() for p in term.split("*")]
            main_arrs = [_encode_column(data, p) for p in parts]
            for arr in main_arrs:
                cols.append(arr)
            n_multi_col_sides = sum(arr.shape[1] > 1 for arr in main_arrs)
            if n_multi_col_sides > 1:
                msg = (
                    f"formula term {term!r} is an interaction between two or more "
                    "categorical variables with >2 levels. The manual (patsy-free) "
                    "fallback only supports elementwise multiplication, which is "
                    "not the correct encoding for this case (it would silently "
                    "produce wrong or shape-mismatched columns). Install patsy for "
                    "full formula support: pip install patsy"
                )
                raise ValueError(msg)
            interaction = main_arrs[0]
            for arr in main_arrs[1:]:
                interaction = interaction * arr
            cols.append(interaction)
        else:
            cols.append(_encode_column(data, term.strip()))
    return np.column_stack(cols) if cols else np.zeros((len(data), 0))


def _encode_column(data: "pd.DataFrame", varname: str) -> np.ndarray:
    import pandas as pd
    col = data[varname]
    if pd.api.types.is_numeric_dtype(col):
        return col.values.astype(float).reshape(-1, 1)
    return pd.get_dummies(col, drop_first=True, dtype=float).values


# ---------------------------------------------------------------------------
# CmdStan path helper
# ---------------------------------------------------------------------------

def _resolve_cmdstan(cmdstan_path: str | None, cmdstanpy: Any) -> None:
    if cmdstan_path is not None:
        cmdstanpy.set_cmdstan_path(cmdstan_path)
    elif not cmdstanpy.cmdstan_path():
        default = os.path.expanduser("~/.cmdstan")
        candidates = sorted(pathlib.Path(default).glob("cmdstan-*"), reverse=True)
        if candidates:
            cmdstanpy.set_cmdstan_path(str(candidates[0]))


# ---------------------------------------------------------------------------
# Stan backend
# ---------------------------------------------------------------------------

def _fit_stan(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    formula_rhs: str,
    data: "pd.DataFrame",
    *,
    parameterization: str,
    chains: int,
    n_iter: int,
    warmup: int,
    seed: int,
    tol: float,
    cmdstan_path: str | None,
    n_gh: int,
    **kwargs: Any,
) -> FitResult:
    try:
        import cmdstanpy  # type: ignore[import]
    except ImportError as e:
        raise ImportError(_CMDSTAN_MSG) from e

    _resolve_cmdstan(cmdstan_path, cmdstanpy)

    counts_mat = _build_count_matrix(participants, n_ratings)
    nsubj = len(participants)
    subject_level = (nsubj == 1)

    if subject_level:
        rhs = formula_rhs.strip()
        if rhs not in ("1", ""):
            msg = (
                f"formula_rhs={formula_rhs!r} requests covariates, but only 1 "
                "participant was provided. The single-subject Stan model has no "
                "covariate design matrix and would silently ignore the formula. "
                "Provide at least 2 participants to fit a covariate regression, "
                "or use formula_rhs='1' for a single-subject fit."
            )
            raise ValueError(msg)
        # Single-subject model: flat priors, 1-D count array
        stan_data: dict[str, Any] = {
            "nratings": n_ratings,
            "counts":   counts_mat[0].tolist(),
            "Tol":      tol,
        }
        if parameterization == "casandre":
            gh_nodes, gh_weights = _gh_quadrature(n_gh)
            stan_data["n_gh"]       = n_gh
            stan_data["gh_nodes"]   = gh_nodes.tolist()
            stan_data["gh_weights"] = gh_weights.tolist()
            stan_data["delta"]      = kwargs.pop("delta", 0.1)
            stan_data["eps"]        = kwargs.pop("eps", 0.05)
        stan_file = str(_STAN_FILES_SUBJECT[parameterization])
    else:
        # Group hierarchical model: 2-D count matrix + optional covariates
        rhs = formula_rhs.strip()
        if rhs in ("1", ""):
            X_cov = np.zeros((nsubj, 0))
        else:
            X_cov = _build_design_matrix(rhs, data)
            if X_cov.shape[0] != nsubj:
                raise ValueError(
                    f"data has {X_cov.shape[0]} rows but participants has {nsubj} entries."
                )

        stan_data = {
            "nsubj":         nsubj,
            "nratings":      n_ratings,
            "hmetad_counts": counts_mat.tolist(),
            "Tol":           tol,
            "p_cov":         X_cov.shape[1],
            "X_cov":         X_cov.tolist(),
        }

        if parameterization == "casandre":
            gh_nodes, gh_weights = _gh_quadrature(n_gh)
            stan_data["n_gh"]       = n_gh
            stan_data["gh_nodes"]   = gh_nodes.tolist()
            stan_data["gh_weights"] = gh_weights.tolist()
            stan_data["delta"]      = kwargs.pop("delta", 0.1)
            stan_data["eps"]        = kwargs.pop("eps", 0.05)
        stan_file = str(_STAN_FILES[parameterization])

    # Stan's default wide random init (unconstrained ~Uniform(-2,2)) can land on
    # a degenerate ordered-criteria configuration for hierarchical fits with many
    # subjects, causing an immediate -inf likelihood that crashes every chain
    # before adaptation even starts. init=0 is a conservative, near-zero starting
    # point that sidesteps this; callers can still override via inits=....
    kwargs.setdefault("inits", 0)

    model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    fit = model.sample(
        data=stan_data,
        chains=chains,
        iter_sampling=n_iter - warmup,
        iter_warmup=warmup,
        seed=seed,
        **kwargs,
    )

    try:
        import arviz as az
        idata = az.from_cmdstanpy(fit)
    except ImportError:
        idata = None
        warnings.warn(
            "arviz not installed — FitResult.idata will be None. "
            "Install with: pip install arviz",
            ImportWarning,
            stacklevel=3,
        )

    return FitResult(idata=idata, r=fit)


# ---------------------------------------------------------------------------
# brms two-stage backend
# ---------------------------------------------------------------------------

def _fit_brms(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    formula: str,
    data: "pd.DataFrame",
    *,
    chains: int,
    n_iter: int,
    warmup: int,
    seed: int,
    **kwargs: Any,
) -> tuple[FitResult, "pd.DataFrame"]:
    try:
        from brmspy import brms  # type: ignore[import]
        import pandas as pd
    except ImportError as e:
        raise ImportError(_BRMSPY_MSG) from e

    mle_df = _compute_participant_estimates(participants, n_ratings)

    if "participant" in data.columns:
        merged = mle_df.merge(data, on="participant", how="left")
    else:
        data_reset = data.reset_index(drop=True).copy()
        data_reset["participant"] = range(len(data_reset))
        merged = mle_df.merge(data_reset, on="participant", how="left")

    valid = merged.dropna(subset=["log_m_ratio"])
    if len(valid) < 3:
        raise ValueError(
            f"Only {len(valid)} participants have valid MLE estimates — need ≥ 3."
        )

    rhs = formula.split("~", 1)[1].strip() if "~" in formula else formula.strip()
    formula_brms = "log_m_ratio ~ " + rhs

    priors = [
        brms.prior("normal(0, 1)", class_="Intercept"),
        brms.prior("normal(0, 1)", class_="b"),
        brms.prior("exponential(1)", class_="sigma"),
        brms.prior("exponential(1)", class_="sd"),
    ]

    _result = brms.brm(
        formula=brms.bf(formula_brms),
        data=valid,
        family="student",
        priors=priors,
        chains=chains,
        iter=n_iter,
        warmup=warmup,
        seed=seed,
        **kwargs,
    )
    return FitResult(idata=_result.idata, r=_result.r), merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_meta_formula(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    formula: str,
    data: "pd.DataFrame",
    *,
    parameterization: str = "mratio",
    backend: str = "stan",
    chains: int = 4,
    n_iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    tol: float = 1e-7,
    n_gh: int = 20,
    cmdstan_path: str | None = None,
    **kwargs: Any,
) -> FitResult | tuple[FitResult, "pd.DataFrame"]:
    """Fit a hierarchical Bayesian metacognition model using a brms-style formula.

    Args:
        participants: List of ``(stim, resp, conf)`` tuples, one per participant.
            ``stim`` and ``resp`` are 0/1 integer arrays; ``conf`` is an integer
            array with values ``1 … n_ratings``.
        n_ratings: Number of confidence rating levels.
        formula: Wilkinson formula string.  The LHS is ignored.  Examples::

                "~ sensory_cond * perform_cond + vviq_c"
                "~ sensory_cond + perform_cond + (1 | group)"   # brms only

        data: :class:`pandas.DataFrame` with one row per participant containing
            all predictor columns named in the formula.  Row order must match
            ``participants``, or include a ``"participant"`` integer column.
        parameterization: Generative model to use:

            - ``"mratio"`` *(default)* — Fleming (2017) HMeta-d.
              Primary group parameter: ``alpha_logMratio`` (log M-ratio).
            - ``"meta_noise"`` — Guggenmos (2022) / Maniscalco & Lau (2014).
              Primary group parameter: ``mu_logSigmaMeta`` (log σ_meta).
            - ``"casandre"`` — Boundy-Singer et al. (2023) CASANDRE.
              Primary group parameter: ``mu_logPhi`` (log meta-uncertainty φ).

        backend: ``"stan"`` (cmdstanpy, default) or ``"brms"`` (two-stage brmspy,
            ``parameterization="mratio"`` only).
        chains: MCMC chains (default 4).
        n_iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations (default 1000).
        seed: Random seed (default 42).
        tol: Probability floor for multinomial cells (default 1e-7). Stan only.
        n_gh: Number of Gauss-Hermite quadrature points for the ``"casandre"``
            model (default 20; increase for more accuracy at higher cost).
        cmdstan_path: Path to CmdStan installation.  If ``None``, auto-detected
            from cmdstanpy config or ``~/.cmdstan/cmdstan-*``. Stan only.
        **kwargs: Forwarded to ``cmdstanpy.CmdStanModel.sample`` (Stan) or
            ``brmspy.brms.brm`` (brms).

    Returns:
        ``FitResult`` — Stan backend.
        ``(FitResult, merged_df)`` — brms backend.

    Examples
    --------
    M-ratio model (default)::

        fit = fit_meta_formula(
            participants, n_ratings=4,
            formula="~ sensory_cond * perform_cond + vviq_c",
            data=df,
            parameterization="mratio",
        )
        # Key posteriors: alpha_logMratio, beta_logMratio, group_Mratio, Mratio[s]

    Meta-noise model::

        fit = fit_meta_formula(
            participants, n_ratings=4,
            formula="~ perform_cond + vviq_c",
            data=df,
            parameterization="meta_noise",
        )
        # Key posteriors: mu_logSigmaMeta, sigma_meta[s], group_sigma_meta

    CASANDRE meta-uncertainty model::

        fit = fit_meta_formula(
            participants, n_ratings=4,
            formula="~ sensory_cond + perform_cond",
            data=df,
            parameterization="casandre",
            n_gh=20,
        )
        # Key posteriors: mu_logPhi, phi[s], group_phi, log_theta[1..K-1]
    """
    if parameterization not in _VALID_PARAMETERIZATIONS:
        raise ValueError(
            f"Unknown parameterization {parameterization!r}. "
            f"Choose from: {_VALID_PARAMETERIZATIONS}"
        )

    if backend == "brms" and parameterization != "mratio":
        raise ValueError(
            "The brms two-stage backend only supports parameterization='mratio'. "
            "For meta_noise or casandre use backend='stan'."
        )

    # Strip and clean formula RHS
    rhs = formula.split("~", 1)[1].strip() if "~" in formula else formula.strip()

    if backend == "stan":
        if "|" in rhs:
            warnings.warn(
                "Random-effects terms (|) are not supported in the Stan backend "
                "and will be ignored. The hierarchical Stan model already pools "
                "over participants. Use backend='brms' (mratio only) for additional "
                "random-effects structure.",
                UserWarning,
                stacklevel=2,
            )
            rhs = " + ".join(t for t in rhs.split("+") if "|" not in t)
        return _fit_stan(
            participants, n_ratings, rhs, data,
            parameterization=parameterization,
            chains=chains, n_iter=n_iter, warmup=warmup, seed=seed,
            tol=tol, cmdstan_path=cmdstan_path, n_gh=n_gh, **kwargs,
        )

    if backend == "brms":
        return _fit_brms(
            participants, n_ratings, formula, data,
            chains=chains, n_iter=n_iter, warmup=warmup, seed=seed, **kwargs,
        )

    raise ValueError(f"Unknown backend {backend!r}. Choose 'stan' or 'brms'.")
