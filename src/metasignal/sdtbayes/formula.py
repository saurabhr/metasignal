"""Formula interface for hierarchical meta-d' estimation.

Provides :func:`fit_meta_formula`, a brms-style entry point that accepts a
Wilkinson formula string and a participant-level DataFrame, then dispatches to
one of two backends:

``backend="stan"``  (default)
    Compiles and runs :file:`stan/hmeta_d.stan` directly via cmdstanpy (no R
    required).  Fixed-effect predictors on log M-ratio are supported via an
    internally built design matrix.  Categorical variables are dummy-coded;
    continuous variables are mean-centred automatically.

``backend="brms"``
    Two-stage approach: Stage 1 computes per-participant MLE log M-ratio; Stage
    2 passes the formula to ``brmspy.brms.brm``.  Any brms formula is accepted,
    including random effects such as ``(1 | participant)``.  Requires R, brms,
    and brmspy.

Formula syntax
--------------
The left-hand side of the formula is ignored — the DV is always ``log_m_ratio``
(for the brms backend) or inferred (for the Stan backend).  Examples::

    "log_m_ratio ~ sensory_cond * perform_cond"
    "log_m_ratio ~ sensory_cond + perform_cond + vviq_c"
    "log_m_ratio ~ sensory_cond + perform_cond + (1 | participant)"  # brms only

The ``data`` DataFrame must have one row per participant.  Its index or a
``"participant"`` column is used to align rows with the ``participants`` list.

Public API
----------
``fit_meta_formula``
``FormulaFitResult``
"""

from __future__ import annotations

import pathlib
import tempfile
import warnings
from typing import Any

import numpy as np

from metasignal.sdtbayes.diagnostics import FitResult
from metasignal.sdtbayes.full_metad import _build_count_matrix
from metasignal.sdtbayes.two_stage import _compute_participant_estimates

_STAN_FILE = pathlib.Path(__file__).parent / "stan" / "hmeta_d.stan"

_BRMSPY_MSG = "brmspy is not installed. Run:\n    pip install metasignal[sdtbayes]"
_CMDSTANPY_MSG = (
    "cmdstanpy is not installed. Run:\n    pip install cmdstanpy\n"
    "and set up CmdStan with:\n    import cmdstanpy; cmdstanpy.install_cmdstan()"
)


# ---------------------------------------------------------------------------
# Design-matrix builder (patsy-based, falls back to manual dummy coding)
# ---------------------------------------------------------------------------

def _build_design_matrix(formula_rhs: str, data: "pd.DataFrame") -> np.ndarray:
    """Return mean-centred design matrix (nsubj × p) from a formula RHS string.

    Uses patsy when available; falls back to manual dummy coding if not.
    Intercept column is always dropped (absorbed by alpha_logMratio in Stan).
    """
    try:
        import patsy  # type: ignore[import]
        _, X = patsy.dmatrices("_y_ ~ " + formula_rhs,
                               {"_y_": np.zeros(len(data)), **dict(data.items())},
                               return_type="matrix")
        # Remove intercept column (patsy adds it by default)
        col_names = X.design_info.column_names
        keep = [i for i, n in enumerate(col_names) if n != "Intercept"]
        X_arr = np.asarray(X)[:, keep].astype(float)
    except ImportError:
        warnings.warn(
            "patsy is not installed — falling back to manual dummy coding. "
            "Install patsy for full formula support: pip install patsy",
            ImportWarning,
            stacklevel=3,
        )
        X_arr = _manual_design_matrix(formula_rhs, data)

    # Mean-centre all columns so alpha_logMratio = group mean at covariate means
    X_arr = X_arr - X_arr.mean(axis=0)
    return X_arr


def _manual_design_matrix(formula_rhs: str, data: "pd.DataFrame") -> np.ndarray:
    """Minimal formula parser: handles + and * for categorical/continuous vars."""
    import pandas as pd

    # Remove random-effects terms (not supported in Stan backend)
    terms = [t.strip() for t in formula_rhs.split("+") if "|" not in t]

    cols: list[np.ndarray] = []

    for term in terms:
        if "*" in term:
            parts = [p.strip() for p in term.split("*")]
            # Add main effects and interaction
            main_arrays = [_encode_column(data, p) for p in parts]
            for arr in main_arrays:
                cols.append(arr)
            # Interaction: element-wise product of dummy columns
            interaction = main_arrays[0]
            for arr in main_arrays[1:]:
                interaction = interaction * arr
            cols.append(interaction)
        else:
            cols.append(_encode_column(data, term.strip()))

    return np.column_stack(cols) if cols else np.zeros((len(data), 0))


def _encode_column(data: "pd.DataFrame", varname: str) -> np.ndarray:
    """Dummy-code a categorical column or return numeric as-is."""
    import pandas as pd

    col = data[varname]
    if pd.api.types.is_numeric_dtype(col):
        return col.values.astype(float).reshape(-1, 1)
    # Categorical: one-hot drop-first
    dummies = pd.get_dummies(col, drop_first=True, dtype=float)
    return dummies.values


# ---------------------------------------------------------------------------
# Stan backend
# ---------------------------------------------------------------------------

def _fit_stan(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    formula_rhs: str,
    data: "pd.DataFrame",
    *,
    chains: int,
    n_iter: int,
    warmup: int,
    seed: int,
    tol: float,
    cmdstan_path: str | None,
    **kwargs: Any,
) -> FitResult:
    try:
        import cmdstanpy  # type: ignore[import]
    except ImportError as e:
        raise ImportError(_CMDSTANPY_MSG) from e

    if cmdstan_path is not None:
        cmdstanpy.set_cmdstan_path(cmdstan_path)
    elif not cmdstanpy.cmdstan_path():
        # Try the default install location used by this project
        import os
        default = os.path.expanduser("~/.cmdstan")
        candidates = sorted(pathlib.Path(default).glob("cmdstan-*"), reverse=True)
        if candidates:
            cmdstanpy.set_cmdstan_path(str(candidates[0]))

    counts_mat = _build_count_matrix(participants, n_ratings)
    nsubj = len(participants)

    # Build design matrix (or empty matrix for intercept-only model)
    rhs = formula_rhs.strip()
    if rhs in ("1", ""):
        X_cov = np.zeros((nsubj, 0))
    else:
        X_cov = _build_design_matrix(rhs, data)
        if X_cov.shape[0] != nsubj:
            raise ValueError(
                f"data has {X_cov.shape[0]} rows but participants has {nsubj} entries."
            )

    p_cov = X_cov.shape[1]

    stan_data = {
        "nsubj":          nsubj,
        "nratings":       n_ratings,
        "hmetad_counts":  counts_mat.tolist(),
        "Tol":            tol,
        "p_cov":          p_cov,
        "X_cov":          X_cov.tolist(),
    }

    model = cmdstanpy.CmdStanModel(stan_file=str(_STAN_FILE))
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

    # Merge participant-level MLE estimates with the supplied covariate data
    if "participant" in data.columns:
        merged = mle_df.merge(data, on="participant", how="left")
    else:
        # Assume data index aligns with participant order
        data_reset = data.reset_index(drop=True)
        data_reset["participant"] = range(len(data_reset))
        merged = mle_df.merge(data_reset, on="participant", how="left")

    valid = merged.dropna(subset=["log_m_ratio"])
    if len(valid) < 3:
        raise ValueError(
            f"Only {len(valid)} participants have valid MLE estimates — need ≥ 3."
        )

    # Normalise formula: ensure LHS is log_m_ratio
    if "~" in formula:
        formula_brms = "log_m_ratio ~ " + formula.split("~", 1)[1].strip()
    else:
        formula_brms = "log_m_ratio ~ " + formula.strip()

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
    backend: str = "stan",
    chains: int = 4,
    n_iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    tol: float = 1e-7,
    cmdstan_path: str | None = None,
    **kwargs: Any,
) -> FitResult | tuple[FitResult, "pd.DataFrame"]:
    """Fit a hierarchical meta-d' model using a brms-style formula.

    Accepts a Wilkinson–Rogers formula string and a participant-level DataFrame,
    builds the appropriate model, and returns a :class:`FitResult`.

    Two backends are available:

    ``backend="stan"`` *(default)*
        Compiles :file:`stan/hmeta_d.stan` via cmdstanpy and runs NUTS
        sampling.  Does not require R.  Supports fixed-effect predictors on
        log M-ratio (conditions, continuous covariates, interactions).
        Random-effects terms (``|``) in the formula are silently ignored
        because the HMeta-d model already places a full hierarchical prior
        over participants.

    ``backend="brms"``
        Two-stage: MLE meta-d' per participant → brms regression on log
        M-ratio.  Requires R, brms, and brmspy.  Supports any brms formula
        including crossed random effects.

    Args:
        participants: List of ``(stim, resp, conf)`` tuples, one per
            participant.  ``stim`` and ``resp`` are 0/1 arrays; ``conf`` is
            an integer array with values ``1 … n_ratings``.
        n_ratings: Number of confidence rating levels.
        formula: Wilkinson formula string.  The LHS is ignored; the RHS
            specifies predictors.  Examples::

                "log_m_ratio ~ sensory_cond * perform_cond"
                "log_m_ratio ~ sensory_cond + perform_cond + vviq_c"
                "log_m_ratio ~ condition + (1 | group)"  # brms backend only

        data: :class:`pandas.DataFrame` with one row per participant.  Must
            contain all predictor columns named in the formula.  Either
            include a ``"participant"`` column (integer, 0-based) or ensure
            row order matches the ``participants`` list.
        backend: ``"stan"`` (cmdstanpy, default) or ``"brms"`` (brmspy two-stage).
        chains: MCMC chains (default 4).
        n_iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup / burn-in iterations (default 1000).
        seed: Random seed (default 42).
        tol: Probability floor for multinomial cells in the Stan model
            (default 1e-7).  Stan backend only.
        cmdstan_path: Path to CmdStan installation.  If ``None``, cmdstanpy's
            configured path is used, with a fallback to ``~/.cmdstan/cmdstan-*``.
            Stan backend only.
        **kwargs: Passed through to ``cmdstanpy.CmdStanModel.sample`` (Stan
            backend) or ``brmspy.brms.brm`` (brms backend).

    Returns:
        ``FitResult`` for the Stan backend.
        ``(FitResult, merged_df)`` for the brms backend, where ``merged_df``
        contains per-participant MLE estimates merged with the covariate data.

    Raises:
        ImportError: If the required backend libraries are not installed.
        ValueError: If ``data`` row count does not match ``participants`` length,
            or if too few valid estimates are available (brms backend).

    Examples
    --------
    Stan backend (no R required)::

        import numpy as np
        import pandas as pd
        from metasignal.sdtbayes import fit_meta_formula

        rng = np.random.default_rng(0)
        N = 40
        participants = [
            (rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.integers(1, 5, 200))
            for _ in range(N)
        ]
        df = pd.DataFrame({
            "participant": range(N),
            "sensory_cond": ["yes", "no"] * 20,
            "perform_cond": (["accuracy"] * 20 + ["speed"] * 20),
            "vviq_c": rng.normal(0, 1, N),
        })

        fit = fit_meta_formula(
            participants, n_ratings=4,
            formula="log_m_ratio ~ sensory_cond * perform_cond + vviq_c",
            data=df,
            backend="stan",
        )
        print(fit.idata)

    brms backend (requires R + brmspy)::

        fit, mle_df = fit_meta_formula(
            participants, n_ratings=4,
            formula="log_m_ratio ~ sensory_cond * perform_cond + vviq_c + (1 | group)",
            data=df,
            backend="brms",
        )
    """
    if backend == "stan":
        # Strip LHS from formula
        rhs = formula.split("~", 1)[1].strip() if "~" in formula else formula.strip()
        # Warn about random-effects terms being ignored
        if "|" in rhs:
            warnings.warn(
                "Random-effects terms (|) are not supported in the Stan backend "
                "and will be ignored. The HMeta-d model already places a full "
                "hierarchical prior over participants. Use backend='brms' for "
                "additional random-effects structure.",
                UserWarning,
                stacklevel=2,
            )
            rhs = " + ".join(t for t in rhs.split("+") if "|" not in t)
        return _fit_stan(
            participants, n_ratings, rhs, data,
            chains=chains, n_iter=n_iter, warmup=warmup, seed=seed,
            tol=tol, cmdstan_path=cmdstan_path, **kwargs,
        )

    if backend == "brms":
        return _fit_brms(
            participants, n_ratings, formula, data,
            chains=chains, n_iter=n_iter, warmup=warmup, seed=seed, **kwargs,
        )

    raise ValueError(f"Unknown backend {backend!r}. Choose 'stan' or 'brms'.")
