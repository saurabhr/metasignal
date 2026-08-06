"""Shared maximum-likelihood estimation engine for ``metasignal.sdtr`` models.

Scoped-down port of the generic estimation machinery in Macho (2020)'s
``SDT-Main.R`` / ``SDT-Auxiliary.R`` (``SDT.Estimate()`` / ``SDT.Statistics()``),
factored out so every model in this subpackage reuses it instead of
re-deriving optimizer/constraint/standard-error handling per model.

Supports the R original's ``fixed`` (parameters excluded from optimization,
pinned to a constant) and ``ident`` (equality constraints between two
positions in the full parameter vector) constraint types. ``functional``
(user-supplied nonlinear constraints) is not implemented — no model in this
subpackage's first phase needs it; add it if a later model does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass
class SDTFitResult:
    """Result of an MLE fit produced by :func:`fit_mle`.

    ``params``/``se`` are free-parameter-length (one entry per optimized
    position); ``full_params`` is full-length, including fixed and
    equality-derived positions.
    """

    params: np.ndarray
    full_params: np.ndarray
    se: np.ndarray
    nll: float
    logL: float
    aic: float
    bic: float
    hessian: np.ndarray | None
    success: bool
    n_obs: int
    n_free_params: int


def _free_indices(n_full: int, fixed_pos: set[int], ident_targets: set[int]) -> list[int]:
    return [i for i in range(n_full) if i not in fixed_pos and i not in ident_targets]


def _numerical_hessian(fn: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central-difference Hessian of ``fn`` at ``x``.

    L-BFGS-B's own ``hess_inv`` is a secant (BFGS) approximation built up
    over the optimizer's iteration history — for small/fast-converging
    problems it can be a poor stand-in for the true curvature (it stays
    close to its identity initialization). A direct finite-difference
    Hessian at the optimum is more reliable and matches what the R
    original does (``numDeriv``) when symbolic derivatives aren't available.
    """
    n = len(x)
    h = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            xpp, xpm = x.copy(), x.copy()
            xmp, xmm = x.copy(), x.copy()
            xpp[i] += eps; xpp[j] += eps
            xpm[i] += eps; xpm[j] -= eps
            xmp[i] -= eps; xmp[j] += eps
            xmm[i] -= eps; xmm[j] -= eps
            val = (fn(xpp) - fn(xpm) - fn(xmp) + fn(xmm)) / (4 * eps * eps)
            h[i, j] = h[j, i] = val
    return h


def expand_params(
    free_params: np.ndarray,
    n_full: int,
    *,
    fixed: Sequence[tuple[int, float]] | None = None,
    ident: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Expand a free-parameter vector into the full ``n_full``-length vector.

    ``fixed``: ``(position, value)`` pairs — ``position`` is excluded from
    the free vector and always set to ``value``.
    ``ident``: ``(source, target)`` pairs — ``full[target]`` is set equal to
    ``full[source]`` after fixed/free values are placed (equality
    constraint; ``target`` is excluded from the free vector).
    """
    fixed = list(fixed or [])
    ident = list(ident or [])
    fixed_pos = {int(p): float(v) for p, v in fixed}
    ident_targets = {int(t) for _, t in ident}

    free_idx = _free_indices(n_full, set(fixed_pos), ident_targets)
    if len(free_idx) != len(free_params):
        raise ValueError(
            f"Expected {len(free_idx)} free parameters given n_full={n_full}, "
            f"{len(fixed_pos)} fixed and {len(ident_targets)} equality-constrained "
            f"positions, but got {len(free_params)}."
        )

    full = np.empty(n_full, dtype=float)
    for i, v in zip(free_idx, free_params):
        full[i] = v
    for pos, val in fixed_pos.items():
        full[pos] = val
    for source, target in ident:
        full[int(target)] = full[int(source)]
    return full


def fit_mle(
    nll_fn: Callable[[np.ndarray], float],
    start: np.ndarray,
    bounds: Sequence[tuple[float, float]],
    *,
    fixed: Sequence[tuple[int, float]] | None = None,
    ident: Sequence[tuple[int, int]] | None = None,
    n_obs: int,
    n_starts: int = 1,
    seed: int = 42,
) -> SDTFitResult:
    """Fit a model by maximum likelihood via ``scipy.optimize.minimize`` (L-BFGS-B).

    Parameters
    ----------
    nll_fn:
        Negative log-likelihood, called with the *full* (expanded) parameter
        vector.
    start, bounds:
        Full-length starting point and box bounds. Entries at fixed or
        equality-target positions are ignored.
    fixed, ident:
        See :func:`expand_params`.
    n_obs:
        Total observation count, for AIC/BIC.
    n_starts:
        Number of optimizer starts. ``1`` uses ``start`` as-is (deterministic).
        For ``n_starts > 1``, additional starts perturb ``start`` with a
        deterministic seeded jitter, keeping the lowest-NLL result — same
        multi-start pattern as ``stdpy.uncertainty.compute_meta_uncertainty``.
    """
    n_full = len(start)
    start = np.asarray(start, dtype=float)
    fixed = list(fixed or [])
    ident = list(ident or [])
    fixed_pos = {int(p) for p, _ in fixed}
    ident_targets = {int(t) for _, t in ident}
    free_idx = _free_indices(n_full, fixed_pos, ident_targets)
    n_free = len(free_idx)

    free_start = start[free_idx]
    free_bounds = [bounds[i] for i in free_idx]

    def objective(free_params: np.ndarray) -> float:
        full = expand_params(free_params, n_full, fixed=fixed, ident=ident)
        return float(nll_fn(full))

    starts = [free_start]
    if n_starts > 1:
        rng = np.random.default_rng(seed)
        for _ in range(n_starts - 1):
            jitter = rng.normal(scale=0.3, size=n_free)
            starts.append(free_start + jitter)

    best = None
    for guess in starts:
        res = minimize(objective, guess, method="L-BFGS-B", bounds=free_bounds)
        if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
            best = res

    full_params = expand_params(best.x, n_full, fixed=fixed, ident=ident)

    se = np.full(n_free, np.nan)
    hessian = _numerical_hessian(objective, best.x)
    try:
        cov = np.linalg.inv(hessian)
        diag = np.diag(cov)
        with np.errstate(invalid="ignore"):
            se = np.sqrt(np.where(diag >= 0, diag, np.nan))
    except np.linalg.LinAlgError:
        pass  # singular Hessian (e.g. a parameter at a boundary) — SEs stay NaN.

    log_l = -float(best.fun)
    aic = 2 * n_free - 2 * log_l
    bic = n_free * np.log(n_obs) - 2 * log_l

    return SDTFitResult(
        params=best.x,
        full_params=full_params,
        se=se,
        nll=float(best.fun),
        logL=log_l,
        aic=aic,
        bic=bic,
        hessian=hessian,
        success=bool(best.success),
        n_obs=n_obs,
        n_free_params=n_free,
    )
