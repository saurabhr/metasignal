"""Base Gaussian signal detection model.

Port of the ``"SDT"`` and ``"Gaussian"``-with-``restriction="standard"``
models from Macho (2020), *SDT-Models in R*, Ch. 3.1–3.2
(https://www.unifr.ch/psycho/fr/assets/public/Forschungseinheiten/sdt/SDT.pdf).

Model
-----
``n_signals`` Gaussian distributions share a single set of ``n_categories - 1``
decision thresholds. Signal 0 is the fixed reference/noise distribution,
``N(0, 1)``; signals ``1 .. n_signals - 1`` have free ``(mean, sd)``
parameters (``restriction="equalvar"`` additionally fixes every ``sd`` to 1,
recovering the classic equal-variance model).

ponytail: per-signal threshold sets (the genuinely new capability of Macho's
``"Gaussian"`` model beyond ``"SDT"``, restriction options ``"no"``/``"symmetric"``)
and non-Gaussian restriction identification schemes are not implemented —
this module always ties all signals to one shared threshold set (§3.2's
``"standard"`` restriction, which the manual states is identical to the
``"SDT"`` model). Extend ``fit_sdt`` if a later use case needs per-signal
thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

from metasignal.sdtr._optimize import SDTFitResult, fit_mle

Restriction = Literal["no", "equalvar"]


@dataclass
class SDTModelFit:
    """Fitted base Gaussian SDT model for one participant/cell."""

    means: np.ndarray          # length n_signals - 1 (signal 0 fixed at 0)
    sds: np.ndarray            # length n_signals - 1 (signal 0 fixed at 1)
    thresholds: np.ndarray     # length n_categories - 1, sorted ascending
    d_a: np.ndarray            # length n_signals - 1
    d_e: np.ndarray            # length n_signals - 1
    A_z: np.ndarray            # length n_signals - 1
    logL: float
    aic: float
    bic: float
    success: bool
    fit: SDTFitResult


def fit_sdt(
    counts,
    *,
    restriction: Restriction = "no",
    n_starts: int = 1,
    seed: int = 42,
) -> SDTModelFit:
    """Fit the base Gaussian SDT model by maximum likelihood.

    Parameters
    ----------
    counts:
        2D array-like, shape ``(n_signals, n_categories)`` — response
        frequency counts. Row 0 is the reference/noise signal; row order
        should go from lowest to highest signal strength. Column order goes
        from the most noise-like response category to the most signal-like.
    restriction:
        ``"no"`` (default): each non-reference signal has a free standard
        deviation (unequal-variance SDT).
        ``"equalvar"``: all standard deviations fixed to 1.
    n_starts:
        Number of optimizer starts (see :func:`metasignal.sdtr._optimize.fit_mle`).
    seed:
        RNG seed for additional multi-start jitter when ``n_starts > 1``.

    Returns
    -------
    SDTModelFit
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or counts.shape[0] < 2 or counts.shape[1] < 2:
        raise ValueError(
            f"counts must be 2D with at least 2 signals and 2 categories, "
            f"got shape {counts.shape}."
        )
    n_signals, n_categories = counts.shape
    n_free_signals = n_signals - 1
    n_thresholds = n_categories - 1

    n_full = 2 * n_free_signals + n_thresholds
    mean_pos = [2 * j for j in range(n_free_signals)]
    sd_pos = [2 * j + 1 for j in range(n_free_signals)]
    thresh_pos = list(range(2 * n_free_signals, n_full))

    start = _starting_values(counts, mean_pos, sd_pos, thresh_pos, n_full)
    bounds = [(-10.0, 10.0)] * n_full
    for p in sd_pos:
        bounds[p] = (0.05, 10.0)

    fixed = [(p, 1.0) for p in sd_pos] if restriction == "equalvar" else None

    def nll(full_params: np.ndarray) -> float:
        return _neg_log_likelihood(full_params, counts, mean_pos, sd_pos, thresh_pos)

    result = fit_mle(
        nll, start, bounds, fixed=fixed, n_obs=int(counts.sum()),
        n_starts=n_starts, seed=seed,
    )

    means = result.full_params[mean_pos]
    sds = result.full_params[sd_pos]
    thresholds = np.sort(result.full_params[thresh_pos])

    d_a = means * np.sqrt(2.0 / (1.0 + sds**2))
    A_z = norm.cdf(d_a / np.sqrt(2.0))
    d_e = _empirical_d(counts)

    return SDTModelFit(
        means=means, sds=sds, thresholds=thresholds,
        d_a=d_a, d_e=d_e, A_z=A_z,
        logL=result.logL, aic=result.aic, bic=result.bic, success=result.success,
        fit=result,
    )


def _starting_values(counts, mean_pos, sd_pos, thresh_pos, n_full) -> np.ndarray:
    """Data-driven starting point: probit thresholds from signal 0, d'-style means."""
    start = np.zeros(n_full)
    for p in sd_pos:
        start[p] = 1.0

    eps = 1e-5
    noise_cum = np.clip(np.cumsum(counts[0]) / counts[0].sum(), eps, 1 - eps)
    thresh_start = norm.ppf(noise_cum[:-1])
    for i, p in enumerate(thresh_pos):
        start[p] = thresh_start[i] if i < len(thresh_start) else 0.0

    mid = counts.shape[1] // 2
    fa = np.clip(counts[0, mid:].sum() / counts[0].sum(), eps, 1 - eps)
    for j, p in enumerate(mean_pos, start=1):
        hit = np.clip(counts[j, mid:].sum() / counts[j].sum(), eps, 1 - eps)
        start[p] = norm.ppf(hit) - norm.ppf(fa)
    return start


def _neg_log_likelihood(full_params, counts, mean_pos, sd_pos, thresh_pos) -> float:
    means = full_params[mean_pos]
    sds = full_params[sd_pos]
    # ponytail: sorting (rather than a monotonicity constraint) keeps thresholds
    # ordered; upgrade to a constrained optimizer (cf. stdpy.metad's SLSQP +
    # monotonicity inequality) if this proves unstable for many rating categories.
    thresholds = np.sort(full_params[thresh_pos])
    edges = np.concatenate([[-np.inf], thresholds, [np.inf]])

    nll = 0.0
    n_signals = counts.shape[0]
    for j in range(n_signals):
        mean_j = 0.0 if j == 0 else means[j - 1]
        sd_j = 1.0 if j == 0 else sds[j - 1]
        cdf = norm.cdf(edges, mean_j, sd_j)
        probs = np.clip(np.diff(cdf), 1e-12, None)
        nll -= np.sum(counts[j] * np.log(probs))
    return nll


def _empirical_d(counts) -> np.ndarray:
    """Empirical d' per non-reference signal via a single median-split collapse.

    ponytail: exact for the 2-category case (the only case validated against
    Macho's manual); for >2 categories this is a coarse approximation
    (collapsing at the middle category) rather than Macho's specific
    empirical-d' procedure, which the manual does not fully specify.
    """
    eps = 1e-5
    mid = counts.shape[1] // 2
    fa = np.clip(counts[0, mid:].sum() / counts[0].sum(), eps, 1 - eps)
    d_e = np.zeros(counts.shape[0] - 1)
    for j in range(1, counts.shape[0]):
        hit = np.clip(counts[j, mid:].sum() / counts[j].sum(), eps, 1 - eps)
        d_e[j - 1] = norm.ppf(hit) - norm.ppf(fa)
    return d_e
