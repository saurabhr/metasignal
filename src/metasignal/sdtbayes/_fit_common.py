"""Shared helpers for the brmspy-backed two-stage sdtbayes wrappers.

Centralizes the brmspy import guard and the per-participant Stage-1 MLE
estimate duplicated across :mod:`two_stage` and :mod:`within_subject`.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

_BRMSPY_MSG = (
    "brmspy is not installed. Run:\n"
    '    pip install "metasignal[sdtbayes] @ git+https://github.com/saurabhr/metasignal.git"'
)


def require_brms() -> Any:
    """Import and return brmspy's ``brms`` module, or raise a clear ImportError."""
    try:
        from brmspy import brms
    except ImportError as e:
        raise ImportError(_BRMSPY_MSG) from e
    return brms


def compute_mle_row(
    stim: np.ndarray,
    resp: np.ndarray,
    conf: np.ndarray,
    n_ratings: int,
    label: str,
) -> dict[str, float]:
    """Fit meta-d' MLE for one participant/condition.

    Returns a dict with ``dprime``, ``c``, ``meta_da``, ``da``, ``m_ratio``,
    ``log_m_ratio``. MLE failures and non-positive M-ratio are treated as
    missing (NaN, with a warning) rather than raised, since callers simply
    drop invalid rows before the group-level fit.
    """
    from metasignal.stdpy.core import compute_sdt_resp, trials_to_counts
    from metasignal.stdpy.metad import fit_meta_d_mle

    stim = np.asarray(stim)
    resp = np.asarray(resp)
    conf = np.asarray(conf)

    try:
        dp, c, _ = compute_sdt_resp(stim, resp)
        nr_s1, nr_s2 = trials_to_counts(stim, resp, conf, n_ratings)
        mle = fit_meta_d_mle(nr_s1, nr_s2)
        meta_da = float(mle["meta_da"])
        da = float(mle["da"])
        m_ratio = float(mle["M_ratio"])
        if m_ratio > 0:
            log_m_ratio = float(np.log(m_ratio))
        else:
            warnings.warn(
                f"{label}: MLE succeeded but M-ratio={m_ratio:.3g} <= 0 "
                "(non-positive metacognitive efficiency); log_m_ratio set to NaN "
                "and this participant will be excluded from log-scale group models.",
                stacklevel=3,
            )
            log_m_ratio = np.nan
    except (ValueError, RuntimeError) as exc:
        warnings.warn(f"{label}: MLE failed ({exc}). Setting estimates to NaN.", stacklevel=3)
        dp = c = meta_da = da = m_ratio = log_m_ratio = np.nan

    return {
        "dprime": float(dp) if not np.isnan(dp) else np.nan,
        "c": float(c) if not np.isnan(c) else np.nan,
        "meta_da": meta_da,
        "da": da,
        "m_ratio": m_ratio,
        "log_m_ratio": log_m_ratio,
    }
