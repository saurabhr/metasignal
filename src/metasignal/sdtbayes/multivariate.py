"""Bivariate hierarchical model for the M-ratio family (log M-ratio and d').

Models log M-ratio and d' jointly using a bivariate normal distribution at
the group level, capturing the cross-participant correlation between
metacognitive efficiency and perceptual sensitivity.

Two-stage approach
------------------
Stage 1 — MLE per participant (meta_da, da, M_ratio, M_diff, log_m_ratio).
Stage 2 — brms multivariate normal regression on (log_m_ratio, dprime) with
           estimated residual correlation (``rescor=TRUE``).

Key posterior parameters
------------------------
- ``b_logmratio_Intercept`` — group mean log M-ratio.
  Convert: ``exp(b_logmratio_Intercept)`` = group mean M-ratio.
- ``b_dprime_Intercept`` — group mean d'.
- ``sigma_logmratio``, ``sigma_dprime`` — between-subject SDs.
- ``rescor__logmratio__dprime`` — posterior correlation between log M-ratio
  and d' across participants.  Positive values mean participants with higher
  d' also tend to have higher metacognitive efficiency.
- ``b_dprime_group1``, ``b_logmratio_group1`` (comparison only) — group B − A
  differences for each outcome.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from metasignal.sdtbayes.diagnostics import FitResult
from metasignal.sdtbayes.two_stage import _compute_participant_estimates

_BRMSPY_MSG = "brmspy is not installed. Run:\n    pip install metasignal[sdtbayes]"


def fit_multivariate_mratio(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    chains: int = 4,
    n_iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    **kwargs: Any,
) -> FitResult:
    """Bivariate hierarchical model for log M-ratio and d' (single group).

    Estimates the joint group-level distribution of metacognitive efficiency
    (log M-ratio) and perceptual sensitivity (d') and their correlation.  This
    answers whether participants with stronger perceptual sensitivity also show
    higher metacognitive efficiency in your sample.

    Args:
        participants: List of ``(stim, resp, conf)`` tuples, one per participant.
        n_ratings: Number of confidence rating categories.
        chains: MCMC chains (default 4).
        n_iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations (default 1000).
        seed: Random seed (default 42).
        **kwargs: Forwarded to ``brmspy.brms.brm``.

    Returns:
        ``FitResult``.  Key parameters:

        - ``b_logmratio_Intercept`` — group mean log M-ratio.
        - ``b_dprime_Intercept`` — group mean d'.
        - ``sigma_logmratio``, ``sigma_dprime`` — between-subject SDs.
        - ``rescor__logmratio__dprime`` — cross-participant correlation.

    Raises:
        RuntimeError: Always — see Notes.
        ValueError: If fewer than 5 participants have valid MLE estimates.

    Notes:
        This function's brmspy path is currently **non-functional**: brms's
        ``rescor`` flag (needed to estimate the residual correlation between
        the two multivariate responses) cannot be threaded through brmspy to
        the R side in this version. Three approaches were tried, each
        failing differently: (1) ``rescor=True`` passed to ``brm()`` reaches
        the low-level cmdstanr sampling call, which rejects it as an unused
        argument; (2) combining ``set_rescor(TRUE)`` into the formula via
        ``+`` fails because brmspy's formula objects can't round-trip through
        a generic ``brms.call()``; (3) passing ``rescor`` via ``formula_args``
        (threaded into ``bf()``) makes brms misinterpret it as a fixed
        distributional parameter name. No working channel was found.

        There is currently no cmdstanpy-backed replacement for jointly
        modelling d' and log M-ratio with an estimated cross-correlation.  If
        you only need d', use :func:`~metasignal.sdtbayes.fit_full_metad`
        (which reports per-subject ``d1``) — it just won't estimate the
        correlation with metacognitive efficiency.

    Example::

        import numpy as np
        import arviz as az
        from metasignal.sdtbayes import fit_multivariate_mratio

        rng = np.random.default_rng(0)
        participants = [
            (rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.integers(1, 5, 200))
            for _ in range(30)
        ]
        fit = fit_multivariate_mratio(participants, n_ratings=4)

        post = az.extract(fit.idata)
        r = post["rescor__logmratio__dprime"].values
        print(f"Correlation(log M-ratio, d'): {r.mean():.3f}  "
              f"[{np.percentile(r, 3):.3f}, {np.percentile(r, 97):.3f}]")
    """
    mle_df = _compute_participant_estimates(participants, n_ratings)
    valid = mle_df.dropna(subset=["log_m_ratio", "dprime"])

    if len(valid) < 5:
        msg = f"Only {len(valid)} participants have valid estimates — need at least 5."
        raise ValueError(msg)

    raise RuntimeError(
        "fit_multivariate_mratio is currently unavailable: brms's rescor flag "
        "(needed for the cross-response correlation) cannot be threaded through "
        "brmspy to R in this version — three distinct approaches were tried and "
        "each failed differently. Use fit_full_metad(...) instead if you only "
        "need d' and log M-ratio separately (no correlation estimate)."
    )


def fit_multivariate_mratio_comparison(
    group_a: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    group_b: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    chains: int = 4,
    n_iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    **kwargs: Any,
) -> FitResult:
    """Bivariate group comparison of log M-ratio and d'.

    Tests whether two groups differ simultaneously in log M-ratio and d',
    capturing the joint posterior of both contrasts and their correlation.

    Key posterior parameters:

    - ``b_logmratio_group1`` — group B − A difference in log M-ratio.
    - ``b_dprime_group1`` — group B − A difference in d'.
    - ``rescor__logmratio__dprime`` — within-group correlation (shared).

    Args:
        group_a: Participants in group A.
        group_b: Participants in group B.
        n_ratings: Number of confidence rating categories.
        chains: MCMC chains (default 4).
        n_iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations (default 1000).
        seed: Random seed (default 42).
        **kwargs: Forwarded to ``brmspy.brms.brm``.

    Returns:
        ``FitResult`` with bivariate group comparison posteriors.

    Raises:
        RuntimeError: Always — see :func:`fit_multivariate_mratio` Notes for
            why brms's ``rescor`` flag cannot currently be used through
            brmspy.

    Example::

        fit = fit_multivariate_mratio_comparison(healthy, patient, n_ratings=4)
        import arviz as az, numpy as np
        post = az.extract(fit.idata)
        d_mr = post["b_logmratio_group1"].values
        d_dp = post["b_dprime_group1"].values
        print(f"P(log M-ratio lower in patients): {(d_mr < 0).mean():.3f}")
        print(f"P(d' lower in patients):          {(d_dp < 0).mean():.3f}")
    """
    raise RuntimeError(
        "fit_multivariate_mratio_comparison is currently unavailable: brms's "
        "rescor flag (needed for the cross-response correlation) cannot be "
        "threaded through brmspy to R in this version. Use "
        "fit_full_metad_comparison(...) instead if you only need the group "
        "difference in log M-ratio (no correlation-with-d' estimate)."
    )
