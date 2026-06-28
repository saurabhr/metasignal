"""Approach 12 — Two-stage within-subject condition comparison.

Models the difference in log M-ratio between two conditions tested on the
**same participants**.  This is the within-subject analogue of the between-
group comparison in :mod:`metasignal.sdtbayes.two_stage`.

Stage 1 — MLE per participant per condition
    :func:`~metasignal.stdpy.fit_meta_d_mle` is run independently on each
    participant's data for condition A and condition B.

Stage 2 — Bayesian paired model
    A mixed-effects model with a ``condition`` fixed effect and a participant
    random intercept captures the within-subject correlation::

        log_m_ratio ~ condition + (1 | participant)

    The coefficient ``b_condition1`` is the posterior difference in log M-ratio
    (condition B − condition A).  Because the random intercept absorbs stable
    between-participant differences, this estimate is substantially more
    powerful than the between-subjects version and directly answers the question
    "does metacognitive efficiency change between conditions?".

Key posterior parameters
------------------------
- ``b_Intercept`` — group mean log M-ratio for condition A.
- ``b_condition1`` — condition B − condition A difference on the log scale.
  ``exp(b_condition1)`` is the M-ratio ratio: values > 1 mean higher
  metacognition in condition B.
- ``sd_participant__Intercept`` — between-participant SD (stable differences
  absorbed by the random intercept).
- ``sigma`` — residual within-participant variability.

References
----------
Nakagawa, S., & Schielzeth, H. (2013). A general and simple method for
obtaining R² from generalized linear mixed-effects models.
*Methods in Ecology and Evolution*, 4(2), 133–142.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from metasignal.sdtbayes.diagnostics import FitResult

_BRMSPY_MSG = "brmspy is not installed. Run:\n    pip install metasignal[sdtbayes]"


def _compute_paired_estimates(
    condition_a: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    condition_b: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
) -> "pd.DataFrame":
    """Run MLE per participant per condition and return a long-format DataFrame.

    Args:
        condition_a: Per-participant ``(stim, resp, conf)`` tuples for condition A.
        condition_b: Per-participant ``(stim, resp, conf)`` tuples for condition B.
            Must be the same length as ``condition_a`` and in the same participant order.
        n_ratings: Number of confidence rating categories.

    Returns:
        Long-format ``pd.DataFrame`` with columns:
        ``participant``, ``condition`` (0=A, 1=B),
        ``dprime``, ``c``, ``meta_da``, ``da``, ``m_ratio``, ``log_m_ratio``.
    """
    import pandas as pd
    from metasignal.stdpy.core import compute_sdt_resp, trials_to_counts
    from metasignal.stdpy.metad import fit_meta_d_mle

    rows = []
    for cond_label, data in ((0, condition_a), (1, condition_b)):
        for pid, (stim, resp, conf) in enumerate(data):
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
                log_m_ratio = float(np.log(m_ratio)) if m_ratio > 0 else np.nan
            except (ValueError, RuntimeError) as exc:
                warnings.warn(
                    f"Participant {pid}, condition {cond_label}: MLE failed ({exc}). "
                    "Setting estimates to NaN.",
                    stacklevel=2,
                )
                dp = c = meta_da = da = m_ratio = log_m_ratio = np.nan

            rows.append({
                "participant": pid,
                "condition": cond_label,
                "dprime": float(dp),
                "c": float(c),
                "meta_da": meta_da,
                "da": da,
                "m_ratio": m_ratio,
                "log_m_ratio": log_m_ratio,
            })

    return pd.DataFrame(rows)


def fit_within_subject_comparison(
    condition_a: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    condition_b: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    chains: int = 4,
    n_iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    **kwargs: Any,
) -> tuple[FitResult, "pd.DataFrame"]:
    """Two-stage within-subject comparison of metacognitive efficiency across conditions.

    Estimates the change in log M-ratio between two conditions tested within
    the same participants.  Because individual participants serve as their own
    controls, this design is statistically more efficient than between-subject
    group comparisons and controls for stable differences in metacognitive
    ability.

    The key estimand is ``b_condition1``, the posterior distribution of the
    within-person condition-B − condition-A difference in log M-ratio.

    Args:
        condition_a: Per-participant ``(stim, resp, conf)`` tuples for condition A.
        condition_b: Per-participant ``(stim, resp, conf)`` tuples for condition B.
            Must have the same length as ``condition_a`` and list participants in
            the same order.
        n_ratings: Number of confidence rating categories.
        chains: MCMC chains (default 4).
        n_iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations (default 1000).
        seed: Random seed (default 42).
        **kwargs: Forwarded to ``brmspy.brms.brm``.

    Returns:
        Tuple of ``(FitResult, mle_df)``.

        - ``FitResult.idata`` — ArviZ InferenceData.  Key parameter:
          ``b_condition1`` = condition B − condition A difference.
        - ``mle_df`` — long-format ``pd.DataFrame`` with Stage-1 MLE estimates
          for all participants × conditions.

    Raises:
        ImportError: If ``brmspy`` is not installed.
        ValueError: If ``condition_a`` and ``condition_b`` have different lengths,
            or if fewer than 3 participants have valid estimates in both conditions.

    Example::

        import numpy as np
        import arviz as az
        from metasignal.sdtbayes import fit_within_subject_comparison

        rng = np.random.default_rng(0)
        n_part = 25

        # Same 25 participants, two experimental conditions
        condition_a = [
            (rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.integers(1, 5, 200))
            for _ in range(n_part)
        ]
        condition_b = [
            (rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.integers(1, 5, 200))
            for _ in range(n_part)
        ]

        fit, mle_df = fit_within_subject_comparison(condition_a, condition_b, n_ratings=4)

        # Posterior probability that condition B has higher metacognition
        post = az.extract(fit.idata)["b_condition1"].values
        print(f"P(M-ratio higher in B): {(post > 0).mean():.3f}")

        # Condition-B minus condition-A M-ratio ratio
        print(f"Median M-ratio ratio B/A: {np.exp(np.median(post)):.3f}")
    """
    try:
        from brmspy import brms
        import pandas as pd
    except ImportError as e:
        raise ImportError(_BRMSPY_MSG) from e

    if len(condition_a) != len(condition_b):
        msg = (
            f"condition_a has {len(condition_a)} participants but condition_b has "
            f"{len(condition_b)}.  Both must list the same participants in the same order."
        )
        raise ValueError(msg)

    mle_df = _compute_paired_estimates(condition_a, condition_b, n_ratings)
    mle_df["condition"] = mle_df["condition"].astype("category")

    valid = mle_df.dropna(subset=["log_m_ratio"])

    # Require each participant to have at least one valid condition
    valid_participants = valid["participant"].nunique()
    if valid_participants < 3:
        msg = (
            f"Only {valid_participants} participants have any valid MLE estimates — "
            "need at least 3."
        )
        raise ValueError(msg)

    formula = brms.bf("log_m_ratio ~ condition + (1 | participant)")
    priors = [
        brms.prior("normal(0, 1)", class_="Intercept"),
        brms.prior("normal(0, 1)", class_="b"),
        brms.prior("exponential(1)", class_="sd"),
        brms.prior("exponential(1)", class_="sigma"),
    ]

    _result = brms.brm(
        formula=formula,
        data=valid,
        family="student",
        priors=priors,
        chains=chains,
        iter=n_iter,
        warmup=warmup,
        seed=seed,
        **kwargs,
    )
    return FitResult(idata=_result.idata, r=_result.r), mle_df
