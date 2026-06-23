"""Option A — Two-stage hierarchical Bayesian meta-d' model.

Stage 1: compute per-participant MLE estimates (meta-d', d', M-ratio) using
``stdpy.fit_meta_d_mle``.

Stage 2: fit a hierarchical Bayesian model on the per-participant log M-ratio
values using brms (via brmspy).  The group-level posterior over
``mu_logMratio`` and ``sigma_logMratio`` gives the group mean and variability
of metacognitive efficiency.

This approach does not propagate Stage-1 estimation uncertainty into Stage 2,
but is computationally fast (Stage 2 runs in seconds) and suitable for the
typical sample sizes in metacognition research (20–50 participants with
100–300 trials each).  For a fully Bayesian single-stage model, see
:mod:`metasignal.sdtbayes.full_metad`.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _compute_participant_estimates(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
) -> "pd.DataFrame":
    """Run MLE on each participant and return a summary DataFrame."""
    import pandas as pd
    from metasignal.stdpy.core import trials_to_counts
    from metasignal.stdpy.metad import fit_meta_d_mle
    from metasignal.stdpy.core import compute_sdt_resp

    rows = []
    for pid, (stim, resp, conf) in enumerate(participants):
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
        except Exception:
            meta_da = da = m_ratio = log_m_ratio = np.nan

        rows.append({
            "participant": pid,
            "dprime": float(dp) if not np.isnan(dp) else np.nan,
            "c": float(c) if not np.isnan(c) else np.nan,
            "meta_da": meta_da,
            "da": da,
            "m_ratio": m_ratio,
            "log_m_ratio": log_m_ratio,
        })

    return pd.DataFrame(rows)


def fit_two_stage_group(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    **kwargs: Any,
) -> tuple[Any, "pd.DataFrame"]:
    """Two-stage hierarchical Bayesian model for group-level M-ratio.

    Runs MLE estimation per participant (Stage 1) then fits a hierarchical
    Bayesian model over log M-ratio across participants (Stage 2).  The key
    group-level parameters are:

    - ``b_Intercept`` — posterior mean log M-ratio (exp gives group mean M-ratio)
    - ``sigma`` — between-subject SD on the log scale

    Args:
        participants: List of ``(stim, resp, conf)`` tuples, one per participant.
        n_ratings: Number of confidence rating categories.
        chains: MCMC chains (default 4).
        iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations (default 1000).
        seed: Random seed (default 42).
        **kwargs: Forwarded to ``brmspy.brms.brm``.

    Returns:
        Tuple of ``FitResult`` and ``pd.DataFrame``.

        The ``FitResult`` has ``.idata`` (ArviZ InferenceData) and ``.r``
        (R handle). The key population-level parameter is ``b_Intercept``
        (group mean log M-ratio).

        The ``pd.DataFrame`` contains per-participant Stage-1 MLE estimates
        (columns: ``participant``, ``dprime``, ``c``, ``meta_da``, ``da``,
        ``m_ratio``, ``log_m_ratio``).

    Raises:
        ImportError: If ``brmspy`` is not installed.

    Example::

        import numpy as np
        from metasignal.sdtbayes import fit_two_stage_group, posterior_summary

        rng = np.random.default_rng(0)
        participants = [
            (rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.integers(1, 5, 200))
            for _ in range(20)
        ]

        fit, mle_df = fit_two_stage_group(participants, n_ratings=4)
        print(mle_df[["participant", "m_ratio", "log_m_ratio"]])
        print(posterior_summary(fit, var_names=["b_Intercept", "sigma"]))
    """
    try:
        from brmspy import brms
    except ImportError as e:
        raise ImportError(
            "brmspy is not installed. Run:\n    pip install metasignal[sdtbayes]"
        ) from e

    mle_df = _compute_participant_estimates(participants, n_ratings)

    valid = mle_df.dropna(subset=["log_m_ratio"])
    if len(valid) < 3:
        msg = f"Only {len(valid)} participants have valid MLE estimates — need at least 3."
        raise ValueError(msg)

    formula = brms.bf("log_m_ratio ~ 1")
    priors = [
        brms.prior("normal(0, 1)", class_="Intercept"),
        brms.prior("exponential(1)", class_="sigma"),
    ]

    fit = brms.brm(
        formula=formula,
        data=valid,
        family="student",
        priors=priors,
        chains=chains,
        iter=iter,
        warmup=warmup,
        seed=seed,
        **kwargs,
    )
    return fit, mle_df


def fit_two_stage_comparison(
    group_a: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    group_b: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    **kwargs: Any,
) -> tuple[Any, "pd.DataFrame"]:
    """Two-stage Bayesian comparison of M-ratio between two groups.

    MLE is run per participant (Stage 1); a Bayesian regression with a
    ``group`` predictor is fit on log M-ratio (Stage 2).  The coefficient
    ``b_group1`` is the posterior difference in log M-ratio (group B − group A).

    Args:
        group_a: Participants in group A.
        group_b: Participants in group B.
        n_ratings: Number of confidence rating categories.
        chains: MCMC chains (default 4).
        iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations (default 1000).
        seed: Random seed (default 42).
        **kwargs: Forwarded to ``brmspy.brms.brm``.

    Returns:
        Tuple of ``FitResult`` and per-participant MLE DataFrame (with a
        ``group`` column, 0 = A, 1 = B).

    Raises:
        ImportError: If ``brmspy`` is not installed.

    Example::

        fit, mle_df = fit_two_stage_comparison(healthy, patient, n_ratings=4)

        # Posterior probability that group B has lower M-ratio than group A
        import arviz as az
        post = az.extract(fit.idata)["b_group1"].values
        print(f"P(group B < group A): {(post < 0).mean():.3f}")
    """
    try:
        from brmspy import brms
    except ImportError as e:
        raise ImportError(
            "brmspy is not installed. Run:\n    pip install metasignal[sdtbayes]"
        ) from e

    import pandas as pd

    df_a = _compute_participant_estimates(group_a, n_ratings)
    df_a["group"] = 0
    df_b = _compute_participant_estimates(group_b, n_ratings)
    df_b["group"] = 1
    # Re-index participant IDs so they are unique across groups
    df_b["participant"] = df_b["participant"] + len(group_a)
    mle_df = pd.concat([df_a, df_b], ignore_index=True)
    mle_df["group"] = mle_df["group"].astype("category")

    valid = mle_df.dropna(subset=["log_m_ratio"])

    formula = brms.bf("log_m_ratio ~ group")
    priors = [
        brms.prior("normal(0, 1)", class_="Intercept"),
        brms.prior("normal(0, 1)", class_="b"),
        brms.prior("exponential(1)", class_="sigma"),
    ]

    fit = brms.brm(
        formula=formula,
        data=valid,
        family="student",
        priors=priors,
        chains=chains,
        iter=iter,
        warmup=warmup,
        seed=seed,
        **kwargs,
    )
    return fit, mle_df
