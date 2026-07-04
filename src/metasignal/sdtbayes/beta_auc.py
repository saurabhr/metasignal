"""Hierarchical Beta regression on Type-2 AUC (AUC2).

Instead of meta-d', models Type-2 AUC directly as the outcome, avoiding the
Gaussian SDT assumption entirely.  Since AUC2 ∈ (0, 1), a Beta likelihood is
natural and well-calibrated.

Two-stage approach
------------------
Stage 1 — compute AUC2 per participant using ``stdpy.compute_type2_auc``.
Stage 2 — fit a Beta regression via brms with the AUC2 values as the
           outcome.  The group mean AUC2 is recovered via the logistic
           transformation of the ``Intercept`` parameter::

               from scipy.special import expit
               group_auc2 = expit(posterior["b_Intercept"].mean())

Key posterior parameters
------------------------
- ``b_Intercept`` — group mean AUC2 on the logit scale.
- ``phi`` — precision (larger = less between-participant variability).
- ``b_group1`` (comparison only) — logit-scale difference, group B − group A.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from metasignal.sdtbayes.diagnostics import FitResult

_BRMSPY_MSG = "brmspy is not installed. Run:\n    pip install metasignal[sdtbayes]"


# ---------------------------------------------------------------------------
# Stage-1 helper
# ---------------------------------------------------------------------------

def _compute_auc2_per_participant(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
) -> "pd.DataFrame":
    """Return a DataFrame with columns [participant, auc2]."""
    import pandas as pd
    from metasignal.stdpy.core import trials_to_counts
    from metasignal.stdpy.type2 import compute_type2_auc

    rows = []
    for pid, (stim, resp, conf) in enumerate(participants):
        stim = np.asarray(stim)
        resp = np.asarray(resp)
        conf = np.asarray(conf)
        try:
            nr_s1, nr_s2 = trials_to_counts(stim, resp, conf, n_ratings)
            auc2 = float(compute_type2_auc(nr_s1, nr_s2))
        except (ValueError, RuntimeError) as exc:  # noqa: BLE001
            warnings.warn(
                f"Participant {pid}: AUC2 computation failed ({exc}). Setting to NaN.",
                stacklevel=3,
            )
            auc2 = float("nan")
        rows.append({"participant": pid, "auc2": auc2})
    return pd.DataFrame(rows)


def _clip_auc2(df: "pd.DataFrame", eps: float = 1e-4) -> "pd.DataFrame":
    """Clip AUC2 away from 0 and 1 so the Beta likelihood is defined."""
    df = df.copy()
    df["auc2"] = df["auc2"].clip(eps, 1.0 - eps)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_beta_auc_group(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    chains: int = 4,
    n_iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    **kwargs: Any,
) -> FitResult:
    """Hierarchical Beta regression on Type-2 AUC for a single group.

    Computes AUC2 per participant (Stage 1) and models the group-level
    distribution using a Beta likelihood (Stage 2).  Unlike meta-d'
    approaches this makes no equal-variance Gaussian SDT assumption —
    AUC2 is a non-parametric measure of metacognitive performance.

    The group mean AUC2 is ``expit(b_Intercept)``.

    Args:
        participants: List of ``(stim, resp, conf)`` tuples, one per participant.
        n_ratings: Number of confidence rating categories.
        chains: MCMC chains (default 4).
        n_iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations (default 1000).
        seed: Random seed (default 42).
        **kwargs: Forwarded to ``brmspy.brms.brm``.

    Returns:
        ``FitResult`` with posterior for ``b_Intercept`` (logit-scale group mean
        AUC2) and ``phi`` (precision).

    Raises:
        ImportError: If ``brmspy`` is not installed.
        ValueError: If fewer than 3 participants have valid AUC2 estimates.

    Example::

        import numpy as np
        from scipy.special import expit
        import arviz as az
        from metasignal.sdtbayes import fit_beta_auc_group

        rng = np.random.default_rng(0)
        participants = [
            (rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.integers(1, 5, 200))
            for _ in range(20)
        ]
        fit = fit_beta_auc_group(participants, n_ratings=4)

        intercept = az.extract(fit.idata)["b_Intercept"].values
        print(f"Group mean AUC2: {expit(intercept).mean():.3f}")
    """
    try:
        from brmspy import brms
    except ImportError as e:
        raise ImportError(_BRMSPY_MSG) from e

    df = _compute_auc2_per_participant(participants, n_ratings)
    valid = _clip_auc2(df.dropna(subset=["auc2"]))

    if len(valid) < 3:
        msg = f"Only {len(valid)} participants have valid AUC2 — need at least 3."
        raise ValueError(msg)

    priors = [
        brms.prior("normal(0, 2)", class_="Intercept"),
        brms.prior("gamma(0.1, 0.1)", class_="phi"),
    ]

    _result = brms.brm(
        formula=brms.bf("auc2 ~ 1"),
        data=valid,
        family="beta",
        priors=priors,
        chains=chains,
        iter=n_iter,
        warmup=warmup,
        seed=seed,
        **kwargs,
    )
    return FitResult(idata=_result.idata, r=_result.r)


def fit_beta_auc_comparison(
    group_a: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    group_b: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    chains: int = 4,
    n_iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    **kwargs: Any,
) -> FitResult:
    """Beta regression comparison of Type-2 AUC between two groups.

    The key posterior parameter ``b_group1`` is the logit-scale difference in
    AUC2 between group B and group A.  Posterior probability that group B has
    higher AUC2::

        post = az.extract(fit.idata)["b_group1"].values
        p_b_higher = (post > 0).mean()

    Args:
        group_a: Participants in group A — list of ``(stim, resp, conf)`` tuples.
        group_b: Participants in group B — list of ``(stim, resp, conf)`` tuples.
        n_ratings: Number of confidence rating categories.
        chains: MCMC chains (default 4).
        n_iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations (default 1000).
        seed: Random seed (default 42).
        **kwargs: Forwarded to ``brmspy.brms.brm``.

    Returns:
        ``FitResult`` with posterior for ``b_Intercept`` (group A logit-AUC2),
        ``b_group1`` (group B − A difference), and ``phi``.

    Raises:
        ImportError: If ``brmspy`` is not installed.
        ValueError: If either group has fewer than 3 participants with valid
            AUC2 estimates.

    Example::

        fit = fit_beta_auc_comparison(healthy, patient, n_ratings=4)
        import arviz as az
        delta = az.extract(fit.idata)["b_group1"].values
        print(f"P(patient AUC2 < healthy): {(delta < 0).mean():.3f}")
    """
    try:
        from brmspy import brms
        import pandas as pd
    except ImportError as e:
        raise ImportError(_BRMSPY_MSG) from e

    df_a = _compute_auc2_per_participant(group_a, n_ratings)
    df_a["group"] = 0
    df_b = _compute_auc2_per_participant(group_b, n_ratings)
    df_b["group"] = 1
    df = pd.concat([df_a, df_b], ignore_index=True)
    df["group"] = df["group"].astype("category")
    valid = _clip_auc2(df.dropna(subset=["auc2"]))

    for g, label in ((0, "A"), (1, "B")):
        n_valid = int((valid["group"] == g).sum())
        if n_valid < 3:
            total = int((df["group"] == g).sum())
            msg = (
                f"Group {label}: only {n_valid} of {total} participants have valid "
                "AUC2 estimates — need at least 3."
            )
            raise ValueError(msg)

    priors = [
        brms.prior("normal(0, 2)", class_="Intercept"),
        brms.prior("normal(0, 1)", class_="b"),
        brms.prior("gamma(0.1, 0.1)", class_="phi"),
    ]

    _result = brms.brm(
        formula=brms.bf("auc2 ~ group"),
        data=valid,
        family="beta",
        priors=priors,
        chains=chains,
        iter=n_iter,
        warmup=warmup,
        seed=seed,
        **kwargs,
    )
    return FitResult(idata=_result.idata, r=_result.r)
