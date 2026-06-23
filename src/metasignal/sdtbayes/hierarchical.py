"""Hierarchical Bayesian meta-d' models using brms (via brmspy).

Model description
-----------------
Confidence ratings are modelled as an **ordered (cumulative logistic)**
outcome.  The key predictor is ``correct`` (1 = accurate response,
0 = error): its population-level coefficient is proportional to meta-d'
at the group level, while the random slope ``(correct | participant)``
captures between-subject variability.

The formula mirrors the HMeta-d formulation of Fleming (2017) but is
specified as a mixed-effects ordered regression rather than a custom
Stan program, making it extensible with standard brms syntax.

References
----------
Fleming, S. M. (2017). HMeta-d: hierarchical Bayesian estimation of
metacognitive efficiency from confidence ratings. *Neuroscience of
Consciousness*, 2017(1), nix007. https://doi.org/10.1093/nc/nix007
"""

from __future__ import annotations

from typing import Any

import numpy as np

from metasignal.sdtbayes.diagnostics import FitResult


def _trials_to_dataframe(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    group_label: int = 0,
    pid_prefix: str = "P",
) -> "pd.DataFrame":
    """Convert a list of (stim, resp, conf) tuples to a long-format DataFrame."""
    import pandas as pd
    frames = []
    for pid, (stim, resp, conf) in enumerate(participants):
        stim = np.asarray(stim, dtype=int)
        resp = np.asarray(resp, dtype=int)
        conf = np.asarray(conf, dtype=int)
        frames.append(pd.DataFrame({
            "participant": f"{pid_prefix}{pid:03d}",
            "stimulus": stim,
            "response": resp,
            "conf": conf,
            "correct": (stim == resp).astype(int),
            "group": group_label,
        }))
    df = pd.concat(frames, ignore_index=True)
    df["conf"] = pd.Categorical(
        df["conf"],
        categories=list(range(1, n_ratings + 1)),
        ordered=True,
    )
    return df


def fit_hierarchical_metad(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    **kwargs: Any,
) -> Any:
    """Fit a hierarchical Bayesian meta-d' model across participants.

    Uses an ordered logistic (cumulative) regression where confidence
    ratings are the outcome and ``correct`` is the key predictor.
    Random slopes per participant allow meta-d' to vary across the group.

    The population-level coefficient ``b_correct`` is proportional to
    group-mean meta-d'; the random slopes in ``r_participant`` capture
    individual meta-d' deviations from the group mean.

    Args:
        participants: List of ``(stim, resp, conf)`` tuples, one per participant.
            ``stim`` and ``resp`` are binary (0/1); ``conf`` is an integer
            from 1 to ``n_ratings``.
        n_ratings: Number of confidence rating categories.
        chains: Number of MCMC chains (default 4).
        iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup (burn-in) iterations per chain (default 1000).
        seed: Random seed for reproducibility (default 42).
        **kwargs: Additional arguments forwarded to ``brmspy.brms.brm``.

    Returns:
        ``FitResult`` with:

        - ``.idata`` — ArviZ ``InferenceData`` object containing posterior
            samples, log-likelihood, and prior predictive (if requested).
        - ``.r`` — Lightweight R object handle for downstream brms calls.

    Raises:
        ImportError: If ``brmspy`` is not installed.

    Example::

        import numpy as np
        from metasignal.sdtbayes import fit_hierarchical_metad, posterior_summary

        rng = np.random.default_rng(0)
        participants = [
            (rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.integers(1, 5, 200))
            for _ in range(20)
        ]

        fit = fit_hierarchical_metad(participants, n_ratings=4)
        print(posterior_summary(fit))
    """
    try:
        from brmspy import brms
    except ImportError as e:
        raise ImportError(
            "brmspy is not installed. Run:\n    pip install metasignal[sdtbayes]"
        ) from e

    df = _trials_to_dataframe(participants, n_ratings)

    formula = brms.bf("conf ~ correct + (correct | participant)")

    priors = [
        brms.prior("normal(0, 2)", class_="b"),
        brms.prior("normal(0, 2)", class_="Intercept"),
        brms.prior("exponential(1)", class_="sd"),
        brms.prior("lkj(2)", class_="cor"),
    ]

    _result = brms.brm(
        formula=formula,
        data=df,
        family="cumulative",
        priors=priors,
        chains=chains,
        iter=iter,
        warmup=warmup,
        seed=seed,
        **kwargs,
    )
    return FitResult(idata=_result.idata, r=_result.r)


def fit_group_comparison(
    group_a: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    group_b: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    **kwargs: Any,
) -> Any:
    """Bayesian comparison of metacognition between two groups.

    Extends the hierarchical meta-d' model with a ``group`` predictor and a
    ``correct × group`` interaction.  The interaction coefficient
    ``b_correct:group`` represents the difference in meta-d' between groups
    on the log-odds scale, with a full posterior distribution.

    Args:
        group_a: Participants in group A — list of ``(stim, resp, conf)`` tuples.
        group_b: Participants in group B — list of ``(stim, resp, conf)`` tuples.
        n_ratings: Number of confidence rating categories.
        chains: Number of MCMC chains (default 4).
        iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations per chain (default 1000).
        seed: Random seed (default 42).
        **kwargs: Additional arguments forwarded to ``brmspy.brms.brm``.

    Returns:
        ``FitResult`` — same structure as :func:`fit_hierarchical_metad`.
        The parameter of interest is ``b_correct:group1`` (group B minus
        group A difference in meta-d').

    Raises:
        ImportError: If ``brmspy`` is not installed.

    Example::

        fit = fit_group_comparison(healthy, patient, n_ratings=4)
        # Posterior probability that group B has lower meta-d' than group A:
        import arviz as az
        post = az.extract(fit.idata)["b_correct:group1"].values
        print(f"P(group B < group A): {(post < 0).mean():.3f}")
    """
    try:
        from brmspy import brms
    except ImportError as e:
        raise ImportError(
            "brmspy is not installed. Run:\n    pip install metasignal[sdtbayes]"
        ) from e

    import pandas as pd
    df_a = _trials_to_dataframe(group_a, n_ratings, group_label=0, pid_prefix="A")
    df_b = _trials_to_dataframe(group_b, n_ratings, group_label=1, pid_prefix="B")
    df = pd.concat([df_a, df_b], ignore_index=True)
    df["group"] = df["group"].astype("category")

    formula = brms.bf("conf ~ correct * group + (correct | participant)")

    priors = [
        brms.prior("normal(0, 2)", class_="b"),
        brms.prior("normal(0, 2)", class_="Intercept"),
        brms.prior("exponential(1)", class_="sd"),
        brms.prior("lkj(2)", class_="cor"),
    ]

    _result = brms.brm(
        formula=formula,
        data=df,
        family="cumulative",
        priors=priors,
        chains=chains,
        iter=iter,
        warmup=warmup,
        seed=seed,
        **kwargs,
    )
    return FitResult(idata=_result.idata, r=_result.r)
