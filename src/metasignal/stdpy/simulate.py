"""Synthetic data simulation for Type-2 SDT experiments.

Functions
---------
``type2_SDT_simuation``
    Gaussian noise model — port of metadpy / Maniscalco & Lau (2012).
    Confidence is a noisy copy of the type-1 internal signal.

``type2_SDT_simuation_bayes``
    Bayesian ideal observer model (Fleming & Dolan 2012 framework).
    Confidence is the posterior probability of being correct P(correct|x),
    optionally corrupted by noise on the log-odds scale.

``ratings2df``
    Convert nR_S1 / nR_S2 count arrays to a trial-level DataFrame.
    Port of metadpy.utils.ratings2df.

``trialSimulation``
    Simulate a single participant using closed-form SDT probabilities
    (multinomial sampling from normal-CDF integrals).
    Port of metadpy.utils.trialSimulation.

``responseSimulation``
    Multi-subject wrapper around trialSimulation.
    Port of metadpy.utils.responseSimulation.

``pairedResponseSimulation``
    Two-condition repeated-measures simulation with between-subject
    variability and correlated M-ratios.
    Port of metadpy.utils.pairedResponseSimulation.

``discreteRatings``
    Bin continuous confidence ratings into discrete levels via quantiles,
    with automatic correction for floor/ceiling biases.
    Port of metadpy.utils.discreteRatings.

Only numpy, scipy, and pandas are required; no metadpy dependency.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid: 1/(1+exp(-x))
from scipy.stats import norm


def type2_SDT_simuation(
    d: float = 1.0,
    noise: Union[float, list[float]] = 0.2,
    c: float = 0.0,
    nRatings: int = 4,
    nTrials: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Type-2 SDT response counts.

    Generates nR_S1 and nR_S2 count arrays compatible with
    :func:`metasignal.stdpy.fit_meta_d_mle` and
    :func:`metasignal.stdpy.trials_to_counts`.

    Parameters
    ----------
    d :
        Type-1 d-prime (sensitivity).
    noise :
        Standard deviation of metacognitive noise added to the type-1
        internal response for the type-2 judgment.
        Pass a two-element list ``[sigma_rS1, sigma_rS2]`` to simulate
        response-conditional noise.
    c :
        Type-1 decision criterion.
    nRatings :
        Number of confidence rating levels.
    nTrials :
        Number of trials to simulate.
    rng :
        Optional :class:`numpy.random.Generator` for reproducibility.
        If ``None``, uses :func:`numpy.random.default_rng`.

    Returns
    -------
    nR_S1, nR_S2 : ndarray
        Response-count vectors of length ``2 * nRatings``.

        *Ordering (same convention as metadpy / Maniscalco & Lau):*

        ``nR_S1[0 : nRatings]``   — S1 stimulus, S1 response, ratings high→low
        ``nR_S1[nRatings : 2*nRatings]`` — S1 stimulus, S2 response, ratings low→high

        Same layout for nR_S2.

    Examples
    --------
    >>> nR_S1, nR_S2 = type2_SDT_simuation(d=1.5, noise=0.0, c=0, nRatings=4, nTrials=1000)
    >>> len(nR_S1)
    8
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- confidence criteria (same placement as metadpy) ---
    c1 = c + np.linspace(-1.5, -0.5, nRatings - 1)  # S1-response side (left of decision)
    c2 = c + np.linspace(0.5, 1.5, nRatings - 1)    # S2-response side (right of decision)

    # --- response-conditional noise ---
    if isinstance(noise, (list, tuple, np.ndarray)):
        response_conditional = True
        sigma_rS1 = float(noise[0])
        sigma_rS2 = float(noise[1])
    else:
        response_conditional = False
        sigma = float(noise)

    S1mu = -d / 2
    S2mu = d / 2

    # Counts: [nRatings] bins per response side
    nC_rS1 = np.zeros(nRatings)   # correct S1 responses (stim=S1, resp=S1)
    nI_rS1 = np.zeros(nRatings)   # incorrect S1 responses (stim=S2, resp=S1)
    nC_rS2 = np.zeros(nRatings)   # correct S2 responses (stim=S2, resp=S2)
    nI_rS2 = np.zeros(nRatings)   # incorrect S2 responses (stim=S1, resp=S2)

    for _ in range(nTrials):
        # Stimulus: 0 = S1, 1 = S2
        s = int(rng.integers(0, 2))

        # Type-1 internal response
        x = rng.normal(S2mu if s == 1 else S1mu, 1.0)

        # Metacognitive (type-2) response: add noise to x
        if response_conditional:
            sigma_now = sigma_rS1 if x < c else sigma_rS2
        else:
            sigma_now = sigma

        x2 = rng.normal(x, sigma_now) if sigma_now > 0 else x

        # Type-1 decision
        resp = int(x >= c)

        # Assign to confidence bin
        if s == 0 and resp == 0:          # S1 stim, S1 response (correct)
            i = np.where(np.append(c1, c) >= x2)[0]
            if len(i):
                nC_rS1[i.min()] += 1

        elif s == 0 and resp == 1:         # S1 stim, S2 response (incorrect)
            i = np.where(np.append(c, c2) <= x2)[0]
            if len(i):
                nI_rS2[i.max()] += 1

        elif s == 1 and resp == 0:         # S2 stim, S1 response (incorrect)
            i = np.where(np.append(c1, c) >= x2)[0]
            if len(i):
                nI_rS1[i.min()] += 1

        else:                              # S2 stim, S2 response (correct)
            i = np.where(np.append(c, c2) <= x2)[0]
            if len(i):
                nC_rS2[i.max()] += 1

    nR_S1 = np.concatenate([nC_rS1, nI_rS2])
    nR_S2 = np.concatenate([nI_rS1, nC_rS2])

    return nR_S1, nR_S2


def type2_SDT_simuation_bayes(
    d: float = 1.0,
    c: float = 0.0,
    nRatings: int = 4,
    nTrials: int = 500,
    meta_noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Type-2 SDT data using a Bayesian ideal observer.

    Unlike :func:`type2_SDT_simuation`, which uses a Gaussian noise model
    (noisy access to the type-1 evidence signal), this function models an
    observer who bases confidence on the **posterior probability of being
    correct** — P(correct | x) — derived via Bayes' theorem.

    A noiseless observer (``meta_noise=0``) is metacognitively ideal:
    meta-d' = d', giving M_ratio = 1 exactly.  Adding ``meta_noise > 0``
    corrupts access to the posterior on the log-odds scale, reducing M_ratio
    below 1 in proportion to the noise.

    This implements the framework described in Fleming & Dolan (2012) and
    separates the metacognitive noise model from the evidence noise model.

    Parameters
    ----------
    d :
        Type-1 d-prime.
    c :
        Type-1 decision criterion.
    nRatings :
        Number of confidence rating levels.
    nTrials :
        Number of trials.
    meta_noise :
        Standard deviation of Gaussian noise added to the **log-odds** of
        P(correct|x) before binning.  Set to 0 for an ideal Bayesian
        observer (M_ratio ≈ 1).  Larger values reduce metacognitive
        efficiency.
    rng :
        Optional :class:`numpy.random.Generator` for reproducibility.

    Returns
    -------
    nR_S1, nR_S2 : ndarray
        Response-count vectors of length ``2 * nRatings``, in the same
        Maniscalco & Lau convention as :func:`type2_SDT_simuation`.

    Notes
    -----
    **How confidence is computed**

    For evidence *x* drawn from N(±d/2, 1):

    .. math::

        P(S2|x) = \\sigma(d \\cdot x)   \\quad (\\text{equal priors, } c=0)

    More generally with non-zero *c* we use the exact Gaussian ratio:

    .. math::

        P(S2|x) = \\frac{\\phi(x - d/2)}{\\phi(x - d/2) + \\phi(x + d/2)}

    where :math:`\\phi` is the standard normal PDF.

    Confidence (posterior probability of being correct):

    .. math::

        p_{\\text{correct}} = P(S2|x) \\text{ if resp=S2, else } 1 - P(S2|x)

    With meta-noise, noise is applied on the log-odds scale before converting
    back to probability:

    .. math::

        \\ell_{\\text{noisy}} = \\text{logit}(p_{\\text{correct}}) + \\epsilon,
        \\quad \\epsilon \\sim N(0, \\sigma_{\\text{meta}})

    **Confidence binning**

    P(correct) ∈ [0.5, 1.0] is divided into *nRatings* equal-width bins.
    The bin edges are placed at ``0.5 + k * 0.5/nRatings`` for
    k = 0, ..., nRatings.  Bin index 0 = lowest confidence (P(correct)
    closest to 0.5), bin index nRatings-1 = highest confidence.

    Examples
    --------
    Ideal observer — expect M_ratio ≈ 1:

    >>> nR_S1, nR_S2 = type2_SDT_simuation_bayes(d=1.5, meta_noise=0.0, nTrials=5000)

    Noisy observer — M_ratio drops below 1:

    >>> nR_S1, nR_S2 = type2_SDT_simuation_bayes(d=1.5, meta_noise=1.0, nTrials=5000)
    """
    if rng is None:
        rng = np.random.default_rng()

    S1mu = -d / 2
    S2mu = d / 2

    nC_rS1 = np.zeros(nRatings)
    nI_rS1 = np.zeros(nRatings)
    nC_rS2 = np.zeros(nRatings)
    nI_rS2 = np.zeros(nRatings)

    # Equal-width bin edges on [0.5, 1.0]
    bin_edges = np.linspace(0.5, 1.0, nRatings + 1)

    for _ in range(nTrials):
        s = int(rng.integers(0, 2))
        x = rng.normal(S2mu if s == 1 else S1mu, 1.0)
        resp = int(x >= c)

        # --- Bayesian posterior P(S2|x) via Gaussian ratio ---
        log_lik_s2 = norm.logpdf(x, S2mu, 1.0)
        log_lik_s1 = norm.logpdf(x, S1mu, 1.0)
        # log P(S2|x) - log P(S1|x) = log_lik_s2 - log_lik_s1  (equal priors)
        log_odds_s2 = log_lik_s2 - log_lik_s1
        p_s2 = float(expit(log_odds_s2))

        # P(correct | x, response)
        p_correct = p_s2 if resp == 1 else 1.0 - p_s2

        # --- optional log-odds noise on the metacognitive posterior ---
        if meta_noise > 0:
            # logit of p_correct, clipped to avoid ±inf
            p_c_clipped = float(np.clip(p_correct, 1e-9, 1 - 1e-9))
            log_odds_c = np.log(p_c_clipped / (1.0 - p_c_clipped))
            log_odds_noisy = log_odds_c + rng.normal(0.0, meta_noise)
            p_correct = float(expit(log_odds_noisy))

        # --- bin P(correct) into 0..nRatings-1 ---
        # np.digitize returns 1-based; clip to [0, nRatings-1]
        conf_bin = int(np.clip(np.digitize(p_correct, bin_edges) - 1, 0, nRatings - 1))

        # --- assign to count arrays ---
        # S1-response counts: index 0 = most confident (mirrors evidence model convention)
        # S2-response counts: index nRatings-1 = most confident
        if s == 0 and resp == 0:        # S1 stim, S1 resp (correct) — most conf = bin 0
            nC_rS1[nRatings - 1 - conf_bin] += 1
        elif s == 0 and resp == 1:      # S1 stim, S2 resp (incorrect)
            nI_rS2[conf_bin] += 1
        elif s == 1 and resp == 0:      # S2 stim, S1 resp (incorrect) — most conf = bin 0
            nI_rS1[nRatings - 1 - conf_bin] += 1
        else:                           # S2 stim, S2 resp (correct)
            nC_rS2[conf_bin] += 1

    nR_S1 = np.concatenate([nC_rS1, nI_rS2])
    nR_S2 = np.concatenate([nI_rS1, nC_rS2])

    return nR_S1, nR_S2


# ---------------------------------------------------------------------------
# ratings2df
# ---------------------------------------------------------------------------

def ratings2df(nR_S1: np.ndarray, nR_S2: np.ndarray) -> "pd.DataFrame":
    """Convert nR_S1 / nR_S2 count arrays to a trial-level DataFrame.

    Port of ``metadpy.utils.ratings2df``.

    Parameters
    ----------
    nR_S1, nR_S2 :
        Count vectors, length ``2 * nRatings``.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``Stimuli`` (0/1), ``Responses`` (0/1), ``Accuracy`` (0/1),
        ``Confidence`` (1..nRatings), ``nTrial``. Rows are shuffled.
    """
    import pandas as _pd
    nR_S1 = np.asarray(nR_S1, dtype=int)
    nR_S2 = np.asarray(nR_S2, dtype=int)
    nRatings = len(nR_S1) // 2

    rows: list = []
    for i in range(nRatings):
        conf = nRatings - i           # high confidence at low index
        for _ in range(nR_S1[i]):
            rows.append({"Stimuli": 0, "Responses": 0, "Accuracy": 1, "Confidence": conf})
        for _ in range(nR_S2[i]):
            rows.append({"Stimuli": 1, "Responses": 0, "Accuracy": 0, "Confidence": conf})
        for _ in range(nR_S1[nRatings + i]):
            rows.append({"Stimuli": 0, "Responses": 1, "Accuracy": 0, "Confidence": i + 1})
        for _ in range(nR_S2[nRatings + i]):
            rows.append({"Stimuli": 1, "Responses": 1, "Accuracy": 1, "Confidence": i + 1})

    df = _pd.DataFrame(rows).sample(frac=1).reset_index(drop=True)
    df["nTrial"] = np.arange(len(df))
    return df


# ---------------------------------------------------------------------------
# trialSimulation
# ---------------------------------------------------------------------------

def trialSimulation(
    d: float = 1.0,
    metad: float = 2.0,
    c: float = 0.0,
    nRatings: int = 4,
    nTrials: int = 500,
    rng: "np.random.Generator | None" = None,
) -> "pd.DataFrame":
    """Simulate a single participant using closed-form SDT probabilities.

    Computes exact multinomial probabilities from normal-CDF integrals and
    draws counts — much faster than trial-by-trial noise simulation.
    Port of ``metadpy.utils.trialSimulation``.

    Parameters
    ----------
    d :
        Type-1 d-prime.
    metad :
        Type-2 meta-d' in d-prime units.
    c :
        Type-1 decision criterion.
    nRatings :
        Number of confidence rating levels.
    nTrials :
        Number of trials.
    rng :
        Optional :class:`numpy.random.Generator` for reproducibility.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``Stimuli``, ``Responses``, ``Accuracy``, ``Confidence``, ``nTrial``.
    """
    if rng is None:
        rng = np.random.default_rng()

    c1 = c + np.linspace(-1.5, -0.5, nRatings - 1)
    c2 = c + np.linspace(0.5,  1.5,  nRatings - 1)

    H  = round((1 - norm.cdf(c,  d / 2)) * (nTrials / 2))
    FA = round((1 - norm.cdf(c, -d / 2)) * (nTrials / 2))
    CR = round(norm.cdf(c, -d / 2)       * (nTrials / 2))
    M  = round(norm.cdf(c,  d / 2)       * (nTrials / 2))

    S1mu = -metad / 2
    S2mu =  metad / 2

    C_area_rS1 = norm.cdf(c, S1mu)
    I_area_rS1 = norm.cdf(c, S2mu)
    C_area_rS2 = 1 - norm.cdf(c, S2mu)
    I_area_rS2 = 1 - norm.cdf(c, S1mu)

    t2c1x = np.concatenate([[-np.inf], c1, [c], c2, [np.inf]])

    prC_rS1, prI_rS1, prC_rS2, prI_rS2 = [], [], [], []
    for i in range(nRatings):
        prC_rS1.append((norm.cdf(t2c1x[i + 1], S1mu) - norm.cdf(t2c1x[i], S1mu)) / C_area_rS1)
        prI_rS1.append((norm.cdf(t2c1x[i + 1], S2mu) - norm.cdf(t2c1x[i], S2mu)) / I_area_rS1)
        prC_rS2.append(
            ((1 - norm.cdf(t2c1x[nRatings + i],     S2mu))
           - (1 - norm.cdf(t2c1x[nRatings + i + 1], S2mu))) / C_area_rS2
        )
        prI_rS2.append(
            ((1 - norm.cdf(t2c1x[nRatings + i],     S1mu))
           - (1 - norm.cdf(t2c1x[nRatings + i + 1], S1mu))) / I_area_rS2
        )

    def _norm(p: list) -> np.ndarray:
        a = np.clip(np.array(p, dtype=float), 0, None)
        return a / a.sum()

    nC_rS1 = rng.multinomial(CR, _norm(prC_rS1))
    nI_rS1 = rng.multinomial(M,  _norm(prI_rS1))
    nC_rS2 = rng.multinomial(H,  _norm(prC_rS2))
    nI_rS2 = rng.multinomial(FA, _norm(prI_rS2))

    nr_s1 = np.concatenate([nC_rS1, nI_rS2])
    nr_s2 = np.concatenate([nI_rS1, nC_rS2])

    return ratings2df(nr_s1, nr_s2)


# ---------------------------------------------------------------------------
# responseSimulation
# ---------------------------------------------------------------------------

def responseSimulation(
    d: float = 1.0,
    metad: float = 2.0,
    c: float = 0.0,
    nRatings: int = 4,
    nTrials: int = 500,
    nSubjects: int = 1,
    rng: "np.random.Generator | None" = None,
) -> "pd.DataFrame":
    """Simulate responses for one or more participants.

    Calls :func:`trialSimulation` once per subject and stacks the results.
    Port of ``metadpy.utils.responseSimulation``.

    Parameters
    ----------
    d :
        Type-1 d-prime (same for all subjects).
    metad :
        Meta-d' (same for all subjects).
    c :
        Type-1 criterion (same for all subjects).
    nRatings :
        Number of confidence rating levels.
    nTrials :
        Trials per subject.
    nSubjects :
        Number of subjects.
    rng :
        Optional :class:`numpy.random.Generator`.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``Stimuli``, ``Responses``, ``Accuracy``, ``Confidence``,
        ``nTrial``, ``Subject``.
    """
    import pandas as _pd
    if rng is None:
        rng = np.random.default_rng()

    frames = []
    for sub in range(nSubjects):
        df = trialSimulation(d=d, metad=metad, c=c,
                             nRatings=nRatings, nTrials=nTrials, rng=rng)
        df["Subject"] = sub
        frames.append(df)

    return _pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# pairedResponseSimulation
# ---------------------------------------------------------------------------

def pairedResponseSimulation(
    d: float = 1.0,
    d_sigma: float = 0.1,
    mRatio=None,
    mRatio_sigma: float = 0.2,
    mRatio_rho: float = 0.0,
    c: float = 0.0,
    c_sigma: float = 0.1,
    nRatings: int = 4,
    nTrials: int = 500,
    nSubjects: int = 20,
    rng: "np.random.Generator | None" = None,
) -> "pd.DataFrame":
    """Simulate a two-condition repeated-measures group design.

    Draws correlated M-ratio pairs per subject from a bivariate normal
    distribution, then simulates both conditions via :func:`trialSimulation`.
    Port of ``metadpy.utils.pairedResponseSimulation``.

    Parameters
    ----------
    d :
        Group-mean type-1 d-prime.
    d_sigma :
        Between-subject SD for d-prime.
    mRatio :
        ``[mRatio_cond1, mRatio_cond2]``. Defaults to ``[1.0, 0.6]``.
    mRatio_sigma :
        SD of M-ratio across subjects (applied to both conditions).
    mRatio_rho :
        Correlation between the two per-subject M-ratios.
    c :
        Group-mean type-1 criterion.
    c_sigma :
        Between-subject SD for criterion.
    nRatings :
        Number of confidence rating levels.
    nTrials :
        Trials per subject per condition.
    nSubjects :
        Number of subjects.
    rng :
        Optional :class:`numpy.random.Generator`.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``Stimuli``, ``Responses``, ``Accuracy``, ``Confidence``,
        ``nTrial``, ``Subject``, ``Condition``.
    """
    import pandas as _pd
    if mRatio is None:
        mRatio = [1.0, 0.6]
    if rng is None:
        rng = np.random.default_rng()

    cov = np.array([
        [mRatio_sigma ** 2, mRatio_rho * mRatio_sigma ** 2],
        [mRatio_rho * mRatio_sigma ** 2, mRatio_sigma ** 2],
    ])

    frames = []
    for sub in range(nSubjects):
        mr_pair = rng.multivariate_normal(mRatio, cov)
        for cond in range(2):
            d_sub     = float(rng.normal(d, d_sigma))
            c_sub     = float(rng.normal(c, c_sigma))
            metad_sub = float(mr_pair[cond] * d_sub)
            df = trialSimulation(d=d_sub, metad=metad_sub, c=c_sub,
                                 nRatings=nRatings, nTrials=nTrials, rng=rng)
            df["Subject"]   = sub
            df["Condition"] = cond
            frames.append(df)

    return _pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# discreteRatings
# ---------------------------------------------------------------------------

def discreteRatings(ratings, nbins: int = 4, verbose: bool = True,
                    ignore_invalid: bool = False):
    """Bin continuous confidence ratings into discrete levels via quantiles.

    Handles floor/ceiling biases by treating extreme values as their own bin
    after re-estimating inner quantile boundaries from the remaining data.
    Port of ``metadpy.utils.discreteRatings``.

    Parameters
    ----------
    ratings :
        Continuous confidence ratings (any numeric scale).
    nbins :
        Target number of discrete bins.
    verbose :
        Print a message when bias correction is applied.
    ignore_invalid :
        If ``False``, raise ``ValueError`` when both floor and ceiling are
        biased. Set ``True`` to proceed anyway.

    Returns
    -------
    discrete : np.ndarray
        Integer ratings from 1 to ``nbins``.
    info : dict
        ``'confBins'``, ``'rebin'`` (``[1]`` if corrected, else ``[0]``),
        ``'binCount'`` (per-bin trial counts).
    """
    ratings = np.asarray(ratings, dtype=float)
    info: dict = {}
    masks: list = []

    conf_bins    = np.quantile(ratings, np.linspace(0, 1, nbins + 1))
    floor_bias   = conf_bins[0] == conf_bins[1]
    ceiling_bias = conf_bins[nbins - 1] == conf_bins[nbins]

    if floor_bias and ceiling_bias:
        if not ignore_invalid:
            raise ValueError(
                "Rating scale has too many identical extreme values to discretise. "
                "Pass ignore_invalid=True to proceed."
            )
        if verbose:
            print("Correcting for bias in both low and high confidence ratings.")
        lo = conf_bins[0]
        hi = conf_bins[-1]
        inner_ratings = ratings[(ratings != lo) & (ratings != hi)]
        n_inner_bins = max(nbins - 2, 0)
        masks.append(ratings == lo)
        if n_inner_bins > 0 and len(inner_ratings) > 0:
            inner = np.quantile(inner_ratings, np.linspace(0, 1, n_inner_bins + 1))
            for b in range(n_inner_bins):
                masks.append(
                    (ratings >= inner[b]) & (ratings <= inner[b + 1])
                    & (ratings != lo) & (ratings != hi)
                )
        masks.append(ratings == hi)
        info["confBins"] = [lo, hi]
        info["rebin"]    = [1]

    elif ceiling_bias:
        if verbose:
            print("Correcting for bias in high confidence ratings.")
        hi = conf_bins[-1]
        inner = np.quantile(ratings[ratings != hi], np.linspace(0, 1, nbins))
        for b in range(len(inner) - 1):
            masks.append((ratings >= inner[b]) & (ratings <= inner[b + 1]))
        masks.append(ratings == hi)
        info["confBins"] = [inner, hi]
        info["rebin"]    = [1]

    elif floor_bias:
        if verbose:
            print("Correcting for bias in low confidence ratings.")
        lo = conf_bins[1]
        masks.append(ratings == lo)
        inner = np.quantile(ratings[ratings != lo], np.linspace(0, 1, nbins))
        for b in range(1, len(inner)):
            masks.append((ratings >= inner[b - 1]) & (ratings <= inner[b]))
        info["confBins"] = [lo, inner]
        info["rebin"]    = [1]

    else:
        for b in range(len(conf_bins) - 1):
            masks.append((ratings >= conf_bins[b]) & (ratings <= conf_bins[b + 1]))
        info["confBins"] = conf_bins
        info["rebin"]    = [0]

    discrete = np.zeros(len(ratings), dtype=int)
    for b, mask in enumerate(masks):
        discrete[mask] = b
    discrete += 1  # 1-based

    info["binCount"] = [int(m.sum()) for m in masks]
    return discrete, info
