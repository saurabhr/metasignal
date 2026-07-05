"""The brmspy-blocked sdtbayes models raise NotImplementedError (not
RuntimeError) — they are permanently unavailable pending an upstream fix,
not a transient runtime failure. See each module's docstring Notes for the
specific brmspy/rpy2 limitation."""

import numpy as np
import pytest

from metasignal.sdtbayes.mixture import fit_mixture_group
from metasignal.sdtbayes.multivariate import (
    fit_multivariate_mratio,
    fit_multivariate_mratio_comparison,
)
from metasignal.sdtbayes.robust import fit_robust_metad, fit_robust_metad_comparison
from metasignal.sdtbayes.statespace import fit_statespace_metad
from metasignal.sdtbayes.variational import fit_full_metad_vi, fit_robust_metad_vi


def _participants(n=10, n_trials=200, seed=0):
    """Confidence correlated with accuracy so meta-d'/M-ratio estimates are
    reliably positive — pure noise makes MLE unstable and trips the
    participant-count guards these functions check before failing."""
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(n):
        stim = rng.integers(0, 2, n_trials)
        resp = stim.copy()
        flip = rng.random(n_trials) < 0.15
        resp[flip] = 1 - resp[flip]
        correct = stim == resp
        p_high = np.where(correct, 0.8, 0.2)
        conf = np.where(rng.random(n_trials) < p_high, 4, 1)
        parts.append((stim, resp, conf))
    return parts


def test_fit_mixture_group_not_implemented():
    with pytest.raises(NotImplementedError, match="brms mixture family"):
        fit_mixture_group(_participants(), n_ratings=4)


def test_fit_multivariate_mratio_not_implemented():
    with pytest.raises(NotImplementedError, match="rescor"):
        fit_multivariate_mratio(_participants(), n_ratings=4)


def test_fit_multivariate_mratio_comparison_not_implemented():
    with pytest.raises(NotImplementedError, match="rescor"):
        fit_multivariate_mratio_comparison(_participants(), _participants(seed=1), n_ratings=4)


def test_fit_robust_metad_not_implemented():
    with pytest.raises(NotImplementedError, match="stanvar"):
        fit_robust_metad(_participants(), n_ratings=4)


def test_fit_robust_metad_comparison_not_implemented():
    with pytest.raises(NotImplementedError, match="stanvar"):
        fit_robust_metad_comparison(_participants(), _participants(seed=1), n_ratings=4)


def test_fit_statespace_metad_not_implemented():
    sessions = [_participants(seed=t) for t in range(3)]
    with pytest.raises(NotImplementedError, match="stanvar"):
        fit_statespace_metad(sessions, n_ratings=4)


def test_fit_full_metad_vi_not_implemented():
    with pytest.raises(NotImplementedError, match="cmdstanpy"):
        fit_full_metad_vi(_participants(), n_ratings=4)


def test_fit_robust_metad_vi_not_implemented():
    with pytest.raises(NotImplementedError, match="cmdstanpy"):
        fit_robust_metad_vi(_participants(), n_ratings=4)
