"""Tests for metasignal.stdpy.metad — meta-d' MLE fitting."""

import numpy as np
import pytest

from metasignal.stdpy.core import trials_to_counts
from metasignal.stdpy.metad import fit_meta_d_mle


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def good_observer_counts():
    """Counts from a simulated observer with strong metacognition."""
    rng = np.random.default_rng(99)
    stim = np.array([0, 1] * 60, dtype=float)
    resp = stim.copy()
    flip = rng.random(120) < 0.1
    resp[flip] = 1 - resp[flip]
    conf = rng.integers(1, 3, 120)
    correct = stim == resp
    conf[correct] = np.where(rng.random(np.sum(correct)) > 0.25, 2, 1)
    conf[~correct] = np.where(rng.random(np.sum(~correct)) > 0.75, 2, 1)
    return trials_to_counts(stim, resp, conf, n_ratings=2)


# ---------------------------------------------------------------------------
# fit_meta_d_mle
# ---------------------------------------------------------------------------

def test_returns_dict(good_observer_counts):
    nr_s1, nr_s2 = good_observer_counts
    result = fit_meta_d_mle(nr_s1, nr_s2)
    assert isinstance(result, dict)


def test_required_keys(good_observer_counts):
    nr_s1, nr_s2 = good_observer_counts
    result = fit_meta_d_mle(nr_s1, nr_s2)
    for key in ("meta_da", "M_ratio", "M_diff"):
        assert key in result, f"Missing key: {key}"


def test_meta_d_positive(good_observer_counts):
    nr_s1, nr_s2 = good_observer_counts
    result = fit_meta_d_mle(nr_s1, nr_s2)
    assert result["meta_da"] > 0.0


def test_m_ratio_positive(good_observer_counts):
    """M-ratio (meta-d'/d') should be positive for a valid observer."""
    nr_s1, nr_s2 = good_observer_counts
    result = fit_meta_d_mle(nr_s1, nr_s2)
    assert result["M_ratio"] > 0.0


def test_mismatched_lengths_raises():
    nr_s1 = np.array([5.0, 10.0, 8.0, 6.0])
    nr_s2 = np.array([3.0, 12.0])
    with pytest.raises(ValueError):
        fit_meta_d_mle(nr_s1, nr_s2)


def test_odd_length_raises():
    nr_s1 = np.array([5.0, 10.0, 8.0])
    nr_s2 = np.array([3.0, 12.0, 7.0])
    with pytest.raises(ValueError):
        fit_meta_d_mle(nr_s1, nr_s2)
