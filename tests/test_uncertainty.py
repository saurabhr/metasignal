"""Tests for metasignal.stdpy.uncertainty — meta-uncertainty model."""

import numpy as np
import pytest

from metasignal.stdpy.uncertainty import compute_meta_uncertainty


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def observer_data():
    """Synthetic observer with 2 confidence ratings."""
    rng = np.random.default_rng(17)
    n = 100
    stim = np.array([0, 1] * (n // 2), dtype=float)
    resp = stim.copy()
    flip = rng.random(n) < 0.12
    resp[flip] = 1 - resp[flip]
    conf = rng.integers(1, 3, n).astype(float)
    correct = stim == resp
    conf[correct] = np.where(rng.random(int(correct.sum())) > 0.3, 2, 1)
    conf[~correct] = np.where(rng.random(int((~correct).sum())) > 0.7, 2, 1)
    return stim, resp, conf


# ---------------------------------------------------------------------------
# compute_meta_uncertainty
# ---------------------------------------------------------------------------

def test_returns_float(observer_data):
    stim, resp, conf = observer_data
    result = compute_meta_uncertainty(stim, resp, conf, n_ratings=2)
    assert isinstance(result, float)


def test_meta_uncertainty_positive(observer_data):
    stim, resp, conf = observer_data
    result = compute_meta_uncertainty(stim, resp, conf, n_ratings=2)
    assert result > 0.0


def test_meta_uncertainty_finite(observer_data):
    stim, resp, conf = observer_data
    result = compute_meta_uncertainty(stim, resp, conf, n_ratings=2)
    assert np.isfinite(result)


def test_meta_uncertainty_within_bounds(observer_data):
    """Result should be within the optimisation bounds [0.01, 5.0]."""
    stim, resp, conf = observer_data
    result = compute_meta_uncertainty(stim, resp, conf, n_ratings=2)
    assert 0.0 < result <= 5.0


def test_matlab_compat_single_start(observer_data):
    stim, resp, conf = observer_data
    rng = np.random.default_rng(0)
    result = compute_meta_uncertainty(
        stim, resp, conf, n_ratings=2, rng=rng, matlab_compat=True
    )
    assert 0.0 < result <= 5.0
    assert np.isfinite(result)
