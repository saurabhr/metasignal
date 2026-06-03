"""Tests for metasignal.stdpy.metanoise — lognormal meta-noise model."""

import numpy as np
import pytest

from metasignal.stdpy.metanoise import compute_meta_noise


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def good_observer():
    """Synthetic observer with decent metacognition for metanoise fitting."""
    rng = np.random.default_rng(42)
    n = 120
    stim = np.array([0, 1] * (n // 2), dtype=float)
    resp = stim.copy()
    flip = rng.random(n) < 0.1
    resp[flip] = 1 - resp[flip]
    conf = rng.integers(1, 3, n).astype(float)
    correct = stim == resp
    conf[correct] = np.where(rng.random(int(correct.sum())) > 0.3, 2, 1)
    conf[~correct] = np.where(rng.random(int((~correct).sum())) > 0.7, 2, 1)
    return stim, resp, conf


# ---------------------------------------------------------------------------
# compute_meta_noise
# ---------------------------------------------------------------------------

def test_returns_dict(good_observer):
    stim, resp, conf = good_observer
    result = compute_meta_noise(stim, resp, conf, n_ratings=2)
    assert isinstance(result, dict)


def test_required_keys(good_observer):
    stim, resp, conf = good_observer
    result = compute_meta_noise(stim, resp, conf, n_ratings=2)
    for key in ("meta_noise", "dprime", "c", "logL"):
        assert key in result, f"Missing key: {key}"


def test_meta_noise_positive(good_observer):
    stim, resp, conf = good_observer
    result = compute_meta_noise(stim, resp, conf, n_ratings=2)
    assert result["meta_noise"] >= 0.0


def test_meta_noise_finite(good_observer):
    stim, resp, conf = good_observer
    result = compute_meta_noise(stim, resp, conf, n_ratings=2)
    assert np.isfinite(result["meta_noise"])


def test_dprime_positive(good_observer):
    stim, resp, conf = good_observer
    result = compute_meta_noise(stim, resp, conf, n_ratings=2)
    assert result["dprime"] > 0.0


def test_c_is_list(good_observer):
    stim, resp, conf = good_observer
    result = compute_meta_noise(stim, resp, conf, n_ratings=2)
    assert isinstance(result["c"], list)
    # For n_ratings=2, there should be 2*n_ratings - 1 = 3 criteria
    assert len(result["c"]) == 3


def test_logl_finite(good_observer):
    stim, resp, conf = good_observer
    result = compute_meta_noise(stim, resp, conf, n_ratings=2)
    assert np.isfinite(result["logL"])
