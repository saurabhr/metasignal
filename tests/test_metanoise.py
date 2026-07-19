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


def test_boundary_rates_use_global_grid():
    """HR/FAR at 0/1 must yield Inf init criteria → global −5:0.01:5 search."""
    from metasignal.stdpy.metanoise import _compute_sdt_criteria

    n = 40
    stim = np.array([0] * (n // 2) + [1] * (n // 2), dtype=float)
    resp = stim.copy()
    conf = np.ones(n, dtype=float)
    conf[stim == 1] = 4
    conf[stim == 0] = 1
    n_ratings = 4
    dprime, c = _compute_sdt_criteria(stim, resp, conf, n_ratings)
    assert np.isinf(c).any(), "expected ±Inf criteria when rates hit 0/1"
    result = compute_meta_noise(stim, resp, conf, n_ratings=n_ratings)
    assert np.isfinite(result["meta_noise"])
    assert np.isfinite(dprime) or np.isinf(dprime)


def test_matlab_artifact_helper():
    from metasignal.stdpy.metanoise import (
        MATLAB_META_NOISE_SEARCH_ARTIFACT,
        is_matlab_meta_noise_artifact,
    )

    assert is_matlab_meta_noise_artifact(MATLAB_META_NOISE_SEARCH_ARTIFACT)
    assert not is_matlab_meta_noise_artifact(0.2)
    arr = np.array([0.2, MATLAB_META_NOISE_SEARCH_ARTIFACT, np.nan])
    mask = is_matlab_meta_noise_artifact(arr)
    assert list(mask) == [False, True, False]
