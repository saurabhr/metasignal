"""Tests for metasignal.stdpy.compute_all — the primary public entry point."""

import numpy as np
import pytest

from metasignal.stdpy.compute_all import compute_all_measures


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def standard_data():
    rng = np.random.default_rng(0)
    stim = np.array([0, 1] * 60, dtype=float)
    resp = stim.copy()
    flip = rng.random(120) < 0.12
    resp[flip] = 1 - resp[flip]
    conf = rng.integers(1, 3, 120).astype(float)
    correct = stim == resp
    conf[correct] = np.where(rng.random(np.sum(correct)) > 0.3, 2, 1)
    return stim, resp, conf


# ---------------------------------------------------------------------------
# Output shape and structure
# ---------------------------------------------------------------------------

def test_output_length(standard_data):
    stim, resp, conf = standard_data
    result = compute_all_measures(stim, resp, conf, n_ratings=2)
    assert result.shape == (20,)


def test_output_not_all_nan(standard_data):
    stim, resp, conf = standard_data
    result = compute_all_measures(stim, resp, conf, n_ratings=2)
    assert not np.all(np.isnan(result))


def test_dprime_at_index_17(standard_data):
    """d' (index 17) should be positive for a non-chance observer."""
    stim, resp, conf = standard_data
    result = compute_all_measures(stim, resp, conf, n_ratings=2)
    assert result[17] > 0.0


def test_mean_conf_at_index_19(standard_data):
    """mean_conf (index 19) should be within the rating scale."""
    stim, resp, conf = standard_data
    result = compute_all_measures(stim, resp, conf, n_ratings=2)
    assert 1.0 <= result[19] <= 2.0


def test_meta_d_at_index_0_positive(standard_data):
    stim, resp, conf = standard_data
    result = compute_all_measures(stim, resp, conf, n_ratings=2)
    assert result[0] > 0.0


def test_auc2_at_index_1_above_chance(standard_data):
    stim, resp, conf = standard_data
    result = compute_all_measures(stim, resp, conf, n_ratings=2)
    assert result[1] > 0.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_all_nan_input_returns_nan_array():
    stim = np.full(20, np.nan)
    resp = np.full(20, np.nan)
    conf = np.full(20, np.nan)
    result = compute_all_measures(stim, resp, conf, n_ratings=2)
    assert np.all(np.isnan(result))


def test_handles_constant_confidence(standard_data):
    """Constant confidence (no type-2 signal) should return NaN array."""
    stim, resp, _ = standard_data
    conf = np.ones(len(stim))
    result = compute_all_measures(stim, resp, conf, n_ratings=2)
    assert np.all(np.isnan(result))
