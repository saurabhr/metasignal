"""Tests for metasignal.stdpy.compute_all — the primary public entry point."""

import numpy as np
import pandas as pd
import pytest

from metasignal.stdpy.compute_all import compute_all_measures, MEASURE_NAMES


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
    assert result.shape == (26,)


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


# ---------------------------------------------------------------------------
# return_type
# ---------------------------------------------------------------------------

def test_return_type_dict_matches_array(standard_data):
    stim, resp, conf = standard_data
    arr = compute_all_measures(stim, resp, conf, n_ratings=2)
    result = compute_all_measures(stim, resp, conf, n_ratings=2, return_type="dict")
    assert isinstance(result, dict)
    assert list(result.keys()) == MEASURE_NAMES
    np.testing.assert_array_equal(list(result.values()), arr)


def test_return_type_dataframe_matches_array(standard_data):
    stim, resp, conf = standard_data
    arr = compute_all_measures(stim, resp, conf, n_ratings=2)
    result = compute_all_measures(stim, resp, conf, n_ratings=2, return_type="dataframe")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 26)
    assert list(result.columns) == MEASURE_NAMES
    np.testing.assert_array_equal(result.iloc[0].to_numpy(), arr)


def test_return_type_dataframe_on_nan_edge_case():
    """The NaN-array early-return paths must also respect return_type."""
    stim = np.full(20, np.nan)
    resp = np.full(20, np.nan)
    conf = np.full(20, np.nan)
    result = compute_all_measures(stim, resp, conf, n_ratings=2, return_type="dataframe")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 26)
    assert result.isna().all(axis=None)


def test_return_type_invalid_raises(standard_data):
    stim, resp, conf = standard_data
    with pytest.raises(ValueError, match="return_type"):
        compute_all_measures(stim, resp, conf, n_ratings=2, return_type="list")
