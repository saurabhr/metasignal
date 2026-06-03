"""Tests for metasignal.stdpy.core — SDT primitives."""

import numpy as np
import pytest

from metasignal.stdpy.core import compute_sdt_resp, trials_to_counts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_data():
    """Perfectly discriminating observer: all hits, no false alarms."""
    stim = np.array([0, 1] * 25, dtype=float)
    resp = np.array([0, 1] * 25, dtype=float)
    return stim, resp


@pytest.fixture
def chance_data():
    """Chance-level observer: hits == false alarms."""
    stim = np.array([0, 0, 1, 1] * 20, dtype=float)
    resp = np.array([0, 1, 0, 1] * 20, dtype=float)
    return stim, resp


@pytest.fixture
def trial_data():
    """Standard synthetic dataset with confidence ratings."""
    rng = np.random.default_rng(42)
    stim = rng.integers(0, 2, 100).astype(float)
    resp = stim.copy()
    # Flip ~15% of responses
    flip = rng.random(100) < 0.15
    resp[flip] = 1 - resp[flip]
    conf = rng.integers(1, 3, 100).astype(float)
    return stim, resp, conf


# ---------------------------------------------------------------------------
# compute_sdt_resp
# ---------------------------------------------------------------------------

def test_perfect_dprime(perfect_data):
    stim, resp = perfect_data
    dprime, c, ln_beta = compute_sdt_resp(stim, resp)
    assert dprime > 3.5


def test_chance_dprime(chance_data):
    stim, resp = chance_data
    dprime, c, _ = compute_sdt_resp(stim, resp)
    assert abs(dprime) < 0.1


def test_criterion_symmetric_observer(perfect_data):
    """Unbiased perfect observer: criterion c should be near 0."""
    stim, resp = perfect_data
    _, c, _ = compute_sdt_resp(stim, resp)
    assert abs(c) < 0.5


def test_returns_three_values(perfect_data):
    stim, resp = perfect_data
    result = compute_sdt_resp(stim, resp)
    assert len(result) == 3


def test_ln_beta_equals_dprime_times_c(perfect_data):
    stim, resp = perfect_data
    dprime, c, ln_beta = compute_sdt_resp(stim, resp)
    assert abs(ln_beta - dprime * c) < 1e-10


# ---------------------------------------------------------------------------
# trials_to_counts
# ---------------------------------------------------------------------------

def test_trials_to_counts_shape(trial_data):
    stim, resp, conf = trial_data
    nr_s1, nr_s2 = trials_to_counts(stim, resp, conf.astype(int), n_ratings=2)
    assert nr_s1.shape == (4,)
    assert nr_s2.shape == (4,)


def test_trials_to_counts_total(trial_data):
    """Total counts must equal number of valid trials."""
    stim, resp, conf = trial_data
    nr_s1, nr_s2 = trials_to_counts(stim, resp, conf.astype(int), n_ratings=2)
    assert int(np.sum(nr_s1) + np.sum(nr_s2)) == len(stim)


def test_trials_to_counts_nonnegative(trial_data):
    stim, resp, conf = trial_data
    nr_s1, nr_s2 = trials_to_counts(stim, resp, conf.astype(int), n_ratings=2)
    assert np.all(nr_s1 >= 0)
    assert np.all(nr_s2 >= 0)


def test_trials_to_counts_filters_bad(trial_data):
    """Trials with out-of-range ratings should be excluded."""
    stim, resp, conf = trial_data
    bad_conf = conf.copy()
    bad_conf[:5] = 99  # invalid rating
    nr_s1_bad, nr_s2_bad = trials_to_counts(stim, resp, bad_conf.astype(int), n_ratings=2)
    nr_s1_ok, nr_s2_ok = trials_to_counts(stim, resp, conf.astype(int), n_ratings=2)
    assert np.sum(nr_s1_bad) + np.sum(nr_s2_bad) < np.sum(nr_s1_ok) + np.sum(nr_s2_ok)
