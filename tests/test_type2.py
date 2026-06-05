"""Tests for metasignal.stdpy.type2 — Type-2 metacognitive measures."""

import numpy as np
import pytest

from metasignal.stdpy.core import trials_to_counts
from metasignal.stdpy.type2 import (
    compute_delta_conf,
    compute_gamma,
    compute_phi,
    compute_type2_auc,
    sdt_expect_conf,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def counts():
    """Synthetic counts for a good metacognitive observer with 2 ratings."""
    rng = np.random.default_rng(7)
    stim = np.array([0, 1] * 50, dtype=float)
    resp = stim.copy()
    flip = rng.random(100) < 0.1
    resp[flip] = 1 - resp[flip]
    conf = rng.integers(1, 3, 100)
    # Higher conf on correct trials
    correct = stim == resp
    conf[correct & (rng.random(100) > 0.3)] = 2
    conf[~correct & (rng.random(100) > 0.7)] = 1
    nr_s1, nr_s2 = trials_to_counts(stim, resp, conf, n_ratings=2)
    return nr_s1, nr_s2


@pytest.fixture
def raw_data():
    rng = np.random.default_rng(7)
    stim = np.array([0, 1] * 50, dtype=float)
    resp = stim.copy()
    flip = rng.random(100) < 0.1
    resp[flip] = 1 - resp[flip]
    conf = rng.integers(1, 3, 100).astype(float)
    return stim, resp, conf


# ---------------------------------------------------------------------------
# compute_type2_auc
# ---------------------------------------------------------------------------

def test_auc2_range(counts):
    nr_s1, nr_s2 = counts
    auc = compute_type2_auc(nr_s1, nr_s2)
    assert 0.5 <= auc <= 1.0, f"AUC2 out of [0.5, 1.0]: {auc}"


def test_auc2_good_observer(counts):
    """Good metacognition should yield AUC2 clearly above chance."""
    nr_s1, nr_s2 = counts
    auc = compute_type2_auc(nr_s1, nr_s2)
    assert auc > 0.55


# ---------------------------------------------------------------------------
# compute_gamma
# ---------------------------------------------------------------------------

def test_gamma_range(counts):
    nr_s1, nr_s2 = counts
    gamma = compute_gamma(nr_s1, nr_s2)
    assert -1.0 <= gamma <= 1.0, f"Gamma out of [-1, 1]: {gamma}"


def test_gamma_positive_for_good_observer(counts):
    nr_s1, nr_s2 = counts
    gamma = compute_gamma(nr_s1, nr_s2)
    assert gamma > 0.0


# ---------------------------------------------------------------------------
# compute_phi
# ---------------------------------------------------------------------------

def test_phi_is_float(counts):
    nr_s1, nr_s2 = counts
    phi = compute_phi(nr_s1, nr_s2)
    assert isinstance(phi, float)


def test_phi_finite(counts):
    nr_s1, nr_s2 = counts
    phi = compute_phi(nr_s1, nr_s2)
    assert np.isfinite(phi)


# ---------------------------------------------------------------------------
# compute_delta_conf
# ---------------------------------------------------------------------------

def test_delta_conf_positive_for_good_observer(counts):
    """Correct trials should receive higher mean confidence than incorrect ones."""
    nr_s1, nr_s2 = counts
    result = compute_delta_conf(nr_s1, nr_s2)
    assert result["delta_conf"] > 0.0


# ---------------------------------------------------------------------------
# sdt_expect_conf
# ---------------------------------------------------------------------------

def test_sdt_expect_conf_keys(counts):
    nr_s1, nr_s2 = counts
    result = sdt_expect_conf(nr_s1, nr_s2)
    for key in ("nR_S1_exp", "nR_S2_exp", "nR_S1_act", "nR_S2_act", "dprime"):
        assert key in result, f"Missing key: {key}"


def test_sdt_expect_conf_totals_preserved(counts):
    """Expected total counts should roughly equal actual total counts."""
    nr_s1, nr_s2 = counts
    result = sdt_expect_conf(nr_s1, nr_s2)
    assert abs(np.sum(result["nR_S1_exp"]) - np.sum(result["nR_S1_act"])) < 1.0
    assert abs(np.sum(result["nR_S2_exp"]) - np.sum(result["nR_S2_act"])) < 1.0
