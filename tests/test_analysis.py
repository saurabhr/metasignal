"""Tests for metasignal.analysis — bootstrap, permutation, group summary."""

import numpy as np
import pytest

from metasignal.analysis.bootstrap import bootstrap_measure
from metasignal.analysis.group import MEASURE_LABELS, group_summary
from metasignal.analysis.permutation import permutation_test


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_observer(seed: int, n: int = 100, error_rate: float = 0.12) -> tuple:
    rng = np.random.default_rng(seed)
    stim = np.array([0, 1] * (n // 2), dtype=float)
    resp = stim.copy()
    flip = rng.random(n) < error_rate
    resp[flip] = 1 - resp[flip]
    conf = rng.integers(1, 3, n).astype(float)
    correct = stim == resp
    conf[correct] = np.where(rng.random(int(correct.sum())) > 0.3, 2, 1)
    conf[~correct] = np.where(rng.random(int((~correct).sum())) > 0.7, 2, 1)
    return stim, resp, conf


@pytest.fixture
def obs_a():
    return _make_observer(seed=0)


@pytest.fixture
def obs_b():
    return _make_observer(seed=1, error_rate=0.3)


# ---------------------------------------------------------------------------
# bootstrap_measure
# ---------------------------------------------------------------------------

def test_bootstrap_returns_tuple(obs_a):
    stim, resp, conf = obs_a
    lo, hi = bootstrap_measure(stim, resp, conf, n_ratings=2, measure_index=17, n_boot=50)
    assert isinstance(lo, float)
    assert isinstance(hi, float)


def test_bootstrap_ci_ordered(obs_a):
    stim, resp, conf = obs_a
    lo, hi = bootstrap_measure(stim, resp, conf, n_ratings=2, measure_index=17, n_boot=100,
                               rng=np.random.default_rng(0))
    assert lo < hi


def test_bootstrap_ci_contains_dprime(obs_a):
    """95% CI for d' (index 17) should be a reasonable positive interval."""
    stim, resp, conf = obs_a
    lo, hi = bootstrap_measure(stim, resp, conf, n_ratings=2, measure_index=17, n_boot=200,
                               rng=np.random.default_rng(42))
    assert lo > 0.0
    assert hi > lo


def test_bootstrap_invalid_index_raises(obs_a):
    stim, resp, conf = obs_a
    with pytest.raises(ValueError):
        bootstrap_measure(stim, resp, conf, n_ratings=2, measure_index=26, n_boot=10)


def test_bootstrap_reproducible(obs_a):
    stim, resp, conf = obs_a
    r1 = bootstrap_measure(stim, resp, conf, n_ratings=2, measure_index=17, n_boot=50,
                           rng=np.random.default_rng(99))
    r2 = bootstrap_measure(stim, resp, conf, n_ratings=2, measure_index=17, n_boot=50,
                           rng=np.random.default_rng(99))
    assert r1 == r2


# ---------------------------------------------------------------------------
# permutation_test
# ---------------------------------------------------------------------------

def test_permutation_returns_two_values(obs_a, obs_b):
    stim_a, resp_a, conf_a = obs_a
    stim_b, resp_b, conf_b = obs_b
    result = permutation_test(
        stim_a, resp_a, conf_a,
        stim_b, resp_b, conf_b,
        n_ratings=2, measure_index=17, n_perm=50,
    )
    assert len(result) == 2


def test_permutation_pvalue_in_range(obs_a, obs_b):
    stim_a, resp_a, conf_a = obs_a
    stim_b, resp_b, conf_b = obs_b
    p, _ = permutation_test(
        stim_a, resp_a, conf_a,
        stim_b, resp_b, conf_b,
        n_ratings=2, measure_index=17, n_perm=100,
        rng=np.random.default_rng(0),
    )
    assert 0.0 <= p <= 1.0


def test_permutation_observed_diff_sign(obs_a, obs_b):
    """obs_a has lower error rate → higher d', so obs_diff should be positive."""
    stim_a, resp_a, conf_a = obs_a
    stim_b, resp_b, conf_b = obs_b
    _, obs_diff = permutation_test(
        stim_a, resp_a, conf_a,
        stim_b, resp_b, conf_b,
        n_ratings=2, measure_index=17, n_perm=50,
        rng=np.random.default_rng(0),
    )
    assert obs_diff > 0.0


def test_permutation_invalid_index_raises(obs_a, obs_b):
    stim_a, resp_a, conf_a = obs_a
    stim_b, resp_b, conf_b = obs_b
    with pytest.raises(ValueError):
        permutation_test(
            stim_a, resp_a, conf_a,
            stim_b, resp_b, conf_b,
            n_ratings=2, measure_index=26, n_perm=10,
        )


# ---------------------------------------------------------------------------
# group_summary
# ---------------------------------------------------------------------------

@pytest.fixture
def group():
    return [_make_observer(seed=i) for i in range(8)]


def test_group_summary_keys(group):
    result = group_summary(group, n_ratings=2)
    for key in ("individual", "mean", "median", "sem", "n_valid", "labels"):
        assert key in result


def test_group_summary_individual_shape(group):
    result = group_summary(group, n_ratings=2)
    assert result["individual"].shape == (8, 26)


def test_group_summary_mean_shape(group):
    result = group_summary(group, n_ratings=2)
    assert result["mean"].shape == (26,)


def test_group_summary_labels(group):
    result = group_summary(group, n_ratings=2)
    assert result["labels"] == MEASURE_LABELS
    assert len(result["labels"]) == 26


def test_group_summary_n_valid_bounded(group):
    result = group_summary(group, n_ratings=2)
    assert np.all(result["n_valid"] >= 0)
    assert np.all(result["n_valid"] <= 8)


def test_group_summary_mean_dprime_positive(group):
    result = group_summary(group, n_ratings=2)
    assert result["mean"][17] > 0.0  # index 17 = dprime


def test_group_summary_sem_nonnegative(group):
    result = group_summary(group, n_ratings=2)
    finite_sem = result["sem"][np.isfinite(result["sem"])]
    assert np.all(finite_sem >= 0.0)
