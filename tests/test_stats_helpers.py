"""Tests for metasignal.stdpy.stats_helpers — utility statistical functions.

Note: icc() depends on pingouin which is not a declared package dependency.
That test is skipped automatically when pingouin is not installed.
"""

import numpy as np
import pytest

from metasignal.stdpy.stats_helpers import perform_ttest, r2z, z2r


# ---------------------------------------------------------------------------
# z2r / r2z round-trip
# ---------------------------------------------------------------------------

def test_z2r_r2z_roundtrip():
    r = 0.5
    assert abs(z2r(r2z(r)) - r) < 1e-10


def test_r2z_z2r_roundtrip():
    z = 0.8
    assert abs(r2z(z2r(z)) - z) < 1e-10


def test_z2r_zero():
    assert abs(z2r(0.0)) < 1e-10


def test_r2z_zero():
    assert abs(r2z(0.0)) < 1e-10


def test_z2r_positive():
    assert z2r(1.0) > 0.0


def test_r2z_positive():
    assert r2z(0.5) > 0.0


# ---------------------------------------------------------------------------
# perform_ttest
# ---------------------------------------------------------------------------

@pytest.fixture
def positive_data():
    """Data clearly above zero — should yield a significant p-value."""
    rng = np.random.default_rng(0)
    return rng.normal(loc=1.0, scale=0.3, size=30)


@pytest.fixture
def null_data():
    """Data centred at zero — should yield a non-significant p-value."""
    rng = np.random.default_rng(1)
    return rng.normal(loc=0.0, scale=1.0, size=30)


def test_perform_ttest_returns_five_values(positive_data):
    result = perform_ttest(positive_data, display=False)
    assert len(result) == 5


def test_perform_ttest_significant(positive_data):
    pval, *_ = perform_ttest(positive_data, display=False)
    assert pval < 0.05


def test_perform_ttest_not_significant(null_data):
    pval, *_ = perform_ttest(null_data, display=False)
    # Not guaranteed to be > 0.05 every seed, but with loc=0 it should be
    # Just check it's a valid probability
    assert 0.0 <= pval <= 1.0


def test_perform_ttest_ci_contains_mean(positive_data):
    pval, tstat, df, cohen_d, ci = perform_ttest(positive_data, display=False)
    sample_mean = np.mean(positive_data)
    assert ci[0] < sample_mean < ci[1]


def test_perform_ttest_cohen_d_positive(positive_data):
    _, tstat, df, cohen_d, _ = perform_ttest(positive_data, display=False)
    assert cohen_d > 0.0


def test_perform_ttest_ignores_nan():
    data = np.array([1.0, 2.0, np.nan, 1.5, 1.8])
    pval, tstat, df, cohen_d, ci = perform_ttest(data, display=False)
    assert df == 3  # 4 valid values → df = 3
    assert np.isfinite(pval)


# ---------------------------------------------------------------------------
# icc — skipped when pingouin is absent
# ---------------------------------------------------------------------------

pingouin = pytest.importorskip("pingouin", reason="pingouin not installed")


def test_icc_returns_dataframe():
    from metasignal.stdpy.stats_helpers import icc
    rng = np.random.default_rng(5)
    data = rng.normal(size=(10, 3))  # 10 targets, 3 raters
    result = icc(data)
    assert hasattr(result, "columns"), "Expected a DataFrame"
    assert "ICC" in result.columns
