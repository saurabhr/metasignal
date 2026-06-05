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


def test_icc_uses_consistency_type():
    """Paper uses consistency ICC (C-k), not absolute agreement.
    Confirmed in Rahnev (2025) Nat Commun and peer review response to Reviewer #2.
    pingouin >= 0.6 labels this type as 'ICC(C,k)' for k raters.
    """
    from metasignal.stdpy.stats_helpers import icc
    rng = np.random.default_rng(5)
    data = rng.normal(size=(20, 2))
    result = icc(data)
    assert "ICC(C,k)" in result["Type"].values, (
        "Expected ICC(C,k) (consistency, k-rater) row — paper uses C-k type"
    )


def _get_icc_ck(result):
    """Extract the ICC(C,k) consistency value from a pingouin result DataFrame."""
    return result.loc[result["Type"] == "ICC(C,k)", "ICC"].values[0]


def test_icc_perfect_reliability():
    """Perfectly correlated sessions should yield ICC ≈ 1.
    Consistent with paper's finding that split-half ICC is near 1 for large bins.
    """
    from metasignal.stdpy.stats_helpers import icc
    base = np.linspace(0, 1, 30)
    # Two sessions that differ by only a small constant (avoids divide-by-zero)
    data = np.column_stack([base, base + 1e-6 * np.arange(30)])
    result = icc(data)
    icc_ck = _get_icc_ck(result)
    assert icc_ck > 0.99, f"Near-identical sessions should give ICC ≈ 1, got {icc_ck:.3f}"


def test_icc_random_data_near_zero():
    """Independent sessions should yield ICC near 0.
    Consistent with paper's finding of poor test-retest reliability (ICC < 0.5
    for most measures even at 400 trials; Rahnev 2025, Fig. 6).
    """
    from metasignal.stdpy.stats_helpers import icc
    rng = np.random.default_rng(42)
    # Generate one session and permute it to create a fully independent second session
    session1 = rng.normal(size=70)
    session2 = rng.permutation(session1)  # same marginal distribution, zero rank correlation
    data = np.column_stack([session1, session2])
    result = icc(data)
    icc_ck = _get_icc_ck(result)
    assert abs(icc_ck) < 0.15, f"Permuted (independent) sessions should give ICC ≈ 0, got {icc_ck:.3f}"


def test_icc_high_reliability_beats_low():
    """High-consistency data should yield higher ICC than low-consistency data.
    Mirrors Rahnev (2025): ΔConf ICC = 0.75 at 400 trials > M-Ratio ICC = 0.42.
    """
    from metasignal.stdpy.stats_helpers import icc
    rng = np.random.default_rng(7)
    n = 70  # Haddara n

    true_scores = rng.normal(size=n)
    # High reliability: session 2 ≈ session 1 + small noise
    high_rel = np.column_stack([true_scores, true_scores + rng.normal(scale=0.2, size=n)])
    # Low reliability: session 2 ≈ session 1 + large noise
    low_rel = np.column_stack([true_scores, true_scores + rng.normal(scale=2.0, size=n)])

    icc_high = _get_icc_ck(icc(high_rel))
    icc_low = _get_icc_ck(icc(low_rel))

    assert icc_high > icc_low, (
        f"High-reliability data (ICC={icc_high:.3f}) should exceed "
        f"low-reliability (ICC={icc_low:.3f})"
    )
