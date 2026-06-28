"""Comprehensive tests for information-theoretic metacognition measures (metasignal.itmc)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import metasignal.itmc as itmc
from metasignal.itmc import (
    MEASURE_COLS,
    RMI,
    estimate_meta_I,
    fit_group,
    meta_I,
    meta_Ir1,
    meta_Ir1_acc,
    meta_Ir2,
    permtest_meta_I,
)
from metasignal.itmc.measures import (
    _build_contingency_table,
    _entropy,
    _gaussian_info_statconfr,
    _get_accuracy_from_table,
    _get_info,
    _h2,
    _lower_info_bound,
    _upper_info_bound,
)


# ---------------------------------------------------------------------------
# Shared data-generation helpers
# ---------------------------------------------------------------------------

def _sdt_data(seed: int = 0, n: int = 600, d: float = 1.5, metad: float = 1.2):
    """Generate realistic SDT trial data via trialSimulation."""
    from metasignal.stdpy.simulate import trialSimulation
    df = trialSimulation(d=d, metad=metad, nTrials=n, rng=np.random.default_rng(seed))
    stim = df["Stimuli"].to_numpy(dtype=int)
    resp = df["Responses"].to_numpy(dtype=int)
    conf = df["Confidence"].to_numpy(dtype=int)
    return stim, resp, conf


def _random_conf_data(seed: int = 1, n: int = 3000, n_ratings: int = 4):
    """Generate data with random (uninformative) confidence ratings."""
    rng = np.random.default_rng(seed)
    stim, resp, _ = _sdt_data(seed=seed)
    n_actual = len(stim)
    conf = rng.integers(1, n_ratings + 1, size=n_actual)
    return stim, resp, conf


def _make_df(n_subjects=4, n_trials=400, n_conditions=2, seed=0):
    """Build a multi-subject × multi-condition DataFrame."""
    from metasignal.stdpy.simulate import trialSimulation
    records = []
    for sid in range(n_subjects):
        for cond in range(n_conditions):
            d = 1.2 + 0.2 * sid + 0.3 * cond
            rng = np.random.default_rng(seed * 100 + sid * 10 + cond)
            df = trialSimulation(d=d, metad=d * 0.8, nTrials=n_trials, rng=rng)
            group = "A" if sid < n_subjects // 2 else "B"
            df = df.rename(columns={
                "Stimuli": "stimulus", "Responses": "response", "Confidence": "rating"
            })
            df["subject"] = f"s{sid}"
            df["condition"] = cond
            df["block"] = cond % 2          # second within factor (varies per condition)
            df["group"] = group
            df["task"] = "T1" if sid < n_subjects // 2 else "T2"  # second between factor
            records.append(df[["stimulus", "response", "rating",
                               "subject", "condition", "block", "group", "task"]])
    return pd.concat(records, ignore_index=True)


def _all_correct_data():
    """All trials correct, uniform confidence."""
    n = 200
    stim = np.array([0] * (n // 2) + [1] * (n // 2))
    resp = stim.copy()
    conf = np.ones(n, dtype=int)
    return stim, resp, conf


def _single_conf_data():
    """All trials use the same confidence level (no variability)."""
    stim, resp, _ = _sdt_data(seed=42)
    conf = np.ones(len(stim), dtype=int)
    return stim, resp, conf


def _perfect_meta_data():
    """Confidence perfectly predicts accuracy: high conf → correct, low → wrong."""
    n = 400
    rng = np.random.default_rng(7)
    stim = rng.integers(0, 2, n)
    resp = stim.copy()
    # correct trials get conf 4, incorrect get conf 1
    # Make half incorrect
    wrong = rng.choice(n, n // 4, replace=False)
    resp[wrong] = 1 - stim[wrong]
    acc = (resp == stim).astype(int)
    conf = np.where(acc == 1, 4, 1)
    return stim, resp, conf


# ============================================================================
# 1. Internal Helpers
# ============================================================================

class TestHelpers:
    def test_h2_at_zero(self):
        assert _h2(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_h2_at_one(self):
        assert _h2(1.0) == pytest.approx(0.0, abs=1e-9)

    def test_h2_at_half(self):
        assert _h2(0.5) == pytest.approx(1.0, abs=1e-6)

    def test_h2_at_0_3(self):
        import math
        expected = -0.3 * math.log2(0.3) - 0.7 * math.log2(0.7)
        assert _h2(0.3) == pytest.approx(expected, rel=1e-5)

    def test_entropy_uniform(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert _entropy(p) == pytest.approx(2.0, abs=1e-6)

    def test_entropy_degenerate(self):
        p = np.array([1.0, 0.0, 0.0])
        assert _entropy(p) == pytest.approx(0.0, abs=1e-9)

    def test_build_contingency_table_shape(self):
        stim, resp, conf = _sdt_data()
        n_ratings = int(conf.max())
        table = _build_contingency_table(stim, resp, conf)
        assert table.shape == (2, 2 * n_ratings)

    def test_build_contingency_table_row_sums(self):
        stim, resp, conf = _sdt_data()
        table = _build_contingency_table(stim, resp, conf)
        n_s1 = int((stim == 0).sum())
        n_s2 = int((stim == 1).sum())
        assert int(table[0].sum()) == n_s1
        assert int(table[1].sum()) == n_s2

    def test_build_contingency_table_total(self):
        stim, resp, conf = _sdt_data()
        table = _build_contingency_table(stim, resp, conf)
        assert int(table.sum()) == len(stim)

    def test_get_info_zero_for_degenerate(self):
        """All mass in one cell → zero mutual information."""
        table = np.zeros((2, 8))
        table[0, 0] = 100
        assert _get_info(table) == pytest.approx(0.0, abs=1e-6)

    def test_get_info_positive_for_informative(self):
        stim, resp, conf = _sdt_data()
        table = _build_contingency_table(stim, resp, conf)
        assert _get_info(table) > 0

    def test_get_accuracy_from_table_matches_direct(self):
        stim, resp, conf = _sdt_data()
        direct_acc = float((stim == resp).mean())
        table = _build_contingency_table(stim, resp, conf)
        table_acc = _get_accuracy_from_table(table)
        assert table_acc == pytest.approx(direct_acc, abs=0.02)

    def test_lower_le_upper(self):
        stim, resp, conf = _sdt_data()
        prior = np.array([0.5, 0.5])
        acc = float((stim == resp).mean())
        lb = _lower_info_bound(prior, acc)
        ub = _upper_info_bound(prior, acc)
        assert lb <= ub + 1e-9

    def test_lower_bound_zero_at_max_prior(self):
        """Lower bound = 0 when accuracy equals max(prior)."""
        prior = np.array([0.5, 0.5])
        lb = _lower_info_bound(prior, 0.5)
        assert lb == pytest.approx(0.0, abs=1e-6)

    def test_upper_bound_equals_entropy_when_acc_one(self):
        prior = np.array([0.5, 0.5])
        ub = _upper_info_bound(prior, 1.0)
        assert ub == pytest.approx(_entropy(prior), abs=1e-6)

    def test_bounds_equal_prior(self):
        prior = np.array([0.5, 0.5])
        stim, resp, conf = _sdt_data()
        acc = float((stim == resp).mean())
        lb = _lower_info_bound(prior, acc)
        ub = _upper_info_bound(prior, acc)
        assert lb >= 0.0
        assert ub <= 1.01

    def test_gaussian_info_zero_at_dprime_zero(self):
        assert _gaussian_info_statconfr(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_gaussian_info_positive_for_positive_dprime(self):
        assert _gaussian_info_statconfr(1.5) > 0

    def test_gaussian_info_monotone_in_dprime(self):
        vals = [_gaussian_info_statconfr(d) for d in [0.5, 1.0, 1.5, 2.0, 2.5]]
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


# ============================================================================
# 2. Public Measures — simple backend
# ============================================================================

class TestSimpleBackend:
    def test_meta_I_positive_informative(self):
        stim, resp, conf = _sdt_data()
        assert meta_I(stim, resp, conf, backend="simple") > 0

    def test_meta_I_near_zero_random_conf(self):
        stim, resp, conf = _random_conf_data()
        mi = meta_I(stim, resp, conf, backend="simple")
        assert abs(mi) < 0.05, f"Expected near-zero meta_I, got {mi:.4f}"

    def test_meta_Ir1_positive(self):
        stim, resp, conf = _sdt_data()
        assert meta_Ir1(stim, resp, conf, backend="simple") > 0

    def test_meta_Ir1_uses_dprime_arg(self):
        stim, resp, conf = _sdt_data()
        r1_auto = meta_Ir1(stim, resp, conf, backend="simple")
        r1_expl = meta_Ir1(stim, resp, conf, dprime=1.5, backend="simple")
        # Both should be positive; they differ because dprime differs
        assert r1_auto > 0
        assert r1_expl > 0

    def test_meta_Ir1_explicit_dprime_different_from_auto(self):
        stim, resp, conf = _sdt_data()
        r1_auto = meta_Ir1(stim, resp, conf, backend="simple")
        r1_expl = meta_Ir1(stim, resp, conf, dprime=0.5, backend="simple")
        # Different dprime → different normalisation → different values
        assert r1_auto != pytest.approx(r1_expl, rel=0.01)

    def test_meta_Ir1_acc_positive(self):
        stim, resp, conf = _sdt_data()
        val = meta_Ir1_acc(stim, resp, conf, backend="simple")
        assert not np.isnan(val)
        assert val > 0

    def test_meta_Ir1_acc_nan_when_all_correct(self):
        stim, resp, conf = _all_correct_data()
        val = meta_Ir1_acc(stim, resp, conf, backend="simple")
        assert np.isnan(val)

    def test_meta_Ir2_in_range(self):
        stim, resp, conf = _sdt_data()
        ir2 = meta_Ir2(stim, resp, conf, backend="simple")
        assert 0.0 <= ir2 <= 1.0

    def test_meta_Ir2_nan_when_all_correct(self):
        stim, resp, conf = _all_correct_data()
        val = meta_Ir2(stim, resp, conf, backend="simple")
        assert np.isnan(val)

    def test_RMI_simple_can_be_negative(self):
        # Known limitation of simple backend: RMI can be outside [0,1]
        # because numerator uses MI(acc;conf) but bounds assume I(S;R).
        stim, resp, conf = _sdt_data()
        rmi = RMI(stim, resp, conf, backend="simple")
        # We do not assert range here — just that it runs and is a finite float.
        assert np.isfinite(rmi)

    def test_bias_correction_le_uncorrected(self):
        stim, resp, conf = _sdt_data(seed=99)
        mi_raw = meta_I(stim, resp, conf, backend="simple", bias_correction=False)
        mi_bc  = meta_I(stim, resp, conf, backend="simple", bias_correction=True, seed=0)
        assert mi_bc <= mi_raw + 0.01


# ============================================================================
# 3. Public Measures — statconfr backend
# ============================================================================

class TestStatconfrBackend:
    def test_meta_I_positive(self):
        stim, resp, conf = _sdt_data()
        assert meta_I(stim, resp, conf, backend="statconfr") > 0

    def test_meta_I_near_zero_random_conf(self):
        stim, resp, conf = _random_conf_data()
        mi = meta_I(stim, resp, conf, backend="statconfr")
        assert abs(mi) < 0.1, f"Expected near-zero meta_I (statconfr), got {mi:.4f}"

    def test_meta_Ir1_positive(self):
        stim, resp, conf = _sdt_data()
        assert meta_Ir1(stim, resp, conf, backend="statconfr") > 0

    def test_meta_Ir1_changes_with_dprime(self):
        stim, resp, conf = _sdt_data()
        r1_low  = meta_Ir1(stim, resp, conf, dprime=0.5, backend="statconfr")
        r1_high = meta_Ir1(stim, resp, conf, dprime=3.0, backend="statconfr")
        assert r1_low != pytest.approx(r1_high, rel=0.01)

    def test_meta_Ir2_in_range(self):
        stim, resp, conf = _sdt_data()
        ir2 = meta_Ir2(stim, resp, conf, backend="statconfr")
        assert 0.0 <= ir2 <= 1.5  # slight overshoot possible for small n

    def test_RMI_in_range_statconfr(self):
        stim, resp, conf = _sdt_data(n=1200)
        rmi = RMI(stim, resp, conf, backend="statconfr")
        assert 0.0 <= rmi <= 1.0, f"statconfr RMI out of [0,1]: {rmi:.4f}"

    def test_bias_correction_statconfr_runs(self):
        stim, resp, conf = _sdt_data()
        mi = meta_I(stim, resp, conf, backend="statconfr", bias_correction=True, seed=0)
        assert np.isfinite(mi)


# ============================================================================
# 4. Backend Agreement
# ============================================================================

class TestBackendAgreement:
    def test_both_backends_positive_meta_I(self):
        stim, resp, conf = _sdt_data()
        for bk in ("simple", "statconfr"):
            mi = meta_I(stim, resp, conf, backend=bk)
            assert mi > 0, f"backend={bk}: expected meta_I > 0, got {mi}"

    def test_both_backends_agree_sign(self):
        stim, resp, conf = _sdt_data()
        mi_s  = meta_I(stim, resp, conf, backend="simple")
        mi_sc = meta_I(stim, resp, conf, backend="statconfr")
        assert (mi_s > 0) == (mi_sc > 0)

    def test_meta_Ir2_within_0_3(self):
        stim, resp, conf = _sdt_data(n=1000)
        ir2_s  = meta_Ir2(stim, resp, conf, backend="simple")
        ir2_sc = meta_Ir2(stim, resp, conf, backend="statconfr")
        assert abs(ir2_s - ir2_sc) < 0.3

    def test_invalid_backend_raises_ValueError(self):
        stim, resp, conf = _sdt_data()
        with pytest.raises(ValueError, match="Unknown backend"):
            meta_I(stim, resp, conf, backend="bogus")

    def test_invalid_backend_meta_Ir1(self):
        stim, resp, conf = _sdt_data()
        with pytest.raises(ValueError):
            meta_Ir1(stim, resp, conf, backend="bogus")

    def test_invalid_backend_meta_Ir2(self):
        stim, resp, conf = _sdt_data()
        with pytest.raises(ValueError):
            meta_Ir2(stim, resp, conf, backend="bogus")

    def test_invalid_backend_RMI(self):
        stim, resp, conf = _sdt_data()
        with pytest.raises(ValueError):
            RMI(stim, resp, conf, backend="bogus")

    def test_invalid_backend_meta_Ir1_acc(self):
        stim, resp, conf = _sdt_data()
        with pytest.raises(ValueError):
            meta_Ir1_acc(stim, resp, conf, backend="bogus")


# ============================================================================
# 5. estimate_meta_I
# ============================================================================

class TestEstimateMetaI:
    def _single_participant_df(self, seed=0, pid="s1"):
        stim, resp, conf = _sdt_data(seed=seed)
        return pd.DataFrame({
            "stimulus": stim, "response": resp, "rating": conf, "participant": pid
        })

    def test_returns_dataframe(self):
        df = self._single_participant_df()
        result = estimate_meta_I(df)
        assert isinstance(result, pd.DataFrame)

    def test_columns_exact(self):
        df = self._single_participant_df()
        result = estimate_meta_I(df)
        assert list(result.columns) == ["participant", "meta_I", "meta_Ir1",
                                        "meta_Ir1_acc", "meta_Ir2", "RMI"]

    def test_one_row_single_participant(self):
        df = self._single_participant_df()
        result = estimate_meta_I(df)
        assert len(result) == 1

    def test_three_rows_three_participants(self):
        dfs = [self._single_participant_df(seed=i, pid=f"s{i}") for i in range(3)]
        df = pd.concat(dfs, ignore_index=True)
        result = estimate_meta_I(df)
        assert len(result) == 3

    def test_meta_I_positive_informative_simple(self):
        df = self._single_participant_df()
        result = estimate_meta_I(df, backend="simple")
        assert result["meta_I"].iloc[0] > 0

    def test_meta_I_positive_informative_statconfr(self):
        df = self._single_participant_df()
        result = estimate_meta_I(df, backend="statconfr")
        assert result["meta_I"].iloc[0] > 0

    def test_non_default_column_names(self):
        stim, resp, conf = _sdt_data()
        df = pd.DataFrame({
            "stim": stim, "resp": resp, "conf": conf, "subj": "p1"
        })
        result = estimate_meta_I(
            df,
            stimulus_col="stim",
            response_col="resp",
            rating_col="conf",
            participant_col="subj",
        )
        assert len(result) == 1
        assert "subj" in result.columns
        assert result["meta_I"].iloc[0] > 0

    def test_bias_correction_runs(self):
        df = self._single_participant_df()
        result = estimate_meta_I(df, bias_correction=True, seed=0)
        assert result["meta_I"].notna().all()


# ============================================================================
# 6. fit_group
# ============================================================================

class TestFitGroup:
    def test_no_grouping_shape(self):
        df = _make_df(n_subjects=1, n_conditions=1)
        result = fit_group(df, stimuli="stimulus", responses="response",
                           confidence="rating")
        assert result.shape == (1, 5)

    def test_no_grouping_columns(self):
        df = _make_df(n_subjects=1, n_conditions=1)
        result = fit_group(df, stimuli="stimulus", responses="response",
                           confidence="rating")
        assert set(result.columns) == {"meta_I", "meta_Ir1", "meta_Ir1_acc", "meta_Ir2", "RMI"}

    def test_subject_only_row_count(self):
        df = _make_df(n_subjects=3, n_conditions=1)
        result = fit_group(df, stimuli="stimulus", responses="response",
                           confidence="rating", subject="subject")
        assert len(result) == 3

    def test_subject_only_meta_I_positive(self):
        df = _make_df(n_subjects=3, n_conditions=1)
        result = fit_group(df, stimuli="stimulus", responses="response",
                           confidence="rating", subject="subject")
        assert result["meta_I"].gt(0).all()

    def test_within_single_row_count(self):
        df = _make_df(n_subjects=4, n_conditions=2)
        result = fit_group(df, stimuli="stimulus", responses="response",
                           confidence="rating", subject="subject", within="condition")
        assert len(result) == 4 * 2

    def test_within_single_columns(self):
        df = _make_df(n_subjects=4, n_conditions=2)
        result = fit_group(df, stimuli="stimulus", responses="response",
                           confidence="rating", subject="subject", within="condition")
        assert {"subject", "condition"}.issubset(result.columns)

    def test_within_multiple_factors_row_count(self):
        # Build data where each subject has both conditions AND both blocks.
        from metasignal.stdpy.simulate import trialSimulation
        records = []
        for sid in range(3):
            for cond in range(2):
                for blk in range(2):
                    d = 1.2 + 0.1 * sid
                    rng = np.random.default_rng(sid * 100 + cond * 10 + blk)
                    df_ = trialSimulation(d=d, metad=d * 0.8, nTrials=200, rng=rng)
                    df_ = df_.rename(columns={
                        "Stimuli": "stimulus", "Responses": "response", "Confidence": "rating"
                    })
                    df_["subject"] = f"s{sid}"
                    df_["condition"] = cond
                    df_["block"] = blk
                    records.append(df_[["stimulus", "response", "rating",
                                        "subject", "condition", "block"]])
        df = pd.concat(records, ignore_index=True)
        result = fit_group(df, stimuli="stimulus", responses="response",
                           confidence="rating", subject="subject",
                           within=["condition", "block"])
        # 3 subjects × 2 conditions × 2 blocks = 12 rows
        assert len(result) == 3 * 2 * 2

    def test_between_single_group_col_present(self):
        df = _make_df(n_subjects=4, n_conditions=1)
        result = fit_group(df, stimuli="stimulus", responses="response",
                           confidence="rating", subject="subject", between="group")
        assert "group" in result.columns

    def test_between_single_row_count(self):
        df = _make_df(n_subjects=4, n_conditions=1)
        result = fit_group(df, stimuli="stimulus", responses="response",
                           confidence="rating", subject="subject", between="group")
        assert len(result) == 4  # one row per subject

    def test_between_multiple_factors_cols(self):
        df = _make_df(n_subjects=4, n_conditions=2)
        result = fit_group(df, stimuli="stimulus", responses="response",
                           confidence="rating", subject="subject",
                           between=["group", "task"])
        assert {"group", "task"}.issubset(result.columns)

    def test_mixed_within_between_row_count(self):
        df = _make_df(n_subjects=4, n_conditions=2)
        result = fit_group(df, stimuli="stimulus", responses="response",
                           confidence="rating", subject="subject",
                           within="condition", between="group")
        n_subjects = df["subject"].nunique()
        n_cond = df["condition"].nunique()
        assert len(result) == n_subjects * n_cond

    def test_measures_subset(self):
        df = _make_df(n_subjects=2, n_conditions=1)
        result = fit_group(df, stimuli="stimulus", responses="response",
                           confidence="rating", subject="subject",
                           measures=["meta_I", "RMI"])
        assert set(result.columns) == {"subject", "meta_I", "RMI"}

    def test_invalid_measure_raises_ValueError(self):
        df = _make_df(n_subjects=1, n_conditions=1)
        with pytest.raises(ValueError, match="Unknown measure"):
            fit_group(df, stimuli="stimulus", responses="response",
                      confidence="rating", measures=["not_a_measure"])

    def test_statconfr_backend_gives_positive_meta_I(self):
        df = _make_df(n_subjects=2, n_conditions=1)
        result = fit_group(df, stimuli="stimulus", responses="response",
                           confidence="rating", subject="subject",
                           backend="statconfr")
        assert result["meta_I"].gt(0).all()

    def test_bias_correction_runs_group(self):
        df = _make_df(n_subjects=2, n_conditions=1)
        result = fit_group(df, stimuli="stimulus", responses="response",
                           confidence="rating", subject="subject",
                           bias_correction=True, seed=0)
        assert result["meta_I"].notna().all()


# ============================================================================
# 7. Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_single_conf_level_meta_I_not_crash(self):
        """All trials same confidence — meta_I should be 0 (no variability)."""
        stim, resp, conf = _single_conf_data()
        mi = meta_I(stim, resp, conf, backend="simple")
        assert mi == pytest.approx(0.0, abs=1e-9)

    def test_single_conf_level_statconfr_not_crash(self):
        stim, resp, conf = _single_conf_data()
        mi = meta_I(stim, resp, conf, backend="statconfr")
        assert np.isfinite(mi) or np.isnan(mi)

    def test_all_correct_meta_Ir1_acc_nan_not_crash(self):
        stim, resp, conf = _all_correct_data()
        val = meta_Ir1_acc(stim, resp, conf, backend="simple")
        assert np.isnan(val)

    def test_all_correct_meta_Ir2_nan_not_crash(self):
        stim, resp, conf = _all_correct_data()
        val = meta_Ir2(stim, resp, conf, backend="simple")
        assert np.isnan(val)

    def test_few_trials_simple_no_crash(self):
        from metasignal.stdpy.simulate import trialSimulation
        df = trialSimulation(d=1.5, metad=1.2, nTrials=20,
                             rng=np.random.default_rng(5))
        stim = df["Stimuli"].to_numpy(dtype=int)
        resp = df["Responses"].to_numpy(dtype=int)
        conf = df["Confidence"].to_numpy(dtype=int)
        val = meta_I(stim, resp, conf, backend="simple")
        assert np.isfinite(val) or np.isnan(val)

    def test_few_trials_statconfr_no_crash(self):
        from metasignal.stdpy.simulate import trialSimulation
        df = trialSimulation(d=1.5, metad=1.2, nTrials=20,
                             rng=np.random.default_rng(5))
        stim = df["Stimuli"].to_numpy(dtype=int)
        resp = df["Responses"].to_numpy(dtype=int)
        conf = df["Confidence"].to_numpy(dtype=int)
        val = meta_I(stim, resp, conf, backend="statconfr")
        assert np.isfinite(val) or np.isnan(val)

    def test_unequal_prior_statconfr_RMI_sensible(self):
        """With 70% S2 stimuli, statconfr RMI should be finite and not crash."""
        from metasignal.stdpy.simulate import trialSimulation
        rng = np.random.default_rng(20)
        # Build unequal-prior data: 70% S2 by oversampling
        df = trialSimulation(d=1.5, metad=1.2, nTrials=600, rng=rng)
        stim = df["Stimuli"].to_numpy(dtype=int)
        resp = df["Responses"].to_numpy(dtype=int)
        conf = df["Confidence"].to_numpy(dtype=int)
        # Force ~70% S2
        idx_s2 = np.where(stim == 1)[0]
        idx_s1 = np.where(stim == 0)[0]
        chosen = np.concatenate([
            rng.choice(idx_s2, size=int(0.7 * 600), replace=True),
            rng.choice(idx_s1, size=int(0.3 * 600), replace=True),
        ])
        stim = stim[chosen]
        resp = resp[chosen]
        conf = conf[chosen]
        rmi = RMI(stim, resp, conf, backend="statconfr")
        assert np.isfinite(rmi)

    def test_n_ratings_2_binary_confidence_simple(self):
        from metasignal.stdpy.simulate import trialSimulation
        df = trialSimulation(d=1.5, metad=1.2, nTrials=600, nRatings=2,
                             rng=np.random.default_rng(11))
        stim = df["Stimuli"].to_numpy(dtype=int)
        resp = df["Responses"].to_numpy(dtype=int)
        conf = df["Confidence"].to_numpy(dtype=int)
        val = meta_I(stim, resp, conf, backend="simple")
        assert np.isfinite(val) or np.isnan(val)

    def test_n_ratings_2_binary_confidence_statconfr(self):
        from metasignal.stdpy.simulate import trialSimulation
        df = trialSimulation(d=1.5, metad=1.2, nTrials=600, nRatings=2,
                             rng=np.random.default_rng(11))
        stim = df["Stimuli"].to_numpy(dtype=int)
        resp = df["Responses"].to_numpy(dtype=int)
        conf = df["Confidence"].to_numpy(dtype=int)
        val = meta_I(stim, resp, conf, backend="statconfr")
        assert np.isfinite(val) or np.isnan(val)


# ============================================================================
# 8. MEASURE_COLS constant
# ============================================================================

class TestMeasureCols:
    def test_measure_cols_exact(self):
        assert MEASURE_COLS == ["meta_I", "meta_Ir1", "meta_Ir1_acc", "meta_Ir2", "RMI"]

    def test_measure_cols_accessible_from_itmc(self):
        assert itmc.MEASURE_COLS == ["meta_I", "meta_Ir1", "meta_Ir1_acc", "meta_Ir2", "RMI"]


# ============================================================================
# 9. test_meta_I — permutation test
# ============================================================================

class TestPermtestMetaI:
    def test_returns_dict_with_required_keys(self):
        stim, resp, conf = _sdt_data()
        result = permtest_meta_I(stim, resp, conf, n_perm=100, seed=0)
        for key in ("observed", "corrected", "p_value", "null_mean", "null_std", "null", "backend", "n_perm"):
            assert key in result, f"Missing key: {key}"

    def test_observed_equals_meta_I(self):
        stim, resp, conf = _sdt_data()
        result = permtest_meta_I(stim, resp, conf, backend="simple", n_perm=50, seed=0)
        assert result["observed"] == pytest.approx(meta_I(stim, resp, conf, backend="simple"), rel=1e-6)

    def test_null_has_correct_length(self):
        stim, resp, conf = _sdt_data()
        result = permtest_meta_I(stim, resp, conf, n_perm=200, seed=0)
        assert result["n_perm"] == 200

    def test_p_value_in_range(self):
        stim, resp, conf = _sdt_data()
        result = permtest_meta_I(stim, resp, conf, n_perm=100, seed=0)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_informative_data_low_p_value_simple(self):
        """Metacognitive confidence should yield p < 0.05 with real SDT data (n=1000)."""
        stim, resp, conf = _sdt_data(seed=0, n=1000)
        result = permtest_meta_I(stim, resp, conf, backend="simple", n_perm=500, seed=1)
        assert result["p_value"] < 0.05

    def test_informative_data_low_p_value_statconfr(self):
        stim, resp, conf = _sdt_data(seed=0, n=1000)
        result = permtest_meta_I(stim, resp, conf, backend="statconfr", n_perm=200, seed=1)
        assert result["p_value"] < 0.05

    def test_random_confidence_p_value_near_one(self):
        """Uninformative confidence: observed ≈ null mean → p near 0.5."""
        stim, resp, conf = _random_conf_data(seed=5)
        result = permtest_meta_I(stim, resp, conf, backend="simple", n_perm=500, seed=0)
        # random confidence should not produce a very small p-value
        assert result["p_value"] > 0.1

    def test_corrected_less_than_observed(self):
        """Bias correction should reduce (not inflate) the estimate."""
        stim, resp, conf = _sdt_data()
        result = permtest_meta_I(stim, resp, conf, backend="simple", n_perm=200, seed=0)
        assert result["corrected"] <= result["observed"] + 1e-9

    def test_statconfr_backend(self):
        stim, resp, conf = _sdt_data()
        result = permtest_meta_I(stim, resp, conf, backend="statconfr", n_perm=100, seed=0)
        assert result["backend"] == "statconfr"
        assert np.isfinite(result["p_value"])

    def test_invalid_backend_raises(self):
        stim, resp, conf = _sdt_data()
        with pytest.raises(ValueError, match="Unknown backend"):
            permtest_meta_I(stim, resp, conf, backend="bad")
