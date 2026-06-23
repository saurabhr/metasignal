"""Tests for metasignal.sdtbayes pure-Python helpers.

Covers every function that does not require the brmspy/Stan runtime:
  - hierarchical._trials_to_dataframe
  - full_metad._build_count_matrix
  - full_metad._group_likelihood_block
  - two_stage._compute_participant_estimates
  - subject_level._extract_type1
  - subject_level._build_counts_vector

Also covers guard logic in fit_two_stage_comparison via a brmspy mock.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def simple_participant(rng):
    """Single participant: 100 trials, 2 ratings, reasonable signal."""
    stim = rng.integers(0, 2, 100)
    resp = stim.copy()
    flip = rng.random(100) < 0.15
    resp[flip] = 1 - resp[flip]
    conf = rng.integers(1, 3, 100)
    return stim, resp, conf


@pytest.fixture
def two_participants(rng):
    """Two participants suitable for two-stage tests."""
    parts = []
    for _ in range(2):
        stim = rng.integers(0, 2, 120)
        resp = stim.copy()
        flip = rng.random(120) < 0.12
        resp[flip] = 1 - resp[flip]
        conf = rng.integers(1, 3, 120)
        parts.append((stim, resp, conf))
    return parts


# Canonical Fleming (2017) / metadpy dataset — 4 ratings
CANONICAL_NR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2])
CANONICAL_NR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45])


# ---------------------------------------------------------------------------
# hierarchical._trials_to_dataframe
# ---------------------------------------------------------------------------

class TestTrialsToDataframe:
    @pytest.fixture(autouse=True)
    def require_pandas(self):
        pytest.importorskip("pandas")

    def _make_df(self, participants, n_ratings=2):
        from metasignal.sdtbayes.hierarchical import _trials_to_dataframe
        return _trials_to_dataframe(participants, n_ratings)

    def test_columns_present(self, simple_participant):
        df = self._make_df([simple_participant])
        for col in ("participant", "stimulus", "response", "conf", "correct", "group"):
            assert col in df.columns

    def test_row_count(self, simple_participant):
        stim, resp, conf = simple_participant
        df = self._make_df([simple_participant])
        assert len(df) == len(stim)

    def test_row_count_multi_participant(self, two_participants):
        df = self._make_df(two_participants)
        total = sum(len(s) for s, _, _ in two_participants)
        assert len(df) == total

    def test_correct_column_binary(self, simple_participant):
        df = self._make_df([simple_participant])
        assert set(df["correct"].unique()).issubset({0, 1})

    def test_correct_column_values(self, simple_participant):
        stim, resp, conf = simple_participant
        df = self._make_df([simple_participant])
        expected_correct = (stim == resp).astype(int)
        assert (df["correct"].values == expected_correct).all()

    def test_conf_is_ordered_categorical(self, simple_participant):
        import pandas as pd
        df = self._make_df([simple_participant])
        assert isinstance(df["conf"].dtype, pd.CategoricalDtype)
        assert df["conf"].cat.ordered

    def test_group_label(self, simple_participant):
        df = self._make_df([simple_participant], n_ratings=2)
        assert (df["group"] == 0).all()

    def test_group_label_custom(self, simple_participant):
        from metasignal.sdtbayes.hierarchical import _trials_to_dataframe
        df = _trials_to_dataframe([simple_participant], n_ratings=2, group_label=1)
        assert (df["group"] == 1).all()

    def test_participant_ids_distinct(self, two_participants):
        df = self._make_df(two_participants)
        assert df["participant"].nunique() == 2


# ---------------------------------------------------------------------------
# full_metad._build_count_matrix
# ---------------------------------------------------------------------------

class TestBuildCountMatrix:
    def _build(self, participants, n_ratings=2):
        from metasignal.sdtbayes.full_metad import _build_count_matrix
        return _build_count_matrix(participants, n_ratings)

    def test_shape(self, two_participants):
        mat = self._build(two_participants, n_ratings=2)
        assert mat.shape == (2, 4 * 2)

    def test_dtype_int(self, two_participants):
        mat = self._build(two_participants, n_ratings=2)
        assert np.issubdtype(mat.dtype, np.integer)

    def test_nonnegative(self, simple_participant):
        mat = self._build([simple_participant], n_ratings=2)
        assert np.all(mat >= 0)

    def test_row_sum_equals_trial_count(self, simple_participant):
        stim, resp, conf = simple_participant
        mat = self._build([simple_participant], n_ratings=2)
        assert mat[0].sum() == len(stim)

    def test_four_rating_shape(self, simple_participant):
        stim, resp, conf = simple_participant
        conf4 = np.clip(conf * 2, 1, 4)
        mat = self._build([(stim, resp, conf4)], n_ratings=4)
        assert mat.shape == (1, 16)


# ---------------------------------------------------------------------------
# full_metad._group_likelihood_block
# ---------------------------------------------------------------------------

class TestGroupLikelihoodBlock:
    def test_returns_string(self):
        from metasignal.sdtbayes.full_metad import _group_likelihood_block
        result = _group_likelihood_block(
            "nsubj_a", "hmetad_counts_a", "Mratio_a", "d1_a", "c1_a", "cS1_a", "cS2_a",
        )
        assert isinstance(result, str)

    def test_contains_correct_variable_names(self):
        from metasignal.sdtbayes.full_metad import _group_likelihood_block
        result = _group_likelihood_block(
            "nsubj_a", "hmetad_counts_a", "Mratio_a", "d1_a", "c1_a", "cS1_a", "cS2_a",
        )
        for name in ("nsubj_a", "hmetad_counts_a", "Mratio_a", "d1_a", "c1_a", "cS1_a", "cS2_a"):
            assert name in result

    def test_contains_multinomial_lpmf(self):
        from metasignal.sdtbayes.full_metad import _group_likelihood_block
        result = _group_likelihood_block(
            "nsubj_b", "hmetad_counts_b", "Mratio_b", "d1_b", "c1_b", "cS1_b", "cS2_b",
        )
        assert "multinomial_lpmf" in result


# ---------------------------------------------------------------------------
# two_stage._compute_participant_estimates
# ---------------------------------------------------------------------------

class TestComputeParticipantEstimates:
    @pytest.fixture(autouse=True)
    def require_pandas(self):
        pytest.importorskip("pandas")

    def _compute(self, participants, n_ratings=2):
        from metasignal.sdtbayes.two_stage import _compute_participant_estimates
        return _compute_participant_estimates(participants, n_ratings)

    def test_columns(self, simple_participant):
        df = self._compute([simple_participant])
        for col in ("participant", "dprime", "c", "meta_da", "da", "m_ratio", "log_m_ratio"):
            assert col in df.columns

    def test_row_count(self, two_participants):
        df = self._compute(two_participants)
        assert len(df) == 2

    def test_valid_participant_finite(self, simple_participant):
        df = self._compute([simple_participant])
        assert np.isfinite(df["meta_da"].iloc[0])

    def test_log_m_ratio_equals_log_m_ratio(self, simple_participant):
        df = self._compute([simple_participant])
        row = df.iloc[0]
        if row["m_ratio"] > 0:
            assert abs(row["log_m_ratio"] - np.log(row["m_ratio"])) < 1e-10

    def test_degenerate_participant_warns_and_returns_nan(self):
        """A participant with only one stimulus class triggers a warning."""
        stim = np.array([0, 0, 0, 0])
        resp = np.array([0, 0, 0, 0])
        conf = np.array([1, 1, 2, 2])
        with pytest.warns(UserWarning, match="MLE failed"):
            df = self._compute([(stim, resp, conf)])
        assert df["log_m_ratio"].isna().all()


# ---------------------------------------------------------------------------
# two_stage group-size guard (fit_two_stage_comparison) — brmspy mocked
# ---------------------------------------------------------------------------

class TestTwoStageComparisonGuard:
    def _make_group(self, n, rng, n_trials=100):
        parts = []
        for _ in range(n):
            stim = rng.integers(0, 2, n_trials)
            resp = stim.copy()
            flip = rng.random(n_trials) < 0.15
            resp[flip] = 1 - resp[flip]
            conf = rng.integers(1, 3, n_trials)
            parts.append((stim, resp, conf))
        return parts

    def _call(self, group_a, group_b):
        from metasignal.sdtbayes.two_stage import fit_two_stage_comparison
        return fit_two_stage_comparison(group_a, group_b, n_ratings=2)

    def _patched_modules(self):
        """Context manager that mocks brmspy and pandas together."""
        pandas = pytest.importorskip("pandas")
        mock_brmspy = MagicMock()
        mock_brmspy.brms = MagicMock()
        return patch.dict("sys.modules", {"brmspy": mock_brmspy}), mock_brmspy, pandas

    def test_raises_when_group_a_too_small(self, rng):
        pandas = pytest.importorskip("pandas")
        mock_brmspy = MagicMock()
        mock_brmspy.brms = MagicMock()
        with patch.dict("sys.modules", {"brmspy": mock_brmspy}):
            group_a = self._make_group(2, rng)
            group_b = self._make_group(5, rng)
            with pytest.raises(ValueError, match="Group A"):
                self._call(group_a, group_b)

    def test_raises_when_group_b_too_small(self, rng):
        pytest.importorskip("pandas")
        mock_brmspy = MagicMock()
        mock_brmspy.brms = MagicMock()
        with patch.dict("sys.modules", {"brmspy": mock_brmspy}):
            group_a = self._make_group(5, rng)
            group_b = self._make_group(1, rng)
            with pytest.raises(ValueError, match="Group B"):
                self._call(group_a, group_b)

    def test_passes_guard_with_sufficient_participants(self, rng):
        pytest.importorskip("pandas")
        mock_brmspy = MagicMock()
        mock_brms = mock_brmspy.brms
        with patch.dict("sys.modules", {"brmspy": mock_brmspy}):
            group_a = self._make_group(5, rng)
            group_b = self._make_group(5, rng)
            self._call(group_a, group_b)
        mock_brms.brm.assert_called_once()


# ---------------------------------------------------------------------------
# subject_level._extract_type1
# ---------------------------------------------------------------------------

class TestExtractType1:
    def _extract(self, nR_S1, nR_S2):
        from metasignal.sdtbayes.subject_level import _extract_type1
        return _extract_type1(np.array(nR_S1), np.array(nR_S2))

    def test_canonical_totals(self):
        t1 = self._extract(CANONICAL_NR_S1, CANONICAL_NR_S2)
        assert t1["S"] == int(CANONICAL_NR_S2.sum())
        assert t1["N"] == int(CANONICAL_NR_S1.sum())

    def test_canonical_hits(self):
        t1 = self._extract(CANONICAL_NR_S1, CANONICAL_NR_S2)
        # Hits = nR_S2[nratings:] with nratings=4
        expected_H = int(CANONICAL_NR_S2[4:].sum())
        assert t1["H"] == expected_H

    def test_canonical_false_alarms(self):
        t1 = self._extract(CANONICAL_NR_S1, CANONICAL_NR_S2)
        expected_FA = int(CANONICAL_NR_S1[4:].sum())
        assert t1["FA"] == expected_FA

    def test_keys_present(self):
        t1 = self._extract(CANONICAL_NR_S1, CANONICAL_NR_S2)
        assert set(t1.keys()) == {"S", "N", "H", "FA"}

    def test_all_nonnegative(self):
        t1 = self._extract(CANONICAL_NR_S1, CANONICAL_NR_S2)
        assert all(v >= 0 for v in t1.values())


# ---------------------------------------------------------------------------
# subject_level._build_counts_vector
# ---------------------------------------------------------------------------

class TestBuildCountsVector:
    def _build(self, nR_S1, nR_S2):
        from metasignal.sdtbayes.subject_level import _build_counts_vector
        return _build_counts_vector(np.array(nR_S1), np.array(nR_S2))

    def test_length(self):
        vec = self._build(CANONICAL_NR_S1, CANONICAL_NR_S2)
        assert len(vec) == len(CANONICAL_NR_S1) + len(CANONICAL_NR_S2)

    def test_length_equals_4_nratings(self):
        vec = self._build(CANONICAL_NR_S1, CANONICAL_NR_S2)
        n_ratings = len(CANONICAL_NR_S1) // 2
        assert len(vec) == 4 * n_ratings

    def test_sum_equals_total_trials(self):
        vec = self._build(CANONICAL_NR_S1, CANONICAL_NR_S2)
        expected = int(CANONICAL_NR_S1.sum() + CANONICAL_NR_S2.sum())
        assert int(vec.sum()) == expected

    def test_dtype_int(self):
        vec = self._build(CANONICAL_NR_S1, CANONICAL_NR_S2)
        assert np.issubdtype(vec.dtype, np.integer)

    def test_nonnegative(self):
        vec = self._build(CANONICAL_NR_S1, CANONICAL_NR_S2)
        assert np.all(vec >= 0)

    def test_matches_concatenation_of_nR_arrays(self):
        """Output must equal [nR_S1 | nR_S2] which is metadpy's extractParameters layout."""
        vec = self._build(CANONICAL_NR_S1, CANONICAL_NR_S2)
        expected = np.concatenate([CANONICAL_NR_S1, CANONICAL_NR_S2])
        np.testing.assert_array_equal(vec, expected)


# ---------------------------------------------------------------------------
# subject_level.fit_subject_level — input validation (no brmspy needed)
# ---------------------------------------------------------------------------

class TestFitSubjectLevelValidation:
    """Input validation runs before optional imports — no brmspy/pandas needed."""

    def test_mismatched_lengths_raises(self):
        from metasignal.sdtbayes.subject_level import fit_subject_level
        with pytest.raises(ValueError, match="same length"):
            fit_subject_level(np.array([1, 2, 3, 4]), np.array([1, 2]))

    def test_odd_length_raises(self):
        from metasignal.sdtbayes.subject_level import fit_subject_level
        with pytest.raises(ValueError, match="even"):
            fit_subject_level(np.array([1, 2, 3]), np.array([1, 2, 3]))
