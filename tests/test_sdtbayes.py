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


# ---------------------------------------------------------------------------
# beta_auc._compute_auc2_per_participant
# ---------------------------------------------------------------------------

class TestComputeAuc2PerParticipant:
    @pytest.fixture(autouse=True)
    def require_pandas(self):
        pytest.importorskip("pandas")

    def _compute(self, participants, n_ratings=2):
        from metasignal.sdtbayes.beta_auc import _compute_auc2_per_participant
        return _compute_auc2_per_participant(participants, n_ratings)

    def test_columns_present(self, simple_participant):
        df = self._compute([simple_participant])
        assert "participant" in df.columns
        assert "auc2" in df.columns

    def test_row_count(self, two_participants):
        df = self._compute(two_participants)
        assert len(df) == 2

    def test_auc2_in_unit_interval(self, simple_participant):
        df = self._compute([simple_participant])
        val = df["auc2"].iloc[0]
        assert 0.0 <= val <= 1.0

    def test_degenerate_participant_returns_nan(self):
        stim = np.array([0, 0, 0, 0])
        resp = np.array([0, 0, 0, 0])
        conf = np.array([1, 1, 2, 2])
        pytest.importorskip("pandas")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = self._compute([(stim, resp, conf)])
        assert df["auc2"].isna().all()


# ---------------------------------------------------------------------------
# beta_auc._clip_auc2
# ---------------------------------------------------------------------------

class TestClipAuc2:
    @pytest.fixture(autouse=True)
    def require_pandas(self):
        pytest.importorskip("pandas")

    def _clip(self, values, eps=1e-4):
        import pandas as pd
        from metasignal.sdtbayes.beta_auc import _clip_auc2
        df = pd.DataFrame({"auc2": values})
        return _clip_auc2(df, eps=eps)

    def test_clips_zero(self):
        df = self._clip([0.0, 0.5, 1.0])
        assert df["auc2"].iloc[0] > 0.0

    def test_clips_one(self):
        df = self._clip([0.0, 0.5, 1.0])
        assert df["auc2"].iloc[2] < 1.0

    def test_does_not_alter_interior(self):
        df = self._clip([0.3, 0.5, 0.7])
        np.testing.assert_allclose(df["auc2"].values, [0.3, 0.5, 0.7])

    def test_does_not_mutate_input(self):
        import pandas as pd
        original = pd.DataFrame({"auc2": [0.0, 0.5]})
        from metasignal.sdtbayes.beta_auc import _clip_auc2
        _clip_auc2(original)
        assert original["auc2"].iloc[0] == 0.0

    def test_respects_custom_eps(self):
        df = self._clip([0.0, 1.0], eps=0.01)
        assert df["auc2"].iloc[0] == pytest.approx(0.01)
        assert df["auc2"].iloc[1] == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# statespace._mle_matrix
# ---------------------------------------------------------------------------

class TestMleMatrix:
    def _make_sessions(self, n_subj, n_sess, rng, n_trials=100):
        sessions = []
        for _ in range(n_sess):
            sess = []
            for _ in range(n_subj):
                stim = rng.integers(0, 2, n_trials)
                resp = stim.copy()
                flip = rng.random(n_trials) < 0.1
                resp[flip] = 1 - resp[flip]
                conf = rng.integers(1, 3, n_trials)
                sess.append((stim, resp, conf))
            sessions.append(sess)
        return sessions

    def _mle(self, sessions, n_ratings=2):
        from metasignal.sdtbayes.statespace import _mle_matrix
        return _mle_matrix(sessions, n_ratings)

    def test_output_shapes(self, rng):
        sessions = self._make_sessions(3, 4, rng)
        log_mr, valid = self._mle(sessions)
        assert log_mr.shape == (3, 4)
        assert valid.shape == (3, 4)

    def test_valid_values_binary(self, rng):
        sessions = self._make_sessions(3, 2, rng)
        _, valid = self._mle(sessions)
        assert set(valid.flatten().tolist()).issubset({0.0, 1.0})

    def test_consistent_participant_count_required(self, rng):
        sessions = self._make_sessions(3, 2, rng)
        bad = sessions[0][:2]  # only 2 participants in session 1
        with pytest.raises(ValueError, match="same number of participants"):
            from metasignal.sdtbayes.statespace import _mle_matrix
            _mle_matrix([sessions[0], bad], n_ratings=2)

    def test_log_mr_zero_where_invalid(self, rng):
        sessions = self._make_sessions(3, 2, rng)
        log_mr, valid = self._mle(sessions)
        invalid_mask = valid == 0.0
        assert np.all(log_mr[invalid_mask] == 0.0)

    def test_good_participants_mostly_valid(self, rng):
        sessions = self._make_sessions(5, 3, rng, n_trials=200)
        _, valid = self._mle(sessions)
        # High-accuracy participants with many trials should mostly converge
        assert valid.sum() >= 5


# ---------------------------------------------------------------------------
# meta_regression._build_regression_stan_blocks
# ---------------------------------------------------------------------------

class TestBuildRegressionStanBlocks:
    def _build(self):
        from metasignal.sdtbayes.meta_regression import _build_regression_stan_blocks
        return _build_regression_stan_blocks()

    def test_returns_four_strings(self):
        result = self._build()
        assert len(result) == 4
        assert all(isinstance(s, str) for s in result)

    def test_data_block_contains_covariate_matrix(self):
        data_block, _, _, _ = self._build()
        assert "X_cov" in data_block
        assert "p_cov" in data_block

    def test_parameters_block_contains_regression_slope(self):
        _, params_block, _, _ = self._build()
        assert "beta_logMratio" in params_block
        assert "alpha_logMratio" in params_block

    def test_tpar_block_contains_dot_product(self):
        _, _, tpar_block, _ = self._build()
        assert "dot_product" in tpar_block

    def test_model_block_contains_priors(self):
        _, _, _, model_block = self._build()
        assert "alpha_logMratio" in model_block
        assert "beta_logMratio" in model_block


# ---------------------------------------------------------------------------
# within_subject._compute_paired_estimates
# ---------------------------------------------------------------------------

class TestComputePairedEstimates:
    @pytest.fixture(autouse=True)
    def require_pandas(self):
        pytest.importorskip("pandas")

    def _compute(self, cond_a, cond_b, n_ratings=2):
        from metasignal.sdtbayes.within_subject import _compute_paired_estimates
        return _compute_paired_estimates(cond_a, cond_b, n_ratings)

    def test_columns_present(self, simple_participant):
        df = self._compute([simple_participant], [simple_participant])
        for col in ("participant", "condition", "dprime", "c", "meta_da",
                    "da", "m_ratio", "log_m_ratio"):
            assert col in df.columns

    def test_row_count_two_conditions(self, simple_participant):
        df = self._compute([simple_participant], [simple_participant])
        assert len(df) == 2  # 1 participant × 2 conditions

    def test_row_count_multiple_participants(self, two_participants):
        df = self._compute(two_participants, two_participants)
        assert len(df) == 4  # 2 participants × 2 conditions

    def test_condition_labels(self, simple_participant):
        df = self._compute([simple_participant], [simple_participant])
        assert set(df["condition"].unique().tolist()) == {0, 1}

    def test_participant_ids_correct(self, two_participants):
        df = self._compute(two_participants, two_participants)
        assert sorted(df["participant"].unique().tolist()) == [0, 1]

    def test_log_m_ratio_finite_for_good_data(self, simple_participant):
        df = self._compute([simple_participant], [simple_participant])
        assert df["log_m_ratio"].notna().any()


# ---------------------------------------------------------------------------
# within_subject.fit_within_subject_comparison — input validation
# ---------------------------------------------------------------------------

class TestFitWithinSubjectComparisonValidation:
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

    def test_raises_on_mismatched_lengths(self, rng):
        pytest.importorskip("pandas")
        from metasignal.sdtbayes.within_subject import fit_within_subject_comparison
        from unittest.mock import MagicMock, patch
        mock_brms = MagicMock()
        with patch.dict("sys.modules", {"brmspy": mock_brms}):
            cond_a = self._make_group(5, rng)
            cond_b = self._make_group(3, rng)
            with pytest.raises(ValueError, match="same"):
                fit_within_subject_comparison(cond_a, cond_b, n_ratings=2)

    def test_passes_with_equal_lengths(self, rng):
        pytest.importorskip("pandas")
        from metasignal.sdtbayes.within_subject import fit_within_subject_comparison
        from unittest.mock import MagicMock, patch
        mock_brmspy = MagicMock()
        mock_brms = mock_brmspy.brms
        with patch.dict("sys.modules", {"brmspy": mock_brmspy}):
            cond_a = self._make_group(5, rng)
            cond_b = self._make_group(5, rng)
            fit_within_subject_comparison(cond_a, cond_b, n_ratings=2)
        mock_brms.brm.assert_called_once()


# ---------------------------------------------------------------------------
# mixture.fit_mixture_group — input validation (brmspy mocked)
# ---------------------------------------------------------------------------

class TestFitMixtureGroupValidation:
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

    def test_raises_when_too_few_participants(self, rng):
        pytest.importorskip("pandas")
        from metasignal.sdtbayes.mixture import fit_mixture_group
        from unittest.mock import MagicMock, patch
        mock_brmspy = MagicMock()
        import warnings
        with patch.dict("sys.modules", {"brmspy": mock_brmspy}):
            # 2-component model needs at least 6 valid participants;
            # use degenerate data so MLE returns NaN for all
            degenerate = [(np.zeros(20, int), np.zeros(20, int),
                           np.ones(20, int))] * 2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(ValueError, match="valid estimates"):
                    fit_mixture_group(degenerate, n_ratings=2, n_components=2)
