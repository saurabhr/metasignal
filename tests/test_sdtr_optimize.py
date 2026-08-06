"""Tests for the shared MLE engine (metasignal.sdtr._optimize)."""

from __future__ import annotations

import numpy as np
import pytest

from metasignal.sdtr._optimize import SDTFitResult, expand_params, fit_mle


# ---------------------------------------------------------------------------
# expand_params
# ---------------------------------------------------------------------------

class TestExpandParams:
    def test_no_constraints_is_identity(self):
        free = np.array([1.0, 2.0, 3.0])
        full = expand_params(free, 3)
        assert np.allclose(full, free)

    def test_fixed_position_uses_fixed_value_not_free_vector(self):
        free = np.array([1.0, 3.0])
        full = expand_params(free, 3, fixed=[(1, 99.0)])
        assert full[1] == 99.0
        assert full[0] == 1.0
        assert full[2] == 3.0

    def test_ident_target_equals_source(self):
        free = np.array([1.0, 2.0])
        full = expand_params(free, 3, ident=[(0, 2)])
        assert full[2] == full[0] == 1.0
        assert full[1] == 2.0

    def test_fixed_and_ident_combined(self):
        free = np.array([5.0])
        full = expand_params(free, 4, fixed=[(1, -1.0)], ident=[(0, 2), (1, 3)])
        assert full[0] == 5.0
        assert full[1] == -1.0
        assert full[2] == 5.0   # ident source 0
        assert full[3] == -1.0  # ident source 1 (a fixed position)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="Expected"):
            expand_params(np.array([1.0]), 3)


# ---------------------------------------------------------------------------
# fit_mle
# ---------------------------------------------------------------------------

class TestFitMle:
    def test_recovers_known_minimum(self):
        """A simple quadratic NLL: minimum at [2, -3]."""
        def nll(p):
            return float(np.sum((p - np.array([2.0, -3.0])) ** 2))

        result = fit_mle(nll, start=np.array([0.0, 0.0]),
                          bounds=[(-10, 10), (-10, 10)], n_obs=100)
        assert isinstance(result, SDTFitResult)
        assert np.allclose(result.params, [2.0, -3.0], atol=1e-3)
        assert result.success

    def test_se_matches_analytic_normal_mean_variance(self):
        """NLL for a Gaussian-mean MLE: SE should match sigma / sqrt(n)."""
        sigma, n = 2.0, 500
        rng = np.random.default_rng(0)
        data = rng.normal(loc=7.0, scale=sigma, size=n)

        def nll(p):
            mu = p[0]
            return float(0.5 * np.sum((data - mu) ** 2) / sigma**2)

        result = fit_mle(nll, start=np.array([0.0]), bounds=[(-50, 50)], n_obs=n)
        assert result.params[0] == pytest.approx(data.mean(), abs=1e-2)
        expected_se = sigma / np.sqrt(n)
        assert result.se[0] == pytest.approx(expected_se, rel=0.05)

    def test_fixed_parameter_never_moves(self):
        """A fixed parameter stays at its pinned value regardless of the NLL's true optimum."""
        def nll(p):
            return float(np.sum((p - np.array([2.0, -3.0])) ** 2))

        result = fit_mle(nll, start=np.array([0.0, 0.0]),
                          bounds=[(-10, 10), (-10, 10)],
                          fixed=[(1, 0.0)], n_obs=10)
        assert result.full_params[1] == 0.0
        assert result.full_params[0] == pytest.approx(2.0, abs=1e-3)

    def test_ident_constraint_keeps_positions_equal(self):
        """An equality constraint forces two full-vector positions to match."""
        def nll(p):
            # True optimum wants p[0]=1, p[2]=5 -- but p[2] is ident-tied to p[0].
            return float((p[0] - 1.0) ** 2 + (p[1] - 4.0) ** 2 + (p[2] - 5.0) ** 2)

        result = fit_mle(nll, start=np.array([0.0, 0.0, 0.0]),
                          bounds=[(-10, 10)] * 3,
                          ident=[(0, 2)], n_obs=10)
        assert result.full_params[0] == result.full_params[2]
        assert result.full_params[1] == pytest.approx(4.0, abs=1e-3)

    def test_multi_start_no_worse_than_single_start(self):
        def nll(p):
            return float(np.sum((p - np.array([2.0, -3.0])) ** 2))

        single = fit_mle(nll, start=np.array([50.0, 50.0]),
                          bounds=[(-100, 100), (-100, 100)], n_obs=10, n_starts=1)
        multi = fit_mle(nll, start=np.array([50.0, 50.0]),
                         bounds=[(-100, 100), (-100, 100)], n_obs=10, n_starts=5, seed=1)
        assert multi.nll <= single.nll + 1e-6
