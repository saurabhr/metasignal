"""Tests for FitResult.convergence_diagnostics().

Regression test for a real bug found by tests/test_sdtbayes_recovery.py:
az.summary()'s default round_to="auto" formats numeric columns (including
r_hat) as display strings in this arviz version, which broke the numeric
comparisons in convergence_diagnostics(). Mocked here so it's caught fast,
without needing a real MCMC fit.
"""

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from metasignal.sdtbayes.diagnostics import FitResult


def _mock_arviz_with_string_summary():
    """Mimic az.summary(round_to="auto")'s display-string columns — the
    exact shape that broke convergence_diagnostics() before the fix."""
    mock_az = MagicMock()

    def _summary(idata, round_to="auto"):
        if round_to == "none":
            return pd.DataFrame({
                "r_hat": [1.001, 1.02],
                "ess_bulk": [800.0, 300.0],
                "ess_tail": [750.0, 350.0],
            }, index=["d1", "meta_d"])
        # "auto" (display) formatting: everything as strings, as the real
        # arviz version does by default.
        return pd.DataFrame({
            "r_hat": ["1.00", "1.02"],
            "ess_bulk": [800, 300],
            "ess_tail": [750, 350],
        }, index=["d1", "meta_d"])

    mock_az.summary.side_effect = _summary
    return mock_az


def test_convergence_diagnostics_returns_numeric_and_bool_columns():
    with patch.dict(sys.modules, {"arviz": _mock_arviz_with_string_summary()}):
        diag = FitResult(idata=None, r=None).convergence_diagnostics()

    assert pd.api.types.is_float_dtype(diag["r_hat"])
    assert pd.api.types.is_float_dtype(diag["ess_bulk"])
    assert pd.api.types.is_bool_dtype(diag["converged"])


def test_convergence_diagnostics_flags_bad_parameter():
    with patch.dict(sys.modules, {"arviz": _mock_arviz_with_string_summary()}):
        diag = FitResult(idata=None, r=None).convergence_diagnostics()

    assert diag.loc["d1", "converged"]
    assert not diag.loc["meta_d", "converged"]  # r_hat=1.02 > 1.01, ess_bulk=300 < 400


def test_convergence_diagnostics_requires_arviz():
    with patch.dict(sys.modules, {"arviz": None}):
        with pytest.raises(ImportError, match="arviz is not installed"):
            FitResult(idata=None, r=None).convergence_diagnostics()
