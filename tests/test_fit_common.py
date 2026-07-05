"""Tests for the shared sdtbayes brmspy-guard and Stage-1 MLE helper."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from metasignal.sdtbayes._fit_common import compute_mle_row, require_brms


def test_require_brms_raises_clear_import_error():
    with patch.dict(sys.modules, {"brmspy": None}):
        with pytest.raises(ImportError, match="brmspy is not installed"):
            require_brms()


def test_require_brms_returns_module_when_available():
    mock_brmspy = MagicMock()
    mock_brmspy.brms = "the-brms-module"
    with patch.dict(sys.modules, {"brmspy": mock_brmspy}):
        assert require_brms() == "the-brms-module"


def test_compute_mle_row_valid_participant():
    rng = np.random.default_rng(0)
    stim = rng.integers(0, 2, 200)
    resp = stim.copy()
    flip = rng.random(200) < 0.15
    resp[flip] = 1 - resp[flip]
    correct = stim == resp
    conf = np.where(rng.random(200) < np.where(correct, 0.8, 0.2), 4, 1)

    row = compute_mle_row(stim, resp, conf, n_ratings=4, label="Participant 0")
    for key in ("dprime", "c", "meta_da", "da", "m_ratio", "log_m_ratio"):
        assert key in row
    assert np.isfinite(row["meta_da"])
    if row["m_ratio"] > 0:
        assert abs(row["log_m_ratio"] - np.log(row["m_ratio"])) < 1e-10


def test_compute_mle_row_degenerate_participant_warns_and_nans():
    stim = np.array([0, 0, 0, 0])
    resp = np.array([0, 0, 0, 0])
    conf = np.array([1, 1, 2, 2])
    with pytest.warns(UserWarning, match="MLE failed"):
        row = compute_mle_row(stim, resp, conf, n_ratings=2, label="Participant 0")
    assert np.isnan(row["log_m_ratio"])
