"""Tests for metasignal.stdpy.group.fit_group group wrapper."""

import numpy as np
import pandas as pd
import pytest

from metasignal.stdpy import (
    fit_group, MEASURE_COLS,
    trialSimulation, responseSimulation, pairedResponseSimulation,
)

TYPE1_COLS = [
    "hit_rate", "fa_rate", "dprime", "criterion", "ln_beta",
    "n_trials", "n_hits", "n_misses", "n_fa", "n_cr",
]
TYPE2_COLS = [
    "meta_d", "AUC2", "Gamma", "Phi", "DeltaConf",
    "M_ratio", "AUC2_ratio", "Gamma_ratio", "Phi_ratio", "DeltaConf_ratio",
    "M_diff", "AUC2_diff", "Gamma_diff", "Phi_diff", "DeltaConf_diff",
    "MetaNoise", "MetaUncertainty", "mean_conf",
    "logL", "AIC", "BIC", "AICc", "k", "n",
]
ALL_MEASURE_COLS = TYPE1_COLS + TYPE2_COLS  # 34 total


# ---------------------------------------------------------------------------
# 1. No grouping — single cell
# ---------------------------------------------------------------------------

def test_single_cell_shape():
    df = trialSimulation(d=1.5, metad=1.5, nTrials=400, rng=np.random.default_rng(0))
    out = fit_group(df, nRatings=4)
    assert out.shape == (1, 34), f"Expected (1,34), got {out.shape}"
    assert list(out.columns) == ALL_MEASURE_COLS


def test_single_cell_dprime_positive():
    df = trialSimulation(d=2.0, metad=2.0, nTrials=500, rng=np.random.default_rng(1))
    out = fit_group(df, nRatings=4)
    assert out["dprime"].iloc[0] > 0


def test_single_cell_mratio_near_one():
    """With metad = d, M_ratio should be close to 1."""
    df = trialSimulation(d=1.5, metad=1.5, nTrials=2000, rng=np.random.default_rng(2))
    out = fit_group(df, nRatings=4)
    mr = out["M_ratio"].iloc[0]
    assert 0.7 < mr < 1.4, f"M_ratio={mr:.3f} far from 1 (unexpected)"


def test_single_cell_type1_measures():
    """Type-1 measures should be present and sensible."""
    df = trialSimulation(d=1.5, metad=1.5, nTrials=500, rng=np.random.default_rng(3))
    out = fit_group(df, nRatings=4)
    assert 0 < out["hit_rate"].iloc[0] < 1
    assert 0 < out["fa_rate"].iloc[0] < 1
    assert out["hit_rate"].iloc[0] > out["fa_rate"].iloc[0]
    assert out["n_trials"].iloc[0] > 0
    assert out["n_hits"].iloc[0] + out["n_misses"].iloc[0] == pytest.approx(out["n_trials"].iloc[0] / 2, abs=5)


# ---------------------------------------------------------------------------
# 2. Subject grouping — no conditions
# ---------------------------------------------------------------------------

def test_subject_grouping_shape():
    df = responseSimulation(d=1.5, metad=1.5, nSubjects=8, nTrials=300,
                            rng=np.random.default_rng(4))
    out = fit_group(df, subject="Subject", nRatings=4)
    assert out.shape[0] == 8
    assert "Subject" in out.columns
    assert set(out.columns[1:]) == set(ALL_MEASURE_COLS)


def test_subject_grouping_subject_ids():
    df = responseSimulation(d=1.5, metad=1.5, nSubjects=5, nTrials=200,
                            rng=np.random.default_rng(5))
    out = fit_group(df, subject="Subject", nRatings=4)
    assert sorted(out["Subject"].tolist()) == list(range(5))


# ---------------------------------------------------------------------------
# 3. Within-subjects factor
# ---------------------------------------------------------------------------

def test_within_subjects_shape():
    df = pairedResponseSimulation(nSubjects=6, nTrials=200, rng=np.random.default_rng(6))
    out = fit_group(df, subject="Subject", within="Condition", nRatings=4)
    assert out.shape[0] == 12, f"Expected 12, got {out.shape[0]}"
    assert "Subject" in out.columns
    assert "Condition" in out.columns


def test_within_mratio_differs_across_conditions():
    df = pairedResponseSimulation(nSubjects=20, nTrials=300,
                                  mRatio=[1.0, 0.6], mRatio_sigma=0.05,
                                  rng=np.random.default_rng(7))
    out = fit_group(df, subject="Subject", within="Condition", nRatings=4)
    mr0 = out[out["Condition"] == 0]["M_ratio"].mean()
    mr1 = out[out["Condition"] == 1]["M_ratio"].mean()
    assert mr0 > mr1, f"Expected cond0 M_ratio ({mr0:.3f}) > cond1 ({mr1:.3f})"


# ---------------------------------------------------------------------------
# 4. Between-subjects factor
# ---------------------------------------------------------------------------

def test_between_subjects_factor():
    rng = np.random.default_rng(8)
    frames = []
    for grp, d_val in [(0, 1.0), (1, 2.0)]:
        sub_df = responseSimulation(d=d_val, metad=d_val, nSubjects=5, nTrials=200, rng=rng)
        sub_df["Subject"] += grp * 5
        sub_df["Group"] = grp
        frames.append(sub_df)
    df = pd.concat(frames, ignore_index=True)

    out = fit_group(df, subject="Subject", between="Group", nRatings=4)
    assert out.shape[0] == 10
    assert "Group" in out.columns
    dprime_grp0 = out[out["Group"] == 0]["dprime"].mean()
    dprime_grp1 = out[out["Group"] == 1]["dprime"].mean()
    assert dprime_grp1 > dprime_grp0


# ---------------------------------------------------------------------------
# 5. Within + between combined
# ---------------------------------------------------------------------------

def test_within_and_between_combined():
    rng = np.random.default_rng(9)
    frames = []
    for grp, mr in [(0, [1.0, 0.8]), (1, [0.7, 0.5])]:
        sub_df = pairedResponseSimulation(nSubjects=4, nTrials=150, mRatio=mr, rng=rng)
        sub_df["Subject"] += grp * 4
        sub_df["Group"] = grp
        frames.append(sub_df)
    df = pd.concat(frames, ignore_index=True)

    out = fit_group(df, subject="Subject", within="Condition", between="Group", nRatings=4)
    assert out.shape[0] == 16
    assert {"Subject", "Condition", "Group"}.issubset(out.columns)


# ---------------------------------------------------------------------------
# 6. measures= selection
# ---------------------------------------------------------------------------

def test_measures_subset_type1_only():
    df = trialSimulation(d=1.5, metad=1.5, nTrials=300, rng=np.random.default_rng(10))
    out = fit_group(df, nRatings=4, measures=["dprime", "hit_rate", "fa_rate"])
    # canonical ordering is preserved regardless of request order
    assert list(out.columns) == ["hit_rate", "fa_rate", "dprime"]


def test_measures_subset_metadpy_style():
    df = trialSimulation(d=1.5, metad=1.5, nTrials=300, rng=np.random.default_rng(11))
    out = fit_group(df, nRatings=4, measures=["dprime", "meta_d", "M_ratio", "M_diff"])
    assert list(out.columns) == ["dprime", "meta_d", "M_ratio", "M_diff"]


def test_measures_single_string():
    df = trialSimulation(d=1.5, metad=1.5, nTrials=300, rng=np.random.default_rng(12))
    out = fit_group(df, nRatings=4, measures="dprime")
    assert list(out.columns) == ["dprime"]


def test_measures_invalid_raises():
    df = trialSimulation(d=1.5, metad=1.5, nTrials=300, rng=np.random.default_rng(13))
    with pytest.raises(ValueError, match="Unknown measure"):
        fit_group(df, nRatings=4, measures="not_a_measure")


# ---------------------------------------------------------------------------
# 7. All measure columns present and MEASURE_COLS consistent
# ---------------------------------------------------------------------------

def test_all_measure_columns_present():
    df = trialSimulation(d=1.5, metad=1.5, nTrials=400, rng=np.random.default_rng(14))
    out = fit_group(df, nRatings=4)
    for col in ALL_MEASURE_COLS:
        assert col in out.columns, f"Missing column: {col}"


def test_measure_cols_export_matches():
    """MEASURE_COLS exported from __init__ should equal the 34-column list."""
    assert MEASURE_COLS == ALL_MEASURE_COLS


# ---------------------------------------------------------------------------
# 8. Custom column names
# ---------------------------------------------------------------------------

def test_custom_column_names():
    df = trialSimulation(d=1.5, metad=1.5, nTrials=300, rng=np.random.default_rng(15))
    df = df.rename(columns={"Stimuli": "stim", "Responses": "resp", "Confidence": "conf"})
    out = fit_group(df, stimuli="stim", responses="resp", confidence="conf", nRatings=4)
    assert out.shape == (1, 34)
