"""Tests for metasignal.stdpy.group.metad group wrapper."""

import numpy as np
import pandas as pd
import pytest

from metasignal.stdpy import metad, trialSimulation, responseSimulation, pairedResponseSimulation

RNG = np.random.default_rng(42)

MEASURE_COLS = [
    "meta_d", "AUC2", "Gamma", "Phi", "DeltaConf",
    "M_ratio", "AUC2_ratio", "Gamma_ratio", "Phi_ratio", "DeltaConf_ratio",
    "M_diff", "AUC2_diff", "Gamma_diff", "Phi_diff", "DeltaConf_diff",
    "MetaNoise", "MetaUncertainty", "dprime", "criterion", "mean_conf",
]


# ---------------------------------------------------------------------------
# 1. No grouping — single cell
# ---------------------------------------------------------------------------

def test_single_cell_shape():
    df = trialSimulation(d=1.5, metad=1.5, nTrials=400, rng=np.random.default_rng(0))
    out = metad(df, nRatings=4)
    assert out.shape == (1, 20), f"Expected (1,20), got {out.shape}"
    assert list(out.columns) == MEASURE_COLS


def test_single_cell_dprime_positive():
    df = trialSimulation(d=2.0, metad=2.0, nTrials=500, rng=np.random.default_rng(1))
    out = metad(df, nRatings=4)
    assert out["dprime"].iloc[0] > 0


def test_single_cell_mratio_near_one():
    """With metad = d, M_ratio should be close to 1."""
    df = trialSimulation(d=1.5, metad=1.5, nTrials=2000, rng=np.random.default_rng(2))
    out = metad(df, nRatings=4)
    mr = out["M_ratio"].iloc[0]
    assert 0.7 < mr < 1.4, f"M_ratio={mr:.3f} far from 1 (unexpected)"


# ---------------------------------------------------------------------------
# 2. Subject grouping — no conditions
# ---------------------------------------------------------------------------

def test_subject_grouping_shape():
    df = responseSimulation(d=1.5, metad=1.5, nSubjects=8, nTrials=300,
                            rng=np.random.default_rng(3))
    out = metad(df, subject="Subject", nRatings=4)
    assert out.shape[0] == 8, f"Expected 8 rows, got {out.shape[0]}"
    assert "Subject" in out.columns
    assert set(out.columns[1:]) == set(MEASURE_COLS)


def test_subject_grouping_subject_ids():
    df = responseSimulation(d=1.5, metad=1.5, nSubjects=5, nTrials=200,
                            rng=np.random.default_rng(4))
    out = metad(df, subject="Subject", nRatings=4)
    assert sorted(out["Subject"].tolist()) == list(range(5))


# ---------------------------------------------------------------------------
# 3. Within-subjects factor
# ---------------------------------------------------------------------------

def test_within_subjects_shape():
    df = pairedResponseSimulation(nSubjects=6, nTrials=200, rng=np.random.default_rng(5))
    out = metad(df, subject="Subject", within="Condition", nRatings=4)
    # 6 subjects × 2 conditions = 12 rows
    assert out.shape[0] == 12, f"Expected 12, got {out.shape[0]}"
    assert "Subject" in out.columns
    assert "Condition" in out.columns


def test_within_mratio_differs_across_conditions():
    """Condition 0 has mRatio=1.0, Condition 1 has mRatio=0.6 — group means should differ."""
    df = pairedResponseSimulation(nSubjects=20, nTrials=300,
                                  mRatio=[1.0, 0.6], mRatio_sigma=0.05,
                                  rng=np.random.default_rng(6))
    out = metad(df, subject="Subject", within="Condition", nRatings=4)
    mr0 = out[out["Condition"] == 0]["M_ratio"].mean()
    mr1 = out[out["Condition"] == 1]["M_ratio"].mean()
    assert mr0 > mr1, f"Expected cond0 M_ratio ({mr0:.3f}) > cond1 ({mr1:.3f})"


# ---------------------------------------------------------------------------
# 4. Between-subjects factor
# ---------------------------------------------------------------------------

def test_between_subjects_factor():
    """Add a Group column and pass it as between."""
    rng = np.random.default_rng(7)
    frames = []
    for grp, d_val in [(0, 1.0), (1, 2.0)]:
        sub_df = responseSimulation(d=d_val, metad=d_val, nSubjects=5, nTrials=200, rng=rng)
        sub_df["Subject"] += grp * 5   # unique subject ids
        sub_df["Group"] = grp
        frames.append(sub_df)
    df = pd.concat(frames, ignore_index=True)

    out = metad(df, subject="Subject", between="Group", nRatings=4)
    assert out.shape[0] == 10, f"Expected 10 rows, got {out.shape[0]}"
    assert "Group" in out.columns

    dprime_grp0 = out[out["Group"] == 0]["dprime"].mean()
    dprime_grp1 = out[out["Group"] == 1]["dprime"].mean()
    assert dprime_grp1 > dprime_grp0, (
        f"Group1 d'={dprime_grp1:.3f} should exceed Group0 d'={dprime_grp0:.3f}"
    )


# ---------------------------------------------------------------------------
# 5. Within + between combined
# ---------------------------------------------------------------------------

def test_within_and_between_combined():
    rng = np.random.default_rng(8)
    frames = []
    for grp, mr in [(0, [1.0, 0.8]), (1, [0.7, 0.5])]:
        sub_df = pairedResponseSimulation(nSubjects=4, nTrials=150, mRatio=mr, rng=rng)
        sub_df["Subject"] += grp * 4
        sub_df["Group"] = grp
        frames.append(sub_df)
    df = pd.concat(frames, ignore_index=True)

    out = metad(df, subject="Subject", within="Condition", between="Group", nRatings=4)
    # 8 subjects × 2 conditions = 16 rows
    assert out.shape[0] == 16
    assert {"Subject", "Condition", "Group"}.issubset(out.columns)


# ---------------------------------------------------------------------------
# 6. Output columns always complete
# ---------------------------------------------------------------------------

def test_all_measure_columns_present():
    df = trialSimulation(d=1.5, metad=1.5, nTrials=400, rng=np.random.default_rng(9))
    out = metad(df, nRatings=4)
    for col in MEASURE_COLS:
        assert col in out.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# 7. Custom column names
# ---------------------------------------------------------------------------

def test_custom_column_names():
    df = trialSimulation(d=1.5, metad=1.5, nTrials=300, rng=np.random.default_rng(10))
    df = df.rename(columns={
        "Stimuli": "stim", "Responses": "resp", "Confidence": "conf"
    })
    out = metad(df, stimuli="stim", responses="resp", confidence="conf", nRatings=4)
    assert out.shape == (1, 20)
