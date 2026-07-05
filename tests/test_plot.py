"""Smoke tests for stdpy.plot — no assertions on pixel content, only that
each function runs headlessly and returns the documented object types."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from metasignal.stdpy.plot import (
    plot_confidence,
    plot_forest,
    plot_measures,
    plot_sanity_check,
    plot_type2roc,
)

NR_S1 = np.array([10.0, 8.0, 5.0, 2.0, 1.0, 2.0, 4.0, 8.0])
NR_S2 = np.array([8.0, 4.0, 2.0, 1.0, 2.0, 5.0, 8.0, 10.0])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plot_confidence_returns_axes():
    ax = plot_confidence(NR_S1, NR_S2)
    assert isinstance(ax, plt.Axes)


def test_plot_type2roc_returns_axes():
    ax = plot_type2roc(NR_S1, NR_S2)
    assert isinstance(ax, plt.Axes)


def test_plot_type2roc_zero_class_does_not_crash():
    # One class entirely empty — must not divide-by-zero into a crash.
    zeros = np.zeros_like(NR_S1)
    ax = plot_type2roc(zeros, NR_S2)
    assert isinstance(ax, plt.Axes)


def test_plot_sanity_check_returns_fig_and_axes():
    fig, axes = plot_sanity_check(NR_S1, NR_S2, meta_d=1.2)
    assert isinstance(fig, plt.Figure)
    assert axes.shape == (3,)


def _fake_group_df():
    return pd.DataFrame({
        "Subject": [1, 2, 3, 1, 2, 3],
        "Condition": ["A", "A", "A", "B", "B", "B"],
        "M_ratio": [0.9, 1.1, 1.0, 0.7, 0.8, 0.75],
    })


def test_plot_forest_returns_axes():
    ax = plot_forest(_fake_group_df(), measure="M_ratio", group_col="Condition")
    assert isinstance(ax, plt.Axes)


def test_plot_measures_returns_fig_and_axes():
    fig, axes = plot_measures(_fake_group_df(), measures="M_ratio", group_col="Condition")
    assert isinstance(fig, plt.Figure)
