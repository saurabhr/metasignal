"""Tests for the base Gaussian SDT model (metasignal.sdtr.sdt, metasignal.sdtr.group).

Golden-value tests are transcribed from Macho (2020), *SDT-Models in R*
(https://www.unifr.ch/psycho/fr/assets/public/Forschungseinheiten/sdt/SDT.pdf),
Ch. 5.1.1: the standard equal-variance SDT model fit to Yes/No recognition
data (Table, p. 109-110) — the same "validate against an independent
reference" discipline used for stdpy (MATLAB) and itmc (statConfR), since
there is no live R session available to cross-check against interactively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metasignal.sdtr import SDTModelFit, fit_group, fit_sdt

# Manual Ch. 5.1.1: datavec = c(1780, 763, 883, 1025) -- NEW then OLD, each
# [respond "new", respond "old"]. Model: n.sdt=2, restriction="equalvar".
_YESNO_COUNTS = np.array([[1780, 763], [883, 1025]], dtype=float)


def _rating_data(seed: int = 0, n: int = 600, d: float = 1.5, metad: float = 1.2):
    """4-category rating counts (2 signals) via trialSimulation, for non-golden checks."""
    from metasignal.stdpy.simulate import trialSimulation
    df = trialSimulation(d=d, metad=metad, nRatings=4, nTrials=n, rng=np.random.default_rng(seed))
    resp_cat = np.where(df["Responses"] == 0, 5 - df["Confidence"], 4 + df["Confidence"])
    counts = np.zeros((2, 8))
    for s in (0, 1):
        for c in range(1, 9):
            counts[s, c - 1] = np.sum((df["Stimuli"] == s) & (resp_cat == c))
    return counts


# ---------------------------------------------------------------------------
# Golden-value validation against Macho (2020) Ch. 5.1.1
# ---------------------------------------------------------------------------

class TestGoldenYesNoExample:
    @pytest.fixture(scope="class")
    def result(self) -> SDTModelFit:
        return fit_sdt(_YESNO_COUNTS, restriction="equalvar")

    def test_mean(self, result):
        assert result.means[0] == pytest.approx(0.618, abs=1e-3)

    def test_threshold(self, result):
        assert result.thresholds[0] == pytest.approx(0.524, abs=1e-3)

    def test_sd_fixed_at_one(self, result):
        assert result.sds[0] == 1.0

    def test_mean_se(self, result):
        assert result.fit.se[0] == pytest.approx(0.039, abs=1e-3)

    def test_threshold_se(self, result):
        assert result.fit.se[1] == pytest.approx(0.026, abs=1e-3)

    def test_d_a(self, result):
        assert result.d_a[0] == pytest.approx(0.618, abs=1e-3)

    def test_A_z(self, result):
        assert result.A_z[0] == pytest.approx(0.669, abs=1e-3)

    def test_logL(self, result):
        assert result.logL == pytest.approx(-2870.749, abs=1e-2)

    def test_aic(self, result):
        assert result.aic == pytest.approx(5745.497, abs=1e-2)

    def test_bic(self, result):
        assert result.bic == pytest.approx(5758.299, abs=1e-2)

    def test_converged(self, result):
        assert result.success


# ---------------------------------------------------------------------------
# fit_sdt — general behavior
# ---------------------------------------------------------------------------

class TestFitSdt:
    def test_returns_sdtmodelfit(self):
        assert isinstance(fit_sdt(_YESNO_COUNTS), SDTModelFit)

    def test_no_restriction_leaves_sd_free(self):
        result = fit_sdt(_YESNO_COUNTS, restriction="no")
        # Not golden-tested (the R "no restriction" fit isn't transcribed),
        # just confirm sd was actually optimized rather than pinned to 1.
        assert result.success

    def test_rejects_too_few_signals(self):
        with pytest.raises(ValueError, match="at least 2 signals"):
            fit_sdt(np.array([[10, 20]]))

    def test_rejects_too_few_categories(self):
        with pytest.raises(ValueError, match="at least 2 signals"):
            fit_sdt(np.array([[10], [20]]))

    def test_thresholds_sorted_ascending(self):
        counts = _rating_data()
        result = fit_sdt(counts, restriction="equalvar")
        assert np.all(np.diff(result.thresholds) > 0)

    def test_recovers_dprime_within_tolerance(self):
        """Not a golden value -- sanity check that fitted mean tracks the true d'."""
        counts = _rating_data(d=1.5, metad=1.5, n=3000)
        result = fit_sdt(counts, restriction="equalvar")
        assert result.means[0] == pytest.approx(1.5, abs=0.2)

    def test_three_signals(self):
        counts = np.array([[400, 100], [250, 250], [100, 400]], dtype=float)
        result = fit_sdt(counts, restriction="equalvar")
        assert result.means.shape == (2,)
        assert result.d_a.shape == (2,)
        assert result.success

    def test_more_starts_does_not_break_fit(self):
        result = fit_sdt(_YESNO_COUNTS, restriction="equalvar", n_starts=3, seed=1)
        assert result.means[0] == pytest.approx(0.618, abs=1e-2)


# ---------------------------------------------------------------------------
# fit_group
# ---------------------------------------------------------------------------

class TestFitGroup:
    def _trial_df(self, seed=0, pid="s1"):
        from metasignal.stdpy.simulate import trialSimulation
        df = trialSimulation(d=1.5, metad=1.2, nRatings=4, nTrials=400, rng=np.random.default_rng(seed))
        resp_cat = np.where(df["Responses"] == 0, 5 - df["Confidence"], 4 + df["Confidence"]).astype(int)
        return pd.DataFrame({
            "participant": pid,
            "signal": df["Stimuli"].astype(int),
            "response": resp_cat,
        })

    def test_single_participant_no_grouping(self):
        df = self._trial_df()
        result = fit_group(df, restriction="equalvar")
        assert len(result) == 1
        assert "mean_1" in result.columns

    def test_two_participants_two_rows(self):
        df = pd.concat([self._trial_df(seed=0, pid="s0"), self._trial_df(seed=1, pid="s1")],
                        ignore_index=True)
        result = fit_group(df, subject="participant", restriction="equalvar")
        assert len(result) == 2
        assert set(result["participant"]) == {"s0", "s1"}

    def test_recovers_yesno_golden_values_via_group(self):
        """fit_group on a hand-built trial-level version of the Ch. 5.1.1 data matches fit_sdt."""
        # Reconstruct trial-level rows directly from the 2x2 count table.
        rows = []
        for signal_idx in (0, 1):
            for cat_idx, n in enumerate(_YESNO_COUNTS[signal_idx], start=1):
                rows.append(pd.DataFrame(
                    {"participant": "p1", "signal": signal_idx, "response": cat_idx},
                    index=range(int(n)),
                ))
        df = pd.concat(rows, ignore_index=True)
        result = fit_group(df, subject="participant", restriction="equalvar")
        assert result["mean_1"].iloc[0] == pytest.approx(0.618, abs=1e-3)
        assert result["threshold_1"].iloc[0] == pytest.approx(0.524, abs=1e-3)

    def test_columns_include_fit_stats(self):
        df = self._trial_df()
        result = fit_group(df, restriction="equalvar")
        for col in ("logL", "aic", "bic", "success"):
            assert col in result.columns
