"""Golden-value regression tests against the canonical Fleming (2017) /
metadpy tutorial-3 dataset — the same reference dataset and expected values
documented in metasignal.sdtbayes.subject_level.fit_subject_level's
docstring (metadpy's hmetad(): d1 ~= 1.53 +/- 0.14, meta_d ~= 1.57 +/- 0.20).

Unlike the rest of the test suite (which only checks sign/range/finiteness),
these assert stdpy's independent MLE implementation reproduces the published
reference numbers within a tolerance wide enough to allow for the expected
MLE-vs-Bayesian-posterior-mean discrepancy, but tight enough to catch a real
regression (sign flip, scaling error, off-by-one count-layout bug).
"""

import numpy as np

from metasignal.stdpy.core import compute_sdt_resp, trials_to_counts
from metasignal.stdpy.metad import fit_meta_d_mle
from metasignal.stdpy.type2 import compute_gamma, compute_phi, compute_type2_auc

# Fleming (2017) / metadpy tutorial-3 canonical dataset — also used in
# tests/test_sdtbayes.py as CANONICAL_NR_S1 / CANONICAL_NR_S2.
NR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2], dtype=float)
NR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45], dtype=float)


def test_fit_meta_d_mle_matches_metadpy_reference():
    result = fit_meta_d_mle(NR_S1, NR_S2)
    # metadpy/HMeta-d reference (see subject_level.py docstring): d1 ~= 1.53
    assert abs(result["da"] - 1.53) < 0.3
    # metadpy/HMeta-d reference: meta_d ~= 1.57
    assert abs(result["meta_da"] - 1.57) < 0.4
    assert 0.7 < result["M_ratio"] < 1.3
    assert result["success"]


def test_compute_sdt_resp_dprime_matches_reference():
    """d' recomputed directly from trial-level type-1 hit/FA rates should
    agree with the meta-d' model's own type-1 d' (da) on the same data."""
    stim, resp, conf = [], [], []
    n_ratings = 4
    for idx in range(n_ratings):
        c = n_ratings - idx
        stim += [0] * int(NR_S1[idx]); resp += [0] * int(NR_S1[idx])
        conf += [c] * int(NR_S1[idx])
        stim += [1] * int(NR_S2[idx]); resp += [0] * int(NR_S2[idx])
        conf += [c] * int(NR_S2[idx])
    for idx in range(n_ratings):
        c = idx + 1
        j = n_ratings + idx
        stim += [0] * int(NR_S1[j]); resp += [1] * int(NR_S1[j])
        conf += [c] * int(NR_S1[j])
        stim += [1] * int(NR_S2[j]); resp += [1] * int(NR_S2[j])
        conf += [c] * int(NR_S2[j])

    stim, resp, conf = np.array(stim), np.array(resp), np.array(conf)
    dprime, _, _ = compute_sdt_resp(stim, resp)
    assert abs(dprime - 1.53) < 0.3

    # Round-trip through trials_to_counts should reproduce the same counts.
    nr_s1, nr_s2 = trials_to_counts(stim, resp, conf, n_ratings)
    assert np.allclose(nr_s1, NR_S1)
    assert np.allclose(nr_s2, NR_S2)


def test_type2_measures_reasonable_for_canonical_dataset():
    """Sanity-anchor the Type-2 descriptive measures against the same
    well-behaved canonical dataset (a good, well-calibrated observer)."""
    auc2 = compute_type2_auc(NR_S1, NR_S2)
    gamma = compute_gamma(NR_S1, NR_S2)
    phi = compute_phi(NR_S1, NR_S2)

    assert 0.65 < auc2 < 0.85
    assert 0.5 < gamma < 0.75
    assert 0.25 < phi < 0.5
