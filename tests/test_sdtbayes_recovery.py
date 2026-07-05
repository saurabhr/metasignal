"""Real MCMC recovery test for fit_subject_level against the canonical
Fleming (2017) / metadpy tutorial-3 dataset — closes the gap flagged in
review: subject_level.py's docstring claims metadpy-reference agreement
(meta_d ~= 1.57 +/- 0.20, d1 ~= 1.53 +/- 0.14) but no test previously
exercised the actual Stan/cmdstanpy fit to verify it.

Requires cmdstanpy + a working CmdStan installation; skipped otherwise.
Marked slow since it runs real MCMC sampling (seconds to ~1 minute).
"""

import numpy as np
import pytest

cmdstanpy = pytest.importorskip("cmdstanpy")
arviz = pytest.importorskip("arviz")

try:
    cmdstanpy.cmdstan_path()
except ValueError:
    pytest.skip("CmdStan is not installed (cmdstanpy.install_cmdstan())", allow_module_level=True)

from metasignal.sdtbayes.subject_level import fit_subject_level  # noqa: E402

# Canonical dataset from Fleming (2017) / metadpy tutorial 3.
NR_S1 = np.array([52, 32, 35, 37, 26, 12, 4, 2])
NR_S2 = np.array([2, 5, 15, 22, 33, 38, 40, 45])


@pytest.mark.slow
def test_fit_subject_level_recovers_metadpy_reference():
    fit = fit_subject_level(NR_S1, NR_S2, chains=4, n_iter=2000, warmup=1000, seed=42)

    post = arviz.extract(fit.idata)
    meta_d_mean = float(post["meta_d"].values.mean())
    d1_mean = float(post["d1"].values.mean())

    # Reference values from metadpy's hmetad() on this exact dataset
    # (documented in subject_level.py's docstring): meta_d ~= 1.57 +/- 0.20,
    # d1 ~= 1.53 +/- 0.14. Tolerance is wide relative to the reported SD to
    # absorb MCMC noise across seeds/platforms while still catching a real
    # regression (sign flip, wrong likelihood, bad prior).
    assert abs(meta_d_mean - 1.57) < 0.5
    assert abs(d1_mean - 1.53) < 0.4

    diag = fit.convergence_diagnostics()
    assert diag["converged"].all(), f"Non-convergent parameters:\n{diag[~diag['converged']]}"
