"""Native Python implementation of meta-signal measures."""

from metasignal.stdpy.core import compute_sdt_resp, trials_to_counts
from metasignal.stdpy.metad import fit_meta_d_mle
from metasignal.stdpy.type2 import (
    sdt_expect_conf,
    compute_type2_auc,
    compute_gamma,
    compute_phi,
    compute_delta_conf,
)
from metasignal.stdpy.uncertainty import compute_meta_uncertainty
from metasignal.stdpy.metanoise import compute_meta_noise
from metasignal.stdpy.compute_all import compute_all_measures
from metasignal.stdpy.group import fit_group, MEASURE_COLS
from metasignal.stdpy.plot import (
    plot_confidence,
    plot_type2roc,
    plot_sanity_check,
    plot_forest,
    plot_measures,
)
from metasignal.stdpy.simulate import (
    type2_SDT_simuation,
    type2_SDT_simuation_bayes,
    ratings2df,
    trialSimulation,
    responseSimulation,
    pairedResponseSimulation,
    discreteRatings,
)

__all__ = [
    "compute_sdt_resp",
    "trials_to_counts",
    "fit_meta_d_mle",
    "sdt_expect_conf",
    "compute_type2_auc",
    "compute_gamma",
    "compute_phi",
    "compute_meta_uncertainty",
    "compute_delta_conf",
    "compute_meta_noise",
    "compute_all_measures",
    "fit_group",
    "MEASURE_COLS",
    "plot_confidence",
    "plot_type2roc",
    "plot_sanity_check",
    "plot_forest",
    "plot_measures",
    "type2_SDT_simuation",
    "type2_SDT_simuation_bayes",
    "ratings2df",
    "trialSimulation",
    "responseSimulation",
    "pairedResponseSimulation",
    "discreteRatings",
]
