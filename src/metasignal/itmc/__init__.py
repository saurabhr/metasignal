"""Information-theoretic metacognition (ITMC) measures.

Implements meta-I and related measures from:
  Dayan P (2023). Metacognitive Information Theory. Open Mind, 7, 392-411.
  https://doi.org/10.1162/opmi_a_00091

Python port of the estimateMetaI() function from:
  statConfR R package (Rausch et al., 2025, JOSS). doi:10.21105/joss.06966
  https://github.com/ManuelRausch/StatConfR
"""

from metasignal.itmc.measures import (
    estimate_meta_I,
    meta_I,
    meta_Ir1,
    meta_Ir1_acc,
    meta_Ir2,
    RMI,
    permtest_meta_I,
)
from metasignal.itmc.group import fit_group, MEASURE_COLS

__all__ = [
    "estimate_meta_I",
    "fit_group",
    "MEASURE_COLS",
    "meta_I",
    "meta_Ir1",
    "meta_Ir1_acc",
    "meta_Ir2",
    "RMI",
    "permtest_meta_I",
]
