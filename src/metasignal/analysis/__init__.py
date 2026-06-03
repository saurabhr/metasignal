"""Statistical inference over SDT and metacognitive measures.

This sub-package operates *over* the point estimates computed by
``metasignal.stdpy``. It provides:

- ``bootstrap`` — non-parametric confidence intervals via resampling
- ``permutation`` — null-distribution tests for condition differences
- ``group`` — group-level summaries and hierarchical model stubs
"""

from metasignal.analysis.bootstrap import bootstrap_measure
from metasignal.analysis.permutation import permutation_test
from metasignal.analysis.group import group_summary

__all__ = [
    "bootstrap_measure",
    "permutation_test",
    "group_summary",
]
