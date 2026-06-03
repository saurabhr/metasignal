"""Deprecated: use metasignal.stdpy.type2 instead."""

import warnings

warnings.warn(
    "metasignal.stdpy.measures is renamed to metasignal.stdpy.type2. "
    "Please update your imports. This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from metasignal.stdpy.type2 import (  # noqa: F401, E402
    sdt_expect_conf,
    compute_type2_auc,
    compute_gamma,
    compute_phi,
    compute_delta_conf,
)
