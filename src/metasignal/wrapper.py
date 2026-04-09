"""Python wrapper for the MATLAB meta-signal analysis functions."""

from __future__ import annotations

import contextlib
import pathlib
from typing import Any

import numpy as np

try:
    import matlab
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False


class MetaSignal:
    """Wrapper class for MATLAB meta-signal functions."""

    def __init__(self, engine: Any | None = None) -> None:
        """Initialize the MetaSignal wrapper.

        Args:
            engine: An existing MATLAB engine instance. If None, a new one will be started.
        """
        if not MATLAB_AVAILABLE:
            raise RuntimeError(
                "MATLAB engine is not available. Please install it with "
                "`pip install metasignal[matlab]` if you have MATLAB locally."
            )

        if engine is None:
            self.eng = matlab.engine.start_matlab()
        else:
            self.eng = engine

        # Add the packaged MATLAB code to the path
        self._setup_path()

    def _setup_path(self) -> None:
        """Add the packaged MATLAB code to the MATLAB path."""
        base_path = pathlib.Path(__file__).parent / "matlab"

        # Add the main directory and its subdirectories
        self.eng.addpath(str(base_path), nargout=0)
        self.eng.addpath(str(base_path / "helperFunctions"), nargout=0)
        self.eng.addpath(str(base_path / "helperFunctions" / "metaMeasures"), nargout=0)
        self.eng.addpath(
            str(base_path / "helperFunctions" / "metaMeasures" / "Mfunctions"),
            nargout=0,
        )
        self.eng.addpath(
            str(base_path / "helperFunctions" / "metaMeasures" / "lognormalMetaNoise"),
            nargout=0,
        )

    def compute_all_measures(
        self, stim: np.ndarray, resp: np.ndarray, conf: np.ndarray, n_ratings: int
    ) -> np.ndarray:
        """Compute all meta-signal measures.

        Args:
            stim: Stimulus data (0/1 or similar).
            resp: Response data (0/1 or similar).
            conf: Confidence ratings.
            n_ratings: Number of rating categories.

        Returns:
            Array of computed measures.
        """
        # Convert numpy arrays to matlab double arrays
        m_stim = matlab.double(stim.tolist())
        m_resp = matlab.double(resp.tolist())
        m_conf = matlab.double(conf.tolist())

        # Call the MATLAB function
        result = self.eng.compute_all_measures(m_stim, m_resp, m_conf, float(n_ratings))

        return np.array(result).flatten()

    def stop(self) -> None:
        """Stop the MATLAB engine."""
        if hasattr(self, "eng"):
            self.eng.quit()
            # Note: We don't delete self.eng here to avoid __del__ issues if called manually
            # but we could.

    def __del__(self) -> None:
        """Ensure the engine is stopped when the object is destroyed."""
        with contextlib.suppress(Exception):
            self.stop()
