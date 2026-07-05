"""Tests for the deprecated MATLAB wrapper's no-MATLAB code path."""

import pytest

from metasignal import wrapper


def test_metasignal_raises_without_matlab(monkeypatch):
    monkeypatch.setattr(wrapper, "MATLAB_AVAILABLE", False)
    with pytest.warns(DeprecationWarning):
        with pytest.raises(RuntimeError, match="MATLAB engine is not available"):
            wrapper.MetaSignal()
