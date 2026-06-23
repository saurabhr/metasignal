"""metasignal package."""

from metasignal.wrapper import MetaSignal
from metasignal import stdpy
from metasignal import analysis

__all__ = ["MetaSignal", "stdpy", "analysis", "bayesian"]


def __getattr__(name: str) -> object:
    if name == "bayesian":
        from metasignal import bayesian as _bayesian
        return _bayesian
    raise AttributeError(f"module 'metasignal' has no attribute {name!r}")
