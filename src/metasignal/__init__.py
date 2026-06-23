"""metasignal package."""

from metasignal.wrapper import MetaSignal
from metasignal import stdpy
from metasignal import analysis

__all__ = ["MetaSignal", "stdpy", "analysis", "sdtbayes"]


def __getattr__(name: str) -> object:
    if name == "sdtbayes":
        import importlib
        return importlib.import_module("metasignal.sdtbayes")
    raise AttributeError(f"module 'metasignal' has no attribute {name!r}")
