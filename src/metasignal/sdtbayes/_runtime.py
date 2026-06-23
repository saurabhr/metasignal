"""One-time brms/Stan runtime setup via brmspy."""

from __future__ import annotations


def setup_runtime(use_prebuilt: bool = True) -> None:
    """Install the brms/Stan toolchain (run once before fitting any model).

    Downloads a prebuilt Stan runtime and installs the brms R package into
    an isolated environment managed by brmspy.  Internet access is required
    on first call; subsequent calls are no-ops if the runtime is already
    present.

    Args:
        use_prebuilt: Use a prebuilt Stan binary (faster). Set ``False`` to
            compile from source (requires a C++ toolchain).

    Example::

        from metasignal.sdtbayes import setup_runtime
        setup_runtime()   # ~5 min on first run
    """
    try:
        from brmspy import brms
    except ImportError as e:
        msg = (
            "brmspy is not installed. Run:\n"
            "    pip install metasignal[sdtbayes]"
        )
        raise ImportError(msg) from e

    with brms.manage() as ctx:
        ctx.install_brms(use_prebuilt=use_prebuilt)
