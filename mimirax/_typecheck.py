"""Runtime type-checking opt-in for mimirax.

Set the environment variable ``MIMIRAX_RUNTIME_TYPECHECK=1`` before importing
``mimirax`` to enable package-wide ``jaxtyping`` + ``beartype`` instrumentation.

The jaxtyping import hook only instruments modules imported *after* it is
installed, so ``mimirax.__init__`` calls :func:`enable_runtime_typecheck` before
importing any mimirax submodules. This is the nornax mechanism, unchanged.
"""

from __future__ import annotations

import os

__all__ = ["enable_runtime_typecheck"]


def enable_runtime_typecheck() -> bool:
    """Install the jaxtyping + beartype import hook when requested via env var.

    Returns
    -------
    bool
        ``True`` when the hook was installed by this call, ``False`` when
        ``MIMIRAX_RUNTIME_TYPECHECK`` is unset or not ``"1"`` and nothing was
        instrumented.
    """
    if os.environ.get("MIMIRAX_RUNTIME_TYPECHECK", "0") != "1":
        return False
    from jaxtyping import install_import_hook

    install_import_hook("mimirax", "beartype.beartype")
    return True
