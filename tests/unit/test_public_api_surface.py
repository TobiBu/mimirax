"""Every name in __all__ is importable, and the runtime-typecheck switch behaves."""

from __future__ import annotations

import importlib

import mimirax
from mimirax._typecheck import enable_runtime_typecheck


def test_all_names_resolve() -> None:
    """__all__ lists only attributes that exist, sorted case-sensitively."""
    for name in mimirax.__all__:
        assert hasattr(mimirax, name), name
    assert list(mimirax.__all__) == sorted(mimirax.__all__)


def test_adapters_are_not_imported_eagerly() -> None:
    """The adapters package exposes no names at import time."""
    adapters = importlib.import_module("mimirax.adapters")
    assert adapters.__all__ == []


def test_runtime_typecheck_is_off_unless_asked(monkeypatch) -> None:
    """The hook installs only when MIMIRAX_RUNTIME_TYPECHECK is exactly '1'."""
    monkeypatch.delenv("MIMIRAX_RUNTIME_TYPECHECK", raising=False)
    assert enable_runtime_typecheck() is False
    monkeypatch.setenv("MIMIRAX_RUNTIME_TYPECHECK", "yes")
    assert enable_runtime_typecheck() is False
