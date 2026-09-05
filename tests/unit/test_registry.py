"""Tests for the generic registry idiom."""

from __future__ import annotations

import pytest

from mimirax._registry import Registry


def test_register_get_and_available_are_consistent() -> None:
    """Registered names come back sorted and resolve to what was registered."""
    reg: Registry[int] = Registry("thing")
    reg.register("b", 2)
    reg.register("a", 1)
    assert reg.available() == ("a", "b")
    assert reg.get("a") == 1 and reg.get(" b ") == 2
    assert "a" in reg and "c" not in reg and 3 not in reg
    assert len(reg) == 2
    assert reg.kind == "thing"


def test_register_refuses_silent_overwrite() -> None:
    """A second registration under the same name needs overwrite=True."""
    reg: Registry[int] = Registry("thing")
    reg.register("a", 1)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("a", 2)
    assert reg.register("a", 2, overwrite=True) == 2
    assert reg.get("a") == 2


def test_register_rejects_empty_name() -> None:
    """An empty or whitespace name is an error."""
    reg: Registry[int] = Registry("thing")
    with pytest.raises(ValueError, match="non-empty"):
        reg.register("   ", 1)


def test_get_unknown_lists_what_exists() -> None:
    """The KeyError for a typo names every registered key."""
    reg: Registry[int] = Registry("thing")
    reg.register("alpha", 1)
    with pytest.raises(KeyError, match="'alpha'"):
        reg.get("alpah")


def test_decorate_registers_and_returns_the_object() -> None:
    """The decorator form leaves the decorated object untouched."""
    reg: Registry[type] = Registry("thing")

    @reg.decorate("k")
    class K:
        pass

    assert reg.get("k") is K
