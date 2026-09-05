"""The adapters import without their solvers and fail loudly when used without them."""

from __future__ import annotations

import builtins
import sys

import jax.numpy as jnp
import pytest

from mimirax import ForceModel, ForwardModel
from mimirax.adapters.jaccpot import JaccpotForceModel
from mimirax.adapters.nornax import NornaxRollout
from mimirax.adapters.odisseo import OdisseoForwardModel
from mimirax.testing import DirectSumGravity


class _FakeSolver:
    """Stands in for jaccpot.FastMultipoleMethod."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def compute_accelerations(self, positions, masses):
        return positions * jnp.sum(masses)


def test_jaccpot_adapter_forwards_to_the_solver() -> None:
    """The adapter is a ForceModel and calls compute_accelerations(positions, masses)."""
    model = JaccpotForceModel(_FakeSolver())
    assert isinstance(model, ForceModel)
    positions = jnp.ones((2, 3))
    out = model.accelerations(
        positions, jnp.asarray([1.0, 2.0]), args={"ignored": True}
    )
    assert jnp.array_equal(out, 3.0 * positions)


def test_jaccpot_from_preset_names_the_extra_when_missing(monkeypatch) -> None:
    """Without jaccpot installed the factory raises an ImportError naming the extra."""
    real_import = builtins.__import__

    def _no_jaccpot(name, *args, **kwargs):
        if name == "jaccpot":
            raise ImportError("no module named jaccpot")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "jaccpot", raising=False)
    monkeypatch.setattr(builtins, "__import__", _no_jaccpot)
    with pytest.raises(ImportError, match=r"mimirax\[jaccpot\]"):
        JaccpotForceModel.from_preset()


def test_jaccpot_from_preset_wraps_a_solver_when_present(monkeypatch) -> None:
    """With a jaccpot module on the path the factory builds and wraps the solver."""
    import types

    fake = types.ModuleType("jaccpot")
    fake.FastMultipoleMethod = _FakeSolver  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jaccpot", fake)
    model = JaccpotForceModel.from_preset(preset="fast", theta=0.6)
    assert isinstance(model.solver, _FakeSolver)
    assert model.solver.kwargs == {"preset": "fast", "theta": 0.6}


def test_rollout_and_odisseo_adapters_are_forward_model_stubs() -> None:
    """Both have the ForwardModel shape and raise until their decisions land."""
    rollout = NornaxRollout(force=DirectSumGravity(), dt=0.01, num_steps=10)
    assert isinstance(rollout, ForwardModel)
    with pytest.raises(NotImplementedError, match="D-026"):
        rollout({"positions": jnp.zeros((2, 3))})
    odisseo = OdisseoForwardModel(config=None, params=None)
    assert isinstance(odisseo, ForwardModel)
    with pytest.raises(NotImplementedError, match="O-4"):
        odisseo({})
