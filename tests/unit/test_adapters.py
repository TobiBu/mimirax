"""The adapters import without their solvers and fail loudly when used without them."""

from __future__ import annotations

import builtins
import sys

import jax
import jax.numpy as jnp
import pytest

from mimirax import ForceModel, ForwardModel, TimeAverage, WeightedKernelSum
from mimirax.adapters.jaccpot import JaccpotForceModel
from mimirax.adapters.nornax import (
    FoldedRollout,
    NornaxRollout,
    SingleRungMutualForce,
)
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


def test_odisseo_adapter_is_a_forward_model_stub() -> None:
    """It has the ForwardModel shape and raises until decision O-4 lands."""
    odisseo = OdisseoForwardModel(config=None, params=None)
    assert isinstance(odisseo, ForwardModel)
    with pytest.raises(NotImplementedError, match="O-4"):
        odisseo({})


# --- the nornax rollout adapter, without nornax installed --------------------------
#
# Everything below runs on a machine that has no solver: nornax is imported
# inside the rollout, so the adapter's construction-time contract -- which
# force can drive which k_max, which leaf is the dynamical mass, the
# single-rung bridge's refusal above level 0 -- is checked here, in CI, and only
# the integration itself waits for the extra
# (``tests/integration/test_m2m_rollout.py``).


def test_rollout_is_a_forward_model_and_needs_nornax_to_run() -> None:
    """The shape is a ForwardModel; calling it without nornax names the extra."""
    rollout = NornaxRollout.tracer(DirectSumGravity(), dt=0.01, num_steps=10)
    assert isinstance(rollout, ForwardModel)
    real_import = builtins.__import__

    def _no_nornax(name, *args, **kwargs):
        if name.startswith("nornax"):
            raise ImportError("no module named nornax")
        return real_import(name, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        for module in [key for key in sys.modules if key.startswith("nornax")]:
            patch.delitem(sys.modules, module, raising=False)
        patch.setattr(builtins, "__import__", _no_nornax)
        with pytest.raises(ImportError, match=r"mimirax\[nornax\]"):
            rollout({"positions": jnp.zeros((2, 3))})


def test_the_two_constructions_differ_only_in_where_the_weights_go() -> None:
    """self_consistent puts the weights in the dynamics; tracer keeps them out."""
    params = {
        "positions": jnp.zeros((3, 3)),
        "velocities": jnp.zeros((3, 3)),
        "weights": jnp.asarray([1.0, 2.0, 3.0]),
    }
    consistent = NornaxRollout.self_consistent(DirectSumGravity(), dt=0.01, num_steps=4)
    assert consistent.weights_are_masses
    assert jnp.array_equal(consistent.dynamical_masses(params), params["weights"])

    tracer = NornaxRollout.tracer(DirectSumGravity(), dt=0.01, num_steps=4)
    assert not tracer.weights_are_masses
    assert tracer.k_max == 0
    # No masses given: unit test particles, which is what an external field wants.
    assert jnp.array_equal(tracer.dynamical_masses(params), jnp.ones(3))
    given = NornaxRollout.tracer(
        DirectSumGravity(), dt=0.01, num_steps=4, masses=jnp.full(3, 0.5)
    )
    assert jnp.array_equal(given.dynamical_masses(params), jnp.full(3, 0.5))


def test_a_mimirax_force_model_is_rejected_above_a_single_rung() -> None:
    """k_max > 0 needs a real MutualForceModel, and the error says which ones."""
    with pytest.raises(ValueError, match="SingleRungMutualForce"):
        NornaxRollout(force=DirectSumGravity(), dt=0.01, num_steps=4, k_max=1)
    with pytest.raises(ValueError, match="num_steps must be >= 1"):
        NornaxRollout(force=DirectSumGravity(), dt=0.01, num_steps=0)


def test_the_bridge_is_the_total_force_at_level_zero_and_refuses_the_rest() -> None:
    """Level 0 is the wrapped model's total; any other level raises."""
    force = DirectSumGravity(softening=0.1)
    bridge = SingleRungMutualForce(force)
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    masses = jnp.asarray([1.0, 2.0, 0.5])
    rung = jnp.zeros(3, dtype=jnp.int32)
    got = bridge.level_accelerations(positions, masses, rung=rung, level=0)
    assert jnp.array_equal(got, force.accelerations(positions, masses))
    with pytest.raises(ValueError, match="k_max=0 bridge"):
        bridge.level_accelerations(positions, masses, rung=rung, level=1)


def test_the_bridge_is_transparent_to_gradients() -> None:
    """Wrapping must not change a derivative; it is a rename, not a computation."""
    force = DirectSumGravity(softening=0.1)
    bridge = SingleRungMutualForce(force)
    positions = jnp.asarray([[0.1, 0.0, 0.0], [1.0, 0.3, 0.0], [0.0, 2.0, 0.4]])
    masses = jnp.asarray([1.0, 2.0, 0.5])
    rung = jnp.zeros(3, dtype=jnp.int32)

    def direct(m):
        return jnp.sum(force.accelerations(positions, m) ** 2)

    def bridged(m):
        return jnp.sum(
            bridge.level_accelerations(positions, m, rung=rung, level=0) ** 2
        )

    assert jnp.allclose(jax.grad(direct)(masses), jax.grad(bridged)(masses), atol=1e-14)


def test_folded_rollout_refuses_an_uneven_split() -> None:
    """An uneven segment split is refused, not silently rounded.

    Under a decaying average the segments are not interchangeable, so a split
    that left a short final segment would weight the trajectory wrongly. The
    error names both numbers.
    """
    observable = TimeAverage(WeightedKernelSum(lambda state: jnp.ones((2, 1))))
    rollout = NornaxRollout.tracer(DirectSumGravity(), dt=0.01, num_steps=24)
    with pytest.raises(ValueError, match="does not divide"):
        FoldedRollout(rollout=rollout, observable=observable, segments=5)
    with pytest.raises(ValueError, match="segments must be >= 1"):
        FoldedRollout(rollout=rollout, observable=observable, segments=0)
    # And the divisors are accepted.
    for segments in (1, 2, 3, 4, 6, 8, 12, 24):
        assert (
            FoldedRollout(
                rollout=rollout, observable=observable, segments=segments
            ).segments
            == segments
        )


def test_balanced_picks_the_best_available_divisor() -> None:
    """`balanced` minimises `num_steps / segments + segments` over the divisors.

    The continuous optimum is `sqrt(num_steps)`; the split has to be even, so it
    takes the largest divisor at or below that. For 24 that is 4 (not 4.9), for
    1024 it is 32 exactly, and for a prime step count there is nothing to do.
    """
    observable = TimeAverage(WeightedKernelSum(lambda state: jnp.ones((2, 1))))

    def balanced_for(steps):
        rollout = NornaxRollout.tracer(DirectSumGravity(), dt=0.01, num_steps=steps)
        return FoldedRollout.balanced(rollout, observable).segments

    assert balanced_for(24) == 4
    assert balanced_for(1024) == 32
    assert balanced_for(256) == 16
    assert balanced_for(100) == 10
    assert balanced_for(23) == 1  # prime: no even split beats one segment
    assert balanced_for(1) == 1


def test_folding_returns_a_forward_model_and_needs_nornax_to_run() -> None:
    """`folding` builds a ForwardModel; running it without nornax names the extra."""
    observable = TimeAverage(WeightedKernelSum(lambda state: jnp.ones((2, 1))))
    folded = NornaxRollout.tracer(DirectSumGravity(), dt=0.01, num_steps=8).folding(
        observable, segments=2
    )
    assert isinstance(folded, FoldedRollout)
    assert isinstance(folded, ForwardModel)
    real_import = builtins.__import__

    def _no_nornax(name, *args, **kwargs):
        if name.startswith("nornax"):
            raise ImportError("no module named nornax")
        return real_import(name, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        for module in [key for key in sys.modules if key.startswith("nornax")]:
            patch.delitem(sys.modules, module, raising=False)
        patch.setattr(builtins, "__import__", _no_nornax)
        with pytest.raises(ImportError, match=r"mimirax\[nornax\]"):
            folded(
                {
                    "positions": jnp.zeros((2, 3)),
                    "velocities": jnp.zeros((2, 3)),
                    "weights": jnp.ones(2),
                }
            )
