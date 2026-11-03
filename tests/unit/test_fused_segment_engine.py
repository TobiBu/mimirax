"""``FusedSegmentEngine``'s bookkeeping, on a stand-in runner (no jaccpot needed).

The engine's logic that does not depend on jaccpot -- one pre-pass per segment,
plans cached and reused while the verdict passes, only the failed segments
re-planned, a stale program refused, the aux behind ``stop_gradient``, lazy plans
for an eager ``init_nodes``, the augmented-Lagrangian driver's prepare / check /
redo loop -- checked on a fake ``ProductionRunner`` whose "plan" is a harmonic
leapfrog with a capacity: a plan made at a node holds while the positions stay
inside 1.5x the extent of its pre-pass, and its verdict bit says when they do
not. The
fused lane itself is ``tests/integration/test_fused_segment_engine.py`` (manual,
jaccpot needed).
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax
import pytest

from mimirax import GaussianLikelihood, InferenceProblem
from mimirax.adapters import jaccpot as adapter
from mimirax.adapters.jaccpot import FusedSegmentEngine
from mimirax.inference.augmented_lagrangian import AugmentedLagrangian
from mimirax.shooting import (
    MultipleShootingForwardModel,
    SegmentEngine,
    SegmentResult,
)


def _harmonic(state, args):
    return -args["k"] * state[:, 0]


class _FakePlan:
    """``GradientPlan``'s surface: ``params``, ``external_fn``, ``forward``,
    ``rollout(inputs, checkpoint=, inner_steps=)``."""

    def __init__(self, state, masses, dt, num_steps, k, slack):
        self.params = {"dt": dt, "force_params": None, "external_params": k}
        self.external_fn = _harmonic
        self.num_steps = num_steps
        # the "capacity": the pre-pass's own extent, with headroom
        self.extent = jnp.inf
        inputs = {
            **self.params,
            "positions": state[:, 0],
            "velocities": state[:, 1],
            "masses": masses,
        }
        self.extent = slack * float(self.rollout(inputs).aux[2])
        out = self.rollout(
            {
                **self.params,
                "positions": state[:, 0],
                "velocities": state[:, 1],
                "masses": masses,
            }
        )
        self.forward = SimpleNamespace(
            state=jnp.stack([out.positions, out.velocities], axis=1),
            handle=SimpleNamespace(force_scale=out.aux[0]),
        )

    def rollout(self, inputs, *, checkpoint="step", inner_steps=None):
        del checkpoint, inner_steps
        x, v, dt = inputs["positions"], inputs["velocities"], inputs["dt"]
        args = inputs["external_params"]
        extent = jnp.asarray(0.0)
        for _ in range(self.num_steps):
            v = v + 0.5 * dt * _harmonic(jnp.stack([x, v], axis=1), args)
            x = x + dt * v
            v = v + 0.5 * dt * _harmonic(jnp.stack([x, v], axis=1), args)
            extent = jnp.maximum(extent, jnp.max(jnp.abs(x)))
        bits = jnp.where(extent > self.extent, 1, 0).astype(jnp.uint32)
        f_b = jnp.sum(x**2, axis=1)  # a floating aux leaf, as dehnen_error's
        return SimpleNamespace(positions=x, velocities=v, aux=(f_b, bits, extent))


class _FakeRunner:
    external_acceleration_fn = staticmethod(_harmonic)

    def __init__(self, slack=1.5):
        self.calls = []
        self.slack = slack

    def gradient_plan(
        self, state, masses, *, dt, num_steps, force_params, external_params
    ):
        del force_params
        self.calls.append(num_steps)
        return _FakePlan(state, masses, dt, num_steps, external_params, self.slack)


@pytest.fixture(autouse=True)
def _verdict(monkeypatch):
    # jaccpot's GRAD_FAIL_MASK stands in as "any bit"
    monkeypatch.setattr(adapter, "_gradient_ok", lambda bits: int(bits) == 0)


def _system():
    key = jax.random.PRNGKey(3)
    x = jax.random.normal(key, (8, 3))
    v = 0.1 * jax.random.normal(jax.random.PRNGKey(4), (8, 3))
    return x, v, jnp.ones(8) / 8


def _objective(engine, calls):
    def objective(theta, nodes, masses):
        calls.append(1)  # trace-time count
        auxes, loss = [], 0.0
        for k in range(engine.num_segments):
            end = engine(
                theta, nodes["positions"][k], nodes["velocities"][k], masses, k
            )
            auxes.append(end.aux)
            # reads the f_b, which is behind stop_gradient
            loss = loss + jnp.sum(end.positions**2) + jnp.sum(end.aux[0])
        return loss, tuple(auxes)

    return jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))


def test_engine_is_a_segment_engine_and_needs_a_plan_when_traced():
    engine = FusedSegmentEngine(_FakeRunner(), (2, 3), dt=0.1)
    assert isinstance(engine, SegmentEngine)
    assert engine.stale_segments == (0, 1)
    x, v, m = _system()
    with pytest.raises(RuntimeError, match="no plan"):
        jax.jit(lambda th: engine(th, x, v, m, 0))({"k": 1.0})
    with pytest.raises(ValueError, match="positive"):
        FusedSegmentEngine(_FakeRunner(), (), dt=0.1)


def test_propagate_plans_along_one_run_and_the_forward_is_the_pre_pass():
    runner = _FakeRunner()
    engine = FusedSegmentEngine(runner, (2, 3), dt=0.1)
    x, v, m = _system()
    theta = {"k": jnp.asarray(1.0)}
    nodes = engine.propagate(theta, x, v, m)
    assert runner.calls == [2, 3] and engine.stale_segments == ()
    assert engine.generation == 2 and engine.plans[0].forward is None
    end0 = engine.ends[0]
    assert jnp.array_equal(nodes["positions"][1], end0[:, 0])
    for k in range(2):
        end = engine(theta, nodes["positions"][k], nodes["velocities"][k], m, k)
        assert isinstance(end, SegmentResult)
        assert jnp.array_equal(
            jnp.stack([end.positions, end.velocities], axis=1), engine.ends[k]
        )
        assert int(end.aux[3]) == k + 1
    # prepare with nothing stale runs no pre-pass: idempotent
    engine.prepare(theta, nodes, m)
    engine.prepare(theta, nodes, m)
    assert runner.calls == [2, 3] and engine.stats["prepares"] == 2


def test_an_eager_init_nodes_plans_lazily_and_prepare_plans_the_rest():
    """``MultipleShootingForwardModel.init_nodes`` called eagerly plans segments
    0..S-2 at the propagated nodes; the driver's first ``prepare`` plans the
    last segment only."""
    runner = _FakeRunner()
    engine = FusedSegmentEngine(runner, (2, 2, 2), dt=0.1)
    x, v, m = _system()
    model = MultipleShootingForwardModel(engine, m, 3, x, v)
    theta = {"k": jnp.asarray(1.0)}
    nodes = model.init_nodes(theta)
    assert runner.calls == [2, 2] and engine.stale_segments == (2,)
    want = engine.propagate(theta, x, v, m)
    assert jnp.array_equal(nodes["positions"], want["positions"][1:])
    runner.calls.clear()
    engine._stale = {2}
    model.prepare({"theta": theta, "nodes": nodes})
    assert runner.calls == [2] and engine.stale_segments == ()
    evaluation = jax.jit(model.evaluate)({"theta": theta, "nodes": nodes})
    assert model.check(evaluation.auxes)
    assert float(jnp.max(jnp.abs(evaluation.defects))) < 1e-12


def test_plans_are_reused_while_the_verdict_holds_and_only_failures_replan():
    runner = _FakeRunner()
    engine = FusedSegmentEngine(runner, (2, 2), dt=0.1, keep_forward=True)
    x, v, m = _system()
    theta = {"k": jnp.asarray(1.0)}
    nodes = engine.propagate(theta, x, v, m)
    assert engine.plans[0].forward is not None
    traces: list = []
    program = _objective(engine, traces)

    # two optimizer-like iterations, slightly moved: one trace, no re-plan
    for step in range(2):
        moved = jax.tree_util.tree_map(lambda a: a * (1.0 + 1e-3 * step), nodes)
        theta_i = {"k": theta["k"] + 1e-3 * step}
        engine.prepare(theta_i, moved, m)
        (_, auxes), grads = program(theta_i, moved, m)
        assert engine.check(auxes)
        assert jnp.isfinite(grads[0]["k"])
    assert len(traces) == 1 and runner.calls == [2, 2]
    assert engine.stats["checks"] == 2

    # segment 1's node far outside its plan's extent: its verdict fails alone
    far = {
        "positions": nodes["positions"].at[1].multiply(3.0),
        "velocities": nodes["velocities"],
    }
    (_, auxes), _ = program(theta, far, m)
    assert not engine.check(auxes)
    assert engine.failed_segments == (1,) and engine.stale_segments == (1,)
    plan0 = engine.plans[0]
    engine.prepare(theta, far, m)
    assert runner.calls == [2, 2, 2] and engine.plans[0] is plan0
    assert engine.generation == 3
    # the compiled program still holds the replaced plan: refused
    (_, auxes), _ = program(theta, far, m)
    with pytest.raises(RuntimeError, match="trace the objective again"):
        engine.check(auxes)
    # traced again (a new function), the redo passes
    program = _objective(engine, traces)
    (_, auxes), _ = program(theta, far, m)
    assert engine.check(auxes) and len(traces) == 2


def test_the_augmented_lagrangian_driver_replans_and_redoes_a_failed_step():
    """M's driver on the fused engine's seam: plans that hold only a hair beyond
    their nodes (slack 1.0001) fail as the optimizer moves the nodes; the driver's
    ``check`` -> ``prepare`` -> retrace -> redo loop re-plans exactly the failed
    segments and the defects still vanish."""
    runner = _FakeRunner(slack=1.0001)
    engine = FusedSegmentEngine(runner, (3, 3, 3), dt=0.05)
    x, v, m = _system()
    model = MultipleShootingForwardModel(engine, m, 3, x, v)
    truth = {"k": jnp.asarray(1.0)}
    nodes = model.init_nodes(truth)
    model.prepare({"theta": truth, "nodes": nodes})
    observed = jax.jit(lambda p: model(p))({"theta": truth, "nodes": nodes})
    observed = observed["positions"].ravel()
    problem = InferenceProblem(
        forward=model,
        observable=lambda state: state["positions"].ravel(),
        likelihood=GaussianLikelihood(sigma=0.05),
        observed=observed,
    )
    start = {
        "theta": {"k": jnp.asarray(1.2)},
        "nodes": jax.tree_util.tree_map(lambda a: a * 1.01, nodes),
    }
    driver = AugmentedLagrangian(
        optimizer=optax.lbfgs(), inner_steps=40, max_outer=10, max_redo=5
    )
    result = driver.minimize(problem, start)
    assert result.num_redos >= 1 and result.num_traces == 1 + result.num_redos
    # every redo re-planned at least one segment, and only failed ones
    assert 3 + result.num_redos <= engine.stats["plans"] <= 3 + 3 * result.num_redos
    assert float(result.outer_violation[-1]) < 1e-8
    assert abs(float(result.params["theta"]["k"]) - 1.0) < 1e-4


def test_prepare_and_check_count_their_arguments():
    engine = FusedSegmentEngine(_FakeRunner(), (2, 2), dt=0.1)
    x, v, m = _system()
    with pytest.raises(ValueError, match="every segment"):
        engine.prepare({"k": 1.0}, [(x, v)], m)
    engine.prepare({"k": jnp.asarray(1.0)}, [(x, v), (x, v)], m)
    with pytest.raises(ValueError, match="one aux per segment"):
        engine.check(())


def test_the_runners_freeze_cache_is_kept_across_pre_passes():
    """Where jaccpot's ``gradient_plan`` takes ``freeze_cache``, the engine passes
    ONE dict to every pre-pass (its freeze program compiles once per key)."""

    class _CachingRunner(_FakeRunner):
        def gradient_plan(self, state, masses, *, freeze_cache=None, **kwargs):
            self.caches = getattr(self, "caches", []) + [freeze_cache]
            return super().gradient_plan(state, masses, **kwargs)

    runner = _CachingRunner()
    engine = FusedSegmentEngine(runner, (1, 1), dt=0.1)
    x, v, m = _system()
    engine.propagate({"k": jnp.asarray(1.0)}, x, v, m)
    assert isinstance(runner.caches[0], dict) and runner.caches[0] is runner.caches[1]
    plain = FusedSegmentEngine(_FakeRunner(), (1,), dt=0.1)
    assert plain._plan_options == {}
