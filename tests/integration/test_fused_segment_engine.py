"""``FusedSegmentEngine`` on jaccpot's fused lane: multiple shooting, segment by segment.

Each shooting segment is a gradient run of jaccpot's ``ProductionRunner``
(``runner.gradient_plan`` at the segment's node, then ``plan.rollout``). The gates
(S10 track J), on CPU with every Pallas kernel and the GPU's walk in interpret
mode, N = 512, float64, order 2, total mass ~1, the harmonic field ``-k x`` with
``theta = {"k": 0.37}``, 2 segments of 2 steps -- jaccpot's
``tests/integration/strict/test_runner_gradient_mode.py`` system, copied -- in the
Plummer lane (geometric MAC) and under ``dehnen_error``:

* G1 -- through :class:`~mimirax.shooting.MultipleShootingForwardModel` at the
  propagated nodes: every segment's traced forward from its node is the runner's
  own trajectory from that node, bitwise (the defect is exactly zero, the end
  state and f_b are the pre-pass's);
* single vs multiple -- the shooting end state against E's single
  ``GradientPlan`` chain over the whole horizon, and the gradient of the end loss
  composed through the node against the chain's (a shooting node is a START
  boundary, a force recomputed at the node; the chain's edge is CARRIED, the last
  step's total acceleration). Measured and bounded per lane;
* FD -- central differences of a loss through both segments in theta and in
  node 1's positions, every force's lists identical on the +-h runs;
* compile once -- the augmented-Lagrangian driver takes two steps at moving
  theta and nodes on ONE trace of its step and no re-plan; a forced cap failure
  (segment 1's far cap cut below its need) makes ``check`` fail for segment 1
  alone, ``prepare`` re-plans it alone, a program traced on the replaced plan is
  refused and the redo passes;
* the aux guard -- a loss that reads the engine's aux (behind ``stop_gradient``)
  traces its gradient; the same through the raw rollout aux is refused;
* forward mode (Plummer lane) -- ``jax.jvp`` in theta through segment 0 under
  ``ad_mode_override("forward")`` and with ``ad_mode="forward"``: the primal the
  runner's end bitwise, the tangent the reverse gradient's transpose and central
  differences'.

MANUAL: a lane takes 16.5 GB (Plummer) / 20.6 GB (``dehnen_error``) of peak RSS
and ~25 min on a loaded CPU host (its programs: the forward, the composed and
E's chain gradients, the shooting gradient, the driver's step; 2026-11-01), and
mimirax's hosted CI installs no jaccpot. Run on a large host with jaccpot (with
``ProductionRunner.gradient_plan``, S9) and its siblings
importable, one process per lane::

    MIMIRAX_RUN_MANUAL_TESTS=1 JAX_PLATFORMS=cpu pytest --no-cov -s \\
        tests/integration/test_fused_segment_engine.py -k plummer
"""

from __future__ import annotations

import copy
import dataclasses
import os
import resource
import time
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

pytest.importorskip("jaccpot", reason="jaccpot is not installed")

pytestmark = pytest.mark.skipif(
    os.environ.get("MIMIRAX_RUN_MANUAL_TESTS") != "1",
    reason="manual: interpret-mode fused-lane gradients, 16-21 GB of RSS on CPU; "
    "set MIMIRAX_RUN_MANUAL_TESTS=1",
)

from mimirax import GaussianLikelihood, InferenceProblem  # noqa: E402
from mimirax.adapters.jaccpot import FusedSegmentEngine  # noqa: E402
from mimirax.inference.augmented_lagrangian import AugmentedLagrangian  # noqa: E402
from mimirax.shooting import (  # noqa: E402
    MultipleShootingForwardModel,
    SegmentEngine,
)

N = 512
LEAF, ORDER, THETA, SOFTENING = 16, 2, 0.6, 1e-2
DT = 1e-3
K_EXT = 0.37
STEPS = (2, 2)

#: what jaccpot's strict tests scrub (a shell's JACCPOT_* must not reach the lane)
_ENV_UNSET = (
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
    "JACCPOT_FLAT_WALK_CAP_HEADROOM",
    "JACCPOT_STATIC_STRICT_FUSED_WALK",
    "JACCPOT_STRICT_SEGMENT_RETRY",
    "JACCPOT_STRICT_CARRY",
    "JACCPOT_STRICT_CARRY_ORDER",
)
#: every Pallas kernel of the lane in interpret mode, the prepacked near layout,
#: the GPU's walk (jaccpot's ``test_grad_rollout_differentiable._ENV``)
_ENV = {
    "JACCPOT_CASCADE_PALLAS": "1",
    "JACCPOT_CASCADE_PALLAS_INTERPRET": "1",
    "JACCPOT_M2L_CSR_INTERPRET": "1",
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_L2P_KERNEL": "interpret",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
    "JACCPOT_WALK_PALLAS_INTERPRET": "1",
}
#: the lanes: name -> MAC
_LANES = {"plummer": "dehnen", "dehnen_error": "dehnen_error"}


@pytest.fixture(scope="module", autouse=True)
def _lane_on_cpu() -> Iterator[None]:
    """The interpret flags, and the large-N profile's backend gate open.

    Yields
    ------
    None
        With the environment set for the module.
    """
    patcher = pytest.MonkeyPatch()
    for key in _ENV_UNSET:
        patcher.delenv(key, raising=False)
    for key, value in _ENV.items():
        patcher.setenv(key, value)
    patcher.setattr(jax, "default_backend", lambda: "gpu")
    yield
    patcher.undo()
    _CACHE.clear()


def _lanes(*names: str) -> list:
    """``names`` as parameters, each in its lane's xdist group."""
    return [
        pytest.param(name, marks=pytest.mark.xdist_group(f"fused_segment_{name}"))
        for name in names
    ]


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _report(name: str, what: str, **values: Any) -> None:
    print(f"\n[fused-segment {name}] {what}: {values} (peak RSS {_rss_gb():.2f} GB)")


# ---------------------------------------------------------------------------
# The system and the lane (jaccpot's strict-test setup).
# ---------------------------------------------------------------------------


def _solver(mac: str) -> Any:
    from jaccpot import FastMultipoleMethod
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    criterion = (
        dict(
            adaptive_eps=1e-5,
            adaptive_error_model="dehnen_paper",
            mac_force_scale_mode="paper_fb",
        )
        if mac == "dehnen_error"
        else {}
    )
    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFTENING,
        working_dtype=jnp.float64,
        use_pallas=True,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=1024,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type=mac,
        ),
        fixed_order=ORDER,
        **criterion,
    )


def _system() -> tuple[Any, Any, Any, Any]:
    """Four tight blobs with velocities, total mass about 1 (jaccpot's ``_system``)."""
    rng = np.random.default_rng(0)
    centres = rng.uniform(-0.8, 0.8, (4, 3))
    pos = np.clip(
        centres[rng.integers(0, 4, N)] + 0.08 * rng.normal(size=(N, 3)), -1, 1
    )
    rng = np.random.default_rng(1)
    vel = 0.1 * rng.normal(size=(N, 3))
    masses = (np.abs(rng.normal(size=N)) + 0.5) / N
    weight = rng.normal(size=(N, 3))
    return tuple(jnp.asarray(a, jnp.float64) for a in (pos, vel, masses, weight))


def _harmonic(state: Any, args: Any) -> Any:
    return -args["k"] * state[:, 0]


def _runner(mac: str) -> Any:
    from jaccpot.production import GalaxySettings, ProductionRunner, SegmentConfig

    return ProductionRunner(
        _solver(mac),
        settings=GalaxySettings(
            leaf_size=LEAF,
            max_order=ORDER,
            theta=THETA,
            mac_type=mac,
            adaptive_eps=1e-5 if mac == "dehnen_error" else None,
            precision="fp64",
        ),
        segment=SegmentConfig(steps=2, diagnostics_every=0),
        external_acceleration_fn=_harmonic,
    )


def _bitwise(got: Any, want: Any) -> int:
    got, want = np.asarray(got), np.asarray(want)
    assert got.dtype == want.dtype and got.shape == want.shape
    got, want = got.reshape(-1), want.reshape(-1)
    return int(np.sum(got.view(np.uint8) != want.view(np.uint8)))


def _theta() -> dict:
    return {"k": jnp.asarray(K_EXT, jnp.float64)}


_CACHE: dict = {}


def _engine(name: str) -> tuple:
    """The lane's engine, planned along one forward run: ``(engine, nodes)``."""
    key = ("engine", name)
    if key not in _CACHE:
        engine = FusedSegmentEngine(
            _runner(_LANES[name]), STEPS, dt=DT, keep_forward=True
        )
        x, v, m, _ = _system()
        nodes = engine.propagate(_theta(), x, v, m)
        _report(name, "pre-passes (cold, then warm)", seconds=engine.stats["plan_s"])
        _CACHE[key] = (engine, nodes)
    return _CACHE[key]


def _model(engine: Any) -> MultipleShootingForwardModel:
    x, v, m, _ = _system()
    return MultipleShootingForwardModel(engine, m, len(STEPS), x, v)


def _params(nodes: dict, theta: Any = None) -> dict:
    """The model's parameters: theta and the free nodes (node 0 is fixed)."""
    return {
        "theta": _theta() if theta is None else theta,
        "nodes": {
            "positions": nodes["positions"][1:],
            "velocities": nodes["velocities"][1:],
        },
    }


# ---------------------------------------------------------------------------
# G1: each segment's forward is the runner's from its node.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _lanes(*_LANES))
def test_each_segments_forward_is_the_runners_from_its_node(name):
    """Every segment is a fresh run from its node (start kind ``"start"``). The
    model's traced evaluation at the propagated nodes: segment 0's end is node 1
    bitwise (defect exactly 0), segment 1's end the runner's from node 1, and every
    segment's f_b the pre-pass's."""
    engine, nodes = _engine(name)
    assert isinstance(engine, SegmentEngine)
    for plan in engine.plans:
        assert [s.kind for s in plan.segments] == ["start"]
    model = _model(engine)
    begun = time.perf_counter()
    evaluation = jax.jit(lambda p: model.evaluate(p))(_params(nodes))
    _report(
        name, "forward (trace + compile + run)", seconds=time.perf_counter() - begun
    )
    assert model.check(evaluation.auxes)
    assert _bitwise(evaluation.defects[0], jnp.zeros_like(evaluation.defects[0])) == 0
    final = evaluation.prediction
    got = jnp.stack([final["positions"], final["velocities"]], axis=1)
    assert _bitwise(got, engine.plans[1].forward.state) == 0
    for k, aux in enumerate(evaluation.auxes):
        f_b = engine.plans[k].forward.handle.force_scale
        if f_b is not None:
            assert _bitwise(aux[0], f_b) == 0, k


# ---------------------------------------------------------------------------
# Single vs multiple shooting.
# ---------------------------------------------------------------------------

#: per lane: bounds on the shooting end vs the chain's (max abs over the state),
#: the end loss, and the composed gradient (max abs error over each array's
#: largest component); measured values in the test's report and the docstring
_CHAIN_BOUNDS = {
    "plummer": {"state": 1e-16, "value": 1e-14, "grad": 1e-12},
    "dehnen_error": {"state": 1e-6, "value": 1e-5, "grad": 1e-5},
}


def _chain(name: str) -> Any:
    """E's single gradient plan over the whole horizon, on the engine's runner."""
    key = ("chain", name)
    if key not in _CACHE:
        engine, _ = _engine(name)
        x, v, m, _ = _system()
        _CACHE[key] = engine.runner.gradient_plan(
            jnp.stack([x, v], axis=1),
            m,
            dt=DT,
            num_steps=sum(STEPS),
            external_params={"k": K_EXT},
        )
    return _CACHE[key]


@pytest.mark.parametrize("name", _lanes(*_LANES))
def test_shooting_at_propagated_nodes_matches_one_chain(name):
    """With node 1 the propagated end of segment 0, the shooting end state against
    E's chain over 4 steps (``start`` + ``carried``), and the gradient of the end
    loss composed through the node -- ``G(theta, x0, v0) = loss(seg1(theta,
    seg0(theta, x0, v0)))`` through the engine -- against the chain's
    ``value_and_grad`` in theta, x0 and v0.

    The boundary kinds differ: the shooting node starts a FRESH force at the node
    (the pre-pass's start force; under ``dehnen_error`` its thresholds from the
    pre-pass's f_b seed at the node); the chain's edge CARRIES the last step's
    total acceleration and f_b. Plummer: the same lists, the same self-gravity;
    the totals may differ by an ulp where XLA contracts the field's multiply into
    the in-scan add (an FMA) while the start total adds it eagerly. Under
    ``dehnen_error`` the node's thresholds pick other lists: the FMM's error
    class (measured 2026-11-01: end state 5.3e-8, positions' gradient 2.7e-7 of
    its largest component, theta's 3.6e-12)."""
    engine, nodes = _engine(name)
    chain = _chain(name)
    assert [s.kind for s in chain.segments] == ["start", "carried"]
    bounds = _CHAIN_BOUNDS[name]
    end = np.asarray(engine.ends[1])
    want = np.asarray(chain.forward.state)
    state_err = float(np.max(np.abs(end - want)))
    _report(
        name,
        "shooting end vs chain",
        differing_elements=int(np.sum(end != want)),
        max_abs=state_err,
    )
    assert state_err <= bounds["state"]

    x0, v0, m, w = _system()

    def end_loss(xe: Any, ve: Any) -> Any:
        return jnp.sum(w * xe) + jnp.sum(w * ve)

    def composed(theta: Any, x: Any, v: Any, mass: Any) -> Any:
        first = engine(theta, x, v, mass, 0)
        second = engine(theta, first.positions, first.velocities, mass, 1)
        return end_loss(second.positions, second.velocities)

    value, (g_theta, g_x, g_v) = jax.jit(jax.value_and_grad(composed, (0, 1, 2)))(
        _theta(), x0, v0, m
    )
    res = chain.value_and_grad(
        lambda out, inputs: end_loss(out.positions, out.velocities),
        wrt=("positions", "velocities", "external_params"),
    )
    pairs = {
        "k": (g_theta["k"], res.grads["external_params"]["k"]),
        "positions": (g_x, res.grads["positions"]),
        "velocities": (g_v, res.grads["velocities"]),
    }
    errors = {}
    for key, (got, ref) in pairs.items():
        ref = np.asarray(ref)
        errors[key] = float(np.max(np.abs(np.asarray(got) - ref))) / float(
            np.max(np.abs(ref))
        )
    _report(
        name,
        "composed gradient vs chain",
        value_diff=float(value - res.value),
        relative=errors,
    )
    assert abs(float(value - res.value)) <= bounds["value"]
    for key, err in errors.items():
        assert err <= bounds["grad"], (key, err)


# ---------------------------------------------------------------------------
# FD through both segments.
# ---------------------------------------------------------------------------


def _shooting_loss(engine: Any) -> Any:
    """A shooting objective through both segments, node 0 fixed.

    ``L(theta, n1, fixed) = <w, x_1 + v_1> + <w2, n1_x - x_0> + <w2, n1_v - v_0>``
    with ``(x_k, v_k)`` segment k's end (the defect's sign is the seam's): the
    loss reads segment 1 from the free node and a linear defect term reads segment
    0, so theta reaches both segments. ``fixed`` holds node 0 and the masses, as
    arguments.
    """
    w = _system()[3]
    w2 = jnp.asarray(np.random.default_rng(5).normal(size=(N, 3)))

    def loss(theta: Any, n1: Any, fixed: dict) -> tuple:
        m = fixed["masses"]
        a = engine(theta, fixed["positions"], fixed["velocities"], m, 0)
        b = engine(theta, n1["positions"], n1["velocities"], m, 1)
        value = (
            jnp.sum(w * b.positions)
            + jnp.sum(w * b.velocities)
            + jnp.sum(w2 * (n1["positions"] - a.positions))
            + jnp.sum(w2 * (n1["velocities"] - a.velocities))
        )
        return value, (a.aux, b.aux)

    return loss


def _fixed() -> dict:
    x, v, m, _ = _system()
    return {"positions": x, "velocities": v, "masses": m}


def _node1(nodes: dict) -> dict:
    return {"positions": nodes["positions"][1], "velocities": nodes["velocities"][1]}


def _shooting_gradient(name: str) -> tuple:
    """The shooting loss's value and gradient in (theta, n1), once per lane."""
    key = ("gradient", name)
    if key not in _CACHE:
        engine, nodes = _engine(name)
        program = jax.jit(
            jax.value_and_grad(_shooting_loss(engine), argnums=(0, 1), has_aux=True)
        )
        begun = time.perf_counter()
        (value, auxes), grads = program(_theta(), _node1(nodes), _fixed())
        assert engine.check(auxes), engine.failed_segments
        _report(
            name,
            "shooting gradient (trace + compile + run)",
            seconds=time.perf_counter() - begun,
        )
        _CACHE[key] = (value, grads)
    return _CACHE[key]


class _Recording:
    """A provider whose differentiable step also sends its lists' digest to the
    host, for the identity check of the +-h runs (jaccpot's FD gate)."""

    def __init__(self, provider: Any, sink: list) -> None:
        self.provider = provider
        self.sink = sink

    def differentiable_step(self, x, m, params, aux):
        f_b = None if aux is None else aux[0]
        stop = jax.lax.stop_gradient
        prepared = self.provider.rebuild(
            stop(x),
            stop(m),
            force_scale=None if f_b is None else stop(f_b),
            keep_far_list=True,
        ).prepared
        jax.debug.callback(
            lambda d: self.sink.append(np.asarray(d)),
            _list_digest(prepared),
            ordered=True,
        )
        return self.provider.differentiable_step(x, m, params, aux)


def _list_digest(prepared: Any) -> Any:
    """Integer fingerprints of a topology's index arrays (jaccpot's)."""
    nl, far = prepared.neighbor_list, prepared.compact_far_pairs
    arrays = (
        prepared.tree.particle_indices,
        nl.offsets,
        nl.neighbors,
        far.sources,
        far.targets,
    )
    digests = [
        jnp.sum(
            jnp.asarray(a, jnp.int64).ravel()
            * (jnp.arange(a.size, dtype=jnp.int64) % 1_000_003 + 1)
        )
        for a in arrays
    ]
    return jnp.stack([*digests, jnp.asarray(far.far_pair_count, jnp.int64)])


def _recording(engine: Any) -> Any:
    """The shooting loss's jitted forward with every force's lists recorded."""
    sink: list = []
    recorded = copy.copy(engine)
    recorded.checkpoint = None
    recorded.plans = [
        dataclasses.replace(
            plan,
            segments=tuple(
                dataclasses.replace(s, provider=_Recording(s.provider, sink))
                for s in plan.segments
            ),
            _programs={},
        )
        for plan in engine.plans
    ]
    loss = _shooting_loss(recorded)

    @jax.jit
    def run(theta, n1, fixed):
        value, auxes = loss(theta, n1, fixed)
        return value, [aux[1] for aux in auxes]

    def call(theta: Any, n1: Any) -> tuple:
        sink.clear()
        value, bits = run(theta, n1, _fixed())
        jax.effects_barrier()
        return float(value), [int(b) for b in bits], np.stack(sink)

    return call


@pytest.mark.parametrize("name", _lanes(*_LANES))
def test_the_shooting_gradient_matches_finite_differences(name):
    """Central differences of the shooting loss (both segments) in theta and in
    node 1's positions, every force's lists (2 x (start + 2 steps)) identical on
    the +-h runs; relative error at most 1e-7 (jaccpot's FD steps)."""
    from jaccpot.runtime.fused_topology import OVERFLOW_PARAMS

    engine, nodes = _engine(name)
    value, (g_theta, g_n1) = _shooting_gradient(name)
    run = _recording(engine)
    theta, n1 = _theta(), _node1(nodes)
    value0, bits0, digests0 = run(theta, n1)
    assert bits0 == [0, 0] and value0 == pytest.approx(float(value), rel=1e-13)
    assert digests0.shape[0] == 2 * (1 + 2)
    d = jnp.asarray(np.random.default_rng(7).normal(size=(N, 3)))
    cases = {
        "k": (
            K_EXT * 1e-3,
            float(g_theta["k"]),
            lambda s: ({"k": theta["k"] + s}, n1),
        ),
        "node1_positions": (
            1e-7,
            float(jnp.sum(g_n1["positions"] * d)),
            lambda s: (theta, {**n1, "positions": n1["positions"] + s * d}),
        ),
    }
    errors = {}
    for which, (h, exact, moved) in cases.items():
        values = []
        for sign in (1.0, -1.0):
            val, bits, digests = run(*moved(sign * h))
            assert all(b & ~OVERFLOW_PARAMS == 0 for b in bits), (which, bits)
            assert np.array_equal(
                digests, digests0
            ), f"{which}: the {'+' if sign > 0 else '-'}h run rebuilt other lists"
            values.append(val)
        fd = (values[0] - values[1]) / (2.0 * h)
        errors[which] = abs(exact - fd) / abs(fd)
        assert exact != 0.0, which
        assert abs(exact - fd) <= 1e-7 * abs(fd), (which, exact, fd)
    _report(name, "FD relative error", **errors)


# ---------------------------------------------------------------------------
# Compile once; a forced cap failure re-plans that segment alone.
# ---------------------------------------------------------------------------


def _shrunk(plan: Any, factor: int = 4) -> Any:
    """``plan`` with its far-pair capacity cut ``factor``-fold (and the frozen
    gradient statics to match), below what its segment needs."""
    seg = plan.segments[0]
    caps = dict(seg.plan["walk_caps"])
    cap = caps["compact_far_pair_capacity"] // factor
    caps["compact_far_pair_capacity"] = cap
    prov = seg.provider
    provider = type(prov)(
        prov.engine,
        leaf_size=prov.leaf_size,
        max_order=prov.max_order,
        theta=prov.theta,
        runtime_overrides=prov.runtime_overrides,
        template=prov.template,
        grad_statics=dataclasses.replace(prov.grad_statics, num_far_pairs=cap),
    )
    shrunk = dataclasses.replace(
        seg, plan={**seg.plan, "walk_caps": caps}, provider=provider
    )
    return dataclasses.replace(plan, segments=(shrunk,), _programs={})


@pytest.mark.parametrize("name", _lanes(*_LANES))
def test_plans_compile_once_and_a_failed_segment_replans_alone(name):
    """M's augmented-Lagrangian driver, two Adam steps from moved theta and nodes:
    ``prepare`` before every step runs no pre-pass, the step is traced ONCE, every
    ``check`` passes. Then segment 1's far cap is cut below its need: ``check``
    fails for segment 1 alone, ``prepare`` re-plans segment 1 alone (segment 0
    keeps its plan object), the program traced on the replaced plan is refused,
    and a redo traced on the new plan passes (on the forward program: a gradient
    redo would compile the whole reverse again; the verdict is the same
    differentiable step's)."""
    engine, nodes = _engine(name)
    model = _model(engine)
    truth = _params(nodes)
    observed = engine.ends[1][:, 0].ravel()
    problem = InferenceProblem(
        forward=model,
        observable=lambda state: state["positions"].ravel(),
        likelihood=GaussianLikelihood(sigma=0.05),
        observed=observed,
    )
    d = jnp.asarray(np.random.default_rng(11).normal(size=(1, N, 3)))
    start = {
        "theta": {"k": truth["theta"]["k"] * (1.0 + 1e-3)},
        "nodes": {
            "positions": truth["nodes"]["positions"] + 1e-4 * d,
            "velocities": truth["nodes"]["velocities"] + 1e-3 * d,
        },
    }
    plans, checks = engine.stats["plans"], engine.stats["checks"]
    begun = time.perf_counter()
    result = AugmentedLagrangian(
        optimizer=optax.adam(1e-4), inner_steps=2, max_outer=1
    ).minimize(problem, start)
    _report(
        name,
        "driver: 2 steps + measurement",
        seconds=time.perf_counter() - begun,
        traces=result.num_traces,
        redos=result.num_redos,
    )
    assert result.num_traces == 1 and result.num_redos == 0
    assert engine.stats["plans"] == plans
    assert engine.stats["checks"] - checks == 3  # two steps and the measurement
    moved = float(
        jnp.max(
            jnp.abs(result.params["nodes"]["positions"] - truth["nodes"]["positions"])
        )
    )
    assert moved > 0.0
    _report(
        name,
        "host cost per iteration",
        check_s=engine.stats["check_s"] / engine.stats["checks"],
        prepare_s_no_replan=engine.stats["prepare_s"] / engine.stats["prepares"],
        node_moved=moved,
    )

    loss = _shooting_loss(engine)
    engine.plans[1] = _shrunk(engine.plans[1])
    engine.generation += 1
    engine._plan_ids[1] = engine.generation
    forward = jax.jit(loss)
    theta, n1, fixed = _theta(), _node1(nodes), _fixed()
    _, auxes = forward(theta, n1, fixed)
    assert not engine.check(auxes)
    assert engine.failed_segments == (1,) and engine.stale_segments == (1,)
    plan0, plans = engine.plans[0], engine.stats["plans"]
    engine.prepare(theta, nodes, fixed["masses"])
    assert engine.plans[0] is plan0 and engine.stats["plans"] == plans + 1
    _report(name, "re-plan of one segment", seconds=engine.stats["plan_s"][-1])
    with pytest.raises(RuntimeError, match="trace the objective again"):
        engine.check(forward(theta, n1, fixed)[1])
    # traced again: a NEW function (jax.jit of `loss` would reuse its trace)
    value, auxes = jax.jit(_shooting_loss(engine))(theta, n1, fixed)
    assert engine.check(auxes)
    want = float(_shooting_gradient(name)[0])
    _report(name, "redo value vs the gradient program's", diff=float(value) - want)
    assert float(value) == pytest.approx(want, rel=1e-12)


# ---------------------------------------------------------------------------
# The aux guard.
# ---------------------------------------------------------------------------


@pytest.mark.xdist_group("fused_segment_dehnen_error")
def test_reading_the_engines_aux_does_not_trip_the_aux_guard_dehnen_error():
    """Under ``dehnen_error`` (f_b a floating aux leaf): a loss reading the
    engine's aux traces its gradient (the aux is behind ``stop_gradient``); the
    same loss on the raw rollout aux is refused by jaccpot's guard. Traced only
    (``lower``), nothing compiled."""
    engine, _ = _engine("dehnen_error")
    x0, v0, m, w = _system()

    def through_engine(theta: Any, mass: Any) -> Any:
        end = engine(theta, x0, v0, mass, 0)
        return jnp.sum(w * end.positions) + jnp.sum(end.aux[0])

    jax.jit(jax.grad(through_engine)).lower(_theta(), m)

    plan = engine.plans[0]

    def through_raw(theta: Any) -> Any:
        out = plan.rollout({**plan.inputs(), "external_params": theta})
        return jnp.sum(w * out.positions) + jnp.sum(out.aux[0])

    with pytest.raises(ValueError, match="reads the rollout's aux"):
        jax.jit(jax.grad(through_raw)).lower(_theta())


# ---------------------------------------------------------------------------
# Forward mode (jaccpot ad_mode="forward", main 1bb30620).
# ---------------------------------------------------------------------------


@pytest.mark.xdist_group("fused_segment_plummer")
def test_forward_mode_through_a_fused_segment_plummer():
    """``jax.jvp`` in theta through segment 0 on the fused lane, both routes: the
    planned engine traced inside ``ad_mode_override("forward")`` (``ad_mode=None``,
    its ``checkpoint="step"``) and a copy with ``ad_mode="forward"`` (the plan's
    rollout under the override, ``checkpoint=None``) -- same plans, no pre-pass.
    The primal is the runner's end from node 0 bitwise, the two tangents agree to
    round-off, and the tangent is the reverse gradient's transpose (the engine's
    ``jax.grad``) and matches central differences of the same plan."""
    from jaccpot.runtime.grad_options import ad_mode_override

    engine, _ = _engine("plummer")
    forward = copy.copy(engine)
    forward.ad_mode = "forward"
    x0, v0, m, w = _system()

    def loss_of(eng: Any) -> Any:
        def loss(theta: Any) -> Any:
            end = eng(theta, x0, v0, m, 0)
            return jnp.sum(w * end.positions) + jnp.sum(w * end.velocities), end

        return loss

    d = {"k": jnp.asarray(1.0, jnp.float64)}
    begun = time.perf_counter()
    with ad_mode_override("forward"):
        (value, end), (tangent, _) = jax.jit(
            lambda t, dt: jax.jvp(loss_of(engine), (t,), (dt,))
        )(_theta(), d)
    (value_f, _), (tangent_f, _) = jax.jit(
        lambda t, dt: jax.jvp(loss_of(forward), (t,), (dt,))
    )(_theta(), d)
    _report(
        "plummer", "forward mode (2 jvp programs)", seconds=time.perf_counter() - begun
    )
    from mimirax.adapters.jaccpot import _gradient_ok

    assert _gradient_ok(end.aux[1])
    got = jnp.stack([end.positions, end.velocities], axis=1)
    assert _bitwise(got, engine.plans[0].forward.state) == 0
    assert _bitwise(value_f, value) == 0
    assert abs(float(tangent_f - tangent)) <= 1e-13 * abs(float(tangent))
    grad = jax.jit(jax.grad(lambda t: loss_of(engine)(t)[0]))(_theta())
    _report(
        "plummer",
        "forward vs reverse",
        tangent=float(tangent),
        reverse=float(grad["k"]),
    )
    assert abs(float(tangent) - float(grad["k"])) <= 1e-11 * abs(float(grad["k"]))
    h = 1e-6
    value_at = jax.jit(lambda t: loss_of(engine)(t)[0])
    fd = (value_at({"k": K_EXT + h}) - value_at({"k": K_EXT - h})) / (2 * h)
    assert abs(float(tangent - fd)) <= 1e-7 * abs(float(fd))
