"""Segments in parallel on jaccpot: the S10 session-2 gates. ``mimirax[jaccpot]``.

One value-and-gradient program per segment, segment ``k`` on ``devices[k %
len(devices)]`` (:class:`~mimirax.inference.parallel.ParallelSegmentObjective`,
``AugmentedLagrangian(devices=...)``), against the single program of session 1.

CPU (CI's solver job): jaccpot's direct sum (``differentiable_rollout``,
``checkpoint="step"``), float64, the Plummer-field system of
``test_multiple_shooting.py``. Two devices when the process has them (run it with
``XLA_FLAGS=--xla_force_host_platform_device_count=2``), else one device twice.

GPU (manual, ``MIMIRAX_RUN_MANUAL_TESTS=1``, two GPUs: cards 6 + 7 through
``on-both-reserved-cards``; functional only, no timing): the direct sum in float64
and jaccpot's fused lane (``FusedSegmentEngine``, ``preset="galaxy"``, float32, a
disc in nornax's Miyamoto-Nagai + NFW field, 2 segments): the parallel gradient
against the same segments on one device and against the one program, the plans'
arrays on their segment's device, the driver, and a failed verdict on one segment
re-planning that segment alone. Run with XLA's fusion autotuner off (jaccpot's
production launch flag), so one program compiles to the same code on both cards::

    XLA_FLAGS=--xla_gpu_experimental_enable_fusion_autotuner=false \\
    MIMIRAX_RUN_MANUAL_TESTS=1 JAX_PLATFORMS=cuda on-both-reserved-cards \\
        python -m pytest -o addopts="" -s tests/integration/test_parallel_segments.py

Measured numbers are quoted in the tests (this machine, jax 0.11.2, jaccpot
``s10/fused-segment-engine`` 5385b164).
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from mimirax import (
    AugmentedLagrangian,
    GaussianLikelihood,
    InferenceProblem,
    MultipleShootingForwardModel,
    ParallelSegmentObjective,
    augmented_lagrangian_value,
)

pytest.importorskip("jaccpot", reason="mimirax[jaccpot] is not installed")
pytest.importorskip(
    "jaccpot.runtime.grad_rollout", reason="jaccpot predates differentiable_rollout"
)

from mimirax.adapters.jaccpot import DirectSumSegmentEngine  # noqa: E402

TRUTH = {"log_mass": jnp.log(jnp.asarray(3.0)), "log_scale": jnp.asarray(0.0)}
START = {"log_mass": jnp.asarray(1.2), "log_scale": jnp.asarray(0.25)}
DT = 0.01
SOFTENING = 0.1


def plummer(x, t, args):
    """A Plummer sphere's field, nornax's ``f(x, t, args)`` convention."""
    del t
    r2 = jnp.sum(x * x, axis=-1, keepdims=True) + jnp.exp(2.0 * args["log_scale"])
    return -jnp.exp(args["log_mass"]) * x * r2**-1.5


def _phase_space(state):
    return jnp.concatenate([state["positions"], state["velocities"]], -1).reshape(-1)


def _devices(platform: str | None = None) -> tuple:
    devices = jax.devices() if platform is None else jax.devices(platform)
    return (devices[0], devices[1] if len(devices) > 1 else devices[0])


def _direct_problem(num_segments: int, n: int = 32, steps: int = 10) -> tuple:
    k_x, k_v = jax.random.split(jax.random.PRNGKey(1))
    x0 = jax.random.normal(k_x, (n, 3))
    v0 = 0.5 * jax.random.normal(k_v, (n, 3))
    masses = jnp.full((n,), 1.0 / n)
    engine = DirectSumSegmentEngine(
        segment_steps=(steps,) * num_segments, dt=DT, field=plummer, softening=SOFTENING
    )
    model = MultipleShootingForwardModel(engine, masses, num_segments, x0, v0)
    observed = _phase_space(model({"theta": TRUTH, "nodes": model.init_nodes(TRUTH)}))
    problem = InferenceProblem(
        model, _phase_space, GaussianLikelihood(sigma=0.01), observed
    )
    nodes = jax.tree_util.tree_map(
        lambda a: a + 0.01 * jnp.sin(jnp.arange(a.size).reshape(a.shape)),
        model.init_nodes(START),
    )
    lam = 0.1 * jnp.cos(jnp.arange((num_segments - 1) * n * 6.0))
    return problem, {"theta": START, "nodes": nodes}, (lam.reshape(-1, n, 6),)


def _one_program(problem, params, multipliers, mu, closure=False):
    """The augmented Lagrangian and its gradient in ONE jitted program."""
    if closure:  # session 1: the data closed over

        def closed(p):
            return augmented_lagrangian_value(problem, p, multipliers, mu)[0]

        return jax.jit(jax.value_and_grad(closed))(params)

    def lagrangian(p, data):
        bound = problem.with_data_arrays(data)
        return augmented_lagrangian_value(bound, p, multipliers, mu)[0]

    return jax.jit(jax.value_and_grad(lagrangian))(params, problem.data_arrays())


def _differ(got, want) -> tuple:
    """``(elements that differ, max |difference| / max |want|)`` over a pytree."""
    count, worst = 0, 0.0
    for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype
        if a.size == 0:
            continue
        count += int(np.sum(a != b))
        scale = max(float(np.max(np.abs(b))), 1e-300)
        worst = max(worst, float(np.max(np.abs(a - b))) / scale)
    return count, worst


def _report(what: str, **values: Any) -> None:
    print(f"\n[parallel-segments] {what}: {values}")


# ---------------------------------------------------------------------------
# CPU: the direct sum.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_segments", [2, 4])
def test_direct_sum_parallel_step_equals_the_one_program(num_segments) -> None:
    """Value and gradient of the augmented Lagrangian, N = 32, 10-step segments.

    Measured (CPU, float64; 1 and 2 forced devices alike): the value and EVERY
    gradient bitwise the one program's (data as arguments), for 2 and for 4
    segments -- jaccpot's segments are ``custom_vjp`` s whose hand-written
    reverses compile to the same code in either program, and each node's
    gradient is a sum of two terms. Against session 1's program (the data
    closed over): the value and the theta gradient bitwise, the node gradients
    not -- 40 of 384 elements by <= 2.7e-16 relative (2 segments), 101 of 1152
    by <= 5.1e-16 (4 segments): with the masses a constant (here 1/N, a splat)
    XLA folds them into the direct sum's arithmetic; the initial condition and
    the data as constants change nothing (each tried alone).
    """
    problem, params, lam = _direct_problem(num_segments)
    mu = jnp.asarray(10.0)
    value, grads = _one_program(problem, params, lam, mu)
    objective = ParallelSegmentObjective(problem, _devices())
    got_value, got_grads, _ = objective.value_and_grad(params, lam, mu)
    assert _differ(got_value, value) == (0, 0.0)
    assert _differ(got_grads, grads) == (0, 0.0)
    old_value, old_grads = _one_program(problem, params, lam, mu, closure=True)
    _report(
        f"S={num_segments}: one program, data as arguments vs closed over",
        value=_differ(value, old_value),
        theta=_differ(grads["theta"], old_grads["theta"]),
        nodes=_differ(grads["nodes"], old_grads["nodes"]),
    )
    assert _differ(value, old_value) == (0, 0.0)
    assert _differ(grads["theta"], old_grads["theta"]) == (0, 0.0)
    assert _differ(grads["nodes"], old_grads["nodes"])[1] < 1e-15


@pytest.mark.parametrize("ambient", [False, True], ids=["engine", "override"])
def test_direct_sum_theta_jacobian_by_forward_mode(ambient) -> None:
    """The ``"jacfwd"`` programs (jaccpot ``ad_mode="forward"``, main 1bb30620)
    on the direct sum, 3 segments: ``df/dtheta`` and ``d defects / d theta``
    against reverse mode on the one program (the default ``checkpoint="step"``
    engine). The forward-mode engine is either ``ad_mode="forward"`` or the
    default engine dispatched inside ``ad_mode_override("forward")``; the
    same objective's reverse programs run beside it, unchanged."""
    from jaccpot.runtime.grad_options import ad_mode_override

    problem, params, lam = _direct_problem(3)
    if not ambient:
        engine = dataclasses.replace(problem.forward.engine, ad_mode="forward")
        model = dataclasses.replace(problem.forward, engine=engine)
        problem = dataclasses.replace(problem, forward=model)
    objective = ParallelSegmentObjective(problem, _devices())
    if ambient:
        with ad_mode_override("forward"):
            f, grad, defects, jac, _ = objective.theta_jacobian(params)
    else:
        f, grad, defects, jac, _ = objective.theta_jacobian(params)

    def evaluate(theta):
        return problem.evaluate({"theta": theta, "nodes": params["nodes"]})

    reference = evaluate(params["theta"])
    assert _differ(f, reference.negative_log_posterior)[1] < 1e-14
    assert _differ(defects, reference.defects[0])[1] < 1e-14
    want_grad = jax.jit(jax.grad(lambda th: evaluate(th).negative_log_posterior))(
        params["theta"]
    )
    want_jac = jax.jit(jax.jacrev(lambda th: evaluate(th).defects[0]))(params["theta"])
    _report(
        f"theta block by jacfwd ({'override' if ambient else 'engine'})",
        grad=_differ(grad, want_grad),
        jac=_differ(jac, want_jac),
    )
    assert _differ(grad, want_grad)[1] < 1e-12
    assert _differ(jac, want_jac)[1] < 1e-12
    # the reverse programs of the same objective, after the forward ones
    value, grads = _one_program(problem, params, lam, jnp.asarray(10.0))
    got_value, got_grads, _ = objective.value_and_grad(params, lam, jnp.asarray(10.0))
    assert _differ(got_grads, grads)[1] < 1e-14
    assert objective.num_jacfwd_traces == 2 and objective.num_traces == 2


@pytest.mark.parametrize("num_segments", [2, 4])
def test_direct_sum_driver_in_parallel(num_segments) -> None:
    """Two outer x 5 inner Adam steps; params, traces and the histories.

    Measured (alone): 2 segments bitwise; 4 segments within 1.8e-16 relative (41
    elements of the parameters after 10 steps, one theta component by 1 ulp): the
    multiplier and penalty terms of three defects are summed per segment, then
    over segments, where the one program sums each over the stacked defects.
    Equal segments share one program: 2 traces (with / without a next node and
    the data) for any S >= 2.
    """
    problem, params, _ = _direct_problem(num_segments)
    kwargs = dict(optimizer=optax.adam(1e-2), inner_steps=5, max_outer=2, mu0=10.0)
    one = AugmentedLagrangian(**kwargs).minimize(problem, params)
    many = AugmentedLagrangian(devices=_devices(), **kwargs).minimize(problem, params)
    diff = _differ(many.params, one.params)
    _report(f"S={num_segments}: driver, parallel vs one program", params=diff)
    # bitwise for S = 2 when run alone; asserted to round-off, because a
    # pytest-xdist worker that ran other tests first compiled one of the two
    # programs differently (15 elements by 2.0e-17 relative)
    assert diff[1] < 1e-14
    assert _differ(many.violation_trace, one.violation_trace)[1] < 1e-13
    assert one.num_traces == 1 and many.num_traces == 2
    assert many.num_redos == 0


def test_direct_sum_recovery_in_parallel_matches_one_program() -> None:
    """The defects vanish under the driver in parallel, as in one program.

    Perturbed nodes (1e-2) and theta, 4 segments of 20 steps, N = 32, Adam: the
    two modes take the same steps to round-off and end at the same estimate
    (theta 8.0e-15 relative apart; the violation 2.8e-2 -> 5.7e-3 over 3 x 60
    steps).
    """
    problem, params, _ = _direct_problem(4, steps=20)
    kwargs = dict(
        optimizer=optax.adam(2e-2), inner_steps=60, max_outer=3, mu0=100.0, tol=1e-6
    )
    one = AugmentedLagrangian(**kwargs).minimize(problem, params)
    many = AugmentedLagrangian(devices=_devices(), **kwargs).minimize(problem, params)
    assert many.outer_steps.tolist() == one.outer_steps.tolist()
    assert float(many.outer_violation[-1]) < float(many.outer_violation[0])
    assert _differ(many.params["theta"], one.params["theta"])[1] < 1e-10
    _report(
        "recovery, 4 x 20 steps",
        violation=many.outer_violation.tolist(),
        theta=jax.tree_util.tree_map(float, many.params["theta"]),
        theta_vs_one=_differ(many.params["theta"], one.params["theta"]),
    )


# ---------------------------------------------------------------------------
# GPU (manual): cards 6 + 7.
# ---------------------------------------------------------------------------


def _gpus() -> list:
    try:
        return jax.devices("gpu")
    except RuntimeError:
        return []


gpu = pytest.mark.skipif(
    os.environ.get("MIMIRAX_RUN_MANUAL_TESTS") != "1" or len(_gpus()) < 2,
    reason="manual: needs two GPUs (on-both-reserved-cards) and "
    "MIMIRAX_RUN_MANUAL_TESTS=1",
)


@gpu
def test_gpu_direct_sum_parallel_equals_one_program() -> None:
    """Float64 direct sum, 2 and 4 segments on two GPUs against one program on one.

    Measured (cards 6 + 7, N = 256, 10-step segments, 2026-11-01): two cards
    against the same segment programs on one card bitwise (S = 2 and 4); against
    the one program the value bitwise and the gradient bitwise for S = 2, one
    element by 1.6e-16 relative for S = 4; the driver (2 x 3 Adam steps) bitwise
    for both; 2 segment traces against the one program's 1.
    """
    gpus = _gpus()
    for num_segments in (2, 4):
        problem, params, lam = _direct_problem(num_segments, n=256)
        mu = jnp.asarray(10.0)
        with jax.default_device(gpus[0]):
            value, grads = _one_program(problem, params, lam, mu)
        objective = ParallelSegmentObjective(problem, gpus[:2])
        got_value, got_grads, _ = objective.value_and_grad(params, lam, mu)
        on_one = ParallelSegmentObjective(problem, gpus[:1])
        one_value, one_grads, _ = on_one.value_and_grad(params, lam, mu)
        _report(
            f"GPU direct sum S={num_segments}",
            value_vs_one_program=_differ(got_value, value),
            grads_vs_one_program=_differ(got_grads, grads),
            grads_two_cards_vs_one_card=_differ(got_grads, one_grads),
            segment_devices=[str(d) for d in objective.segment_devices],
        )
        assert _differ(got_value, value)[1] < 1e-14
        assert _differ(got_grads, grads)[1] < 1e-13
        assert _differ(got_grads, one_grads) == (0, 0.0)
        kwargs = dict(optimizer=optax.adam(1e-2), inner_steps=3, max_outer=2)
        with jax.default_device(gpus[0]):
            one = AugmentedLagrangian(**kwargs).minimize(problem, params)
        many = AugmentedLagrangian(devices=gpus[:2], **kwargs).minimize(problem, params)
        _report(
            f"GPU direct sum S={num_segments} driver",
            params=_differ(many.params, one.params),
            traces=(one.num_traces, many.num_traces),
        )
        assert _differ(many.params, one.params)[1] < 1e-13
        assert many.num_traces == 2


# --- the fused lane ------------------------------------------------------------

FUSED_N = int(os.environ.get("MIMIRAX_PARALLEL_FUSED_N", "20000"))
FUSED_STEPS = 3
FUSED_DT = 0.01
FUSED_TRUTH = (float(np.log(5.0)), float(np.log(3.0)))


def _galaxy_field(x, t, theta):
    """nornax's Miyamoto-Nagai disc (fixed) plus an NFW halo ``(log M, log r_s)``."""
    from nornax import potentials

    field = potentials.CompositePotential(
        disc=potentials.MiyamotoNagai(), halo=potentials.NFW()
    )
    disc = potentials.MiyamotoNagaiParams(M=1.0, a=1.0, b=0.1)
    halo = potentials.NFWParams(
        M=jnp.exp(theta["log_halo_mass"]), r_s=jnp.exp(theta["log_halo_scale"])
    )
    return field(x, t, {"disc": disc, "halo": halo})


def _theta32(values) -> dict:
    return {
        "log_halo_mass": jnp.asarray(values[0], jnp.float32),
        "log_halo_scale": jnp.asarray(values[1], jnp.float32),
    }


def _disc(n: int) -> tuple:
    """Near-circular orbits at R in [0.5, 3], float32 (track F's system)."""
    rng = np.random.default_rng(3)
    radius = rng.uniform(0.5, 3.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    z = 0.05 * rng.normal(size=n)
    x = np.stack([radius * np.cos(phi), radius * np.sin(phi), z], -1)
    theta = {k: jnp.asarray(v, jnp.float64) for k, v in _theta32(FUSED_TRUTH).items()}
    a = np.asarray(_galaxy_field(jnp.asarray(x), 0.0, theta))
    radial = -(a[:, 0] * np.cos(phi) + a[:, 1] * np.sin(phi))
    speed = np.sqrt(radius * radial)
    v = np.stack([-speed * np.sin(phi), speed * np.cos(phi), np.zeros(n)], -1)
    m = np.full((n,), 1e-2 / n)
    return tuple(jnp.asarray(a_, jnp.float32) for a_ in (x, v, m))


_FUSED: dict = {}


def _fused_setup(gpus) -> dict:
    """The fused system, planned once per process (both GPU tests use it)."""
    if "setup" not in _FUSED:
        _FUSED["setup"] = _build_fused(gpus)
    return _FUSED["setup"]


def _build_fused(gpus) -> dict:
    from jaccpot import FastMultipoleMethod
    from jaccpot.production import ProductionRunner, SegmentConfig

    from mimirax.adapters.jaccpot import FusedSegmentEngine, rows_field

    with jax.default_device(gpus[0]):
        x, v, m = _disc(FUSED_N)
        eps = 0.015 * (FUSED_N / 1e5) ** (-1.0 / 3.0)
        runner = ProductionRunner(
            FastMultipoleMethod(preset="galaxy", softening=eps),
            segment=SegmentConfig(steps=FUSED_STEPS, diagnostics_every=0),
            external_acceleration_fn=rows_field(_galaxy_field),
        )
        engine = FusedSegmentEngine(runner, (FUSED_STEPS, FUSED_STEPS), dt=FUSED_DT)
        truth = _theta32(FUSED_TRUTH)
        nodes = engine.propagate(truth, x, v, m)
        end = np.asarray(engine.ends[-1], np.float64)
        observed = jnp.asarray(
            np.concatenate([end[:, 0], end[:, 1]], -1).reshape(-1)
            + 1e-3 * np.random.default_rng(7).normal(size=6 * FUSED_N)
        )
        model = MultipleShootingForwardModel(engine, m, 2, x, v)

        def observable(state):
            rows = jnp.concatenate([state["positions"], state["velocities"]], -1)
            return rows.astype(jnp.float64).reshape(-1)

        problem = InferenceProblem(
            model, observable, GaussianLikelihood(sigma=1e-3), observed
        )
        d = jnp.asarray(np.random.default_rng(11).normal(size=(1, FUSED_N, 3)))
        params = {
            "theta": _theta32((FUSED_TRUTH[0] + 1e-3, FUSED_TRUTH[1] - 1e-3)),
            "nodes": {
                "positions": nodes["positions"][1:] + (1e-5 * d).astype(jnp.float32),
                "velocities": nodes["velocities"][1:] + (1e-5 * d).astype(jnp.float32),
            },
        }
        lam = (jnp.asarray(1e-2 * np.cos(np.arange(FUSED_N * 6.0))).reshape(1, -1, 6),)
        lam = (lam[0].astype(jnp.float32),)
    return dict(engine=engine, problem=problem, params=params, lam=lam, x=x, v=v, m=m)


def _shrunk(plan: Any, factor: int = 4) -> Any:
    """``plan`` with its far-pair capacity cut ``factor``-fold (and the frozen
    gradient statics to match), below what its segment needs (session 1's
    ``test_fused_segment_engine._shrunk``)."""
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


def _largest_constant(text: str) -> int:
    """The element count of the largest non-splat ``stablehlo.constant`` in ``text``."""
    import re

    largest = 0
    pattern = (
        r'stablehlo\.constant dense<("0x[0-9A-Fa-f]+"|\[[^>]*?\])> : tensor<([^>]*)>'
    )
    for _, shape in re.findall(pattern, text):
        dims = [int(d) for d in shape.split("x")[:-1] if d.isdigit()]
        largest = max(largest, int(np.prod(dims)) if dims else 1)
    return largest


def _device_arrays(root: Any, skip: tuple = ("engine",)) -> list:
    """Every ``jax.Array`` reachable from ``root`` (attributes, containers), with
    its path, skipping attributes named in ``skip``."""
    found, seen = [], set()

    def walk(obj: Any, path: str) -> None:
        if id(obj) in seen or obj is None:
            return
        seen.add(id(obj))
        if isinstance(obj, jax.Array):
            found.append(
                (path, obj.shape, str(obj.dtype), sorted(obj.devices(), key=str))
            )
        elif isinstance(obj, dict):
            for k, val in obj.items():
                walk(val, f"{path}[{k!r}]")
        elif isinstance(obj, (list, tuple)):
            for i, val in enumerate(obj):
                walk(val, f"{path}[{i}]")
        elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            for f in dataclasses.fields(obj):
                if f.name not in skip:
                    walk(getattr(obj, f.name), f"{path}.{f.name}")
        elif hasattr(obj, "__dict__") and not isinstance(obj, type):
            for k, val in vars(obj).items():
                if k not in skip and not callable(val):
                    walk(val, f"{path}.{k}")

    walk(root, "plan")
    return found


@gpu
def test_gpu_fused_lane_segments_on_two_cards() -> None:
    """The fused lane, float32, ``preset="galaxy"`` (``dehnen_error``), N = 2e4, 2
    segments x 3 steps, segment 0 on card 6 and segment 1 on card 7.

    * the parallel value and gradient against the same per-segment programs on
      ONE card and against the one program (session 1's step) on one card;
    * the plans: what device arrays a ``GradientPlan`` holds, and each segment's
      arrays (dt, G / softening, the f_b seed) placed on its card;
    * a segment program's HLO holds no large constant (the seeds, the masses,
      node 0 and the data are arguments).

    Measured (cards 6 + 7, 2026-11-01, fusion autotuner off): two cards against
    the same programs on one card BITWISE, and against the one program (both
    segments and the sum in one jitted value-and-gradient on card 6) BITWISE too
    (value 5.98e4, theta and node gradients); a plan's device arrays are its
    scalar ``dt`` and the pre-pass's ``external_params`` (the field's theta,
    replaced by the traced theta), on the pre-pass's device -- the f_b seeds are
    host numpy, the providers and segment plans hold no device buffer; each
    segment's arrays placed on its card; the largest constant of a segment
    program 4 elements, 20000 (= N, the seed) with the plan's own arrays. 17 min
    for the test (compiles; shared host).
    """
    gpus = _gpus()[:2]
    s = _fused_setup(gpus)
    engine, problem, params, lam = s["engine"], s["problem"], s["params"], s["lam"]
    mu = jnp.asarray(10.0)

    # what a plan holds on a device (the pre-pass's forward is dropped)
    payload = [_device_arrays(plan) for plan in engine.plans]
    _report("device arrays in each GradientPlan (engine skipped)", payload=payload)

    two = ParallelSegmentObjective(problem, gpus)
    value, grads, evaluation = two.value_and_grad(params, lam, mu)
    assert engine.check(evaluation.aux)
    placed = {
        key: sorted(
            {str(d) for leaf in jax.tree_util.tree_leaves(val) for d in leaf.devices()}
        )
        for key, val in engine._placed.items()
    }
    _report("segment arrays placed", placed=placed)
    for k, device in enumerate(gpus):
        arrays = engine.segment_arrays(k, device=device)
        leaves = jax.tree_util.tree_leaves(arrays)
        assert leaves and all(leaf.devices() == {device} for leaf in leaves)
        assert all(
            next(iter(leaf.devices())) == device
            for leaf in jax.tree_util.tree_leaves(evaluation.aux[k])
        )
    # the segment programs hold no [N]-sized constant. (Their text is ~1e8
    # characters at any N: the Pallas kernels' Triton IR in the custom calls'
    # backend_config, ~650 of them; not data.) With the plan's own arrays (no
    # ``arrays``) segment 0's f_b seed is an [N] constant.
    largest = {}
    for k, device in enumerate(gpus):
        states = two._states(params)
        args = (
            jax.device_put(params["theta"], device),
            jax.device_put(states[k], device),
            jax.device_put(states[k + 1], device) if k == 0 else None,
            jax.device_put(lam[0][0], device) if k == 0 else None,
            jax.device_put(mu, device),
            two._masses[device],
            two._observations[k],
            engine.segment_arrays(k, device=device),
        )
        program = two._program("value_and_grad", k)
        largest[k] = _largest_constant(program.lower(*args).as_text())
        if k == 0:
            largest["0, plan's own arrays"] = _largest_constant(
                program.lower(*args[:-1], None).as_text()
            )
    _report("largest constant in a segment program (elements)", largest=largest)
    assert largest[0] < FUSED_N and largest[1] < FUSED_N
    assert largest["0, plan's own arrays"] >= FUSED_N

    one_card = ParallelSegmentObjective(problem, gpus[:1])
    value_1, grads_1, _ = one_card.value_and_grad(params, lam, mu)
    with jax.default_device(gpus[0]):
        value_p, grads_p = _one_program(problem, params, lam, mu)
    _report(
        "fused gradient: two cards vs one card (same programs)",
        value=_differ(value, value_1),
        grads=_differ(grads, grads_1),
    )
    _report(
        "fused gradient: two cards vs the one program",
        value=_differ(value, value_p),
        theta=_differ(grads["theta"], grads_p["theta"]),
        nodes=_differ(grads["nodes"], grads_p["nodes"]),
        value_abs=float(value),
    )
    assert _differ(grads, grads_1)[1] < 1e-5
    assert _differ(value, value_p)[1] < 1e-5
    assert _differ(grads["theta"], grads_p["theta"])[1] < 1e-4
    assert _differ(grads["nodes"], grads_p["nodes"])[1] < 1e-4


@gpu
def test_gpu_fused_lane_driver_and_a_failed_verdict_on_two_cards() -> None:
    """The driver (Adam) on two cards: one trace per segment, no re-plan; then a far
    cap cut below its need in segment 1: ``check`` fails for segment 1 alone,
    ``prepare`` re-plans it alone, only its program is traced again, the redo
    passes. The system of :func:`test_gpu_fused_lane_segments_on_two_cards`.

    Measured (cards 6 + 7, 2026-11-01): 2 Adam steps, 2 traces, no redo, no
    pre-pass (L 59793.40 -> 58843.31, violation 5.0e-5 / 6.0e-5 absolute); the cut
    cap: 1 redo, 1 pre-pass (segment 1's), 3 traces (segment 0 once, segment 1 on
    the cut plan and on its new one). 19.5 min for the test (compiles).
    """
    gpus = _gpus()[:2]
    s = _fused_setup(gpus)
    engine, problem, params = s["engine"], s["problem"], s["params"]
    plans = engine.stats["plans"]

    # the driver in parallel: no re-plan, one trace per segment
    driver = AugmentedLagrangian(
        optimizer=optax.adam(1e-5), inner_steps=2, max_outer=1, devices=gpus
    )
    result = driver.minimize(problem, params)
    _report(
        "driver, 2 Adam steps on two cards",
        traces=result.num_traces,
        redos=result.num_redos,
        lagrangian=result.lagrangian_trace.tolist(),
        violation=result.violation_trace.tolist(),
    )
    assert engine.stats["plans"] == plans
    assert result.num_traces == 2 and result.num_redos == 0

    # segment 1's far cap cut below its need: re-planned alone, redone
    engine.plans[1] = _shrunk(engine.plans[1])
    engine.generation += 1
    engine._plan_ids[1] = engine.generation
    plan0 = engine.plans[0]
    result = AugmentedLagrangian(
        optimizer=optax.adam(1e-5), inner_steps=1, max_outer=1, devices=gpus
    ).minimize(problem, params)
    _report(
        "a failed verdict on segment 1",
        traces=result.num_traces,
        redos=result.num_redos,
        plans=engine.stats["plans"] - plans,
    )
    assert result.num_redos == 1
    assert engine.plans[0] is plan0
    assert engine.stats["plans"] == plans + 1
    # segment 0 once; segment 1 on the shrunk plan, then on its new plan
    assert result.num_traces == 3
