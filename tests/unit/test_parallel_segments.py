"""Segments in parallel (S10 session 2) on the pure-JAX segment engine: no solver needed.

:class:`~mimirax.inference.parallel.ParallelSegmentObjective` and
``AugmentedLagrangian(devices=...)`` against the single program they replace,
on :class:`~mimirax.testing.VerletSegmentEngine`; the problem's data as
arguments; the split over segments and its refusals. In-process the tests use
the devices this process has (two forced CPU devices, or one device twice:
the placement code runs either way); ``test_two_forced_cpu_devices`` runs the
same comparison in a subprocess with
``XLA_FLAGS=--xla_force_host_platform_device_count=2``. The direct-sum and GPU
gates are ``tests/integration/test_parallel_segments.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from mimirax import (
    AugmentedLagrangian,
    FunctionConstraint,
    GaussianLikelihood,
    GaussianPrior,
    IdentityObservable,
    InferenceProblem,
    MultipleShootingForwardModel,
    ParallelSegmentObjective,
    SegmentResult,
    SegmentSplit,
    augmented_lagrangian_value,
)
from mimirax.shooting import BoundSegmentEngine, segment_key
from mimirax.testing import LinearForwardModel, VerletSegmentEngine


def harmonic(x, t, args):
    """``a = -k x``, nornax's ``f(x, t, args)`` convention."""
    del t
    return -args["k"] * x


N = 8


def _system(seed=0):
    k_x, k_v = jax.random.split(jax.random.PRNGKey(seed))
    positions = jax.random.normal(k_x, (N, 3))
    velocities = 0.3 * jax.random.normal(k_v, (N, 3))
    # unequal masses: a splat constant is folded differently from an argument
    masses = (1.0 + 0.1 * jnp.arange(N)) / N
    return positions, velocities, masses


def _devices():
    """Two devices: the first two of this process, or its only one twice."""
    devices = jax.devices()
    return (devices[0], devices[1] if len(devices) > 1 else devices[0])


def _phase_space(state):
    return jnp.concatenate([state["positions"], state["velocities"]], -1).reshape(-1)


def _segment_wise(state):
    """``[S, N, 3]`` ends -> ``[S, 6 N]``: one row per segment."""
    rows = jnp.concatenate([state["positions"], state["velocities"]], -1)
    return rows.reshape(rows.shape[0], -1)


def _problem(
    num_segments=3,
    steps=4,
    output="final",
    free_initial=False,
    priors=(),
    sigma=0.05,
    engine=None,
):
    x0, v0, masses = _system()
    if engine is None:
        engine = VerletSegmentEngine(
            segment_steps=(steps,) * num_segments, dt=0.02, field=harmonic
        )
    model = MultipleShootingForwardModel(
        engine,
        masses,
        num_segments,
        x0,
        v0,
        free_initial=free_initial,
        output=output,
    )
    truth = {"k": jnp.asarray(1.0)}
    observable = _phase_space if output == "final" else _segment_wise
    observed = observable(model({"theta": truth, "nodes": model.init_nodes(truth)}))
    observed = observed + 0.01 * jnp.sin(jnp.arange(observed.size)).reshape(
        observed.shape
    )
    problem = InferenceProblem(
        model, observable, GaussianLikelihood(sigma=sigma), observed, priors=priors
    )
    start = {"k": jnp.asarray(1.2)}
    nodes = jax.tree_util.tree_map(
        lambda a: a + 0.01 * jnp.cos(jnp.arange(a.size).reshape(a.shape)),
        model.init_nodes(start),
    )
    return problem, {"theta": start, "nodes": nodes}


def _multipliers(problem, params):
    shape = jax.eval_shape(problem.evaluate, params).defects[0].shape
    return (0.1 * jnp.sin(jnp.arange(int(np.prod(shape)))).reshape(shape),)


def _one_program(problem, params, multipliers, mu):
    """The augmented Lagrangian and its gradient in ONE program, data as arguments."""

    def lagrangian(p, data):
        bound = problem.with_data_arrays(data)
        return augmented_lagrangian_value(bound, p, multipliers, mu)[0]

    return jax.jit(jax.value_and_grad(lagrangian))(params, problem.data_arrays())


def _max_relative(got, want):
    worst = 0.0
    for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype
        if a.size == 0:
            continue
        scale = max(float(np.max(np.abs(b))), 1e-300)
        worst = max(worst, float(np.max(np.abs(a - b))) / scale)
    return worst


# --- the split ------------------------------------------------------------------


def test_final_output_puts_the_whole_likelihood_on_the_last_segment() -> None:
    problem, _ = _problem(num_segments=3)
    split = problem.segment_split()
    assert isinstance(split, SegmentSplit)
    assert split.model is problem.forward
    assert split.observed_segments == (2,)
    assert split.observations[:2] == (None, None)
    assert split.observations[2]["likelihood"] is None  # a Python-number sigma


def test_segment_ends_split_slices_the_data_and_a_per_segment_sigma() -> None:
    problem, _ = _problem(num_segments=3, output="segment_ends")
    sigma = 0.05 + 0.01 * jnp.arange(3.0)[:, None] * jnp.ones((3, 6 * N))
    problem = InferenceProblem(
        problem.forward,
        problem.observable,
        GaussianLikelihood(sigma=sigma),
        problem.observed,
    )
    split = problem.segment_split()
    assert split.observed_segments == (0, 1, 2)
    for k, obs in enumerate(split.observations):
        assert obs["observed"].shape == (1, 6 * N)
        assert jnp.array_equal(obs["likelihood"]["sigma"], sigma[k : k + 1])


@pytest.mark.parametrize(
    "case, match",
    [
        ("constraints", "explicit constraints"),
        ("not_shooting", "split_objective"),
        ("not_gaussian", "GaussianLikelihood"),
        ("not_segment_wise", "segment-wise"),
    ],
)
def test_problems_that_do_not_split_are_refused(case, match) -> None:
    problem, _ = _problem(num_segments=2, output="segment_ends")
    if case == "constraints":
        problem = InferenceProblem(
            problem.forward,
            problem.observable,
            problem.likelihood,
            problem.observed,
            constraints=(FunctionConstraint(lambda p: p["theta"]["k"] - 1.0),),
        )
    elif case == "not_shooting":
        problem = InferenceProblem(
            LinearForwardModel(jnp.eye(2), jnp.zeros(2)),
            IdentityObservable(),
            GaussianLikelihood(),
            jnp.zeros(2),
        )
    elif case == "not_gaussian":

        @dataclass(frozen=True)
        class Laplace:
            def log_prob(self, predicted, observed):
                return -jnp.sum(jnp.abs(predicted - observed))

        problem = InferenceProblem(
            problem.forward, problem.observable, Laplace(), problem.observed
        )
    else:
        problem = InferenceProblem(
            problem.forward,
            _phase_space,
            problem.likelihood,
            problem.observed.reshape(-1),
        )
    with pytest.raises(ValueError, match=match):
        problem.segment_split()
    with pytest.raises(ValueError, match=match):
        AugmentedLagrangian(devices=_devices()).minimize(
            problem, {"theta": {"k": jnp.asarray(1.0)}, "nodes": None}
        )


# --- data as arguments ----------------------------------------------------------


def test_data_arrays_round_trip_and_leave_the_program() -> None:
    """The data are arguments of the driver's jitted step, not constants of it."""
    problem, params = _problem(num_segments=2)
    sigma = jnp.full(problem.observed.shape, 0.0371234567)
    problem = InferenceProblem(
        problem.forward,
        problem.observable,
        GaussianLikelihood(sigma=sigma),
        problem.observed,
    )
    data = problem.data_arrays()
    assert data["likelihood"]["sigma"] is sigma
    assert data["forward"]["engine"] is None  # the Verlet engine has no arrays
    again = problem.with_data_arrays(data)
    assert jnp.array_equal(again.log_posterior(params), problem.log_posterior(params))
    assert jnp.array_equal(
        problem.with_data_arrays({"observed": problem.observed}).observed,
        problem.observed,
    )

    driver = AugmentedLagrangian(optimizer=optax.adam(1e-2))
    step, measure = driver._compile(problem, [0])
    multipliers = _multipliers(problem, params)
    opt_state = driver.optimizer.init(params)
    mu = jnp.asarray(1.0)
    text = step.lower(params, opt_state, multipliers, mu, data).as_text()
    text += measure.lower(params, data).as_text()
    # the masses (1.1 / 8, ...), the initial condition and sigma would be dense
    # constants; only scalars and splats may be embedded
    for value in (0.0371234567, float(problem.forward.masses[1])):
        assert f"{value:.6e}"[:8] not in text, value
    assert "dense<[" not in text


def test_gaussian_likelihood_data_arrays() -> None:
    assert GaussianLikelihood(sigma=0.3).data_arrays() is None
    likelihood = GaussianLikelihood(sigma=jnp.ones(2))
    arrays = likelihood.data_arrays()
    assert arrays is not None
    assert likelihood.with_data_arrays(None) is likelihood
    bound = likelihood.with_data_arrays({"sigma": 2.0 * jnp.ones(2)})
    assert jnp.allclose(bound.log_prob(jnp.ones(2), jnp.zeros(2)), -0.25)


# --- parallel == one program ----------------------------------------------------


@pytest.mark.parametrize(
    "num_segments, output, free_initial, with_prior",
    [
        (2, "final", False, False),
        (3, "final", False, True),
        (4, "segment_ends", False, False),
        (3, "final", True, False),
        (1, "final", False, False),
    ],
)
def test_parallel_value_and_gradient_equal_the_one_program(
    num_segments, output, free_initial, with_prior
) -> None:
    """Round-off at most (the module docstring of ``mimirax.inference.parallel``)."""
    priors = (GaussianPrior(mean=0.5, sigma=2.0),) if with_prior else ()
    problem, params = _problem(
        num_segments=num_segments,
        output=output,
        free_initial=free_initial,
        priors=priors,
    )
    multipliers = _multipliers(problem, params)
    mu = jnp.asarray(7.0)
    want_value, want_grads = _one_program(problem, params, multipliers, mu)
    objective = ParallelSegmentObjective(problem, _devices())
    value, grads, evaluation = objective.value_and_grad(params, multipliers, mu)
    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(
        want_grads
    )
    assert _max_relative(value, want_value) < 1e-14
    assert _max_relative(grads, want_grads) < 1e-13
    reference = problem.evaluate(params)
    assert _max_relative(evaluation.negative_log_posterior, reference[0]) < 1e-14
    assert _max_relative(evaluation.defects, reference.defects) == 0.0
    measured = objective.evaluate(params)
    assert _max_relative(measured.negative_log_posterior, reference[0]) < 1e-14
    assert _max_relative(measured.defects, reference.defects) == 0.0
    assert len(measured.aux) == num_segments


def test_equal_segments_share_one_program_compiled_once_per_device() -> None:
    """Four equal segments, ``"final"``: two keys (with / without a next node and
    data), one trace each, one compile per device a key runs on; nothing is
    traced or compiled again across steps."""
    problem, params = _problem(num_segments=4)
    objective = ParallelSegmentObjective(problem, _devices())
    multipliers = _multipliers(problem, params)
    for step in range(3):
        objective.value_and_grad(params, multipliers, jnp.asarray(1.0 + step))
        objective.evaluate(params)
    assert objective.num_traces == 2
    assert objective.num_forward_traces == 2
    assert segment_key(problem.forward.engine, 0) == segment_key(
        problem.forward.engine, 3
    )
    distinct = len(set(_devices()))
    # keys: segments 0-2 (next node, no data) on both devices; segment 3 on one
    compiles = {
        key: program._cache_size() for key, program in objective._programs.items()
    }
    assert sorted(compiles.values()) == sorted([distinct, 1, distinct, 1])


def test_the_driver_in_parallel_equals_the_one_program() -> None:
    """Two outer x 4 inner Adam steps from perturbed nodes; equal to round-off."""
    problem, params = _problem(num_segments=3)
    kwargs = dict(optimizer=optax.adam(1e-2), inner_steps=4, max_outer=2, mu0=10.0)
    one = AugmentedLagrangian(**kwargs).minimize(problem, params)
    many = AugmentedLagrangian(devices=_devices(), **kwargs).minimize(problem, params)
    assert _max_relative(many.params, one.params) < 1e-13
    assert _max_relative(many.lagrangian_trace, one.lagrangian_trace) < 1e-13
    assert _max_relative(many.violation_trace, one.violation_trace) < 1e-13
    assert _max_relative(many.multipliers, one.multipliers) < 1e-12
    assert one.num_traces == 1
    assert many.num_traces == 2  # segments 0-1 share a key, segment 2 is its own
    assert {
        d for leaf in jax.tree_util.tree_leaves(many.params) for d in leaf.devices()
    } == {_devices()[0]}


def test_a_line_search_is_refused_in_parallel() -> None:
    problem, params = _problem(num_segments=2)
    with pytest.raises(NotImplementedError, match="value_fn"):
        AugmentedLagrangian(
            optimizer=optax.lbfgs(), devices=_devices(), inner_steps=1, max_outer=1
        ).minimize(problem, params)
    result = AugmentedLagrangian(
        optimizer=optax.lbfgs(linesearch=None),
        devices=_devices(),
        inner_steps=2,
        max_outer=1,
    ).minimize(problem, params)
    assert result.objective_trace.shape == (2,)


@dataclass
class _PlannedEngine:
    """The Verlet engine with host ``prepare`` / ``check`` and scripted verdicts."""

    inner: VerletSegmentEngine
    keyed: bool = False
    verdicts: list = field(default_factory=list)
    prepares: int = 0
    generation: int = 0

    def __call__(self, theta, positions, velocities, masses, segment):
        x, v, _ = self.inner(theta, positions, velocities, masses, segment)
        return SegmentResult(x, v, jnp.asarray(segment, dtype=jnp.int32))

    def prepare(self, theta, nodes, masses):
        del theta, masses
        assert nodes["positions"].shape[0] == len(self.inner.segment_steps)
        self.prepares += 1

    def check(self, auxes):
        assert [int(a) for a in auxes] == list(range(len(auxes)))
        verdict = self.verdicts.pop(0) if self.verdicts else True
        if not verdict:
            self.generation += 1
        return verdict

    def __getattr__(self, name):
        # a keyed engine re-keys its segments at every failed check
        if name == "segment_key" and self.keyed:
            return lambda k: ("planned", k, self.generation)
        raise AttributeError(name)


@pytest.mark.parametrize("keyed", [False, True])
def test_a_failed_check_re_prepares_and_redoes_the_step_in_parallel(keyed) -> None:
    """A failed verdict: prepare again and redo the step on new programs."""
    _, params = _problem(num_segments=2)
    x0, v0, masses = _system()
    engine = _PlannedEngine(
        VerletSegmentEngine(segment_steps=(4, 4), dt=0.02, field=harmonic),
        keyed=keyed,
        verdicts=[True, False, True],
    )
    model = MultipleShootingForwardModel(engine, masses, 2, x0, v0)
    reference, _ = _problem(num_segments=2)
    problem = InferenceProblem(
        model, _phase_space, reference.likelihood, reference.observed
    )
    driver = AugmentedLagrangian(
        optimizer=optax.adam(1e-2), inner_steps=3, max_outer=1, devices=_devices()
    )
    result = driver.minimize(problem, params)
    assert result.num_redos == 1
    # two segments (two keys), traced again after the failed check
    assert result.num_traces == 4
    # one prepare per step, one per redo, one before the outer measurement
    assert engine.prepares == 3 + 1 + 1
    plain = AugmentedLagrangian(
        optimizer=optax.adam(1e-2), inner_steps=3, max_outer=1
    ).minimize(problem, params)
    assert _max_relative(result.params, plain.params) < 1e-13

    hopeless = _PlannedEngine(engine.inner, keyed=keyed, verdicts=[False] * 10)
    model = MultipleShootingForwardModel(hopeless, masses, 2, x0, v0)
    problem = InferenceProblem(
        model, _phase_space, reference.likelihood, reference.observed
    )
    with pytest.raises(RuntimeError, match="check failed"):
        AugmentedLagrangian(max_redo=2, devices=_devices()).minimize(problem, params)


def test_bound_engine_forwards_the_arrays() -> None:
    calls = []

    def engine(theta, positions, velocities, masses, segment, arrays=None):
        calls.append((segment, arrays))
        return SegmentResult(positions, velocities, None)

    bound = BoundSegmentEngine(engine, ("a", "b"))
    x = jnp.zeros((2, 3))
    bound(None, x, x, jnp.ones(2), 1)
    assert calls == [(1, "b")]


def test_parallel_objective_validation() -> None:
    problem, _ = _problem(num_segments=2)
    with pytest.raises(ValueError, match="at least one device"):
        ParallelSegmentObjective(problem, ())


_SUBPROCESS = textwrap.dedent("""
    import jax, optax
    assert len(jax.devices()) == 2, jax.devices()
    from tests.unit import test_parallel_segments as t

    problem, params = t._problem(num_segments=4)
    objective = t.ParallelSegmentObjective(problem, jax.devices())
    assert [d.id for d in objective.segment_devices] == [0, 1, 0, 1]
    multipliers = t._multipliers(problem, params)
    mu = t.jnp.asarray(3.0)
    want = t._one_program(problem, params, multipliers, mu)
    raw, _ = objective._dispatch("value_and_grad", params, multipliers[0], mu)
    devices = [next(iter(c.penalty.devices())).id for c in raw]
    assert devices == [0, 1, 0, 1], devices
    gathered, _ = objective.dispatch(params, multipliers, mu)
    assert {next(iter(c.penalty.devices())).id for c in gathered} == {0}
    value, grads, _ = objective.value_and_grad(params, multipliers, mu)
    assert t._max_relative(value, want[0]) < 1e-14
    assert t._max_relative(grads, want[1]) < 1e-13
    kwargs = dict(optimizer=optax.adam(1e-2), inner_steps=3, max_outer=2)
    one = t.AugmentedLagrangian(**kwargs).minimize(problem, params)
    many = t.AugmentedLagrangian(devices=jax.devices(), **kwargs).minimize(
        problem, params
    )
    assert t._max_relative(many.params, one.params) < 1e-13
    print("OK", many.num_traces)
    """)


def test_two_forced_cpu_devices() -> None:
    """The comparison on two real (forced) CPU devices, segments 0-3 on 0, 1, 0, 1."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ)
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2"
    ).strip()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([root, env.get("PYTHONPATH", "")])
    done = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    assert done.stdout.strip().startswith("OK"), done.stdout


def test_lbfgs_with_the_host_line_search_drives_the_defects_to_zero() -> None:
    """L-BFGS without optax's line search, Armijo backtracking on the host.

    Session 1's gate (``test_shooting.test_shooting_defects_vanish_under_the_driver``:
    perturbed nodes and theta, 3 segments) in the per-segment mode: the violation
    below 1e-8, theta to 1e-6 of the truth, one trace per segment key.
    """
    problem, _ = _problem(num_segments=3, steps=5)
    model = problem.forward
    truth = {"k": jnp.asarray(1.0)}
    observed = _phase_space(model({"theta": truth, "nodes": model.init_nodes(truth)}))
    problem = InferenceProblem(
        model, _phase_space, GaussianLikelihood(sigma=0.05), observed
    )
    start = {"k": jnp.asarray(1.25)}
    nodes = jax.tree_util.tree_map(
        lambda a: a + 0.01 * jnp.sin(jnp.arange(a.size).reshape(a.shape)),
        model.init_nodes(start),
    )
    driver = AugmentedLagrangian(
        optimizer=optax.lbfgs(linesearch=None),
        line_search="backtracking",
        devices=_devices(),
        inner_steps=100,
        max_outer=15,
        mu0=10.0,
    )
    result = driver.minimize(problem, {"theta": start, "nodes": nodes})
    assert result.converged
    assert float(result.outer_violation[-1]) < 1e-8
    assert abs(float(result.params["theta"]["k"]) - 1.0) < 1e-6
    assert result.num_traces == 2


def test_line_search_validation() -> None:
    problem, params = _problem(num_segments=2)
    with pytest.raises(ValueError, match="per-segment mode"):
        AugmentedLagrangian(line_search="backtracking").minimize(problem, params)
    with pytest.raises(ValueError, match="line_search must be"):
        AugmentedLagrangian(line_search="zoom", devices=_devices()).minimize(
            problem, params
        )
