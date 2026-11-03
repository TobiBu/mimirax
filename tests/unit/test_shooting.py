"""Multiple shooting's core on the pure-JAX segment engine: no solver needed.

The model (:class:`~mimirax.MultipleShootingForwardModel`), the problem's
constraint slot (:class:`~mimirax.InferenceProblem` ``constraints`` and the
``predict_with_defects`` capability) and the augmented-Lagrangian driver, on
:class:`~mimirax.testing.VerletSegmentEngine`. The same gates on jaccpot's
direct sum are in ``tests/integration/test_multiple_shooting.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import optax
import pytest

from mimirax import (
    AugmentedLagrangian,
    AugmentedLagrangianResult,
    Constraint,
    ForwardModel,
    FunctionConstraint,
    GaussianLikelihood,
    GaussianPrior,
    IdentityObservable,
    InferenceProblem,
    MultipleShootingForwardModel,
    SegmentEngine,
    SegmentResult,
    TotalMassConstraint,
    augmented_lagrangian_value,
    constraint_violation,
    defect_violation,
    quadratic_penalty,
)
from mimirax.testing import LinearForwardModel, VerletSegmentEngine


def harmonic(x, t, args):
    """``a = -k x``: the field with a traced parameter, nornax's convention."""
    del t
    return -args["k"] * x


def _system(n=8, seed=0):
    k_x, k_v = jax.random.split(jax.random.PRNGKey(seed))
    positions = jax.random.normal(k_x, (n, 3))
    velocities = 0.3 * jax.random.normal(k_v, (n, 3))
    masses = jnp.full((n,), 1.0 / n)
    return positions, velocities, masses


def _model(num_segments=3, steps=5, **kwargs):
    x0, v0, masses = _system()
    engine = VerletSegmentEngine(
        segment_steps=(steps,) * num_segments,
        dt=0.02,
        field=harmonic,
        vjp_only=kwargs.pop("vjp_only", False),
    )
    model = MultipleShootingForwardModel(engine, masses, num_segments, x0, v0, **kwargs)
    return model, x0, v0


def _phase_space(state):
    return jnp.concatenate([state["positions"], state["velocities"]], -1).reshape(-1)


THETA = {"k": jnp.asarray(1.3)}


# --- the model ------------------------------------------------------------------


def test_the_model_satisfies_the_protocols() -> None:
    """A forward model, a constraint; the engine double is a segment engine."""
    model, _, _ = _model()
    assert isinstance(model, ForwardModel)
    assert isinstance(model, Constraint)
    assert isinstance(model.engine, SegmentEngine)


def test_init_nodes_gives_zero_defects_and_the_chained_prediction() -> None:
    """Nodes from a forward run: the defects vanish exactly, the ends chain.

    EXACTLY when the nodes and the defects come from the same compilation mode
    (both jitted here): a segment from a node is the same program as the
    segment that produced the node. Eager against jitted differs by round-off.
    """
    model, x0, v0 = _model(num_segments=3)
    nodes = jax.jit(model.init_nodes)(THETA)
    assert nodes["positions"].shape == (2, 8, 3)
    params = {"theta": THETA, "nodes": nodes}
    prediction, defects = jax.jit(model.predict_with_defects)(params)
    assert defects.shape == (2, 8, 6)
    assert float(jnp.max(jnp.abs(defects))) == 0.0
    # the same three segments chained by hand
    x, v = x0, v0
    for k in range(3):
        x, v, _ = model.engine(THETA, x, v, model.masses, k)
    assert jnp.allclose(prediction["positions"], x, rtol=0, atol=1e-14)
    assert jnp.allclose(prediction["velocities"], v, rtol=0, atol=1e-14)
    assert jnp.array_equal(model(params)["positions"], prediction["positions"])
    # eager is another program than the jitted init_nodes: round-off only
    assert float(jnp.max(jnp.abs(model.defect(params)))) < 1e-14


def test_defect_convention_is_node_minus_end() -> None:
    """nornax's ``shooting_defect``: ``next_node - segment_end``, positions first."""
    model, _, _ = _model(num_segments=2)
    nodes = model.init_nodes(THETA)
    shifted = {
        "positions": nodes["positions"] + 0.5,
        "velocities": nodes["velocities"] - 0.25,
    }
    defects = model.defect({"theta": THETA, "nodes": shifted})
    assert jnp.allclose(defects[..., :3], 0.5)
    assert jnp.allclose(defects[..., 3:], -0.25)


def test_one_segment_is_single_shooting_with_no_defects() -> None:
    """S = 1: no nodes, an empty ``[0, N, 6]`` defect, violation zero."""
    model, x0, v0 = _model(num_segments=1)
    nodes = model.init_nodes(THETA)
    assert nodes["positions"].shape == (0, 8, 3)
    prediction, defects = model.predict_with_defects({"theta": THETA, "nodes": nodes})
    assert defects.shape == (0, 8, 6)
    assert float(defect_violation((defects,))) == 0.0
    x, v, _ = model.engine(THETA, x0, v0, model.masses, 0)
    assert jnp.array_equal(prediction["positions"], x)


def test_free_initial_and_segment_ends() -> None:
    """A free node 0 and the ``segment_ends`` output, with their shapes."""
    model, x0, v0 = _model(num_segments=3, free_initial=True, output="segment_ends")
    assert model.num_nodes == 3
    nodes = model.init_nodes(THETA, x0, v0)
    assert nodes["positions"].shape == (3, 8, 3)
    assert jnp.array_equal(nodes["positions"][0], x0)
    params = {"theta": THETA, "nodes": nodes}
    evaluation = model.evaluate(params)
    assert evaluation.prediction["positions"].shape == (3, 8, 3)
    assert evaluation.defects.shape == (2, 8, 6)
    assert evaluation.auxes == (None, None, None)
    # the ends are the next nodes
    assert jnp.allclose(
        evaluation.prediction["positions"][:2], nodes["positions"][1:], atol=1e-14
    )
    starts = model.start_states(params)
    assert jnp.array_equal(starts["positions"], nodes["positions"])


def test_model_validation() -> None:
    """Bad static configuration and mis-sized nodes are refused."""
    _, _, masses = _system()
    engine = VerletSegmentEngine(segment_steps=(2, 2), dt=0.02)
    with pytest.raises(ValueError, match="num_segments"):
        MultipleShootingForwardModel(engine, masses, 0, free_initial=True)
    with pytest.raises(ValueError, match="output"):
        MultipleShootingForwardModel(engine, masses, 2, free_initial=True, output="x")
    with pytest.raises(ValueError, match="required"):
        MultipleShootingForwardModel(engine, masses, 2)
    free = MultipleShootingForwardModel(engine, masses, 2, free_initial=True)
    with pytest.raises(ValueError, match="start state"):
        free.init_nodes(THETA)
    model, _, _ = _model(num_segments=3)
    nodes = model.init_nodes(THETA)
    short = jax.tree_util.tree_map(lambda a: a[:1], nodes)
    with pytest.raises(ValueError, match="must hold 2 states"):
        model({"theta": THETA, "nodes": short})


@dataclass
class _PlannedEngine:
    """A Verlet engine with the optional host capabilities, recording the calls."""

    inner: VerletSegmentEngine
    verdicts: list = field(default_factory=list)
    prepared: list = field(default_factory=list)
    checked: list = field(default_factory=list)

    def __call__(self, theta, positions, velocities, masses, segment):
        x, v, _ = self.inner(theta, positions, velocities, masses, segment)
        return SegmentResult(x, v, jnp.asarray(segment, dtype=jnp.int32))

    def prepare(self, theta, nodes, masses):
        self.prepared.append((nodes["positions"].shape, masses.shape))

    def check(self, auxes):
        self.checked.append(tuple(int(a) for a in auxes))
        return self.verdicts.pop(0) if self.verdicts else True


def test_prepare_and_check_are_delegated() -> None:
    """The model hands the engine every segment's start state and every aux."""
    model, x0, v0 = _model(num_segments=3)
    engine = _PlannedEngine(model.engine)
    planned = MultipleShootingForwardModel(engine, model.masses, 3, x0, v0)
    params = {"theta": THETA, "nodes": planned.init_nodes(THETA)}
    planned.prepare(params)
    assert engine.prepared == [((3, 8, 3), (8,))]
    evaluation = planned.evaluate(params)
    assert planned.check(evaluation.auxes) is True
    assert engine.checked == [(0, 1, 2)]
    # an engine without the capabilities: prepare is a no-op, check passes
    model.prepare(params)
    assert model.check((None, None, None)) is True


# --- the problem ----------------------------------------------------------------


def _shooting_problem(num_segments=3, penalty_weight=1.0, **kwargs):
    model, x0, v0 = _model(num_segments=num_segments, **kwargs)
    truth = {"k": jnp.asarray(1.0)}
    observed = _phase_space(model({"theta": truth, "nodes": model.init_nodes(truth)}))
    problem = InferenceProblem(
        forward=model,
        observable=_phase_space,
        likelihood=GaussianLikelihood(sigma=0.05),
        observed=observed,
        penalty_weight=penalty_weight,
    )
    return problem, model, truth


def test_unconstrained_problem_is_unchanged() -> None:
    """No constraints, no defects: the old expression, bit for bit."""
    model = LinearForwardModel(jnp.eye(3), jnp.ones(3))
    problem = InferenceProblem(
        model,
        IdentityObservable(),
        GaussianLikelihood(),
        jnp.asarray([1.0, 2.0, 3.0]),
        priors=(GaussianPrior(),),
    )
    params = jnp.asarray([0.1, -0.2, 0.3])
    assert not problem.is_constrained
    want = problem.log_likelihood(params) + problem.log_prior(params)
    assert jnp.array_equal(problem.log_posterior(params), want)
    evaluation = problem.evaluate(params)
    assert evaluation.defects == () and evaluation.aux is None
    assert jnp.array_equal(evaluation.negative_log_posterior, -want)


def test_constrained_problem_adds_the_quadratic_penalty_from_one_pass() -> None:
    """Forward defects (same pass) and explicit constraints, through ``quadratic_penalty``."""
    problem, model, _ = _shooting_problem(penalty_weight=3.0)
    nodes = model.init_nodes({"k": jnp.asarray(1.2)})
    params = {
        "theta": {"k": jnp.asarray(1.2)},
        "nodes": jax.tree_util.tree_map(lambda a: a + 0.01, nodes),
    }
    assert problem.is_constrained
    evaluation = problem.evaluate(params)
    assert len(evaluation.defects) == 1
    expected = quadratic_penalty(model, params, 3.0)
    assert jnp.allclose(problem.penalty(evaluation.defects), expected, rtol=1e-14)
    want = -(evaluation.negative_log_posterior + expected)
    assert jnp.allclose(problem.log_posterior(params), want, rtol=1e-14)
    assert jnp.allclose(problem.negative_log_posterior(params), -want, rtol=1e-14)

    # an explicit constraint on top
    extra = FunctionConstraint(lambda p: p["theta"]["k"] - 1.0)
    both = InferenceProblem(
        problem.forward,
        problem.observable,
        problem.likelihood,
        problem.observed,
        constraints=(extra,),
    )
    defects = both.evaluate(params).defects
    assert len(defects) == 2
    assert jnp.allclose(defects[1], 0.2)


def test_one_forward_pass_per_evaluation() -> None:
    """``evaluate`` (and so the penalised posterior) runs every segment ONCE."""
    calls = []

    @dataclass(frozen=True)
    class Counting:
        inner: VerletSegmentEngine

        def __call__(self, theta, positions, velocities, masses, segment):
            calls.append(segment)
            return self.inner(theta, positions, velocities, masses, segment)

    problem, model, truth = _shooting_problem()
    counting = MultipleShootingForwardModel(
        Counting(model.engine),
        model.masses,
        3,
        model.initial_positions,
        model.initial_velocities,
    )
    problem = InferenceProblem(
        counting, problem.observable, problem.likelihood, problem.observed
    )
    params = {"theta": truth, "nodes": counting.init_nodes(truth)}
    calls.clear()
    problem.log_posterior(params)
    assert calls == [0, 1, 2]
    calls.clear()
    jax.grad(problem.negative_log_posterior)(params)
    assert calls == [0, 1, 2]


def test_predict_with_defects_without_evaluate() -> None:
    """A forward model offering only ``predict_with_defects`` is consumed too."""

    @dataclass(frozen=True)
    class Minimal:
        def __call__(self, params):
            return params

        def predict_with_defects(self, params):
            return params, params[:1] - 1.0

    problem = InferenceProblem(
        Minimal(), IdentityObservable(), GaussianLikelihood(), jnp.zeros(2)
    )
    evaluation = problem.evaluate(jnp.asarray([3.0, 0.0]))
    assert evaluation.aux is None
    assert jnp.allclose(evaluation.defects[0], 2.0)


# --- the augmented-Lagrangian driver ----------------------------------------------


def test_augmented_lagrangian_on_a_linear_problem_with_a_constraint() -> None:
    """``min |x - y|^2 / 2`` s.t. ``sum x = 1``: the closed-form solution and multiplier."""
    model = LinearForwardModel(jnp.eye(3), jnp.zeros(3))
    observed = jnp.asarray([1.0, 2.0, 3.0])
    problem = InferenceProblem(
        model,
        IdentityObservable(),
        GaussianLikelihood(),
        observed,
        constraints=(TotalMassConstraint(1.0),),
    )
    driver = AugmentedLagrangian(
        optimizer=optax.lbfgs(), inner_steps=50, max_outer=30, tol=1e-10
    )
    result = driver.minimize(problem, jnp.zeros(3))
    assert isinstance(result, AugmentedLagrangianResult)
    assert result.converged
    # x = y - (sum y - 1) / 3; the multiplier of sum x - 1 is (sum y - 1) / 3
    want = observed - (jnp.sum(observed) - 1.0) / 3.0
    assert jnp.allclose(result.params, want, atol=1e-8)
    assert float(constraint_violation(TotalMassConstraint(1.0), result.params)) < 1e-10
    assert jnp.allclose(result.multipliers[0], 5.0 / 3.0, atol=1e-6)
    assert result.num_traces == 1
    assert result.objective_trace.shape == result.violation_trace.shape
    assert int(jnp.sum(result.outer_steps)) == result.objective_trace.shape[0]


def test_augmented_lagrangian_unconstrained_is_a_plain_fit() -> None:
    """No defects: no multipliers, violation zero, Adam by default."""
    model = LinearForwardModel(jnp.eye(2), jnp.zeros(2))
    problem = InferenceProblem(
        model, IdentityObservable(), GaussianLikelihood(), jnp.asarray([1.0, -1.0])
    )
    result = AugmentedLagrangian(inner_steps=20, max_outer=3).minimize(
        problem, jnp.zeros(2)
    )
    assert result.multipliers == ()
    assert float(jnp.max(result.violation_trace)) == 0.0
    assert result.num_traces == 1


def test_augmented_lagrangian_value_and_mismatch() -> None:
    """``L = f + lambda . c + mu / 2 |c|^2``; a multiplier count mismatch raises."""
    problem, model, truth = _shooting_problem()
    nodes = jax.tree_util.tree_map(lambda a: a + 0.02, model.init_nodes(truth))
    params = {"theta": truth, "nodes": nodes}
    evaluation = problem.evaluate(params)
    lam = (jnp.full_like(evaluation.defects[0], 0.3),)
    value, again = augmented_lagrangian_value(problem, params, lam, 7.0)
    c = evaluation.defects[0]
    want = evaluation.negative_log_posterior + 0.3 * jnp.sum(c) + 3.5 * jnp.sum(c * c)
    assert jnp.allclose(value, want, rtol=1e-14)
    assert jnp.array_equal(again.defects[0], c)
    with pytest.raises(ValueError, match="multipliers"):
        augmented_lagrangian_value(problem, params, (), 1.0)


@pytest.mark.parametrize("vjp_only", [False, True])
def test_shooting_defects_vanish_under_the_driver(vjp_only: bool) -> None:
    """Perturbed nodes and theta: the violation goes below 1e-8, the fit converges.

    ``vjp_only=True`` makes every segment a ``custom_vjp``: forward-mode
    autodiff anywhere on the driver's path would raise, so passing proves the
    path is VJP-only.
    """
    problem, model, truth = _shooting_problem(vjp_only=vjp_only)
    start = {"k": jnp.asarray(1.25)}
    nodes = jax.tree_util.tree_map(
        lambda a: a + 0.01 * jnp.sin(jnp.arange(a.size).reshape(a.shape)),
        model.init_nodes(start),
    )
    driver = AugmentedLagrangian(
        optimizer=optax.lbfgs(), inner_steps=100, max_outer=15, mu0=10.0
    )
    result = driver.minimize(problem, {"theta": start, "nodes": nodes})
    assert result.converged
    assert float(result.outer_violation[-1]) < 1e-8
    assert abs(float(result.params["theta"]["k"]) - 1.0) < 1e-6
    assert result.num_traces == 1


def test_vjp_only_engine_refuses_forward_mode() -> None:
    """The guard is live: ``jax.jvp`` through a ``vjp_only`` segment raises."""
    model, _, _ = _model(num_segments=2, vjp_only=True)
    params = {"theta": THETA, "nodes": model.init_nodes(THETA)}
    with pytest.raises(TypeError, match="custom_vjp"):
        jax.jvp(lambda p: model(p)["positions"], (params,), (params,))


@dataclass
class _FlakyForward:
    """A linear forward model whose ``check`` fails the first ``failures`` times."""

    failures: int
    prepares: int = 0

    def __call__(self, params):
        return params

    def prepare(self, params):
        self.prepares += 1

    def check(self, aux):
        if self.failures > 0:
            self.failures -= 1
            return False
        return True


def test_a_failed_check_re_prepares_and_redoes_the_step() -> None:
    """``check`` False: prepare again, retrace, redo; too many failures raise."""
    flaky = _FlakyForward(failures=1)
    problem = InferenceProblem(
        flaky, IdentityObservable(), GaussianLikelihood(), jnp.asarray([1.0])
    )
    driver = AugmentedLagrangian(optimizer=optax.sgd(0.5), inner_steps=3, max_outer=2)
    result = driver.minimize(problem, jnp.zeros(1))
    assert result.num_redos == 1
    assert result.num_traces == 2  # one re-plan, one retrace
    # one prepare per step, one per redo, one per outer measurement
    assert flaky.prepares == 3 * 2 + 1 + 2

    hopeless = _FlakyForward(failures=100)
    problem = InferenceProblem(
        hopeless, IdentityObservable(), GaussianLikelihood(), jnp.asarray([1.0])
    )
    with pytest.raises(RuntimeError, match="check failed"):
        AugmentedLagrangian(max_redo=2).minimize(problem, jnp.zeros(1))


def test_a_failed_check_on_the_outer_measurement_re_prepares() -> None:
    """The measurement after the inner loop is checked and redone too."""
    flaky = _FlakyForward(failures=0)
    problem = InferenceProblem(
        flaky, IdentityObservable(), GaussianLikelihood(), jnp.asarray([1.0])
    )
    driver = AugmentedLagrangian(optimizer=optax.sgd(0.5), inner_steps=1, max_outer=1)

    original_check = flaky.check
    calls = []

    def check(aux):
        calls.append(1)
        return len(calls) != 2 and original_check(aux)

    object.__setattr__(flaky, "check", check)
    result = driver.minimize(problem, jnp.zeros(1))
    assert len(calls) == 2
    # step, measurement, and the re-prepare before the re-measurement (the step
    # itself is not re-run, so it is not retraced)
    assert flaky.prepares == 3
    assert result.num_traces == 1
