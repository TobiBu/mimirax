"""Multiple shooting on jaccpot's direct-sum rollout: the S10 gates. ``mimirax[jaccpot]``.

:class:`~mimirax.adapters.jaccpot.DirectSumSegmentEngine` runs each segment as
jaccpot's ``differentiable_rollout`` (production velocity Verlet, the
``checkpoint="step"`` hand-written reverse, a ``custom_vjp``) over the direct
sum, plus a Plummer field whose parameters ``theta = {log_mass, log_scale}`` are
the rollout's ``external_args``. CPU, float64, ``N`` 32-64. The gates:

1. single vs multiple: nodes from a forward run give zero defects, the
   prediction of the single rollout over the whole horizon, and its gradient in
   theta;
2. defects -> 0 under the augmented-Lagrangian driver from perturbed nodes;
3. recovery of the field's parameters from the final state, 2 and 4 segments
   against single shooting;
4. the augmented Lagrangian's gradient in theta and in the nodes against
   central finite differences;
5. VJP-only: forward mode through a segment raises, and the driver's step is
   traced once.

Measured numbers are quoted in the tests that use them (this machine, jax
0.11.2, jaccpot ``s9/session2`` 342d75f9).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

from mimirax import (
    AugmentedLagrangian,
    GaussianLikelihood,
    InferenceProblem,
    MultipleShootingForwardModel,
    SegmentEngine,
    augmented_lagrangian_value,
)
from mimirax.adapters.jaccpot import DirectSumSegmentEngine, rows_field

pytest.importorskip("jaccpot", reason="mimirax[jaccpot] is not installed")
pytest.importorskip(
    "jaccpot.runtime.grad_rollout", reason="jaccpot predates differentiable_rollout"
)

TRUTH = {"log_mass": jnp.log(jnp.asarray(3.0)), "log_scale": jnp.asarray(0.0)}
DT = 0.01
SOFTENING = 0.1


def plummer(x, t, args):
    """A Plummer sphere's field, nornax's ``f(x, t, args)`` convention."""
    del t
    r2 = jnp.sum(x * x, axis=-1, keepdims=True) + jnp.exp(2.0 * args["log_scale"])
    return -jnp.exp(args["log_mass"]) * x * r2**-1.5


def _cluster(n, seed=1):
    """A self-gravitating cloud (total mass 1, G = 1) inside the field."""
    k_x, k_v = jax.random.split(jax.random.PRNGKey(seed))
    positions = jax.random.normal(k_x, (n, 3))
    velocities = 0.5 * jax.random.normal(k_v, (n, 3))
    return positions, velocities, jnp.full((n,), 1.0 / n)


def _model(n, segment_steps, **engine_kwargs):
    x0, v0, masses = _cluster(n)
    engine = DirectSumSegmentEngine(
        segment_steps=tuple(segment_steps),
        dt=DT,
        field=plummer,
        softening=SOFTENING,
        **engine_kwargs,
    )
    model = MultipleShootingForwardModel(engine, masses, len(segment_steps), x0, v0)
    return model, engine, x0, v0, masses


def _phase_space(state):
    return jnp.concatenate([state["positions"], state["velocities"]], -1).reshape(-1)


def _problem(model, *, sigma=0.01, noise=0.0, seed=7):
    """Observe the final phase-space state at the true parameters."""
    truth = {"theta": TRUTH, "nodes": jax.jit(model.init_nodes)(TRUTH)}
    clean = _phase_space(jax.jit(lambda p: model(p))(truth))
    observed = clean + noise * jax.random.normal(jax.random.PRNGKey(seed), clean.shape)
    return InferenceProblem(
        model, _phase_space, GaussianLikelihood(sigma=sigma), observed
    )


def _relative(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


# --- the engine -------------------------------------------------------------------


def test_engine_contract() -> None:
    """A segment engine; validated lengths; the field adapter reads positions only."""
    model, engine, x0, v0, masses = _model(32, (3, 4))
    assert isinstance(engine, SegmentEngine)
    assert engine.num_segments == 2
    with pytest.raises(ValueError, match="segment_steps"):
        DirectSumSegmentEngine(segment_steps=(), dt=DT)
    with pytest.raises(ValueError, match="segment_steps"):
        DirectSumSegmentEngine(segment_steps=(3, 0), dt=DT)
    rows = jnp.stack([x0, v0], axis=1)
    assert jnp.array_equal(rows_field(plummer)(rows, TRUTH), plummer(x0, 0.0, TRUTH))
    end = engine(TRUTH, x0, v0, masses, 1)
    assert end.aux is None
    assert end.positions.shape == x0.shape
    # self-gravity only: theta is not used
    free = DirectSumSegmentEngine(segment_steps=(2,), dt=DT, softening=SOFTENING)
    assert jnp.all(jnp.isfinite(free(None, x0, v0, masses, 0).positions))


# --- gate 1: single vs multiple ---------------------------------------------------


def test_single_vs_multiple_shooting() -> None:
    """Nodes = propagated ends: zero defects, the single rollout's state and gradient.

    Four segments of 10 steps against one 40-step ``differentiable_rollout``,
    N = 64, both jitted. Measured:

    * defects: exactly zero (a segment from a node is the program that made the
      node);
    * prediction: BITWISE equal to the single rollout. Each segment computes
      its start force from the node, which is the single rollout's
      end-of-step force at the same positions, so every step sees the same
      inputs; asserted at 1e-14 so a compiler that fuses the two programs
      differently cannot fail it;
    * d(objective)/d(theta): round-off, 5.0e-15 (log_mass) and 0 (log_scale)
      relative. Not
      bitwise because the reverse passes associate differently: in the single
      rollout the start-force cotangent at a seam flows through the carried
      acceleration into the previous step's force VJP; in the chained one each
      segment's start force has its own VJP, summed into the node's cotangent.

    The gradient compared is the TOTAL derivative ``d/dtheta J(theta,
    init_nodes(theta))``: at fixed nodes the multiple-shooting objective sees
    only the last segment's dependence on theta, and the two agree only once
    the nodes move with theta (the reduced problem single shooting solves).
    """
    model, engine, x0, v0, masses = _model(64, (10, 10, 10, 10))
    nodes = jax.jit(model.init_nodes)(TRUTH)
    params = {"theta": TRUTH, "nodes": nodes}
    prediction, defects = jax.jit(model.predict_with_defects)(params)
    single = jax.jit(lambda th: engine.rollout(th, x0, v0, masses, 40))(TRUTH)
    assert float(jnp.max(jnp.abs(defects))) == 0.0
    for got, want in (
        (prediction["positions"], single.positions),
        (prediction["velocities"], single.velocities),
    ):
        assert float(jnp.max(jnp.abs(got - want))) <= 1e-14 * float(
            jnp.max(jnp.abs(want))
        )

    def multiple_objective(theta):
        state = model({"theta": theta, "nodes": model.init_nodes(theta)})
        return jnp.sum(state["positions"] ** 2) + jnp.sum(state["velocities"] ** 2)

    def single_objective(theta):
        end = engine.rollout(theta, x0, v0, masses, 40)
        return jnp.sum(end.positions**2) + jnp.sum(end.velocities**2)

    got = jax.jit(jax.grad(multiple_objective))(TRUTH)
    want = jax.jit(jax.grad(single_objective))(TRUTH)
    for name in TRUTH:
        relative = abs(float(got[name] - want[name])) / abs(float(want[name]))
        assert relative < 1e-13, f"{name}: {relative:.3e}"


# --- gate 2: defects -> 0 ---------------------------------------------------------


def test_defects_vanish_under_the_augmented_lagrangian() -> None:
    """Perturbed nodes (1e-2) and theta: violation < 1e-8 relative, fit converged.

    N = 32, four segments of 20 steps, L-BFGS inner loop, mu0 = 100, the
    tolerance relative to the largest node component (2.74). Measured: 3 outer
    iterations, 337 inner steps, 9.5 s on CPU; the violation went 4.9e-2 ->
    3.3e-6 -> 3.3e-6 -> 2.0e-8 (7.4e-9 relative: the driver stops at the first
    outer iteration below the tolerance whose objective has stagnated), the
    objective to 3.7e-13, theta to 9e-9 of the truth; the step was traced once
    (``/export/scratch/tbuck/s9-scratch/s10/m/gates-2-3-measured.txt``).
    """
    model, *_ = _model(32, (20, 20, 20, 20))
    problem = _problem(model)
    start = {"log_mass": jnp.asarray(1.2), "log_scale": jnp.asarray(0.25)}
    nodes = jax.tree_util.tree_map(
        lambda a: a + 0.01 * jax.random.normal(jax.random.PRNGKey(3), a.shape),
        jax.jit(model.init_nodes)(start),
    )
    params = {"theta": start, "nodes": nodes}
    scale = float(
        jnp.max(jnp.abs(jnp.concatenate([nodes["positions"], nodes["velocities"]])))
    )
    initial = float(jnp.max(jnp.abs(model.defect(params))))
    driver = AugmentedLagrangian(
        optimizer=optax.lbfgs(),
        inner_steps=200,
        max_outer=20,
        tol=1e-8,
        violation_scale=scale,
        mu0=100.0,
    )
    result = driver.minimize(problem, params)
    assert result.converged
    assert result.num_traces == 1
    final = float(result.outer_violation[-1])
    assert final / scale < 1e-8, f"violation {initial:.3e} -> {final:.3e}"
    assert float(result.outer_objective[-1]) < 1e-8
    for name in TRUTH:
        assert abs(float(result.params["theta"][name] - TRUTH[name])) < 1e-5
    # the driver's own measure agrees with an independent evaluation
    again = float(jnp.max(jnp.abs(model.defect(result.params))))
    assert again == pytest.approx(final, rel=1e-6, abs=1e-14)


# --- gate 3: recovery -------------------------------------------------------------


@pytest.mark.parametrize("num_segments", [2, 4])
def test_recover_field_parameters_against_single_shooting(num_segments: int) -> None:
    """Recover a Plummer field from the final state; multiple vs single shooting.

    N = 32 self-gravitating particles, 80 steps (t = 0.8, about two crossing
    times), the final phase-space state observed with Gaussian noise 1e-3
    (sigma = 1e-3); start at log_mass + 0.2, log_scale + 0.25. Both converge
    to the SAME estimate (the problems have the same minimizer once the defects
    vanish), within the noise of the truth. Measured (L-BFGS inner loop, mu0 =
    100, ``tol`` 1e-8 absolute):

    ========  =====  ======  ======  ==============  ==============  =========
    segments  outer  inner   CPU s   log_mass error  log_scale err.  vs single
    ========  =====  ======  ======  ==============  ==============  =========
    1         2      15      1.8     +2.78e-4        +2.89e-4        --
    2         14     901     18.7    +2.78e-4        +2.89e-4        2e-8
    4         14     1834    36.2    +2.78e-4        +2.89e-4        9e-8
    ========  =====  ======  ======  ==============  ==============  =========

    On this short horizon single shooting is easy, and multiple shooting pays
    for its ``6 N (S - 1)`` node variables in iterations; its case is the long,
    chaotic horizon where the single-shot objective is rough.
    """
    steps = 80 // num_segments
    model, *_ = _model(32, (steps,) * num_segments)
    single_model, *_ = _model(32, (80,))
    sigma = 1e-3
    problem = _problem(model, sigma=sigma, noise=sigma)
    single_problem = InferenceProblem(
        single_model, _phase_space, GaussianLikelihood(sigma=sigma), problem.observed
    )
    start = {
        "log_mass": TRUTH["log_mass"] + 0.2,
        "log_scale": TRUTH["log_scale"] + 0.25,
    }
    driver = AugmentedLagrangian(
        optimizer=optax.lbfgs(), inner_steps=200, max_outer=20, mu0=100.0
    )
    single = driver.minimize(
        single_problem,
        {"theta": start, "nodes": single_model.init_nodes(start)},
    )
    multiple = driver.minimize(
        problem, {"theta": start, "nodes": jax.jit(model.init_nodes)(start)}
    )
    assert single.converged and multiple.converged
    assert multiple.num_traces == 1
    for name in TRUTH:
        got = float(multiple.params["theta"][name])
        assert abs(got - float(single.params["theta"][name])) < 1e-6, name
        assert abs(got - float(TRUTH[name])) < 5e-3, name


# --- gate 4: finite differences ---------------------------------------------------


def test_augmented_lagrangian_gradient_against_finite_differences() -> None:
    """``dL/dtheta`` and ``dL/dnodes`` (a random direction) against central FD.

    N = 32, four segments of 20 steps, perturbed nodes, random multipliers,
    mu = 10. Measured ``|AD - FD| / |AD|`` against ``h``:

    ======  =========  ==========  ======
    h       log_mass   log_scale   nodes
    ======  =========  ==========  ======
    1e-3    3.8e-6     4.9e-6      1.8e-4
    1e-4    3.8e-8     4.9e-8      1.8e-6
    1e-5    4.0e-10    5.0e-10     1.8e-8
    1e-6    2.1e-10    1.4e-10     6.6e-11
    1e-7    7.1e-10    2.0e-9      1.0e-10
    ======  =========  ==========  ======

    a clean ``h^2`` truncation down to the round-off floor at 1e-6, the step
    used here (``/export/scratch/tbuck/s9-scratch/s10/m/gates-1-4-measured.txt``).
    """
    model, *_ = _model(32, (20, 20, 20, 20))
    problem = _problem(model)
    k_nodes, k_lam, k_dir = jax.random.split(jax.random.PRNGKey(11), 3)
    theta = {"log_mass": jnp.asarray(1.2), "log_scale": jnp.asarray(0.1)}
    nodes = jax.tree_util.tree_map(
        lambda a: a + 0.01 * jax.random.normal(k_nodes, a.shape),
        model.init_nodes(theta),
    )
    params = {"theta": theta, "nodes": nodes}
    shape = problem.evaluate(params).defects[0].shape
    multipliers = (0.5 * jax.random.normal(k_lam, shape),)

    def lagrangian(p):
        return augmented_lagrangian_value(problem, p, multipliers, 10.0)[0]

    value = jax.jit(lagrangian)
    grads = jax.jit(jax.grad(lagrangian))(params)
    h = 1e-6
    for name in theta:
        plus = {"theta": {**theta, name: theta[name] + h}, "nodes": nodes}
        minus = {"theta": {**theta, name: theta[name] - h}, "nodes": nodes}
        fd = (value(plus) - value(minus)) / (2 * h)
        ad = grads["theta"][name]
        assert abs(float(fd - ad)) / abs(float(ad)) < 1e-8, name
    direction = jax.tree_util.tree_map(
        lambda a: jax.random.normal(k_dir, a.shape), nodes
    )
    plus = {
        "theta": theta,
        "nodes": jax.tree_util.tree_map(lambda a, d: a + h * d, nodes, direction),
    }
    minus = {
        "theta": theta,
        "nodes": jax.tree_util.tree_map(lambda a, d: a - h * d, nodes, direction),
    }
    fd = (value(plus) - value(minus)) / (2 * h)
    ad = sum(
        jnp.vdot(g, d)
        for g, d in zip(
            jax.tree_util.tree_leaves(grads["nodes"]),
            jax.tree_util.tree_leaves(direction),
        )
    )
    assert abs(float(fd - ad)) / abs(float(ad)) < 1e-8


# --- gate 5: VJP only -------------------------------------------------------------


def test_forward_mode_is_refused_through_a_checkpointed_segment() -> None:
    """``checkpoint="step"`` is a ``custom_vjp``: ``jax.jvp`` through it raises.

    So every driver run above (whose engine is checkpointed) proves its path is
    VJP-only. The plain-scan engine (``checkpoint=None``) is the control: there
    forward mode goes through.
    """
    model, *_ = _model(32, (3, 3))
    params = {"theta": TRUTH, "nodes": model.init_nodes(TRUTH)}

    def final_positions(theta):
        return model({"theta": theta, "nodes": params["nodes"]})["positions"]

    with pytest.raises(TypeError, match="custom_vjp"):
        jax.jvp(final_positions, (TRUTH,), (TRUTH,))
    plain, *_ = _model(32, (3, 3), checkpoint=None)
    out, tangent = jax.jvp(
        lambda th: plain({"theta": th, "nodes": params["nodes"]})["positions"],
        (TRUTH,),
        (TRUTH,),
    )
    assert jnp.all(jnp.isfinite(tangent))
    reference = model({"theta": TRUTH, "nodes": params["nodes"]})["positions"]
    assert _relative(out, reference) < 1e-14
