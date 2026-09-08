"""Both made-to-measure iterations, on a problem whose gradient is known by hand.

The rollout is deliberately absent here: these tests fix the *inference* half of
the method against closed-form expressions, on a "trajectory" that is a stack of
fixed snapshots. Whether autodiff through a real N-body integration agrees with a
finite difference is the integration tests' question
(``tests/integration/test_m2m_rollout.py``); whether the weight update is the
Syer & Tremaine update is this file's, and it is the more basic of the two.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

from mimirax import (
    METHODS,
    EntropyPrior,
    GaussianLikelihood,
    GaussianRadialBins,
    InferenceProblem,
    LogTransform,
    MadeToMeasure,
    Optimizer,
    SoftplusTransform,
    TimeAverage,
    WeightedKernelSum,
    degenerate_directions,
    fisher_information,
    made_to_measure,
)

STEPS = 4
SIGMA = 0.05
MU = 1.0e-3


class _FrozenTrajectory:
    """A ForwardModel that returns fixed snapshots with the weights broadcast in.

    Stands in for :class:`mimirax.adapters.nornax.NornaxRollout` with the orbits
    held fixed -- the tracer construction's defining property, and the setting in
    which the classic force of change is the right algorithm. Using it here
    isolates the *inference* half: a discrepancy in this file cannot be blamed on
    the integrator.

    Attributes
    ----------
    positions : Array
        ``(t, n, 3)`` recorded positions.
    velocities : Array
        ``(t, n, 3)`` recorded velocities.
    """

    def __init__(self, positions, velocities) -> None:
        self.positions = positions
        self.velocities = velocities

    def __call__(self, params):
        weights = jnp.asarray(params["weights"])
        return {
            "positions": self.positions,
            "velocities": self.velocities,
            "weights": jnp.broadcast_to(weights, (STEPS, weights.size)),
        }


def _problem(key, *, n, centres, width):
    """Build a frozen-orbit tracer problem with mock data from known weights."""
    k_pos, k_vel, k_w = jax.random.split(key, 3)
    forward = _FrozenTrajectory(
        jax.random.normal(k_pos, (STEPS, n, 3)),
        0.4 * jax.random.normal(k_vel, (STEPS, n, 3)),
    )
    kernel = GaussianRadialBins(centres=jnp.asarray(centres), width=width)
    observable = TimeAverage(WeightedKernelSum(kernel))
    truth = 0.5 + jax.random.uniform(k_w, (n,))
    observed = observable(forward({"weights": truth}))
    problem = InferenceProblem(
        forward=forward,
        observable=observable,
        likelihood=GaussianLikelihood(sigma=SIGMA),
        observed=observed,
        priors=(EntropyPrior(mu=MU),),
    )
    return problem, truth, {"weights": jnp.ones(n)}, kernel, observable


@pytest.fixture()
def degenerate_problem(key):
    """20 weights against 6 observables: the under-determined case, on purpose.

    Three bins times two moments cannot pin down twenty weights, and this
    fixture is where that is *reported* rather than tuned away (see
    ``test_the_under_determined_problem_is_reported_as_such``). It is the right
    fixture for identities -- the autodiff-versus-hand-written bracket -- and the
    wrong one for comparing two iterations' end points, which is what the
    well-posed fixture is for.
    """
    return _problem(key, n=20, centres=[0.6, 1.2, 1.8], width=0.5)


@pytest.fixture()
def well_posed_problem(key):
    """6 weights against 16 observables, Fisher condition number 122.

    Eight narrow bins times two moments over six weights, so the maximum of the
    objective is a point and not a valley. Only here does "the two iterations
    reach the same stationary point" have a sharp meaning.
    """
    return _problem(
        key, n=6, centres=[0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.4], width=0.2
    )


def test_made_to_measure_is_a_registered_optimizer() -> None:
    """It satisfies the Optimizer protocol and is the registry's M2M entry."""
    method = made_to_measure()
    assert isinstance(method, Optimizer)
    assert METHODS.get("made_to_measure") is MadeToMeasure


def test_minimize_needs_an_optax_rule() -> None:
    """No step size is guessed: minimize without a rule fails, naming the factory."""
    with pytest.raises(ValueError, match="made_to_measure"):
        MadeToMeasure().minimize(
            lambda p: jnp.sum(p["weights"]), {"weights": jnp.ones(3)}
        )


def test_missing_weight_leaf_is_named() -> None:
    """A parameter dict that spells the weights differently fails loudly."""
    method = made_to_measure()
    with pytest.raises(KeyError, match=r"\('mass',\)"):
        method.minimize(lambda p: jnp.sum(p["mass"]), {"mass": jnp.ones(3)})


def test_autodiff_reproduces_the_syer_tremaine_bracket(degenerate_problem) -> None:
    """``jax.grad`` of the objective *is* the classic force-of-change bracket.

    The identification the whole module rests on. Syer & Tremaine assemble

        dF/dw_i = sum_j Delta_j K_ji / sigma_j^2  -  mu dS/dw_i

    by hand, with ``Delta_j`` the residual of the time-averaged model against
    the data and ``K`` the observable's kernel. Here that expression is written
    out from the kernel and the residual and compared with autodiff.

    **The reference contains no autodiff at all**, which is the point: the
    entropy term is the closed form ``dS/dw = -log(w / w0)`` rather than
    ``jax.grad`` of the prior, so nothing in the comparison shares a code path
    with the quantity under test. An earlier version of this test differentiated
    the prior, which would have hidden an error in the prior's gradient by
    making it appear on both sides.

    Why it matters that this test exists at all:
    :meth:`~mimirax.inference.MadeToMeasure.force_of_change` calls ``jax.grad``
    on the *same* objective :meth:`~mimirax.inference.MadeToMeasure.minimize`
    does, so the classic iteration is an oracle for the *iteration and its fixed
    point* and is **not** an independent check on the gradient. The gradient's
    oracles are this closed form, the finite-difference curves in
    ``tests/integration/test_m2m_rollout.py``, and forward-mode autodiff there.
    """
    problem, _, start, kernel, observable = degenerate_problem
    weights = start["weights"] * 1.3
    params = {"weights": weights}

    trajectory = problem.forward(params)
    residual = observable(trajectory) - problem.observed
    per_step = jnp.stack(
        [
            kernel({leaf: value[t] for leaf, value in trajectory.items()})
            for t in range(STEPS)
        ]
    )
    averaged_kernel = jnp.mean(per_step, axis=0)  # (n, m)
    chi_term = averaged_kernel @ (residual / SIGMA**2)
    # dS/dw = -log(w / w0) in closed form, w0 = 1. F = chi^2/2 - mu S, so the
    # prior contributes +mu log(w). No autodiff anywhere in this expression.
    entropy_term = MU * jnp.log(weights)
    by_hand = chi_term + entropy_term

    by_autodiff = jax.grad(problem.negative_log_posterior)(params)["weights"]
    assert jnp.allclose(by_autodiff, by_hand, rtol=1e-12, atol=1e-12)


def test_force_of_change_is_the_multiplicative_classic_update(
    degenerate_problem,
) -> None:
    """One step is exactly ``w * exp(-epsilon * dF/dw)``, weights only."""
    problem, _, start, _, _ = degenerate_problem
    epsilon = 1.0e-6
    method = MadeToMeasure(num_steps=1, epsilon=epsilon)
    result = method.force_of_change(problem.negative_log_posterior, start)
    gradient = jax.grad(problem.negative_log_posterior)(start)["weights"]
    want = start["weights"] * jnp.exp(-epsilon * gradient)
    assert jnp.allclose(result.params["weights"], want, atol=1e-14)
    assert float(result.objective_trace[0]) == pytest.approx(
        float(problem.negative_log_posterior(start))
    )


def test_force_of_change_cannot_step_a_weight_negative(degenerate_problem) -> None:
    """The multiplicative form keeps ``w >= 0`` for any epsilon; it underflows instead.

    ``w * exp(-epsilon dF/dw)`` cannot change sign, whatever the step size does.
    The historical Euler spelling ``w (1 - epsilon dF/dw)`` can and does: at the
    uniform start of this problem ``max |dF/dw|`` is 5.1e2, so any
    ``epsilon > 2e-3`` sends some weight straight through zero. That is the
    reason for the multiplicative choice, and the package's rule -- positivity
    through a reparameterization, never a clip -- is the same statement.

    What too large an epsilon does instead is measured here, on chi-squared
    alone: at ``epsilon = 1e-2`` the weights underflow to **exactly zero**,
    finite and non-negative, and the fit is destroyed rather than corrupted.
    Two further failure modes are worth naming because a caller will meet them:
    at ``epsilon = 1e-1`` the *other* sign overflows (``exp`` of a large
    positive number) and the weights become ``inf`` and then ``nan``; and with
    the entropy prior in the objective, a weight that reaches zero makes the
    prior ``nan`` by design, so the collapse shows up as ``nan`` from the first
    step rather than as a run of zeros.
    """
    problem, _, start, _, observable = degenerate_problem
    chi_squared_only = InferenceProblem(
        forward=problem.forward,
        observable=observable,
        likelihood=problem.likelihood,
        observed=problem.observed,
    )
    objective = chi_squared_only.negative_log_posterior
    gradient = jax.grad(objective)(start)["weights"]
    assert float(jnp.max(jnp.abs(gradient))) > 1.0e2

    collapsed = MadeToMeasure(num_steps=20, epsilon=1.0e-2).force_of_change(
        objective, start
    )
    weights = collapsed.params["weights"]
    assert jnp.all(jnp.isfinite(weights))
    assert jnp.all(weights >= 0.0)
    assert float(jnp.min(weights)) == 0.0

    with_prior = MadeToMeasure(num_steps=20, epsilon=1.0e-2).force_of_change(
        problem.negative_log_posterior, start
    )
    assert jnp.all(jnp.isnan(with_prior.params["weights"]))


def test_minimize_stays_positive_and_lowers_the_objective(degenerate_problem) -> None:
    """Descent in log-weights: the objective falls and the weights stay positive."""
    problem, _, start, _, _ = degenerate_problem
    result = made_to_measure(learning_rate=0.05, num_steps=300).minimize(
        problem.negative_log_posterior, start
    )
    assert jnp.all(result.params["weights"] > 0.0)
    assert float(result.objective_trace[-1]) < float(result.objective_trace[0])
    assert set(result.params) == {"weights"}


def test_minimize_moves_the_weights_and_nothing_else(degenerate_problem) -> None:
    """Made-to-measure fits weights. Every other leaf comes back untouched.

    This is a regression test for a real defect. `minimize` used to optimize
    *every* leaf of ``params``, so a fit handed the ``{"positions",
    "velocities", "weights"}`` dict a rollout needs silently had **448** free
    parameters for 64 particles instead of 64 -- against ten observables -- and
    moved the positions by 17 % and the velocities by 47 % while its docstring
    said it was recovering weights. Every recovery and convergence number
    measured before the fix was a number about a different, much larger
    inference problem.

    Asserted **exactly zero** movement, not a tolerance: a frozen leaf never
    enters the optimizer's state, so there is no update to be small.
    """
    problem, _, start, _, _ = degenerate_problem
    params = {
        "weights": start["weights"],
        "offset": jnp.asarray([3.0, -1.0]),
        "label": jnp.zeros(4),
    }

    def objective(p):
        return problem.negative_log_posterior({"weights": p["weights"]}) + jnp.sum(
            p["offset"] ** 2
        )

    result = made_to_measure(0.05, 200).minimize(objective, params)
    assert jnp.array_equal(result.params["offset"], params["offset"])
    assert jnp.array_equal(result.params["label"], params["label"])
    assert not jnp.array_equal(result.params["weights"], params["weights"])
    assert list(result.params) == list(params)


def test_also_fit_opts_into_the_larger_inference_problem(degenerate_problem) -> None:
    """Fitting more than the weights is legitimate, and has to be asked for.

    Syer & Tremaine's method extends to the initial conditions, and
    :attr:`~mimirax.inference.MadeToMeasure.also_fit` is how that is spelled. It
    also checks the reparameterization stays off those leaves: an ``offset``
    starting at zero would be ``-inf`` in log space, so a fit that returns
    finite values here is a fit that transformed only the weights.
    """
    problem, _, start, _, _ = degenerate_problem
    params = {"weights": start["weights"], "offset": jnp.zeros(2)}

    def objective(p):
        return problem.negative_log_posterior({"weights": p["weights"]}) + jnp.sum(
            (p["offset"] - 2.0) ** 2
        )

    result = made_to_measure(0.05, 400, also_fit=("offset",)).minimize(
        objective, params
    )
    assert jnp.all(jnp.isfinite(result.params["offset"]))
    assert jnp.all(jnp.isfinite(result.params["weights"]))
    assert jnp.all(result.params["weights"] > 0.0)
    # It moved toward 2.0, so the leaf really was free.
    assert float(jnp.min(result.params["offset"])) > 0.5


def test_a_bare_array_of_weights_round_trips_through_both_iterations() -> None:
    """params may be the weight array itself, not only a mapping."""
    target = jnp.asarray([0.5, 2.0, 1.0])

    def objective(w):
        return 0.5 * jnp.sum((w - target) ** 2)

    start = jnp.ones(3)
    fit = made_to_measure(learning_rate=0.1, num_steps=600).minimize(objective, start)
    assert jnp.allclose(fit.params, target, atol=1e-6)
    foc = MadeToMeasure(num_steps=4000, epsilon=1.0e-2).force_of_change(
        objective, start
    )
    assert jnp.allclose(foc.params, target, atol=1e-6)


def test_the_transform_is_a_knob_and_both_choices_agree(well_posed_problem) -> None:
    """Softplus instead of log reaches the same optimum, only along another path.

    The reparameterization is a metric choice, not part of the objective -- no
    log-Jacobian is added anywhere -- so it must not move the answer. Measured
    at 5000 Adam steps, learning rate 0.02, on the well-posed problem: the two
    optima agree to **2.8e-7** relative on the weights. On the degenerate
    problem the same comparison gives 4.8e-3, and that is not a bug in either
    transform: along a direction the data do not constrain, where the fit stops
    is set by the path, and the two paths differ. Which is precisely why this
    claim is only testable on a well-posed problem.
    """
    problem, _, start, _, _ = well_posed_problem
    objective = problem.negative_log_posterior
    common = dict(optimizer=optax.adam(0.02), num_steps=5000)
    with_log = MadeToMeasure(transform=LogTransform(), **common).minimize(
        objective, start
    )
    with_softplus = MadeToMeasure(transform=SoftplusTransform(), **common).minimize(
        objective, start
    )
    difference = float(
        jnp.linalg.norm(with_log.params["weights"] - with_softplus.params["weights"])
        / jnp.linalg.norm(with_log.params["weights"])
    )
    assert difference < 1.0e-5, f"the two transforms' optima differ by {difference:.3e}"


def test_classic_and_differentiable_reach_the_same_stationary_point(
    well_posed_problem,
) -> None:
    """Correctness item 4 in the frozen-orbit setting, with the numbers stated.

    Both iterations minimize the same ``F(w)`` and both are stationary exactly
    where ``dF/dw = 0``, so on a problem whose optimum is a point they must
    land on the same weights. Measured on the well-posed fixture: ``|dF/dw|``
    falls from 1.86e2 at the uniform start to **1.0e-7** for 5000 Adam steps at
    a learning rate of 0.02, and to **2.5e-13** for 5000 classic steps at
    ``epsilon = 1e-3``. The two weight vectors then agree to **4.8e-11**
    relative and the objectives to **1.7e-18** absolute -- the classic iteration
    is the sharper of the two here, because a fixed-metric descent settles onto
    a well-conditioned optimum while Adam keeps a residual oscillation from its
    moment estimates (its ``|dF/dw|`` wobbles over 4.7e-13 to 2.9e-2 across
    3000 to 8000 steps, which is why the bound asserted for it is 1e-4 and not
    the measured value).

    The same comparison on the degenerate fixture puts the two end points 2.2e-2
    apart while both gradients are below 1.4e-4. That is the honest result, not
    a failure: the objective has a 14-dimensional flat valley there, the
    stationary *set* is not a point, and no amount of convergence makes two
    different descents agree on where in it to stop.
    """
    problem, truth, start, _, _ = well_posed_problem
    objective = problem.negative_log_posterior
    gradient = jax.jit(jax.grad(objective))

    assert float(jnp.linalg.norm(gradient(start)["weights"])) > 1.0e2

    differentiable = made_to_measure(learning_rate=0.02, num_steps=5000).minimize(
        objective, start
    )
    classic = MadeToMeasure(num_steps=5000, epsilon=1.0e-3).force_of_change(
        objective, start
    )
    for label, result, bound in (
        ("differentiable", differentiable, 1.0e-4),
        ("classic", classic, 1.0e-10),
    ):
        norm = float(jnp.linalg.norm(gradient(result.params)["weights"]))
        assert norm < bound, f"{label} stopped at |dF/dw| = {norm:.3e}"

    relative = float(
        jnp.linalg.norm(differentiable.params["weights"] - classic.params["weights"])
        / jnp.linalg.norm(classic.params["weights"])
    )
    assert relative < 1.0e-9, f"the two end points differ by {relative:.3e}"
    assert (
        float(abs(differentiable.objective_trace[-1] - classic.objective_trace[-1]))
        < 1.0e-12
    )
    # Both recover the weights that made the data, which is the point of it all.
    for result in (differentiable, classic):
        error = float(
            jnp.linalg.norm(result.params["weights"] - truth) / jnp.linalg.norm(truth)
        )
        assert error < 1.0e-4, f"weight error {error:.3e}"


def test_the_under_determined_problem_is_reported_as_such(degenerate_problem) -> None:
    """20 weights against 6 observables: the degeneracy is measured, not tuned away.

    The fit drives chi-squared to **2.5e-8** and still leaves the weights
    **23 %** away from the ones that made the data, because six numbers cannot
    determine twenty. The Fisher information says so exactly: at the uniform
    start its smallest eigenvalue **is** ``mu`` -- the entropy prior's own
    curvature ``mu / w`` at ``w = 1`` -- so along those directions the *data
    contribute nothing at all* and the prior alone decides the answer.
    Exactly **fourteen** directions are null: the time-averaged kernel has full
    rank 6 (singular values 4.43, 1.25, 0.671, 0.464, 0.226, 0.106 -- no
    redundancy among the six observables), so six directions carry information
    and the remaining fourteen carry none. :func:`~mimirax.degenerate_directions`
    reports **fifteen** at ``rtol = 1e-3``, and the fifteenth is a property of
    the tolerance rather than of the problem: the weakest *informative* Fisher
    eigenvalue is 4.50 against a largest of 7.84e3, a ratio of 5.7e-4, which is
    under the threshold without being zero. Worth knowing before an ``rtol`` is
    read as a rank -- so both numbers are asserted below.

    The bound on that smallest eigenvalue is ``1e-6`` relative, and the reason it
    is not tighter is the eigensolver rather than the physics: ``eigh`` returns
    ``mu`` to 7.6e-10 relative on darwin/arm64 and to 1.1e-9 on CI's linux
    build. The claim under test is that the data contribute *nothing* along the
    direction; 1e-6 states that without also asserting LAPACK's last bits.

    This is the diagnostic the module ships instead of a choice of observables
    that makes the recovery look good.
    """
    problem, truth, start, _, _ = degenerate_problem
    fisher = fisher_information(problem.negative_log_posterior, start)
    eigenvalues, directions = degenerate_directions(fisher, rtol=1.0e-3)
    assert float(eigenvalues[0]) == pytest.approx(MU, rel=1e-6)
    assert directions.shape[1] >= 14
    assert directions.shape[1] == 15

    fit = made_to_measure(learning_rate=0.05, num_steps=20000).minimize(
        problem.negative_log_posterior, start
    )
    chi2 = -2.0 * float(problem.log_likelihood(fit.params))
    error = float(
        jnp.linalg.norm(fit.params["weights"] - truth) / jnp.linalg.norm(truth)
    )
    assert chi2 < 1.0e-6, f"chi2 {chi2:.3e}"
    assert 0.1 < error < 0.5, f"weight error {error:.3e}"


# --- optimizer choice: schedules, quasi-Newton rules, and two stages ---------------
#
# `minimize` takes any optax rule, and that is not a decorative claim: a
# quasi-Newton or line-search rule needs `update` to be handed the objective's
# value and the objective itself, which the first version of this module did not
# do. These tests hold that seam open. Which rule is *best* is measured on the
# real problems in tests/integration/test_m2m_rollout.py and reported in
# reports/M2M_module.md, not decided here.


def test_a_learning_rate_schedule_is_accepted(well_posed_problem) -> None:
    """`made_to_measure` takes an optax schedule where a float goes.

    Measured on the degenerate 64-weight tracer problem,
    ``optax.exponential_decay(0.05, 2000, 0.3)`` reaches ``|dF/dw| = 2.7e-4``
    against the best constant rate's ``7.1e-4``, so this is a knob worth having
    reachable rather than a generality for its own sake.
    """
    problem, _, start, _, _ = well_posed_problem
    schedule = optax.cosine_decay_schedule(0.05, 2000)
    result = made_to_measure(schedule, 2000).minimize(
        problem.negative_log_posterior, start
    )
    assert result.objective_trace.size == 2000
    assert jnp.all(jnp.isfinite(result.objective_trace))
    assert float(result.objective_trace[-1]) < float(result.objective_trace[0])


def test_lbfgs_runs_through_minimize(well_posed_problem) -> None:
    """A line-search rule works, which needs more than a gradient in `update`.

    ``optax.lbfgs`` is a ``GradientTransformationExtraArgs``: its ``update``
    requires ``value``, ``grad`` and ``value_fn``, because a line search
    evaluates the objective at trial points. Handing it only a gradient raises
    ``TypeError: ... missing 3 required keyword-only arguments: 'value', 'grad',
    and 'value_fn'``, which is exactly what this module did before
    :meth:`~mimirax.inference.MadeToMeasure._descend` existed. Thirty LBFGS
    steps reach the well-posed problem's optimum here; on the degenerate tracer
    problem fifty of them reach a lower objective than ten thousand Adam steps.
    """
    problem, truth, start, _, _ = well_posed_problem
    result = MadeToMeasure(optimizer=optax.lbfgs(), num_steps=30).minimize(
        problem.negative_log_posterior, start
    )
    assert jnp.all(jnp.isfinite(result.objective_trace))
    assert jnp.all(result.params["weights"] > 0.0)
    error = float(
        jnp.linalg.norm(result.params["weights"] - truth) / jnp.linalg.norm(truth)
    )
    assert error < 1.0e-4, f"weight error {error:.3e}"


def test_a_refine_stage_concatenates_its_trace(well_posed_problem) -> None:
    """Adam then LBFGS: one FitResult, one trace, `num_steps + refine_steps` long.

    The two-stage fit the caller asked for is expressible in one object, and the
    trace is continuous across the seam so convergence can still be read from
    the result alone. Note what the *measurements* say about it, in the module's
    report: on both real test problems the best single-stage fit beats the
    two-stage one, because Adam's first stage has already chosen a point in the
    objective's flat valley that the refinement cannot leave. The stage exists
    so the comparison can be run, not because it wins.
    """
    problem, _, start, _, _ = well_posed_problem
    objective = problem.negative_log_posterior
    result = made_to_measure(0.02, 500, refine=optax.lbfgs(), refine_steps=20).minimize(
        objective, start
    )
    assert result.objective_trace.size == 520
    assert jnp.all(jnp.isfinite(result.objective_trace))
    # The seam is a continuation, not a restart: the refinement's first recorded
    # value is the objective at the point Adam left off.
    single = made_to_measure(0.02, 500).minimize(objective, start)
    assert float(result.objective_trace[500]) == pytest.approx(
        float(objective(single.params)), rel=1e-10
    )
    assert float(result.objective_trace[-1]) <= float(result.objective_trace[499])


def test_refine_is_ignored_without_steps(well_posed_problem) -> None:
    """A refine rule with zero steps changes nothing, so the knob is safe to set."""
    problem, _, start, _, _ = well_posed_problem
    objective = problem.negative_log_posterior
    plain = made_to_measure(0.02, 200).minimize(objective, start)
    with_zero = made_to_measure(
        0.02, 200, refine=optax.lbfgs(), refine_steps=0
    ).minimize(objective, start)
    assert with_zero.objective_trace.size == plain.objective_trace.size
    assert jnp.allclose(
        with_zero.params["weights"], plain.params["weights"], atol=1e-14
    )
