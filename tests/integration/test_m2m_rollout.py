"""Made-to-measure through a real nornax rollout. ``mimirax[nornax]``.

Correctness items 2 to 5 and 7 of the made-to-measure module's plan, in order:
the gradient through the integration against a finite difference in both
constructions, recovery in the tracer case with its degeneracy reported, the
classic and differentiable iterations against each other on that problem,
recovery in the self-consistent case with the chained-segments property, and
nornax's own conformance kit run on the bridge this package wrote.

Every finite-difference tolerance here is set from a measured ``|AD - FD|``
versus ``h`` curve quoted in the test that uses it. None is inherited: the
EDDA programme's FMM gradient tests found an inherited step hiding a 1.5e-5
round-off, and that is the whole reason these curves are in the docstrings.

The rollouts are run with ``reassign_rungs=False`` wherever a finite difference
is taken. Rung assignment is a discrete function of the acceleration that nornax
severs from the gradient, so the map is only globally smooth in the continuous
state with the schedule frozen; with it live, an FD step that crosses a rung
boundary compares two different maps and the disagreement is not a gradient
error. The recovery tests leave it on, which is production behaviour.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from mimirax import (
    EntropyPrior,
    GaussianLikelihood,
    GaussianRadialBins,
    InferenceProblem,
    MadeToMeasure,
    TimeAverage,
    WeightedKernelSum,
    degenerate_directions,
    fisher_information,
    made_to_measure,
)
from mimirax.adapters.nornax import NornaxRollout, SingleRungMutualForce
from mimirax.testing import DirectSumGravity, SoftenedPointMassField

nornax = pytest.importorskip("nornax", reason="mimirax[nornax] is not installed")

from nornax.conformance import check_mutual_force_model  # noqa: E402
from nornax.forces.mutual_direct import MutualDirectSumGravity  # noqa: E402
from nornax.solvers.leapfrog_kdk import (  # noqa: E402
    shooting_defect,
    shooting_node,
)

SOFTENING = 0.2
SIGMA = 0.05
MU = 1.0e-3


def _plummer_like(key, n, *, scale=1.0, speed=0.3):
    """A small centrally concentrated cloud on mildly bound orbits."""
    k_pos, k_vel = jax.random.split(key)
    radius = scale * jax.random.uniform(k_pos, (n,), minval=0.3, maxval=2.0)
    direction = jax.random.normal(k_pos, (n, 3))
    direction = direction / jnp.linalg.norm(direction, axis=1, keepdims=True)
    positions = radius[:, None] * direction
    velocities = speed * jax.random.normal(k_vel, (n, 3))
    return positions, velocities


def _external_field():
    """The tracer construction's potential: one softened point mass at the origin."""
    return SoftenedPointMassField(
        sources=jnp.zeros((1, 3)),
        source_masses=jnp.asarray([1.0]),
        softening=SOFTENING,
    )


def _tracer_problem(key, *, n=64, num_steps=24, dt=0.02):
    """Build a tracer M2M problem with mock data from known weights."""
    k_system, k_weights = jax.random.split(key)
    positions, velocities = _plummer_like(k_system, n)
    rollout = NornaxRollout.tracer(
        _external_field(), dt=dt, num_steps=num_steps, reassign_rungs=False
    )
    observable = TimeAverage(
        WeightedKernelSum(
            GaussianRadialBins(
                centres=jnp.asarray([0.4, 0.8, 1.2, 1.6, 2.0]), width=0.3
            )
        )
    )
    truth = 0.5 + jax.random.uniform(k_weights, (n,))
    observed = observable(
        rollout({"positions": positions, "velocities": velocities, "weights": truth})
    )
    problem = InferenceProblem(
        forward=rollout,
        observable=observable,
        likelihood=GaussianLikelihood(sigma=SIGMA),
        observed=observed,
        priors=(EntropyPrior(mu=MU),),
    )
    start = {
        "positions": positions,
        "velocities": velocities,
        "weights": jnp.ones(n),
    }
    return problem, truth, start


def _directional(objective, params, direction, h):
    """Central-difference derivative of ``objective`` along a weight direction."""

    def at(step):
        moved = dict(params)
        moved["weights"] = params["weights"] + step * direction
        return objective(moved)

    return float((at(h) - at(-h)) / (2.0 * h))


def _fd_curve(objective, params, direction, steps):
    """Return ``{h: |AD - FD|}`` so a tolerance can be read off a measurement."""
    ad = float(jnp.sum(jax.grad(objective)(params)["weights"] * direction))
    return ad, {
        h: abs(ad - _directional(objective, params, direction, h)) for h in steps
    }


# --- item 2: the gradient through the rollout --------------------------------------


def test_gradient_through_a_self_consistent_rollout_matches_a_finite_difference(
    key,
) -> None:
    """Case (a): the weights are the masses, so the gradient flows through the forces.

    The measurement this module exists to make. ``d(objective)/d(weights)`` here
    passes through every force evaluation of a 12-base-step block-step rollout
    of 16 mutually attracting bodies, and it is compared with a central
    difference of the same map along a random weight direction.

    Measured ``|AD - FD|`` against ``h``, central difference, float64, frozen
    rung schedule (AD = 1.280339e+02, objective 2.98)::

        h        1e-2      1e-3      1e-4      1e-5      1e-6      1e-7      1e-8
        |AD-FD|  2.03e-02  2.03e-04  2.03e-06  2.29e-08  1.07e-07  8.45e-07  2.64e-06
        rel      1.58e-4   1.58e-6   1.58e-8   1.79e-10  8.37e-10  6.60e-9   2.06e-8

    A clean ``O(h^2)`` truncation branch from 1e-2 down to 1e-5 -- two decades
    of accuracy per decade of step, a central difference's signature, and in
    itself the evidence that the map is smooth in the weights all the way
    through twelve base steps of the integration -- and an ``O(eps/h)``
    round-off branch from 1e-6 up. The minimum is at h = 1e-5, **1.79e-10
    relative**; the tolerance below is 1e-8 relative there, about 56 times the
    measured value, which bounds the gradient without being a fit to one number.
    """
    k_system, k_weights, k_direction = jax.random.split(key, 3)
    n = 16
    positions, velocities = _plummer_like(k_system, n)
    weights = 0.5 + jax.random.uniform(k_weights, (n,))
    rollout = NornaxRollout.self_consistent(
        MutualDirectSumGravity(G=1.0, softening=SOFTENING, k_max=1),
        dt=0.02,
        num_steps=12,
        k_max=1,
        reassign_rungs=False,
    )
    observable = TimeAverage(
        WeightedKernelSum(
            GaussianRadialBins(centres=jnp.asarray([0.5, 1.0, 1.5]), width=0.4)
        )
    )
    params = {"positions": positions, "velocities": velocities, "weights": weights}
    observed = observable(rollout(params)) + 0.05
    problem = InferenceProblem(
        forward=rollout,
        observable=observable,
        likelihood=GaussianLikelihood(sigma=SIGMA),
        observed=observed,
        priors=(EntropyPrior(mu=MU),),
    )
    objective = problem.negative_log_posterior

    direction = jax.random.normal(k_direction, (n,))
    direction = direction / jnp.linalg.norm(direction)
    ad, curve = _fd_curve(objective, params, direction, (1.0e-5,))
    assert abs(ad) > 1.0, f"AD = {ad:.3e} is too small for a relative bound"
    assert curve[1.0e-5] <= 1.0e-8 * abs(ad), f"|AD - FD| = {curve[1.0e-5]:.3e}"


def test_the_self_consistent_gradient_actually_flows_through_the_dynamics(key) -> None:
    """The claim behind case (a): the same fit with fixed masses has another gradient.

    A gradient that agreed with the tracer construction's would mean the weights
    were not reaching the forces at all, and the finite-difference check above
    would pass just as happily. So this compares the two constructions on the
    *same* system and the same observable: the self-consistent gradient differs
    from the tracer one by a factor of order unity in norm and is not parallel
    to it, which is the part no classic made-to-measure code computes.
    """
    k_system, k_weights = jax.random.split(key)
    n = 16
    positions, velocities = _plummer_like(k_system, n)
    weights = 0.5 + jax.random.uniform(k_weights, (n,))
    params = {"positions": positions, "velocities": velocities, "weights": weights}
    force = MutualDirectSumGravity(G=1.0, softening=SOFTENING, k_max=0)
    observable = TimeAverage(
        WeightedKernelSum(
            GaussianRadialBins(centres=jnp.asarray([0.5, 1.0, 1.5]), width=0.4)
        )
    )

    def gradient_of(rollout):
        observed = observable(rollout(params)) + 0.05
        problem = InferenceProblem(
            forward=rollout,
            observable=observable,
            likelihood=GaussianLikelihood(sigma=SIGMA),
            observed=observed,
        )
        return jax.grad(problem.negative_log_posterior)(params)["weights"]

    common = dict(dt=0.02, num_steps=12, reassign_rungs=False)
    consistent = gradient_of(NornaxRollout.self_consistent(force, **common))
    tracer = gradient_of(NornaxRollout.tracer(force, masses=weights, **common))
    assert jnp.all(jnp.isfinite(consistent))
    cosine = float(
        jnp.dot(consistent, tracer)
        / (jnp.linalg.norm(consistent) * jnp.linalg.norm(tracer))
    )
    assert abs(cosine) < 0.999, f"the two gradients are parallel (cos = {cosine:.6f})"


def test_gradient_through_a_tracer_rollout_matches_a_finite_difference(key) -> None:
    """Case (b): the weights reach the observable only, and that path is exact.

    The orbits do not depend on the weights here, so the objective is a
    *quadratic* in them plus the entropy term, and the gradient is available in
    closed form. It is still worth an FD check, because what is under test is
    that nothing in the rollout leaks a weight dependence into the trajectory.

    Measured ``|AD - FD|`` against ``h``, central difference, float64
    (AD = 7.137474e+03, objective 1.00e+04)::

        h        1e-2      1e-3      1e-4      1e-5      1e-6      1e-7      1e-8
        |AD-FD|  2.13e-09  6.69e-10  1.70e-08  8.07e-08  2.72e-06  4.09e-05  6.50e-04
        rel      2.98e-13  9.38e-14  2.39e-12  1.13e-11  3.81e-10  5.73e-9   9.11e-8

    **There is no truncation branch**, and its absence is the result. A central
    difference is exact on a quadratic, and with the orbits independent of the
    weights the chi-squared term *is* quadratic in them; the only curvature left
    is the entropy prior's, whose third derivative ``mu / w^2`` at ``mu = 1e-3``
    is far below the round-off floor. So the curve is pure ``O(eps/h)``
    round-off, rising monotonically from h = 1e-3 -- which is a sharper
    statement than any tolerance: had a weight dependence leaked into the
    trajectory, the map would not be quadratic and an ``O(h^2)`` branch would
    have appeared at large h. The minimum is 9.38e-14 relative at h = 1e-3, the
    tolerance is 1e-11 relative there, and the trajectory's independence of the
    weights is *also* asserted exactly, below.
    """
    k_problem, k_direction = jax.random.split(key)
    problem, _, start = _tracer_problem(k_problem, n=32, num_steps=12)
    n = start["weights"].size
    objective = problem.negative_log_posterior
    params = dict(start)
    params["weights"] = start["weights"] * 1.3

    direction = jax.random.normal(k_direction, (n,))
    direction = direction / jnp.linalg.norm(direction)
    ad, curve = _fd_curve(objective, params, direction, (1.0e-3,))
    assert curve[1.0e-3] <= 1.0e-11 * abs(ad), f"|AD - FD| = {curve[1.0e-3]:.3e}"

    # Exactly, not to a tolerance: the recorded orbits must not move at all.
    trajectory = jax.jacobian(
        lambda w: problem.forward({**params, "weights": w})["positions"]
    )(params["weights"])
    assert float(jnp.max(jnp.abs(trajectory))) == 0.0


# --- item 3: recovery in the tracer case, with its degeneracy reported --------------


def test_tracer_recovery_from_uniform_weights(key) -> None:
    """Item 3: recover 64 weights from five binned moments' time averages.

    Sixty-four weights against ten observables (five soft radial bins times two
    moments), started from uniform, 10000 Adam steps at a learning rate of 0.02
    through a 24-base-step rollout. What is reported is what was measured:

    * chi-squared falls from **8.82e+03** to **5.35e-11**, so the time-averaged
      observables are reproduced far below the assumed noise;
    * the weights end **0.284** (relative L2) from the ones that made the data;
    * every weight stays positive, the smallest being 0.918.

    Those numbers together *are* the result: the fit is essentially perfect and
    the weights are not recovered, because ten numbers cannot determine
    sixty-four. The Fisher information says it exactly. Its smallest eigenvalue
    **is** ``mu = 1e-3`` -- the entropy prior's own curvature ``mu / w``, with no
    contribution from the data at all, asserted to ``1e-6`` relative because
    ``eigh``'s own round-off differs by ~1e-9 between platforms -- and only
    **nine**
    eigenvalues rise above ``1e-3`` of the largest (5.78e1 to 2.57e4), so
    :func:`~mimirax.degenerate_directions` reports **55** degenerate directions
    out of 64. Along those the answer is the prior's, which is what an entropy
    prior is for and what must not be reported as a recovery.

    Choosing observables until the weight error looked small would have hidden
    the structure of the problem. Whether a real made-to-measure target is
    identifiable is a question for the experiment that poses it, and this is the
    diagnostic that answers it there.
    """
    problem, truth, start = _tracer_problem(key, n=64, num_steps=24)
    objective = problem.negative_log_posterior

    initial_chi2 = -2.0 * float(problem.log_likelihood(start))
    fit = made_to_measure(learning_rate=0.02, num_steps=10000).minimize(
        objective, start
    )
    final_chi2 = -2.0 * float(problem.log_likelihood(fit.params))
    error = float(
        jnp.linalg.norm(fit.params["weights"] - truth) / jnp.linalg.norm(truth)
    )
    assert initial_chi2 > 1.0e3
    assert final_chi2 < 1.0e-8, f"chi2 {initial_chi2:.3e} -> {final_chi2:.3e}"
    assert jnp.all(fit.params["weights"] > 0.0)
    # The weight error stays large, and that is the reported result.
    assert 0.15 < error < 0.5, f"weight error {error:.3e}"

    # The degeneracy that explains it, over the weights alone.
    def over_weights(weights):
        return objective({**start, "weights": weights})

    fisher = fisher_information(over_weights, start["weights"])
    eigenvalues, directions = degenerate_directions(fisher, rtol=1.0e-3)
    assert float(eigenvalues[0]) == pytest.approx(MU, rel=1e-6)
    assert directions.shape[1] == 55


# --- item 4: the classic iteration against the differentiable one -------------------


def test_classic_and_differentiable_agree_on_the_tracer_problem(key) -> None:
    """Item 4: both iterations, same objective, same rollout, measured agreement.

    Adam (learning rate 0.02, 10000 steps) reaches ``|dF/dw| = 7.1e-4``; the
    classic force of change (``epsilon = 3e-5``, 30000 steps) reaches
    **4.1e-6**. Their objectives then agree to **8.6e-05** absolute, on an
    objective whose value is -6.33e-2 and whose starting value was 4.41e+03.
    Their weight vectors are **0.173** apart in relative L2.

    The second number is not a defect and is why the comparison is worth
    running. The stationary *set* of this objective is 55-dimensional (see the
    recovery test), so two descents in two different metrics stop at two
    different points of the same flat valley and no step count closes the gap.
    The objectives agree because a shared stationary set constrains them; the
    weights do not because nothing constrains them to. The two end points are
    also differently far from the truth -- 0.284 for Adam against 0.217 for the
    classic iteration -- and neither is the better answer: both are points the
    data cannot distinguish.

    The sharp version of this comparison, two end points agreeing to 4.8e-11
    relative, is in ``tests/unit/test_m2m.py`` on a problem with more
    observables than weights. Read together they say that the classic and
    differentiable variants implement the same method, and that a
    made-to-measure fit's weights are only as identifiable as its observables
    make them.
    """
    problem, truth, start = _tracer_problem(key, n=64, num_steps=24)
    objective = problem.negative_log_posterior
    gradient = jax.jit(jax.grad(objective))
    differentiable = made_to_measure(learning_rate=0.02, num_steps=10000).minimize(
        objective, start
    )
    classic = MadeToMeasure(num_steps=30000, epsilon=3.0e-5).force_of_change(
        objective, start
    )
    for label, result, bound in (
        ("differentiable", differentiable, 1.0e-2),
        ("classic", classic, 1.0e-4),
    ):
        weights = result.params["weights"]
        assert jnp.all(jnp.isfinite(weights)), label
        assert jnp.all(weights > 0.0), label
        norm = float(jnp.linalg.norm(gradient(result.params)["weights"]))
        assert norm < bound, f"{label} stopped at |dF/dw| = {norm:.3e}"

    objective_gap = float(
        abs(differentiable.objective_trace[-1] - classic.objective_trace[-1])
    )
    assert objective_gap < 1.0e-3, f"objectives differ by {objective_gap:.3e}"

    # And the disagreement the flat valley leaves, reported rather than hidden.
    weight_gap = float(
        jnp.linalg.norm(differentiable.params["weights"] - classic.params["weights"])
        / jnp.linalg.norm(classic.params["weights"])
    )
    assert 0.05 < weight_gap < 0.3, f"weights differ by {weight_gap:.3e}"
    for result in (differentiable, classic):
        error = float(
            jnp.linalg.norm(result.params["weights"] - truth) / jnp.linalg.norm(truth)
        )
        assert 0.15 < error < 0.5, f"weight error {error:.3e}"


def test_a_too_large_adam_step_diverges_on_the_tracer_problem(key) -> None:
    """The differentiable variant is not step-size-free either, and it says so.

    At a learning rate of 0.05 this fit reaches chi-squared 3.2e-9 after 3000
    steps and then **leaves**: at 20000 steps the objective is back up at
    2.24e+02 and chi-squared at 4.27e+01. The objective's curvature spans seven
    orders of magnitude here (Fisher condition number 2.6e+07), and once Adam's
    normalized steps are inside the flat valley they wander out of the basin
    they found. So the classic iteration's hand-tuned ``epsilon`` is not the
    only step size a made-to-measure fit has to think about -- which is worth
    stating plainly, because it is the obvious argument *for* the differentiable
    variant and it does not hold unconditionally. Read ``objective_trace``; it
    is returned for this.
    """
    problem, _, start = _tracer_problem(key, n=64, num_steps=24)
    objective = problem.negative_log_posterior
    early = made_to_measure(learning_rate=0.05, num_steps=3000).minimize(
        objective, start
    )
    late = made_to_measure(learning_rate=0.05, num_steps=20000).minimize(
        objective, start
    )
    assert -2.0 * float(problem.log_likelihood(early.params)) < 1.0e-6
    assert -2.0 * float(problem.log_likelihood(late.params)) > 1.0
    assert float(late.objective_trace[-1]) > float(early.objective_trace[-1])


# --- item 5: the self-consistent case ----------------------------------------------


@pytest.mark.parametrize("k_max", [0, 1])
def test_self_consistent_recovery_from_a_perturbation(key, k_max: int) -> None:
    """Item 5: 32 mutually attracting bodies, weights recovered from a perturbation.

    The weights *are* the masses, so every gradient step moves the orbits and
    the gradient runs through every force evaluation. From a 10 % random
    perturbation of the true weights, 400 Adam steps at a learning rate of 0.01
    give, measured:

    ==========  ==================  =========================
    ``k_max``   chi-squared         weight error (relative L2)
    ==========  ==================  =========================
    0           7.82e+04 -> 1.13e-01  0.0877 -> 0.0854
    1           8.03e+04 -> 1.18e-01  0.0877 -> 0.0854
    ==========  ==================  =========================

    Nearly six orders of magnitude off chi-squared and **2.6 % off the weight
    error**. As in the tracer case that gap is the problem's, not the method's,
    and here it can be attributed exactly: with 32 weights against 10
    observables, only **four** Fisher directions rise above ``1e-3`` of the
    largest (condition number 6.1e+09), the random perturbation puts 47.3 % of
    its norm in that four-dimensional subspace, and removing all of it would
    leave 0.0773 -- so 0.0854 after 400 steps is a partly converged fit of the
    identifiable part and nothing more. Longer runs bear that out: 2000 steps at
    a learning rate of 0.005 take chi-squared to 7.9e-10 and leave the weight
    error at 0.0858.

    What is asserted below is the direction and the finiteness. The absolute
    residual a self-consistent fit reaches at N = 32 is an experiment's result,
    and the experiment is Jaccpot-Dynamics I's.
    """
    k_system, k_weights, k_perturb = jax.random.split(key, 3)
    n = 32
    positions, velocities = _plummer_like(k_system, n)
    truth = 0.5 + jax.random.uniform(k_weights, (n,))
    rollout = NornaxRollout.self_consistent(
        MutualDirectSumGravity(G=1.0, softening=SOFTENING, k_max=k_max),
        dt=0.02,
        num_steps=12,
        k_max=k_max,
    )
    observable = TimeAverage(
        WeightedKernelSum(
            GaussianRadialBins(
                centres=jnp.asarray([0.4, 0.8, 1.2, 1.6, 2.0]), width=0.3
            )
        )
    )
    truth_params = {
        "positions": positions,
        "velocities": velocities,
        "weights": truth,
    }
    observed = observable(rollout(truth_params))
    problem = InferenceProblem(
        forward=rollout,
        observable=observable,
        likelihood=GaussianLikelihood(sigma=SIGMA),
        observed=observed,
        priors=(EntropyPrior(mu=MU, reference=truth),),
    )
    perturbed = truth * (1.0 + 0.1 * jax.random.normal(k_perturb, (n,)))
    start = {**truth_params, "weights": perturbed}

    initial_chi2 = -2.0 * float(problem.log_likelihood(start))
    fit = made_to_measure(learning_rate=0.01, num_steps=400).minimize(
        problem.negative_log_posterior, start
    )
    final_chi2 = -2.0 * float(problem.log_likelihood(fit.params))
    assert jnp.all(jnp.isfinite(fit.params["weights"]))
    assert jnp.all(fit.params["weights"] > 0.0)
    assert (
        final_chi2 < 1.0e-3 * initial_chi2
    ), f"k_max={k_max}: chi2 {initial_chi2:.3e} -> {final_chi2:.3e}"
    before = float(jnp.linalg.norm(perturbed - truth) / jnp.linalg.norm(truth))
    after = float(
        jnp.linalg.norm(fit.params["weights"] - truth) / jnp.linalg.norm(truth)
    )
    assert after < before, f"k_max={k_max}: weight error {before:.3e} -> {after:.3e}"


@pytest.mark.parametrize("k_max", [0, 1])
def test_chained_segments_equal_one_rollout_with_weights_free(key, k_max: int) -> None:
    """Two shooting segments are one rollout, and the weights may be the free variable.

    nornax's chained-segments property, re-checked with the *weights* as the
    quantity being differentiated rather than the boundary state. Twelve base
    steps in one rollout, against six then six through a fresh
    ``shooting_node`` at the seam: the matching defect vanishes to round-off,
    and so does the difference between the two gradients with respect to the
    weights. This is what makes multiple shooting available to a made-to-measure
    fit -- a segment boundary is a place a tree can be rebuilt without breaking
    the derivative.

    Measured with ``reassign_rungs=False``: the states match to 1e-14 relative
    and the weight gradients to 1e-12 relative. With rung reassignment live the
    seam re-derives ``acc`` and can land on a different rung than the continuous
    rollout's mid-point would have, so the property is a statement about the
    frozen-schedule map, as nornax's own tests have it.
    """
    k_system, k_weights = jax.random.split(key)
    n = 16
    positions, velocities = _plummer_like(k_system, n)
    weights = 0.5 + jax.random.uniform(k_weights, (n,))
    force = MutualDirectSumGravity(G=1.0, softening=SOFTENING, k_max=k_max)
    common = dict(dt=0.02, k_max=k_max, reassign_rungs=False)
    whole = NornaxRollout.self_consistent(force, num_steps=12, **common)
    first = NornaxRollout.self_consistent(force, num_steps=6, **common)
    second = NornaxRollout.self_consistent(force, num_steps=6, base_index=6, **common)

    def midpoint(w):
        return first.final_state(
            {"positions": positions, "velocities": velocities, "weights": w}
        )

    def chained(w):
        middle = midpoint(w)
        return second.final_state(
            {
                "positions": middle.positions,
                "velocities": middle.velocities,
                "weights": w,
            }
        )

    def one_shot(w):
        return whole.final_state(
            {"positions": positions, "velocities": velocities, "weights": w}
        )

    for leaf in ("positions", "velocities"):
        got = getattr(chained(weights), leaf)
        want = getattr(one_shot(weights), leaf)
        relative = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
        assert relative < 1.0e-14, f"{leaf} differs by {relative:.3e}"

    # The seam's projection is the identity on the free variables and recomputes
    # only the derived ones: the defect between the segment end and the node
    # built from it is exactly zero, while acc and rung come back from the force.
    middle = midpoint(weights)
    node = shooting_node(
        middle.positions,
        middle.velocities,
        weights,
        force,
        k_max=k_max,
        dt_max=0.02,
        base_index=6,
    )
    defect_positions, defect_velocities = shooting_defect(middle, node)
    assert float(jnp.max(jnp.abs(defect_positions))) == 0.0
    assert float(jnp.max(jnp.abs(defect_velocities))) == 0.0
    assert jnp.allclose(node.acc, middle.acc, atol=1e-14)

    def summary(fn, w):
        end = fn(w)
        return jnp.sum(end.positions**2) + jnp.sum(end.velocities**2)

    chained_gradient = jax.grad(lambda w: summary(chained, w))(weights)
    single_gradient = jax.grad(lambda w: summary(one_shot, w))(weights)
    relative = float(
        jnp.linalg.norm(chained_gradient - single_gradient)
        / jnp.linalg.norm(single_gradient)
    )
    assert relative < 1.0e-12, f"the weight gradients differ by {relative:.3e}"


# --- item 7: nornax's conformance kit on the bridge --------------------------------


def test_the_bridge_conforms_over_a_mutual_force_model() -> None:
    """nornax's kit on ``SingleRungMutualForce(DirectSumGravity())`` at k_max = 0.

    The bridge this package wrote, checked by the contract's own kit rather than
    by an assertion of ours: shape, dtype, finiteness, the level partition, and
    the momentum residual that is the defining property of a
    ``MutualForceModel`` (nornax's D-007). It passes, and the whole report is
    the assertion message if it ever stops.
    """
    positions = jnp.asarray(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [0.0, 1.5, 0.3],
            [0.4, -0.5, 1.0],
            [0.3, 0.2, -1.2],
            [-0.7, 0.9, 0.4],
        ]
    )
    masses = jnp.asarray([1.0, 2.0, 0.5, 1.5, 0.25, 0.8])
    bridge = SingleRungMutualForce(DirectSumGravity(softening=SOFTENING))
    report = check_mutual_force_model(
        bridge,
        positions,
        masses,
        k_max=0,
        rung=jnp.zeros(positions.shape[0], dtype=jnp.int32),
        dt_max=0.02,
    )
    report.raise_for_failures()
    assert report.passed


def test_the_bridge_over_an_external_field_fails_conformance_on_momentum() -> None:
    """And it must: an external potential does not conserve the particles' momentum.

    The finding, stated as a test rather than left as a caveat. The tracer
    construction's double is an *external* field, so the antisymmetry the block
    schedule's level split relies on does not hold for it; the conformance kit
    says so on the momentum row, with a residual of order unity instead of
    1e-13. That is why the tracer construction is single-rung -- at ``k_max = 0``
    there is one level, every particle is kicked at every boundary, and the
    momentum clause is never used -- and it is why the bridge refuses to be
    driven above level 0 instead of quietly reporting the total there.
    """
    positions = jnp.asarray(
        [[0.6, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.7], [0.5, 0.5, 0.5]]
    )
    masses = jnp.ones(4)
    bridge = SingleRungMutualForce(_external_field())
    report = check_mutual_force_model(
        bridge,
        positions,
        masses,
        k_max=0,
        rung=jnp.zeros(4, dtype=jnp.int32),
        dt_max=0.02,
    )
    failed = [check.name for check in report.failures]
    assert not report.passed
    assert any("momentum" in name for name in failed), failed
    assert all(
        "momentum" in name for name in failed
    ), f"only the momentum clause may fail here; got {failed}"
