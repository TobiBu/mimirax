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
import numpy as np
import optax  # noqa: E402
import pytest

from mimirax import (
    EntropyPrior,
    GaussianLikelihood,
    GaussianRadialBins,
    IdentityObservable,
    InferenceProblem,
    MadeToMeasure,
    TimeAverage,
    WeightedKernelSum,
    degenerate_directions,
    effective_parameters,
    fisher_information,
    made_to_measure,
)
from mimirax.adapters.nornax import (
    FoldedRollout,
    NornaxRollout,
    SingleRungMutualForce,
)
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


def _log_spaced(key, n, *, rmin=0.3, rmax=4.0, speed=0.3):
    """Log-spaced radii, so ``|a|`` spans a factor of ~20 and rungs actually split.

    :func:`_plummer_like`'s uniform radii give ``|a|`` a factor-3 spread, and a
    factor of 3 in acceleration is a factor of 1.7 in
    ``dt = eta sqrt(eps / |a|)`` -- not enough to straddle a power of two, so
    every particle lands on the same rung whatever ``k_max`` says. That is how
    the module's first ``k_max = 1`` tests came to be *vacuous* in the only
    dimension they were meant to exercise, and it is the same trap jaccpot names
    for an FMM with no far pairs. This system exists so the multi-rung tests
    have more than one rung to test.
    """
    k_r, k_dir, k_v = jax.random.split(key, 3)
    u = jax.random.uniform(k_r, (n,))
    radius = rmin * (rmax / rmin) ** u
    direction = jax.random.normal(k_dir, (n, 3))
    direction = direction / jnp.linalg.norm(direction, axis=1, keepdims=True)
    return radius[:, None] * direction, speed * jax.random.normal(k_v, (n, 3))


def _rung_histogram(state) -> dict[int, int]:
    """Return how many particles sit on each rung, as a plain dict.

    Parameters
    ----------
    state : Any
        A nornax ``BlockStepState``.

    Returns
    -------
    dict[int, int]
        Rung -> count. Asserted non-trivial wherever a test claims to be
        exercising more than one rung.
    """
    rungs = np.asarray(state.rung)
    values, counts = np.unique(rungs, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def _forward_mode_directional(objective, params, direction):
    """Directional derivative by ``jax.jvp``: forward mode, an independent path.

    Reverse mode and forward mode share the primal computation and nothing else
    -- different transformation rules, different traced graph, no reuse of the
    VJP machinery. Agreement between them is therefore a real check on the
    gradient, and unlike a finite difference it carries no step-size error at
    all, so it can be asserted at round-off rather than at a measured tolerance.
    """
    tangent = jax.tree_util.tree_map(jnp.zeros_like, params)
    tangent["weights"] = direction
    _, out = jax.jvp(objective, (params,), (tangent,))
    return float(out)


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
    """The module's one claim classic made-to-measure cannot make, quantified.

    When the weights are the masses, ``dF/dw`` splits into an **observable path**
    -- which is exactly Syer & Tremaine's bracket -- and a **dynamical path**
    through every force evaluation of the integration, which the classic
    algorithm cannot contain because it assumes the orbits are fixed. This test
    isolates the second term by running the *same* system, orbits and observable
    twice: once with the weights as the masses, and once with them held out of
    the dynamics at the same numerical values, so the difference of the two
    gradients *is* the dynamical path.

    Measured on this test's own system, N = 24, softening 0.2, ``dt = 0.02``:

    ================  ====================  ======================
    integration time  ``|g_dyn|/|g_full|``  cosine(full, classic)
    ================  ====================  ======================
    t = 0.02  (1)     0.127                 0.9951
    t = 0.04  (2)     0.235                 0.9859
    t = 0.08  (4)     0.429                 0.9617
    t = 0.16  (8)     0.633                 0.9263
    t = 0.32  (16)    **0.685**             0.8523
    t = 0.64  (32)    **0.677**             0.8626
    t = 1.28  (64)    0.536                 0.9360
    t = 2.56  (128)   **0.974**             **0.3022**
    ================  ====================  ======================

    **One base step already puts 13 % of the gradient outside the classic
    bracket, and eight put 63 % there.** By a fraction of a dynamical time the
    omitted term is about 68 % of the gradient's norm and misdirects the step by
    ~30 degrees. Beyond that the growth is *not* monotone -- 0.54 at t = 1.28,
    0.97 at t = 2.56 -- because the orbits are mixing and the gradient's
    structure changes with them; the honest statement is that the share rises
    fast and then fluctuates between a half and all of it, not that it converges
    to a constant.

    Push the self-gravity harder and it is worse. At softening 0.05, t = 0.64:
    the dynamical term is **100 %** of the norm and the cosine drops to
    **0.30** -- the classic bracket is then 72 degrees away from the descent
    direction, and no step size rescues that.

    This is the number the paper should quote, and it is also why a
    finite-difference check that merely passes is not enough: the FD test above
    would pass just as happily if the weights never reached the forces at all.
    This is the test that says they do.

    **Prior art, so the claim is stated at the right strength.** The
    orbit-response term is not new. Bovy, Kawata & Hunt (2018,
    arXiv:1704.03884) compute the analogous derivative for *external-potential*
    parameters by finite differencing the orbit integration -- one extra
    integration per parameter -- and it is entirely practical for the one or two
    parameters they fit. They also state that they ignore self-gravity, so the
    regime measured here is outside their method by construction. What reverse
    mode changes is the cost: ``O(1)`` in the parameter count instead of
    ``O(n)``, which is the difference between one rollout and 25 of them here,
    and between one and a million for a galaxy.
    """
    k_system, k_weights = jax.random.split(key)
    n = 24
    steps = 32
    positions, velocities = _plummer_like(k_system, n)
    weights = 0.5 + jax.random.uniform(k_weights, (n,))
    params = {
        "positions": positions,
        "velocities": velocities,
        "weights": weights,
    }
    force = MutualDirectSumGravity(G=1.0, softening=SOFTENING, k_max=0)
    observable = TimeAverage(
        WeightedKernelSum(
            GaussianRadialBins(
                centres=jnp.asarray([0.4, 0.8, 1.2, 1.6, 2.0]), width=0.3
            )
        )
    )
    common = dict(dt=0.02, num_steps=steps, reassign_rungs=False)
    self_consistent = NornaxRollout.self_consistent(force, k_max=0, **common)
    # The same orbits and the same observable, with the weights out of the
    # dynamics: its gradient is the observable path alone, i.e. the classic
    # bracket. `masses=weights` keeps the trajectory identical.
    observable_path_only = NornaxRollout.tracer(force, masses=weights, **common)
    observed = observable(self_consistent(params)) + 0.05

    def gradient_of(rollout):
        problem = InferenceProblem(
            forward=rollout,
            observable=observable,
            likelihood=GaussianLikelihood(sigma=SIGMA),
            observed=observed,
            priors=(EntropyPrior(mu=MU),),
        )
        return jax.grad(problem.negative_log_posterior)(params)["weights"]

    full = gradient_of(self_consistent)
    classic = gradient_of(observable_path_only)
    dynamical = full - classic
    assert jnp.all(jnp.isfinite(full))

    share = float(jnp.linalg.norm(dynamical) / jnp.linalg.norm(full))
    cosine = float(
        jnp.dot(full, classic) / (jnp.linalg.norm(full) * jnp.linalg.norm(classic))
    )
    assert share > 0.5, f"the dynamical term is only {share:.3f} of the gradient"
    assert cosine < 0.95, f"the classic bracket is aligned to {cosine:.6f}"

    # And it grows with integration time, which is what makes it a dynamical
    # effect rather than a constant offset: one base step gives 0.127 against
    # 0.677 at thirty-two.
    brief = NornaxRollout.self_consistent(
        force, k_max=0, dt=0.02, num_steps=1, reassign_rungs=False
    )
    brief_only = NornaxRollout.tracer(
        force, masses=weights, dt=0.02, num_steps=1, reassign_rungs=False
    )
    brief_observed = observable(brief(params)) + 0.05

    def brief_gradient(rollout):
        problem = InferenceProblem(
            forward=rollout,
            observable=observable,
            likelihood=GaussianLikelihood(sigma=SIGMA),
            observed=brief_observed,
            priors=(EntropyPrior(mu=MU),),
        )
        return jax.grad(problem.negative_log_posterior)(params)["weights"]

    brief_full = brief_gradient(brief)
    brief_share = float(
        jnp.linalg.norm(brief_full - brief_gradient(brief_only))
        / jnp.linalg.norm(brief_full)
    )
    assert brief_share < 0.2, f"one step already gives {brief_share:.3f}"
    assert share > 3.0 * brief_share


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
    moments), started from uniform, through a 24-base-step rollout, **weights
    only** -- the positions and velocities are fixed inputs and are asserted
    unmoved. (They were not always: see
    ``test_minimize_moves_the_weights_and_nothing_else``. Every number here is
    a re-measurement after that fix.)

    Measured, 200 LBFGS steps:

    * chi-squared falls from **8.82e+03** to **1.36e-08**;
    * the weights end **0.217** (relative L2) from the ones that made the data;
    * every weight stays positive, the smallest being 0.666.

    Those two numbers together are the result, and the reason is now measured
    rather than asserted. The objective is **strictly convex** in the weights
    (Hessian ``K̄Σ⁻¹K̄ᵀ + mu diag(1/w)``, smallest eigenvalue ``mu / max w > 0``),
    so it has a **unique** minimiser, and 0.217 is that minimiser's distance
    from the truth -- not an optimizer artefact, and not a point on a flat
    valley. What makes it 0.217 rather than 0 is how much the data actually say:
    :func:`~mimirax.effective_parameters` returns **9.9999**, i.e. the ten
    observables determine ten degrees of freedom, and the other 54 are the
    entropy prior's answer. The smallest Fisher eigenvalue *is* ``mu``, with no
    contribution from the data at all.

    No amount of optimizer, step size or ``mu`` tuning moves 9.9999. More
    information means more or better observables, and that is the honest reading
    of a made-to-measure fit whose chi-squared looks perfect.
    """
    problem, truth, start = _tracer_problem(key, n=64, num_steps=24)
    objective = problem.negative_log_posterior

    initial_chi2 = -2.0 * float(problem.log_likelihood(start))
    fit = MadeToMeasure(optimizer=optax.lbfgs(), num_steps=200).minimize(
        objective, start
    )
    final_chi2 = -2.0 * float(problem.log_likelihood(fit.params))
    error = float(
        jnp.linalg.norm(fit.params["weights"] - truth) / jnp.linalg.norm(truth)
    )
    assert initial_chi2 > 1.0e3
    assert final_chi2 < 1.0e-6, f"chi2 {initial_chi2:.3e} -> {final_chi2:.3e}"
    assert jnp.all(fit.params["weights"] > 0.0)
    # Weights only: the initial conditions are inputs, not parameters.
    for leaf in ("positions", "velocities"):
        assert jnp.array_equal(fit.params[leaf], start[leaf]), leaf
    # The weight error stays large, and that is the reported result.
    assert 0.15 < error < 0.3, f"weight error {error:.3e}"

    # Why it is large: the data determine ten degrees of freedom, not 64.
    def over_weights(weights):
        return objective({**start, "weights": weights})

    fisher = fisher_information(over_weights, start["weights"])
    eigenvalues, directions = degenerate_directions(fisher, rtol=1.0e-3)
    assert float(eigenvalues[0]) == pytest.approx(MU, rel=1e-6)
    assert directions.shape[1] == 55

    data_curvature = fisher_information(
        lambda w: -problem.log_likelihood({**start, "weights": w}),
        start["weights"],
    )
    prior_curvature = fisher_information(
        lambda w: -problem.log_prior({**start, "weights": w}), start["weights"]
    )
    effective = float(effective_parameters(data_curvature, prior_curvature))
    assert effective == pytest.approx(
        problem.observed.size, rel=1.0e-3
    ), f"the data determined {effective:.4f} of 64 weights"


def test_the_tracer_objective_has_one_minimum_and_everything_agrees_on_it(
    key,
) -> None:
    """Strictly convex in the weights, so Newton from anywhere finds the same point.

    The correction that reorganised the rest of this file. The tracer objective
    is ``½‖(K̄ᵀw − Y)/σ‖² − mu S(w)``: chi-squared is convex in ``w`` because the
    model is *linear* in the weights, ``−mu S`` is convex because ``S`` is
    concave, and the sum's Hessian ``K̄Σ⁻¹K̄ᵀ + mu diag(1/w)`` has smallest
    eigenvalue ``mu / max w > 0``. So the minimiser is **unique** -- there is no
    flat valley of stationary points, only a very anisotropic bowl with
    condition number 2.6e+07.

    Measured: a damped Newton iteration in ``log w`` from four starting points
    (uniform, half the truth, twice the truth, random) reaches the same weights
    to **1.2e-15** relative, with the objective identical to twelve digits
    (-6.328701800494e-02) and ``|dF/dw|`` of order 1e-11.

    This is why the earlier reading of this suite was wrong. The spread between
    optimizers -- weight errors from 0.246 to 2.409 in one table -- was **not**
    different optimizers stopping at different valid optima. It was
    non-convergence along the low-curvature directions of a single bowl, made
    much worse by the over-parameterisation defect. With that fixed and a
    quasi-Newton rule, every method that converges lands on the same answer.
    """
    problem, truth, start = _tracer_problem(key, n=64, num_steps=24)
    objective = problem.negative_log_posterior

    def over_weights(weights):
        return objective({**start, "weights": weights})

    # Strict convexity, at three unrelated points.
    for weights in (
        start["weights"],
        truth,
        0.3 + 2.0 * jax.random.uniform(jax.random.fold_in(key, 11), (64,)),
    ):
        eigenvalues = jnp.linalg.eigvalsh(jax.hessian(over_weights)(weights))
        smallest = float(eigenvalues[0])
        # Weyl: lambda_min(A + B) >= lambda_min(A) + lambda_min(B), and the data
        # term is positive semi-definite while the prior's is mu / w. So the
        # bound is mu / max(w) -- a bound, not an equality: the data term need
        # not vanish along the eigenvector that attains it. Measured 6.92e-04
        # against a bound of 6.72e-04 at the truth, 4.37e-04 against 4.36e-04 at
        # a random point.
        #
        # The slack on the bound is the eigensolver's, and its size is derivable
        # rather than guessed: `eigvalsh` resolves an eigenvalue to about
        # `eps * lambda_max`, which here is 2.2e-16 * 2.57e+04 = 5.7e-12, i.e.
        # 5.7e-09 relative to a smallest eigenvalue of 1e-03. At the uniform
        # point, where the bound is attained exactly, the computed value sits
        # 6e-09 below it -- so the tolerance is 1e-07 relative, two decades
        # above eps * cond and still four decades tighter than the effect
        # being tested.
        assert smallest > 0.0, f"smallest eigenvalue {smallest:.3e}"
        assert smallest >= MU / float(jnp.max(weights)) * (1.0 - 1.0e-7)

    def newton(weights, steps=60):
        for _ in range(steps):
            gradient = jax.grad(over_weights)(weights)
            hessian = jax.hessian(over_weights)(weights)
            # Newton in u = log w, so the iterate cannot leave w > 0.
            gradient_u = gradient * weights
            hessian_u = hessian * weights[:, None] * weights[None, :] + jnp.diag(
                gradient_u
            )
            step = jnp.clip(jnp.linalg.solve(hessian_u, -gradient_u), -1.0, 1.0)
            weights = jnp.exp(jnp.log(weights) + step)
        return weights

    solutions = [
        newton(w0)
        for w0 in (
            jnp.ones(64),
            0.5 * truth,
            2.0 * truth,
            0.3 + 2.0 * jax.random.uniform(jax.random.fold_in(key, 13), (64,)),
        )
    ]
    reference = solutions[0]
    for other in solutions[1:]:
        relative = float(
            jnp.linalg.norm(other - reference) / jnp.linalg.norm(reference)
        )
        assert relative < 1.0e-12, f"Newton solutions differ by {relative:.3e}"
    assert float(jnp.linalg.norm(jax.grad(over_weights)(reference))) < 1.0e-8

    # And a quasi-Newton fit through the public API reaches it too.
    fit = MadeToMeasure(optimizer=optax.lbfgs(), num_steps=500).minimize(
        objective, start
    )
    relative = float(
        jnp.linalg.norm(fit.params["weights"] - reference) / jnp.linalg.norm(reference)
    )
    assert relative < 1.0e-3, f"lbfgs is {relative:.3e} from the unique minimum"


# --- item 4: the classic iteration against the differentiable one -------------------


def test_classic_and_differentiable_agree_on_the_tracer_problem(key) -> None:
    """Item 4: both iterations, same objective, same rollout, measured agreement.

    Re-measured weights-only, and the answer is much better than the first
    reading of it. The classic force of change (``epsilon = 3e-5``, 30000 steps)
    reaches the unique minimum to **3.3e-08** relative -- the best of any method
    tried, because its ``diag(w)`` preconditioner happens to suit this bowl's
    geometry. Against it:

    ==============================  =====================  =============
    rule                            distance to minimum    weight error
    ==============================  =====================  =============
    classic FOC, ``eps = 3e-5``     **3.3e-08**            0.2169
    ``lbfgs()`` x200                6.1e-04                0.2170
    ``adam(0.1)`` x10000            1.9e-02                0.2196
    ==============================  =====================  =============

    Adam and the classic iteration differ by 1.9e-02 in the weights and
    1.1e-05 in the objective; LBFGS and the classic iteration by 6.1e-04. The
    residual disagreement is Adam not converging along the low-curvature
    directions, **not** two methods choosing different points -- there is only
    one point (see
    ``test_the_tracer_objective_has_one_minimum_and_everything_agrees_on_it``).

    The earlier version of this test reported a 0.173 weight gap and explained
    it as "the size of the flat valley". That explanation was wrong on both
    counts: the fit was over-parameterised, and the objective has a unique
    minimum. What survives is the weaker and true claim -- the two iterations
    implement the same method and agree to the accuracy each is run to.
    """
    problem, truth, start = _tracer_problem(key, n=64, num_steps=24)
    objective = problem.negative_log_posterior
    classic = MadeToMeasure(num_steps=30000, epsilon=3.0e-5).force_of_change(
        objective, start
    )
    quasi_newton = MadeToMeasure(optimizer=optax.lbfgs(), num_steps=200).minimize(
        objective, start
    )
    adam = made_to_measure(0.1, 10000).minimize(objective, start)

    for label, result in (
        ("classic", classic),
        ("lbfgs", quasi_newton),
        ("adam", adam),
    ):
        assert jnp.all(jnp.isfinite(result.params["weights"])), label
        assert jnp.all(result.params["weights"] > 0.0), label
        error = float(
            jnp.linalg.norm(result.params["weights"] - truth) / jnp.linalg.norm(truth)
        )
        assert 0.2 < error < 0.24, f"{label} weight error {error:.4f}"

    reference = classic.params["weights"]
    for label, result, bound in (
        ("lbfgs", quasi_newton, 5.0e-3),
        ("adam", adam, 5.0e-2),
    ):
        gap = float(
            jnp.linalg.norm(result.params["weights"] - reference)
            / jnp.linalg.norm(reference)
        )
        assert gap < bound, f"{label} is {gap:.3e} from the classic iteration"

    objective_gap = float(
        abs(quasi_newton.objective_trace[-1] - classic.objective_trace[-1])
    )
    assert objective_gap < 1.0e-5, f"objectives differ by {objective_gap:.3e}"


def test_adam_does_not_settle_on_this_objective(key) -> None:
    """Adam orbits the minimum instead of settling, and the trace is how you see it.

    Not divergence -- the earlier reading of this, taken from the
    over-parameterised fit, called it that. Measured weights-only at learning
    rate 0.1: the objective is ``+5.25e+01`` after 3000 steps and
    ``-6.328e-02`` after 20000; at 0.02 it is ``-6.3270e-02`` after 3000 and
    slightly *worse*, ``-6.3236e-02``, after 20000. Every Adam rate from 0.01 to
    0.5 ends between 1.9e-02 and 2.8e-02 from the unique minimum whatever the
    schedule, while LBFGS reaches 6.1e-05.

    The cause is the bowl's condition number, 2.6e+07: Adam's per-coordinate
    normalization keeps taking steps of order the learning rate along directions
    whose curvature is ``mu``, so it cannot come to rest there. That is a
    property of the objective, and the reason the module does not hard-code a
    rule. Read ``fit.objective_trace``, which is returned for exactly this.
    """
    problem, _, start = _tracer_problem(key, n=64, num_steps=24)
    objective = problem.negative_log_posterior
    short = made_to_measure(0.1, 3000).minimize(objective, start)
    long = made_to_measure(0.1, 20000).minimize(objective, start)
    # More steps is better here, but non-monotonically: the short run is above
    # zero while the long one has come back down.
    assert float(short.objective_trace[-1]) > 1.0
    assert float(long.objective_trace[-1]) < 0.0
    # And at a smaller rate more steps is *worse*, which is the point.
    fewer = made_to_measure(0.02, 3000).minimize(objective, start)
    more = made_to_measure(0.02, 20000).minimize(objective, start)
    assert float(more.objective_trace[-1]) > float(fewer.objective_trace[-1])


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

    Re-measured weights-only after the over-parameterisation fix, and with a
        quasi-Newton rule for comparison:

        ==========  ====================  ==================  =============
        rule        ``chi²``              weight error
        ==========  ====================  ==================  =============
        ``adam(0.01)`` x400   8.03e+04 -> 3.33e+00   0.0877 -> 0.0749
        ``adam(0.01)`` x2000  8.03e+04 -> 2.30e-02   0.0877 -> 0.0759
        ``lbfgs()`` x300      8.03e+04 -> **5.11e-06**  0.0877 -> **0.0688**
        ==========  ====================  ==================  =============

        So the fit does reduce the weight error, by 22 % of it with LBFGS, and it
        stalls there. With 32 weights against 10 observables only **four** Fisher
        directions rise above ``1e-3`` of the largest (condition number 6.1e+09),
        the random perturbation puts 47.3 % of its norm in that four-dimensional
        subspace, and removing all of it would leave 0.0773 -- which LBFGS beats,
        because the entropy prior is centred on the truth here and supplies real
        information along the rest. On this problem the classic force of change
        diverges at every ``epsilon`` tried and a fixed-step SGD goes non-finite:
        the objective is far stiffer than the tracer one (start ``|dF/dw| =
        5.1e+05``).

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
    fit = MadeToMeasure(optimizer=optax.lbfgs(), num_steps=300).minimize(
        problem.negative_log_posterior, start
    )
    # Weights only: the initial conditions are inputs, not parameters.
    for leaf in ("positions", "velocities"):
        assert jnp.array_equal(fit.params[leaf], start[leaf]), leaf
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


# --- how far the autodiff gradient has actually been checked -----------------------
#
# The finite-difference curves above are one oracle, and a finite difference is
# the weakest one available: it has a step-size error that must be argued about,
# and on a degenerate problem it probes a single direction. The three tests below
# are the stronger checks, and they exist because the classic
# `force_of_change` iteration is *not* one -- it calls `jax.grad` on the same
# objective `minimize` does, so it validates the iteration and its fixed point
# and shares every possible error in the gradient itself.


def test_the_tracer_gradient_matches_a_closed_form_with_no_autodiff(key) -> None:
    """The tracer objective's gradient in closed form, against autodiff.

    With the orbits independent of the weights the whole objective is available
    on paper. Writing ``Kbar`` for the time-averaged kernel and ``w0 = 1``:

        F(w)      = 0.5 |(Kbar^T w - Y) / sigma|^2  -  mu S(w)
        dF/dw     = Kbar (Kbar^T w - Y) / sigma^2   +  mu log(w)

    using ``dS/dw = -log(w / w0)``. **No autodiff appears anywhere in that
    reference** -- not even for the entropy term -- so this is a genuinely
    independent oracle for the gradient through a real 12-base-step rollout,
    and unlike a finite difference it has no step-size error to argue about.

    Measured: ``|AD - closed form| / |closed form| = 6.5e-16`` on a gradient of
    norm 7.68e+03, worst component 1.6e-12 absolute. That is round-off, and it
    covers the whole chain -- the rollout's records, the weights broadcast along
    the time axis, the time average, the kernel contraction, the likelihood and
    the prior.
    """
    problem, _, start = _tracer_problem(key, n=32, num_steps=12)
    params = {**start, "weights": start["weights"] * 1.3}
    kernel = GaussianRadialBins(
        centres=jnp.asarray([0.4, 0.8, 1.2, 1.6, 2.0]), width=0.3
    )

    trajectory = problem.forward(params)
    steps = trajectory["positions"].shape[0]
    averaged_kernel = jnp.mean(
        jnp.stack(
            [
                kernel({leaf: value[t] for leaf, value in trajectory.items()})
                for t in range(steps)
            ]
        ),
        axis=0,
    )
    residual = averaged_kernel.T @ params["weights"] - problem.observed
    closed_form = averaged_kernel @ (residual / SIGMA**2) + MU * jnp.log(
        params["weights"]
    )

    by_autodiff = jax.grad(problem.negative_log_posterior)(params)["weights"]
    relative = float(
        jnp.linalg.norm(by_autodiff - closed_form) / jnp.linalg.norm(closed_form)
    )
    assert relative < 1.0e-13, f"AD vs closed form: {relative:.3e}"


def test_forward_and_reverse_mode_agree_through_the_rollout(key) -> None:
    """``jax.jvp`` against ``jax.grad``: two independent autodiff transformations.

    A stronger check than a finite difference where it applies, because it has
    no truncation error -- any disagreement is a bug in one of the two paths,
    not a step size. Measured: 4.1e-16 relative in the tracer construction,
    4.4e-16 in the self-consistent one, and 2.6e-16 on the production
    (``reassign_rungs=True``) path, which no finite difference can check at all
    because its schedule is not a smooth function of the weights.
    """
    k_problem, k_direction = jax.random.split(key)
    problem, _, start = _tracer_problem(k_problem, n=32, num_steps=12)
    n = start["weights"].size
    params = {**start, "weights": start["weights"] * 1.3}
    direction = jax.random.normal(k_direction, (n,))
    direction = direction / jnp.linalg.norm(direction)

    objective = problem.negative_log_posterior
    reverse = float(jnp.sum(jax.grad(objective)(params)["weights"] * direction))
    forward = _forward_mode_directional(objective, params, direction)
    assert abs(reverse - forward) <= 1.0e-13 * abs(reverse)

    # And on the self-consistent construction, where the weights move the orbits.
    k_system, k_weights, k_dir = jax.random.split(key, 3)
    m = 16
    positions, velocities = _plummer_like(k_system, m)
    weights = 0.5 + jax.random.uniform(k_weights, (m,))
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
    consistent = {
        "positions": positions,
        "velocities": velocities,
        "weights": weights,
    }
    problem = InferenceProblem(
        forward=rollout,
        observable=observable,
        likelihood=GaussianLikelihood(sigma=SIGMA),
        observed=observable(rollout(consistent)) + 0.05,
        priors=(EntropyPrior(mu=MU),),
    )
    objective = problem.negative_log_posterior
    direction = jax.random.normal(k_dir, (m,))
    direction = direction / jnp.linalg.norm(direction)
    reverse = float(jnp.sum(jax.grad(objective)(consistent)["weights"] * direction))
    forward = _forward_mode_directional(objective, consistent, direction)
    assert abs(reverse - forward) <= 1.0e-13 * abs(reverse)


# --- the multi-rung path, which the k_max=1 tests above do NOT exercise ------------


def _multi_rung_problem(key, *, dt, reassign_rungs, **kwargs):
    """Build a genuinely multi-rung self-consistent problem and its pieces.

    ``dt`` is a parameter because it decides the *schedule*, and the schedule is
    what these tests are about: at ``dt = 0.1`` the frozen and production
    schedules nearly coincide, and at ``dt = 0.05`` they do not.
    """
    k_system, k_weights = jax.random.split(key)
    n = 16
    positions, velocities = _log_spaced(k_system, n, rmax=4.0)
    weights = 0.5 + jax.random.uniform(k_weights, (n,))
    force = MutualDirectSumGravity(G=1.0, softening=SOFTENING, k_max=2)
    rollout = NornaxRollout.self_consistent(
        force,
        dt=dt,
        num_steps=8,
        k_max=2,
        eta=0.1,
        eps=1.0,
        reassign_rungs=reassign_rungs,
        **kwargs,
    )
    observable = TimeAverage(
        WeightedKernelSum(
            GaussianRadialBins(centres=jnp.asarray([0.5, 1.0, 2.0, 3.0]), width=0.4)
        )
    )
    params = {
        "positions": positions,
        "velocities": velocities,
        "weights": weights,
    }
    return rollout, observable, params


def _objective_for(rollout, observable, observed):
    """The made-to-measure objective for one rollout and one data vector."""
    return InferenceProblem(
        forward=rollout,
        observable=observable,
        likelihood=GaussianLikelihood(sigma=SIGMA),
        observed=observed,
        priors=(EntropyPrior(mu=MU),),
    ).negative_log_posterior


def test_the_multi_rung_gradient_is_right_and_the_schedule_is_not_vacuous(
    key,
) -> None:
    """A three-rung rollout: FD, forward mode, and the rung histogram asserted.

    The ``k_max = 1`` tests earlier in this file pass, and until this test was
    written they were **vacuous** in exactly the dimension they were meant to
    cover: with :func:`_plummer_like`'s narrow acceleration spread every
    particle lands on rung 0, so ``k_max = 1`` silently runs the single-rung
    reduced case and the block schedule is never exercised. The histogram is
    asserted here for the same reason jaccpot asserts ``num_far_pairs > 0``
    before believing an FMM number.

    On the log-spaced system the schedule is ``{0: 7, 1: 6, 2: 3}`` -- all three
    levels evaluated, real sub-step structure. Measured there, frozen schedule,
    AD = 7.412948423875e+01::

        h     1e-2     1e-3     1e-4     1e-5     1e-6     1e-7     1e-8
        rel   1.06e-4  1.06e-6  1.06e-8  8.95e-11 8.17e-10 4.30e-9  1.33e-7

    The same clean ``O(h^2)`` truncation branch and round-off floor as the
    single-rung case, minimum 9.0e-11 relative at h = 1e-5. Forward mode agrees
    with reverse mode to **5.8e-16**, and that is the assertion that really pins
    the multi-rung arithmetic: it has no step-size error to trade against.
    """
    rollout, observable, params = _multi_rung_problem(
        key, dt=0.05, reassign_rungs=False
    )
    histogram = _rung_histogram(rollout.final_state(params))
    assert len(histogram) == 3, f"the rung schedule is vacuous: {histogram}"
    assert histogram == {0: 7, 1: 6, 2: 3}

    objective = _objective_for(rollout, observable, observable(rollout(params)) + 0.05)
    n = params["weights"].size
    direction = jax.random.normal(jax.random.fold_in(key, 3), (n,))
    direction = direction / jnp.linalg.norm(direction)

    ad, curve = _fd_curve(objective, params, direction, (1.0e-5,))
    assert curve[1.0e-5] <= 1.0e-9 * abs(ad), f"|AD - FD| = {curve[1.0e-5]:.3e}"
    forward = _forward_mode_directional(objective, params, direction)
    assert abs(ad - forward) <= 1.0e-13 * abs(ad)


@pytest.mark.parametrize(
    ("label", "knobs", "bound"),
    [
        ("checkpoint=False", {"checkpoint": False}, 0.0),
        (
            "checkpoint_substeps=True",
            {"checkpoint": True, "checkpoint_substeps": True},
            1.0e-13,
        ),
    ],
)
def test_rematerialization_does_not_move_the_multi_rung_gradient(
    key, label, knobs, bound
) -> None:
    """`checkpoint` and `checkpoint_substeps` are memory schedules, not results.

    Measured on the three-rung system, relative to the default gradient:
    ``checkpoint=False`` is **exactly zero** -- bit-identical, which is what
    rematerializing the same arithmetic should give -- and
    ``checkpoint_substeps=True`` is **8.8e-15**, round-off rather than zero,
    because remating a sub-step boundary's kick changes the order the pair
    contributions are summed in. Worth stating precisely rather than rounding to
    "unchanged": one of these two knobs is exact and the other is exact to
    round-off, and a future change that made either worse should fail here.

    ``checkpoint_substeps`` is exposed on the adapter because of what nornax's
    rollout says about it -- it bounds the per-base-step backward memory to one
    boundary's pair tensors, and deep-``k_max`` gradients otherwise run out of
    memory. That is precisely the regime a made-to-measure fit at FMM scale
    lands in, so leaving it reachable only by bypassing the adapter would have
    meant the adapter could not express the fit the module exists for.
    """
    rollout, observable, params = _multi_rung_problem(key, dt=0.1, reassign_rungs=False)
    observed = observable(rollout(params)) + 0.05
    baseline = jax.grad(_objective_for(rollout, observable, observed))(params)[
        "weights"
    ]
    variant, _, _ = _multi_rung_problem(key, dt=0.1, reassign_rungs=False, **knobs)
    got = jax.grad(_objective_for(variant, observable, observed))(params)["weights"]
    relative = float(jnp.linalg.norm(got - baseline) / jnp.linalg.norm(baseline))
    assert relative <= bound, f"{label} moved the gradient by {relative:.3e}"


@pytest.mark.parametrize(
    ("dt", "frozen_schedule", "live_schedule", "lo", "hi"),
    [
        (0.1, {1: 7, 2: 9}, {1: 4, 2: 12}, 1.0e-3, 0.5),
        (0.05, {0: 7, 1: 6, 2: 3}, {0: 7, 1: 2, 2: 7}, 0.3, 1.5),
    ],
)
def test_the_production_gradient_tracks_the_schedule_it_realised(
    key, dt, frozen_schedule, live_schedule, lo, hi
) -> None:
    """`reassign_rungs=True` is a different map, and how different is not bounded.

    Every finite-difference test in this file freezes the schedule, because a
    finite difference of the production path compares two rollouts that may have
    realised *different* schedules. That leaves the production default -- what a
    real fit runs -- unmeasured, and this test measures it.

    The answer is not a number, it is a dependence. The production gradient
    differs from the frozen-schedule one by however much the two **schedules**
    differ, and that is set by the configuration:

    ======  =========================  =======================  =========  ======
    ``dt``  frozen schedule            production schedule      relative   cosine
    ======  =========================  =======================  =========  ======
    0.1     ``{1: 7, 2: 9}``           ``{1: 4, 2: 12}``        7.2e-02    0.998
    0.05    ``{0: 7, 1: 6, 2: 3}``     ``{0: 7, 1: 2, 2: 7}``   **0.65**   0.953
    ======  =========================  =======================  =========  ======

    Three particles move rung between the two ``dt = 0.05`` schedules and the
    gradient changes by **65 %** in norm; four move at ``dt = 0.1`` and it
    changes by 7 %. The size of the effect is set by the schedule difference and
    by nothing a caller can read off in advance. On other configurations tried
    while writing this test the two gradients came out **nearly antiparallel**
    (cosine -0.987), so 65 % is not a ceiling.

    Neither gradient is wrong. nornax severs the rung assignment from the
    gradient, so each is the exact gradient of the map its own forward pass
    realised -- which this test also asserts, forward against reverse mode at
    round-off on the production path, where no finite difference can check
    anything. What the table means is that the production objective is only
    *piecewise* smooth in the weights: a weight change that moves a particle
    across a rung boundary lands on a neighbouring map with a kink between. So a
    finite-difference check on a frozen schedule does **not** license the
    production path, and a fit that must descend one smooth objective should
    freeze the schedule. That caveat is now on the adapter, where a caller will
    see it.
    """
    frozen_rollout, observable, params = _multi_rung_problem(
        key, dt=dt, reassign_rungs=False
    )
    live_rollout, _, _ = _multi_rung_problem(key, dt=dt, reassign_rungs=True)
    assert _rung_histogram(frozen_rollout.final_state(params)) == frozen_schedule
    assert _rung_histogram(live_rollout.final_state(params)) == live_schedule

    observed = observable(frozen_rollout(params)) + 0.05
    frozen = jax.grad(_objective_for(frozen_rollout, observable, observed))(params)[
        "weights"
    ]
    live_objective = _objective_for(live_rollout, observable, observed)
    live = jax.grad(live_objective)(params)["weights"]

    relative = float(jnp.linalg.norm(live - frozen) / jnp.linalg.norm(frozen))
    assert lo <= relative <= hi, f"relative difference {relative:.3e}"

    # The production gradient is still an exact gradient -- of its own map.
    n = params["weights"].size
    direction = jax.random.normal(jax.random.fold_in(key, 7), (n,))
    direction = direction / jnp.linalg.norm(direction)
    reverse = float(jnp.sum(live * direction))
    forward = _forward_mode_directional(live_objective, params, direction)
    assert abs(reverse - forward) <= 1.0e-13 * abs(reverse)


# --- which optimizer, measured on the real problem --------------------------------


def test_a_well_conditioned_rule_is_what_reaches_the_minimum(key) -> None:
    """The measured answer to "which rule should a made-to-measure fit use".

    The objective's Fisher condition number is **2.6e+07** on the tracer problem
    and **6.1e+09** on the self-consistent one, so this is a question about
    conditioning, not about step sizes. Measured, weights only, distance to the
    unique minimum in relative L2:

    ================================  ==============  =============
    rule                              tracer          weight error
    ================================  ==============  =============
    classic FOC, ``eps = 3e-5``       **3.3e-08**     0.2169
    ``lbfgs()`` x500                  6.1e-05         0.2169
    ``lbfgs()`` x200                  6.1e-04         0.2170
    ``sgd(1e-5)``                     1.2e-02         0.2182
    ``adam``, every rate 0.01 … 0.5   1.9e-02 … 2.8e-02  0.2195 … 0.2206
    ``adam`` + cosine / exp decay     1.9e-02         0.2193 … 0.2195
    ================================  ==============  =============

    And on the self-consistent problem (N = 32, ``k_max = 1``, start
    ``|dF/dw| = 5.1e+05``), where the orbits move with the weights:

    ================================  ==========  ==========  =============
    rule                              ``|g|``     ``chi2``    weight error
    ================================  ==========  ==========  =============
    ``lbfgs()`` x300                  **6.6e-02** **5.1e-06** **0.0688**
    ``adam(0.01)`` x2000              1.4e+00     2.3e-02     0.0759
    ``sgd(1e-6)``                     non-finite  —           —
    classic FOC, any ``eps`` tried    diverges    —           —
    ================================  ==========  ==========  =============

    Three things worth keeping. **Adam plateaus**: every rate and every schedule
    tried lands 1.9e-02 to 2.8e-02 from the minimum, because its per-coordinate
    normalization keeps taking steps of order the learning rate along directions
    whose curvature is ``mu``. **The classic reweighting is the best rule on the
    tracer problem** -- Syer & Tremaine's ``diag(w)`` preconditioner suits that
    bowl, which is a nice thing for a 1996 algorithm to be right about -- and it
    **diverges** on the self-consistent one, where it is also the wrong
    gradient. **LBFGS is the only rule that works well on both**, and it is
    therefore the one to reach for; it is also the reason `minimize` had to learn
    to pass a line search its ``value``/``grad``/``value_fn``.

    So the answer to "should we use a more involved scheme, given the eigenvalue
    spread" is yes, and it is not a speed-up: at this conditioning a first-order
    rule does not reach the answer at all, and the answer is well defined
    (the objective is strictly convex) so failing to reach it is purely the
    optimizer's fault.
    """
    problem, truth, start = _tracer_problem(key, n=64, num_steps=24)
    objective = problem.negative_log_posterior

    quasi_newton = MadeToMeasure(optimizer=optax.lbfgs(), num_steps=500).minimize(
        objective, start
    )
    classic = MadeToMeasure(num_steps=30000, epsilon=3.0e-5).force_of_change(
        objective, start
    )
    adam = made_to_measure(0.1, 10000).minimize(objective, start)
    scheduled = made_to_measure(
        optax.exponential_decay(0.1, 2000, 0.3), 10000
    ).minimize(objective, start)

    for label, result in (
        ("lbfgs", quasi_newton),
        ("classic", classic),
        ("adam", adam),
        ("adam+schedule", scheduled),
    ):
        assert jnp.all(jnp.isfinite(result.params["weights"])), label
        assert jnp.all(result.params["weights"] > 0.0), label

    reference = classic.params["weights"]

    def distance(result):
        return float(
            jnp.linalg.norm(result.params["weights"] - reference)
            / jnp.linalg.norm(reference)
        )

    # The well-conditioned rules reach the minimum; Adam plateaus an order of
    # magnitude short of them, and a schedule does not rescue it.
    assert distance(quasi_newton) < 1.0e-3, f"lbfgs {distance(quasi_newton):.3e}"
    assert distance(adam) > 10.0 * distance(quasi_newton)
    assert distance(scheduled) > 10.0 * distance(quasi_newton)
    assert 1.0e-2 < distance(adam) < 5.0e-2


# --- folding the observable into the rollout: the memory schedule -----------------


@pytest.mark.parametrize("decay_steps", [None, 6.0])
@pytest.mark.parametrize("segments", [1, 2, 4, 6, 12, 24])
def test_folding_reproduces_the_stacked_path_exactly(key, decay_steps, segments):
    """`FoldedRollout` is a memory schedule, so it must not move the answer.

    The whole justification for folding the observable into the rollout and
    chaining checkpointed segments is that it computes the same number more
    cheaply. This asserts that, in the value **and** in the gradient, over every
    divisor of a 24-step rollout and for both averaging modes. Measured: value
    agreement 1.1e-16 for the mean and 6.0e-16 for the decaying average,
    gradients 1.0e-16 to 6.0e-16 -- round-off, independent of the segment count.

    The decaying average is the case worth parametrising: the segments are not
    interchangeable there, so a fold that lost the global step order, or a
    normalization that used the per-segment count instead of the total, would
    show up here and nowhere else.
    """
    k_system, k_weights = jax.random.split(key)
    n, steps = 32, 24
    positions, velocities = _log_spaced(k_system, n)
    weights = 0.5 + jax.random.uniform(k_weights, (n,))
    params = {
        "positions": positions,
        "velocities": velocities,
        "weights": weights,
    }
    rollout = NornaxRollout.self_consistent(
        MutualDirectSumGravity(G=1.0, softening=SOFTENING, k_max=0),
        dt=0.02,
        num_steps=steps,
        k_max=0,
        reassign_rungs=False,
    )
    observable = TimeAverage(
        WeightedKernelSum(
            GaussianRadialBins(centres=jnp.asarray([0.5, 1.0, 2.0]), width=0.4)
        ),
        decay_steps=decay_steps,
    )
    folded = rollout.folding(observable, segments=segments)

    stacked = observable(rollout(params))
    got = folded(params)
    relative = float(jnp.linalg.norm(got - stacked) / jnp.linalg.norm(stacked))
    assert relative < 1.0e-14, f"folding moved the value by {relative:.3e}"

    def through_folded(w):
        return jnp.sum(folded({**params, "weights": w}))

    def through_stacked(w):
        return jnp.sum(observable(rollout({**params, "weights": w})))

    folded_gradient = jax.grad(through_folded)(weights)
    stacked_gradient = jax.grad(through_stacked)(weights)
    relative = float(
        jnp.linalg.norm(folded_gradient - stacked_gradient)
        / jnp.linalg.norm(stacked_gradient)
    )
    assert relative < 1.0e-13, f"folding moved the gradient by {relative:.3e}"


def test_folding_and_segmenting_cut_the_gradient_s_memory(key) -> None:
    """The point of the exercise, measured with XLA's own accounting.

    Peak scratch for one gradient of the full negative log posterior, from
    ``memory_analysis().temp_size_in_bytes``:

    =====  ======  ==========  ==============  ====================  =========
    ``n``  ``t``   stacked     folded, ``S=1`` folded, ``balanced``  reduction
    =====  ======  ==========  ==============  ====================  =========
    64     64      1.8 MB      2.0 MB          1.0 MB  (S=8)         1.8x
    64     256     6.8 MB      5.5 MB          1.2 MB  (S=16)        5.8x
    64     1024    27.1 MB     19.5 MB         1.5 MB  (S=32)        **17.9x**
    256    256     28.6 MB     31.4 MB         14.2 MB (S=16)        2.0x
    256    1024    108.1 MB    87.3 MB         15.6 MB (S=32)        6.9x
    =====  ======  ==========  ==============  ====================  =========

    Two things to read off. The stacked column is **linear in t** and the
    balanced column is nearly flat, which is the ``O(sqrt(t))`` law. And folding
    *alone* (``S = 1``) buys little or nothing -- it removes the stacked
    trajectory but leaves the scan's per-step carries -- so the segmenting is
    where the win is, and folding is what makes segmenting possible.

    The reduction is smaller at ``n = 256`` (6.9x against 17.9x) and that is
    honest rather than disappointing: segmenting shrinks the ``O(t * n)`` term
    and leaves the direct sum's ``O(n^2)`` per-step scratch untouched. An FMM
    changes that term to ``O(n log n)`` but it stays per-step and stays outside
    what this buys.
    """
    n, steps = 64, 256
    k_system, k_weights = jax.random.split(key)
    positions, velocities = _log_spaced(k_system, n)
    params = {
        "positions": positions,
        "velocities": velocities,
        "weights": 0.5 + jax.random.uniform(k_weights, (n,)),
    }
    rollout = NornaxRollout.self_consistent(
        MutualDirectSumGravity(G=1.0, softening=SOFTENING, k_max=0),
        dt=0.02,
        num_steps=steps,
        k_max=0,
        reassign_rungs=False,
    )
    observable = TimeAverage(
        WeightedKernelSum(
            GaussianRadialBins(centres=jnp.asarray([0.5, 1.0, 2.0, 3.0]), width=0.4)
        )
    )

    def scratch(forward, inner, observed):
        problem = InferenceProblem(
            forward=forward,
            observable=inner,
            likelihood=GaussianLikelihood(sigma=SIGMA),
            observed=observed,
            priors=(EntropyPrior(mu=MU),),
        )
        compiled = (
            jax.jit(jax.grad(problem.negative_log_posterior)).lower(params).compile()
        )
        return compiled.memory_analysis().temp_size_in_bytes

    balanced = FoldedRollout.balanced(rollout, observable)
    assert balanced.segments == 16, balanced.segments

    stacked_bytes = scratch(rollout, observable, observable(rollout(params)) + 0.05)
    folded_bytes = scratch(balanced, IdentityObservable(), balanced(params) + 0.05)
    assert folded_bytes < stacked_bytes / 3.0, (
        f"stacked {stacked_bytes / 1e6:.1f} MB vs folded "
        f"{folded_bytes / 1e6:.1f} MB"
    )
