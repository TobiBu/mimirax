"""Tests for residual, convergence and degeneracy diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mimirax import (
    EntropyPrior,
    GaussianRadialBins,
    WeightedKernelSum,
    degenerate_directions,
    effective_parameters,
    effective_parameters_from_jacobian,
    fisher_information,
    has_converged,
    l_curve_corner,
    normalized_residuals,
    profile_likelihood,
    relative_change,
    residuals,
    rms_residual,
)


def test_residual_helpers() -> None:
    """Signed, normalized and rms residuals agree with their definitions."""
    p = jnp.asarray([1.0, 3.0])
    o = jnp.asarray([0.0, 1.0])
    assert jnp.array_equal(residuals(p, o), jnp.asarray([1.0, 2.0]))
    assert jnp.array_equal(normalized_residuals(p, o, 2.0), jnp.asarray([0.5, 1.0]))
    assert float(rms_residual(p, o)) == pytest.approx((2.5) ** 0.5)


def test_convergence_from_a_trace() -> None:
    """A geometrically decaying trace converges once its relative step is below rtol."""
    trace = 1.0 + 0.5 ** jnp.arange(40.0)
    assert relative_change(trace).shape == (39,)
    assert has_converged(trace, rtol=1.0e-6, window=5)
    assert not has_converged(trace[:8], rtol=1.0e-6, window=5)
    assert not has_converged(trace[:3], rtol=1.0e-6, window=5)


def test_fisher_information_of_a_quadratic_is_its_hessian() -> None:
    """For 0.5 x^T H x the Fisher matrix is H, in ravel order over a pytree."""
    h = jnp.asarray([[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]])

    def nlp(p):
        x = jnp.concatenate([p["u"], p["v"]])
        return 0.5 * x @ h @ x

    params = {"u": jnp.zeros(2), "v": jnp.zeros(1)}
    fisher = fisher_information(nlp, params)
    assert jnp.allclose(fisher, h, atol=1.0e-14)
    eigenvalues, directions = degenerate_directions(fisher, rtol=1.0e-10)
    assert eigenvalues.shape == (3,) and directions.shape == (3, 1)
    assert jnp.allclose(
        jnp.abs(directions[:, 0]), jnp.asarray([0.0, 0.0, 1.0]), atol=1e-14
    )


def test_profile_likelihood_is_a_stub() -> None:
    """profile_likelihood raises until a method session writes it."""
    with pytest.raises(NotImplementedError):
        profile_likelihood(
            lambda p: jnp.sum(p**2), jnp.zeros(2), jnp.asarray([1.0, 0.0])
        )


def test_effective_parameters_counts_what_the_data_determined() -> None:
    """`tr(H_data (H_data + H_prior)^-1)`: 1 per constrained direction, 0 per prior one.

    Built so the answer is exact. Three parameters, a prior of curvature `mu` on
    all of them, and data curvature that is large on one direction, comparable to
    the prior on the second and zero on the third. The effective count is then
    `1 + 1/2 + 0` to the accuracy of the "large" direction.

    This is the diagnostic to reach for instead of counting eigenvalues below an
    `rtol`: it needs no threshold, and on the made-to-measure tracer problem it
    returns exactly the number of observables while a threshold count reports
    55 of 64 degenerate directions.
    """
    mu = 1.0e-3
    prior = mu * jnp.eye(3)
    data = jnp.diag(jnp.asarray([1.0e6 * mu, mu, 0.0]))
    got = float(effective_parameters(data, prior))
    assert got == pytest.approx(1.0e6 / (1.0e6 + 1.0) + 0.5, rel=1e-12)


def test_effective_parameters_brackets_are_zero_and_d() -> None:
    """An overwhelming prior determines nothing; a dominant likelihood determines all."""
    data = jnp.diag(jnp.asarray([2.0, 5.0, 1.0]))
    assert float(effective_parameters(data, 1.0e12 * jnp.eye(3))) == pytest.approx(
        0.0, abs=1e-10
    )
    assert float(effective_parameters(data, 1.0e-9 * jnp.eye(3))) == pytest.approx(
        3.0, rel=1e-8
    )


def test_effective_parameters_from_jacobian_is_the_dense_route_exactly() -> None:
    """Same quantity, every input form, to round-off on a well-conditioned problem.

    Twelve observables of seven parameters, unequal noise, and an entropy prior
    at unequal weights, so the threshold ``mu / w_i`` differs per parameter --
    the case a single threshold cannot represent. The prior's curvature is
    taken from :class:`EntropyPrior` itself rather than written down, and it is
    diagonal. Its diagonal, the same matrix dense, a pre-whitened Jacobian, a
    scalar ``sigma``, and A2's prior-whitened form ``sum d_k / (d_k + mu)`` over
    the spectrum of ``C^(-1/2) J diag(sqrt(w))`` all give the dense route's
    answer; so does a dense, correlated prior, through its Cholesky factor.
    """
    k_jac, k_sigma, k_weights, k_prior = jax.random.split(jax.random.PRNGKey(3), 4)
    jacobian = jax.random.normal(k_jac, (12, 7))
    sigma = 0.5 + jax.random.uniform(k_sigma, (12,))
    weights = 0.5 + jax.random.uniform(k_weights, (7,))
    mu = 0.3
    prior = EntropyPrior(mu=mu, reference=1.0)
    hessian = fisher_information(lambda w: -prior.log_prob({"weights": w}), weights)
    assert jnp.allclose(hessian, jnp.diag(mu / weights), rtol=1e-12, atol=0.0)
    curvature = jnp.diag(hessian)

    whitened = jacobian / sigma[:, None]
    dense = float(effective_parameters(whitened.T @ whitened, hessian))
    for got in (
        effective_parameters_from_jacobian(jacobian, curvature, sigma=sigma),
        effective_parameters_from_jacobian(jacobian, hessian, sigma=sigma),
        effective_parameters_from_jacobian(whitened, curvature),
    ):
        assert float(got) == pytest.approx(dense, rel=1e-12)
    spectrum = jnp.linalg.svd(whitened * jnp.sqrt(weights), compute_uv=False) ** 2
    assert float(jnp.sum(spectrum / (spectrum + mu))) == pytest.approx(dense, rel=1e-12)

    scalar = effective_parameters_from_jacobian(jacobian, curvature, sigma=0.7)
    for got in (
        effective_parameters_from_jacobian(jacobian / 0.7, curvature),
        effective_parameters_from_jacobian(
            jacobian, curvature, sigma=jnp.asarray([0.7])
        ),
    ):
        assert float(got) == pytest.approx(float(scalar), rel=1e-12)

    q = jax.random.normal(k_prior, (7, 7))
    correlated = q @ q.T / 7.0 + 0.2 * jnp.eye(7)
    assert float(
        effective_parameters_from_jacobian(whitened, correlated)
    ) == pytest.approx(
        float(effective_parameters(whitened.T @ whitened, correlated)), rel=1e-12
    )


def test_effective_parameters_from_jacobian_cannot_exceed_the_observable_count() -> (
    None
):
    """Five observables of nine parameters: at most five, none under an overwhelming prior.

    ``J^T C^-1 J`` has rank at most ``min(m, d)``, and the SVD route sees only
    ``min(m, d)`` singular values, so the bound holds by construction rather
    than to a tolerance.
    """
    jacobian = jax.random.normal(jax.random.PRNGKey(5), (5, 9))
    weak = float(effective_parameters_from_jacobian(jacobian, jnp.full((9,), 1e-12)))
    assert 4.999 < weak <= 5.0
    strong = effective_parameters_from_jacobian(jacobian, jnp.full((9,), 1e12))
    assert float(strong) == pytest.approx(0.0, abs=1e-10)


def _circular_orbits_in_radial_bins(
    mu: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The probe's ``radial_wide`` instrument on 32 circular tracer orbits.

    Thirty-two soft radial bins of width 0.6 on ``[0.4, 2.0]`` and two moments,
    so ``m = 64`` against 32 weights; orbits at radii 0.5 to 2.0 in a unit
    point mass softened at 0.2; the 2 % relative noise model; and
    ``EntropyPrior(mu)`` at the uniform ``w0 = mean(truth)``, whose curvature
    there is ``mu / w0``. A circular orbit keeps its radius and its speed, so
    one snapshot *is* the time average and no rollout is needed. The
    observable is linear in the weights, so the AD Jacobian is the kernel,
    rounded in whatever precision is active.

    Returns the Jacobian at ``w0``, ``sigma``, and the prior's diagonal.
    """
    n = 32
    radius = jnp.linspace(0.5, 2.0, n)
    phase = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    speed = radius / (radius**2 + 0.2**2) ** 0.75
    zeros = jnp.zeros(n)
    state = {
        "positions": jnp.stack(
            [radius * jnp.cos(phase), radius * jnp.sin(phase), zeros], axis=1
        ),
        "velocities": jnp.stack(
            [-speed * jnp.sin(phase), speed * jnp.cos(phase), zeros], axis=1
        ),
    }
    observable = WeightedKernelSum(
        GaussianRadialBins(centres=jnp.linspace(0.4, 2.0, 32), width=0.6)
    )
    truth = 1.0 + (jax.random.uniform(jax.random.PRNGKey(0), (n,)) - 0.5)
    truth = truth / jnp.sum(truth)
    observed = observable({**state, "weights": truth})
    sigma = 0.02 * jnp.maximum(jnp.abs(observed), 1.0e-3 * jnp.max(jnp.abs(observed)))
    w0 = jnp.full((n,), jnp.mean(truth))
    jacobian = jax.jacfwd(lambda w: observable({**state, "weights": w}))(w0)
    return jacobian, sigma, mu / w0


def test_in_single_precision_the_dense_route_fails_and_the_jacobian_route_does_not() -> (
    None
):
    """Defect 1 of the EDDA Fisher probe, reproduced, and the SVD route fixing it.

    ``d_max / threshold`` is 1.6e12 here: 2e5 times the ``1 / eps`` at which a
    float32 ``J^T C^-1 J`` loses the directions at the threshold, and 40 times
    below the ``1 / eps^2`` up to which the SVD of the whitened Jacobian keeps
    them. The reference is the same float32 Jacobian analysed in double, so
    only the linear algebra differs between the three numbers.

    THE MECHANISM, ASSERTED. The Gram matrix's smallest diagonal entry is
    ~1e5, whose float32 spacing is ~8e-3; the prior adds 3.2e-6. So
    ``H_data + H_prior`` is ``H_data`` **bit for bit**, the prior never reaches
    the solve, and the dense route returns ``tr(F F^-1)`` of a numerically
    singular ``F`` -- rounding, not information, and independent of ``mu``.

    Measured on an M1 CPU: reference 8.560, dense route 31.39, this route
    8.563. Over 40 variants of this problem (seeds and orbit phases, measured
    once, not in this test) this route's worst float32 error was 0.011 here
    and 0.065 at the probe's ten-times-lower ``mu``, where ``d_max /
    threshold`` is 1.6e13 as on its ``radial_wide`` set; the dense route was
    never closer than 9.5. The probe's own numbers on that set: 28.19 dense,
    13.937 by SVD, 13.920 the reference. This test sits at ``mu = 1e-7`` for
    the margin, not because the route fails at 1e-8.

    x64 is switched off for the test, not just the inputs cast: under
    ``jax_enable_x64`` the dense route's identity is float64 and promotes its
    solve, which then returns ``d`` -- also wrong, but not what the probe ran.
    """
    with jax.enable_x64(False):
        jacobian, sigma, curvature = _circular_orbits_in_radial_bins(mu=1.0e-7)
        assert jacobian.dtype == jnp.float32 and curvature.dtype == jnp.float32
        whitened = jacobian / sigma[:, None]
        gram = whitened.T @ whitened
        assert jnp.array_equal(gram + jnp.diag(curvature), gram)
        dense = effective_parameters(gram, jnp.diag(curvature))
        svd = effective_parameters_from_jacobian(jacobian, curvature, sigma=sigma)
        assert dense.dtype == jnp.float32 and svd.dtype == jnp.float32

    reference_jacobian = np.asarray(jacobian, np.float64)
    reference_sigma = np.asarray(sigma, np.float64)
    spectrum = (
        np.linalg.svd(reference_jacobian / reference_sigma[:, None], compute_uv=False)
        ** 2
    )
    threshold = float(curvature[0])
    reference = float(np.sum(spectrum / (spectrum + threshold)))
    eps = float(np.finfo(np.float32).eps)
    assert 1.0 / eps < float(spectrum.max()) / threshold < 1.0 / eps**2
    assert abs(float(dense) - reference) > 1.0
    assert abs(float(svd) - reference) < 0.05


def test_in_double_precision_the_jacobian_route_reaches_a_closed_form() -> None:
    """Defect 2: the dense route's own double-precision floor, against an exact answer.

    ``A = U diag(sqrt(d)) V^T`` with orthonormal ``U``, ``V`` and a spectrum
    ``d`` spanning 1e13 down to 1e-26 of a unit threshold -- the probe's radial
    sets' dynamic range -- has ``sum d / (d + 1)`` as its answer in closed
    form. Measured on an M1 CPU at this seed: this route 1.0e-11 off, the dense
    route 7.8e-5. Over 20 seeds (once, not in this test) the worst for this
    route was 5.4e-11 and the best for the dense route 2.1e-5; the probe found
    the dense route 6.6e-5 to 9.7e-4 from the SVD route on its radial sets.
    """
    rng = np.random.default_rng(0)
    u, _ = np.linalg.qr(rng.normal(size=(64, 32)))
    v, _ = np.linalg.qr(rng.normal(size=(32, 32)))
    spectrum = np.logspace(13.0, -26.0, 32)
    whitened = jnp.asarray((u * np.sqrt(spectrum)) @ v.T)
    exact = float(np.sum(spectrum / (spectrum + 1.0)))
    svd = float(effective_parameters_from_jacobian(whitened, jnp.ones(32)))
    dense = float(effective_parameters(whitened.T @ whitened, jnp.eye(32)))
    assert svd == pytest.approx(exact, abs=1e-9)
    assert abs(dense - exact) > 1e-6


def test_a_hessian_away_from_a_fit_is_not_a_fisher_matrix() -> None:
    """Defect 3: residual curvature counted as information; the Jacobian route is immune.

    Two observables of six parameters, ``y = (sum x^2, sum sin x)``. At an
    exact fit the residuals vanish, the Hessian of the negative log likelihood
    is ``J^T C^-1 J``, and the two routes agree. Move the data off the
    prediction and the Hessian gains ``sum_j (r_j / sigma^2) d^2 y_j``, here
    negative definite and of full rank: the Hessian route then reports more
    than all six parameters determined -- which only an indefinite curvature
    can -- while the Jacobian route, which never sees the residuals, still
    says two. The probe's 32.000009 for ten observables of 32 weights is the
    same thing, at a point that was not a fit.
    """

    def predict(x):
        return jnp.stack([jnp.sum(x * x), jnp.sum(jnp.sin(x))])

    x = jnp.linspace(0.2, 1.2, 6)
    sigma = jnp.asarray([0.1, 0.1])
    prior = jnp.full((6,), 1.0e-3)
    jacobian = jax.jacfwd(predict)(x)
    gauss_newton = float(
        effective_parameters_from_jacobian(jacobian, prior, sigma=sigma)
    )
    assert 1.99 < gauss_newton < 2.0

    def hessian_of_the_likelihood(observed):
        return fisher_information(
            lambda z: 0.5 * jnp.sum(((predict(z) - observed) / sigma) ** 2), x
        )

    at_fit = hessian_of_the_likelihood(predict(x))
    assert float(effective_parameters(at_fit, jnp.diag(prior))) == pytest.approx(
        gauss_newton, rel=1e-9
    )
    off_fit = hessian_of_the_likelihood(predict(x) + jnp.asarray([2.0, 0.5]))
    assert float(jnp.min(jnp.linalg.eigvalsh(off_fit))) < 0.0
    assert float(effective_parameters(off_fit, jnp.diag(prior))) > 6.0


def test_effective_parameters_from_jacobian_differentiates_like_the_dense_route() -> (
    None
):
    """Its gradients in the Jacobian and in the prior are the dense route's.

    So it can stand in as a design objective. Checked where both routes are
    exact: a well-conditioned problem, in double precision.
    """
    jacobian = jax.random.normal(jax.random.PRNGKey(7), (5, 9))
    prior = 0.5 + jax.random.uniform(jax.random.PRNGKey(8), (9,))

    def dense(j, p):
        return effective_parameters(j.T @ j, jnp.diag(p))

    for argnum in (0, 1):
        got = jax.grad(effective_parameters_from_jacobian, argnums=argnum)(
            jacobian, prior
        )
        want = jax.grad(dense, argnums=argnum)(jacobian, prior)
        assert jnp.allclose(got, want, rtol=1e-10, atol=1e-13)


def test_effective_parameters_from_jacobian_rejects_shapes_it_cannot_read() -> None:
    """A Jacobian that is not a matrix, a prior of the wrong size, a sigma per column."""
    jacobian = jnp.ones((4, 3))
    with pytest.raises(ValueError, match=r"\(m, d\)"):
        effective_parameters_from_jacobian(jnp.ones(3), jnp.ones(3))
    with pytest.raises(ValueError, match="prior_curvature"):
        effective_parameters_from_jacobian(jacobian, jnp.ones(4))
    with pytest.raises(ValueError, match="prior_curvature"):
        effective_parameters_from_jacobian(jacobian, jnp.ones((3, 4)))
    with pytest.raises(ValueError, match="sigma"):
        effective_parameters_from_jacobian(jacobian, jnp.ones(3), sigma=jnp.ones(3))
    with pytest.raises(ValueError, match="sigma"):
        effective_parameters_from_jacobian(
            jacobian, jnp.ones(3), sigma=jnp.ones((4, 1))
        )


def _tikhonov_curve(singular, truth, noise, mus):
    """Exact L-curve of a diagonal Tikhonov problem, by construction.

    ``x(mu)_i = s_i b_i / (s_i^2 + mu)``, so the misfit, the norm and the true
    error are all closed forms and no optimizer is in the loop.
    """
    b = singular * truth + noise
    misfit = jnp.stack([jnp.sum((mu / (singular**2 + mu) * b) ** 2) for mu in mus])
    norm = jnp.stack([jnp.sum((singular * b / (singular**2 + mu)) ** 2) for mu in mus])
    return misfit, norm


def test_l_curve_corner_lands_between_the_two_singular_value_scales() -> None:
    """On a problem with a spectral gap the corner separates signal from noise.

    Three singular values at 1 and three at ``delta``, with the truth living
    wholly in the first subspace. A useful ``mu`` damps the second and spares
    the first, i.e. it lies between ``delta^2`` and 1, and that is what the
    corner returns.
    """
    delta = 1.0e-4
    singular = jnp.asarray([1.0, 1.0, 1.0, delta, delta, delta])
    truth = jnp.asarray([1.0, -0.5, 0.8, 0.0, 0.0, 0.0])
    noise = jnp.asarray([1.0, -2.0, 0.5, 1.5, -1.0, 0.7]) * 1.0e-3
    mus = jnp.logspace(-12.0, 4.0, 65)
    misfit, norm = _tikhonov_curve(singular, truth, noise, mus)

    corner, curvature = l_curve_corner(mus, misfit, norm)
    assert delta**2 < float(corner) < 1.0
    assert curvature.shape == mus.shape
    # The ends are not evaluated, so they cannot be selected.
    assert bool(jnp.all(jnp.isnan(curvature[:3])))
    assert bool(jnp.all(jnp.isnan(curvature[-3:])))
    assert bool(jnp.all(jnp.isfinite(curvature[3:-3])))


def test_l_curve_corner_is_invariant_to_rescaling_either_axis() -> None:
    """Scaling the misfit or the norm shifts a log axis and cannot move a corner.

    Worth pinning because it is the property that makes the criterion usable
    at all: it must not depend on the units the misfit and the norm happen to
    be measured in, only on the shape of the curve.
    """
    singular = jnp.asarray([1.0, 1.0, 1.0, 1.0e-3, 1.0e-3, 1.0e-3])
    truth = jnp.asarray([1.0, -0.5, 0.8, 0.0, 0.0, 0.0])
    noise = jnp.asarray([1.0, -2.0, 0.5, 1.5, -1.0, 0.7]) * 1.0e-3
    mus = jnp.logspace(-12.0, 4.0, 65)
    misfit, norm = _tikhonov_curve(singular, truth, noise, mus)

    base = float(l_curve_corner(mus, misfit, norm)[0])
    assert float(l_curve_corner(mus, 1.0e7 * misfit, norm)[0]) == base
    assert float(l_curve_corner(mus, misfit, 1.0e-5 * norm)[0]) == base


def test_l_curve_corner_ignores_the_spurious_curvature_at_the_ends() -> None:
    """A huge kink in the last point must not be reported as the corner.

    One-sided differences at an end routinely produce the largest curvature in
    the array, and taking ``gradient`` twice spreads one bad endpoint two
    points inward -- which is why *three* points at each end are excluded, not
    two. Here the curve has a genuine corner in the middle and a planted spike
    in the final point.
    """
    mus = jnp.logspace(-6.0, 6.0, 41)
    s = jnp.log(mus)
    # A single smooth bend at log mu = 0, plus a spike in the final point.
    misfit = jnp.exp(s)
    norm = jnp.exp(-2.0 * jnp.logaddexp(0.0, s / 2.0))
    spiked = norm.at[-1].set(norm[-1] * 1.0e6)

    clean_corner = float(l_curve_corner(mus, misfit, norm)[0])
    spiked_corner = float(l_curve_corner(mus, misfit, spiked)[0])
    assert clean_corner == spiked_corner
    assert spiked_corner < float(mus[-4])


def test_l_curve_corner_rejects_input_it_cannot_read() -> None:
    """Mismatched shapes, too few points, and a grid that is not increasing."""
    mus = jnp.logspace(-3.0, 3.0, 11)
    ok = jnp.ones_like(mus)
    with pytest.raises(ValueError, match="same shape"):
        l_curve_corner(mus, ok[:-1], ok)
    with pytest.raises(ValueError, match="one-dimensional"):
        l_curve_corner(mus[None, :], ok[None, :], ok[None, :])
    with pytest.raises(ValueError, match="at least 9 points"):
        l_curve_corner(mus[:8], ok[:8], ok[:8])
    with pytest.raises(ValueError, match="strictly increasing"):
        l_curve_corner(mus[::-1], ok, ok)
    with pytest.raises(ValueError, match="strictly positive"):
        l_curve_corner(mus.at[0].set(-1.0), ok, ok)
