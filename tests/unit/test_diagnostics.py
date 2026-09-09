"""Tests for residual, convergence and degeneracy diagnostics."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from mimirax import (
    degenerate_directions,
    effective_parameters,
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
