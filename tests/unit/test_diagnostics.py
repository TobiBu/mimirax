"""Tests for residual, convergence and degeneracy diagnostics."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from mimirax import (
    degenerate_directions,
    effective_parameters,
    fisher_information,
    has_converged,
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
