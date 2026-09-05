"""Tests for residual, convergence and degeneracy diagnostics."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from mimirax import (
    degenerate_directions,
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
