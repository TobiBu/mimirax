"""The doubles' hand-written gradients agree with jax to round-off."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from mimirax.testing import (
    DirectSumGravity,
    LinearForwardModel,
    SoftenedPointMassField,
    make_linear_problem,
)


def test_linear_model_exact_gradient_matches_autodiff(key) -> None:
    """2 A^T (A x + b - y) equals jax.grad of the sum of squares."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    model = LinearForwardModel(
        jax.random.normal(k1, (7, 3)), jax.random.normal(k2, (7,))
    )
    x = jax.random.normal(k3, (3,))
    y = jax.random.normal(k4, (7,))
    autodiff = jax.grad(lambda p: jnp.sum((model(p) - y) ** 2))(x)
    assert jnp.allclose(
        autodiff, model.exact_gradient_sum_of_squares(x, y), atol=1.0e-12
    )


def test_point_mass_field_tidal_tensor_matches_jacobian(key) -> None:
    """The closed-form tidal tensor equals the diagonal blocks of jacfwd(accelerations)."""
    k1, k2, k3 = jax.random.split(key, 3)
    field = SoftenedPointMassField(
        sources=jax.random.normal(k1, (4, 3)),
        source_masses=jax.random.uniform(k2, (4,), minval=0.5, maxval=2.0),
        softening=0.1,
        gravitational_constant=1.3,
    )
    positions = 2.0 * jax.random.normal(k3, (3, 3))
    masses = jnp.ones(3)
    full = jax.jacfwd(lambda x: field.accelerations(x, masses))(positions)  # (3,3,3,3)
    diagonal_blocks = jnp.einsum("iaib->iab", full)
    assert jnp.allclose(diagonal_blocks, field.tidal_tensor(positions), atol=1.0e-12)
    # Test particles: no cross-particle coupling.
    off = full - jnp.einsum("iab,ij->iajb", diagonal_blocks, jnp.eye(3))
    assert jnp.allclose(off, 0.0, atol=1.0e-14)


def test_point_mass_field_single_source_has_newtonian_magnitude() -> None:
    """One unit mass at the origin, eps = 0: |a| = G / r^2 pointing inward."""
    field = SoftenedPointMassField(jnp.zeros((1, 3)), jnp.ones(1))
    acc = field.accelerations(jnp.asarray([[2.0, 0.0, 0.0]]), jnp.ones(1))
    assert jnp.allclose(acc, jnp.asarray([[-0.25, 0.0, 0.0]]), atol=1.0e-15)


def test_direct_sum_conserves_momentum_and_matches_two_body(small_system) -> None:
    """sum m a = 0 to round-off, and two bodies give the textbook pair force."""
    positions, _, masses = small_system
    gravity = DirectSumGravity(softening=0.05)
    acc = gravity.accelerations(positions, masses)
    assert jnp.allclose(
        jnp.sum(masses[:, None] * acc, axis=0), jnp.zeros(3), atol=1.0e-13
    )
    two = DirectSumGravity().accelerations(positions[:2], masses[:2])
    # Separation 2 along x: a_0 = +m_1 / 4, a_1 = -m_0 / 4.
    assert jnp.allclose(
        two, jnp.asarray([[0.5, 0.0, 0.0], [-0.25, 0.0, 0.0]]), atol=1e-15
    )


def test_direct_sum_gradient_is_finite_and_antisymmetric_in_masses(
    small_system,
) -> None:
    """The where-masked self term leaves no NaN in the gradient, even at eps = 0."""
    positions, _, masses = small_system
    gravity = DirectSumGravity()
    grad = jax.grad(lambda x: jnp.sum(gravity.accelerations(x, masses) ** 2))(positions)
    assert bool(jnp.all(jnp.isfinite(grad)))
    dmass = jax.jacfwd(lambda m: gravity.accelerations(positions, m))(masses)
    assert bool(jnp.all(jnp.isfinite(dmass)))
    # Particle i's acceleration does not depend on its own mass.
    assert jnp.allclose(jnp.einsum("iai->ia", dmass), 0.0, atol=1.0e-15)


def test_make_linear_problem_truth_is_the_maximum_a_posteriori(key) -> None:
    """With no noise and no priors, the gradient of the objective vanishes at the truth."""
    problem, truth = make_linear_problem(key, n_params=3, n_data=9)
    grad = jax.grad(problem.negative_log_posterior)(truth)
    assert jnp.allclose(grad, jnp.zeros(3), atol=1.0e-12)
    assert float(problem.negative_log_posterior(truth)) == 0.0
