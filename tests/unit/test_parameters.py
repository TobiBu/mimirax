"""Tests for parameter pytrees, reparameterizations, gauge fixing and constraints."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from mimirax import parameters as P

TRANSFORMS = [
    P.IdentityTransform(),
    P.LogTransform(),
    P.SoftplusTransform(),
    P.AffineTransform(scale=2.5, shift=-0.75),
]


def test_ravel_round_trips_a_nested_pytree() -> None:
    """ravel flattens to one vector and its inverse rebuilds the structure exactly."""
    params = {"a": jnp.arange(6.0).reshape(2, 3), "b": (jnp.ones(2), jnp.asarray(3.0))}
    flat, unravel = P.ravel(params)
    assert flat.shape == (9,)
    assert P.num_parameters(params) == 9
    rebuilt = unravel(flat)
    assert jnp.array_equal(rebuilt["a"], params["a"])
    assert jnp.array_equal(rebuilt["b"][0], params["b"][0])
    assert float(rebuilt["b"][1]) == 3.0


@pytest.mark.parametrize("transform", TRANSFORMS, ids=lambda t: type(t).__name__)
def test_transform_inverse_round_trips(transform) -> None:
    """inverse(forward(u)) == u to round-off, for every registered transform."""
    u = jnp.linspace(-2.0, 2.0, 7)
    assert jnp.allclose(transform.inverse(transform.forward(u)), u, atol=1.0e-12)


@pytest.mark.parametrize("transform", TRANSFORMS, ids=lambda t: type(t).__name__)
def test_transform_log_abs_det_jacobian_matches_autodiff(transform) -> None:
    """The closed-form log|det J| equals the sum of log|d forward / du| from jax."""
    u = jnp.linspace(-1.5, 1.5, 5)
    jac = jax.jacfwd(transform.forward)(u)
    expected = jnp.sum(jnp.log(jnp.abs(jnp.diag(jac))))
    assert jnp.allclose(transform.log_abs_det_jacobian(u), expected, atol=1.0e-12)


def test_registry_holds_the_four_transforms() -> None:
    """The reparameterization registry resolves names to the classes."""
    assert P.REPARAMETERIZATIONS.available() == (
        "affine",
        "identity",
        "log",
        "softplus",
    )
    assert P.REPARAMETERIZATIONS.get("log") is P.LogTransform


def test_constrain_and_unconstrain_apply_per_key_and_pass_others_through() -> None:
    """Only keys named in transforms are mapped; the total log-Jacobian is the sum."""
    transforms = {"mass": P.LogTransform(), "scale": P.AffineTransform(scale=3.0)}
    unconstrained = {
        "mass": jnp.asarray([0.0, 1.0]),
        "scale": jnp.asarray(2.0),
        "x": jnp.ones(3),
    }
    constrained = P.constrain(unconstrained, transforms)
    assert jnp.allclose(constrained["mass"], jnp.exp(unconstrained["mass"]))
    assert float(constrained["scale"]) == 6.0
    assert constrained["x"] is unconstrained["x"]
    back = P.unconstrain(constrained, transforms)
    for k in unconstrained:
        assert jnp.allclose(back[k], unconstrained[k], atol=1.0e-12)
    expected = jnp.sum(unconstrained["mass"]) + jnp.log(3.0)
    assert jnp.allclose(
        P.log_abs_det_jacobian(unconstrained, transforms), expected, atol=1e-12
    )


def test_gauge_fixing_zeroes_centre_of_mass_and_momentum(small_system) -> None:
    """After recentring and boosting, COM and net momentum vanish to round-off."""
    positions, velocities, masses = small_system
    assert jnp.linalg.norm(P.centre_of_mass(positions, masses)) > 1.0e-3
    assert jnp.linalg.norm(P.net_momentum(velocities, masses)) > 1.0e-3
    recentred = P.remove_centre_of_mass(positions, masses)
    boosted = P.remove_net_momentum(velocities, masses)
    assert jnp.allclose(P.centre_of_mass(recentred, masses), jnp.zeros(3), atol=1.0e-14)
    assert jnp.allclose(P.net_momentum(boosted, masses), jnp.zeros(3), atol=1.0e-14)
    # Relative geometry is untouched.
    assert jnp.allclose(
        recentred[1] - recentred[0], positions[1] - positions[0], atol=1e-14
    )


def test_gauge_fixing_is_differentiable() -> None:
    """The recentring map has the analytic Jacobian (I - 1 m^T / M) on every row."""
    masses = jnp.asarray([1.0, 3.0])
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    jac = jax.jacfwd(P.remove_centre_of_mass)(positions, masses)  # (2, 3, 2, 3)
    d_row0_d_pos1 = jac[0, :, 1, :]
    expected = -(masses[1] / jnp.sum(masses)) * jnp.eye(3)
    assert jnp.allclose(d_row0_d_pos1, expected, atol=1.0e-14)


def test_total_mass_constraint_defect_and_penalty() -> None:
    """The defect is sum - total; the penalty is weight * defect^2; violation is |defect|."""
    constraint = P.TotalMassConstraint(total=10.0)
    masses = jnp.asarray([1.0, 2.0, 3.0])
    assert float(constraint.defect(masses)) == -4.0
    assert float(P.quadratic_penalty(constraint, masses, weight=0.5)) == 8.0
    assert float(P.constraint_violation(constraint, masses)) == 4.0
    grad = jax.grad(lambda m: P.quadratic_penalty(constraint, m, weight=0.5))(masses)
    assert jnp.allclose(grad, -4.0 * jnp.ones(3), atol=1.0e-14)


def test_function_constraint_wraps_a_defect_function() -> None:
    """FunctionConstraint forwards to its callable, pytree in, array out."""
    constraint = P.FunctionConstraint(lambda p: p["x"] - p["y"])
    params = {"x": jnp.asarray([1.0, 2.0]), "y": jnp.asarray([1.0, 0.0])}
    assert jnp.array_equal(constraint.defect(params), jnp.asarray([0.0, 2.0]))
    assert float(P.constraint_violation(constraint, params)) == 2.0
