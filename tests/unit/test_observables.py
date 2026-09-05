"""Tests for the reference observables and their registry."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from mimirax import OBSERVABLES, IdentityObservable, ProjectedPositions, make_observable


def test_registry_and_factory() -> None:
    """Both reference observables are registered and constructible by name."""
    assert OBSERVABLES.available() == ("identity", "projected_positions")
    assert isinstance(make_observable("identity"), IdentityObservable)
    obs = make_observable("projected_positions", axes=(2,))
    assert isinstance(obs, ProjectedPositions) and obs.axes == (2,)
    with pytest.raises(KeyError):
        make_observable("kinematic_map")


def test_identity_returns_the_state_as_an_array() -> None:
    """A list state comes back as an array with the same values."""
    out = IdentityObservable()([1.0, 2.0])
    assert isinstance(out, jax.Array) and jnp.array_equal(out, jnp.asarray([1.0, 2.0]))


def test_projected_positions_selects_axes_particle_major() -> None:
    """(n, 3) -> (n * len(axes),) with particle-major ordering, and a permutation Jacobian."""
    positions = jnp.arange(12.0).reshape(4, 3)
    out = ProjectedPositions(axes=(0, 2))(positions)
    assert jnp.array_equal(out, jnp.asarray([0.0, 2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0]))
    jac = jax.jacfwd(ProjectedPositions(axes=(0, 2)))(positions).reshape(8, 12)
    assert jnp.array_equal(jnp.sum(jac, axis=1), jnp.ones(8))
    assert jnp.array_equal(jnp.sum(jac, axis=0), jnp.asarray([1.0, 0.0, 1.0] * 4))
