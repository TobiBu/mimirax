"""Shared test fixtures for mimirax."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture(autouse=True)
def _enable_x64() -> None:
    """Enable x64 for deterministic numerical tests."""
    jax.config.update("jax_enable_x64", True)


@pytest.fixture()
def key() -> jax.Array:
    """A fixed PRNG key."""
    return jax.random.PRNGKey(0)


@pytest.fixture()
def small_system() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Five bodies with unequal masses and a non-zero net momentum."""
    positions = jnp.asarray(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, -0.5, 1.0],
            [0.3, 0.2, -1.2],
        ],
        dtype=jnp.float64,
    )
    velocities = jnp.asarray(
        [
            [0.0, 0.2, 0.0],
            [0.0, -0.2, 0.1],
            [0.1, 0.0, 0.0],
            [0.0, 0.0, -0.3],
            [0.2, 0.2, 0.2],
        ],
        dtype=jnp.float64,
    )
    masses = jnp.asarray([1.0, 2.0, 0.5, 1.5, 0.25], dtype=jnp.float64)
    return positions, velocities, masses
