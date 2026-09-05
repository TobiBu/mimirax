"""A force-model double drives a static reconstruction, Paper I section 7 in miniature."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from mimirax import (
    GaussianLikelihood,
    IdentityObservable,
    InferenceProblem,
    OptaxOptimizer,
    fisher_information,
)
from mimirax.testing import SoftenedPointMassField


def test_recover_a_single_source_position_from_field_samples(key) -> None:
    """Accelerations at 12 tracers pin down one softened point mass's position."""
    k_tracers, k_guess = jax.random.split(key)
    tracers = 3.0 * jax.random.normal(k_tracers, (12, 3))
    true_source = jnp.asarray([[0.4, -0.3, 0.2]])
    unit_mass = jnp.ones(1)

    def forward(source):
        field = SoftenedPointMassField(source, unit_mass, softening=0.2)
        return field.accelerations(tracers, jnp.ones(12)).reshape(-1)

    observed = forward(true_source)
    problem = InferenceProblem(
        forward, IdentityObservable(), GaussianLikelihood(0.01), observed
    )
    guess = true_source + 0.3 * jax.random.normal(k_guess, (1, 3))
    fit = OptaxOptimizer(optax.adam(0.02), num_steps=1500).minimize(
        problem.negative_log_posterior, guess
    )
    assert jnp.allclose(fit.params, true_source, atol=1.0e-5)
    fisher = fisher_information(problem.negative_log_posterior, fit.params)
    assert fisher.shape == (3, 3)
    assert bool(jnp.all(jnp.linalg.eigvalsh(fisher) > 0.0)), "the problem is well posed"
