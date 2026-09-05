"""End to end on the double: an optax fit recovers a linear-Gaussian truth."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from mimirax import (
    GaussianPrior,
    InferenceProblem,
    OptaxOptimizer,
    has_converged,
    rms_residual,
)
from mimirax.testing import make_linear_problem


def test_gradient_descent_recovers_the_truth_without_noise(key) -> None:
    """With exact data the MAP is the truth; Adam gets within 1e-6 of it."""
    problem, truth = make_linear_problem(key, n_params=4, n_data=16)
    fit = OptaxOptimizer(optax.adam(0.05), num_steps=2000).minimize(
        problem.negative_log_posterior, jnp.zeros(4)
    )
    assert has_converged(fit.objective_trace, rtol=1.0e-8, window=10)
    assert jnp.allclose(fit.params, truth, atol=1.0e-6)
    assert float(rms_residual(problem.predict(fit.params), problem.observed)) < 1.0e-6


def test_prior_pulls_the_map_toward_its_mean(key) -> None:
    """A Gaussian prior at zero shrinks the MAP, and Adam lands on the closed-form ridge solution."""
    problem, truth = make_linear_problem(key, n_params=3, n_data=6, sigma=1.0)
    prior_sigma = 0.3
    shrunk = InferenceProblem(
        problem.forward,
        problem.observable,
        problem.likelihood,
        problem.observed,
        priors=(GaussianPrior(mean=0.0, sigma=prior_sigma),),
    )
    free = (
        OptaxOptimizer(optax.adam(0.02), num_steps=6000)
        .minimize(problem.negative_log_posterior, jnp.zeros(3))
        .params
    )
    # The regularized objective is strongly convex (curvature >= 1 / prior_sigma^2),
    # so plain gradient descent converges geometrically where Adam would plateau.
    regularized = (
        OptaxOptimizer(optax.sgd(0.02), num_steps=6000)
        .minimize(shrunk.negative_log_posterior, jnp.zeros(3))
        .params
    )
    assert jnp.allclose(free, truth, atol=1.0e-5)
    assert float(jnp.linalg.norm(regularized)) < float(jnp.linalg.norm(free))
    # Linear-Gaussian with a Gaussian prior has the closed-form ridge solution.
    a, b = problem.forward.matrix, problem.forward.offset
    lhs = a.T @ a + jnp.eye(3) / prior_sigma**2
    expected = jnp.linalg.solve(lhs, a.T @ (problem.observed - b))
    assert jnp.allclose(regularized, expected, atol=1.0e-5)
    grad = jax.grad(shrunk.negative_log_posterior)(regularized)
    assert jnp.allclose(grad, jnp.zeros(3), atol=1.0e-3)
