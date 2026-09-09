"""Tests for the inference problem, the optax optimizer and the method stubs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

from mimirax import (
    HMC,
    METHODS,
    NUTS,
    FitResult,
    GaussianLikelihood,
    GaussianPrior,
    IdentityObservable,
    InferenceProblem,
    L2Regularizer,
    MeanFieldVI,
    NelderMead,
    OptaxOptimizer,
    adam,
)
from mimirax.testing import LinearForwardModel


def test_methods_registry_names() -> None:
    """The six planned methods are registered under stable names."""
    assert METHODS.available() == (
        "data_resampling",
        "hmc",
        "made_to_measure",
        "mean_field_vi",
        "nelder_mead",
        "nuts",
        "optax",
    )
    assert METHODS.get("optax") is OptaxOptimizer


def test_inference_problem_composes_likelihood_and_priors() -> None:
    """log_posterior = log_likelihood + sum(log_prior), and the negative is the negative."""
    model = LinearForwardModel(jnp.eye(2), jnp.zeros(2))
    observed = jnp.asarray([1.0, -1.0])
    problem = InferenceProblem(
        forward=model,
        observable=IdentityObservable(),
        likelihood=GaussianLikelihood(),
        observed=observed,
        priors=(GaussianPrior(), L2Regularizer(0.5)),
    )
    params = jnp.asarray([0.0, 0.0])
    assert jnp.array_equal(problem.predict(params), params)
    assert float(problem.log_likelihood(params)) == pytest.approx(-1.0)
    assert float(problem.log_prior(params)) == 0.0
    assert float(problem.log_posterior(params)) == pytest.approx(-1.0)
    assert float(problem.negative_log_posterior(params)) == pytest.approx(1.0)
    no_prior = InferenceProblem(
        model, IdentityObservable(), GaussianLikelihood(), observed
    )
    assert float(no_prior.log_prior(params)) == 0.0


def test_optax_optimizer_reaches_the_quadratic_minimum_and_records_the_trace() -> None:
    """Plain SGD on a well-conditioned quadratic converges; the trace starts at f(x0)."""
    target = jnp.asarray([1.0, -2.0, 0.5])

    def objective(x):
        return 0.5 * jnp.sum((x - target) ** 2)

    x0 = jnp.zeros(3)
    result = OptaxOptimizer(optax.sgd(0.5), num_steps=60).minimize(objective, x0)
    assert isinstance(result, FitResult)
    assert result.objective_trace.shape == (60,)
    assert float(result.objective_trace[0]) == pytest.approx(float(objective(x0)))
    assert bool(jnp.all(jnp.diff(result.objective_trace) <= 0.0))
    assert jnp.allclose(result.params, target, atol=1.0e-12)


def test_optax_optimizer_preserves_pytree_structure_and_is_jittable() -> None:
    """A dict of parameters comes back as a dict; the whole minimize traces under jit."""
    optimizer = adam(learning_rate=0.1, num_steps=50)

    def objective(p):
        return jnp.sum(p["a"] ** 2) + jnp.sum((p["b"] - 1.0) ** 2)

    p0 = {"a": jnp.ones(2), "b": jnp.zeros(3)}
    result = jax.jit(lambda p: optimizer.minimize(objective, p))(p0)
    assert set(result.params) == {"a", "b"}
    assert float(result.objective_trace[-1]) < float(result.objective_trace[0])


def test_stub_methods_raise_not_implemented() -> None:
    """Every stub fixes its signature and raises, so a caller learns early.

    MadeToMeasure is no longer among them: it landed with the made-to-measure
    module and is exercised in ``tests/unit/test_m2m.py``.
    """
    log_density = lambda p: -jnp.sum(p**2)  # noqa: E731
    params = jnp.zeros(2)
    key = jax.random.PRNGKey(0)
    for sampler in (HMC(), NUTS(), MeanFieldVI()):
        with pytest.raises(NotImplementedError):
            sampler.sample(log_density, params, key=key, num_samples=4)
    with pytest.raises(NotImplementedError):
        NelderMead().minimize(log_density, params)
