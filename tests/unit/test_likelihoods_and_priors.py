"""Tests for the Gaussian likelihood, the priors and their registries."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from mimirax import (
    LIKELIHOODS,
    PRIORS,
    EntropyPrior,
    GaussianLikelihood,
    GaussianPrior,
    L2Regularizer,
    chi_squared,
)


def test_registries_hold_the_shipped_classes() -> None:
    """Names resolve to classes."""
    assert LIKELIHOODS.available() == ("gaussian",)
    assert LIKELIHOODS.get("gaussian") is GaussianLikelihood
    assert PRIORS.available() == ("entropy", "gaussian", "l2")


def test_gaussian_log_prob_is_minus_half_chi_squared() -> None:
    """log_prob == -chi^2 / 2, and its gradient is -(p - o) / sigma^2."""
    p = jnp.asarray([1.0, 2.0, 3.0])
    o = jnp.asarray([1.5, 2.0, 2.0])
    like = GaussianLikelihood(sigma=0.5)
    assert float(chi_squared(p, o, 0.5)) == pytest.approx(5.0)
    assert float(like.log_prob(p, o)) == pytest.approx(-2.5)
    grad = jax.grad(like.log_prob)(p, o)
    assert jnp.allclose(grad, -(p - o) / 0.25, atol=1.0e-14)


def test_gaussian_prior_sums_over_pytree_leaves() -> None:
    """A dict of leaves is treated as one long vector."""
    prior = GaussianPrior(mean=1.0, sigma=2.0)
    params = {"a": jnp.asarray([3.0, 1.0]), "b": jnp.asarray(-1.0)}
    # z = (1, 0, -1) -> -0.5 * 2 = -1
    assert float(prior.log_prob(params)) == pytest.approx(-1.0)


def test_l2_regularizer_is_a_zero_mean_gaussian() -> None:
    """L2 with weight w equals a Gaussian prior with sigma = 1 / sqrt(2 w)."""
    params = jnp.asarray([0.3, -1.2, 2.0])
    w = 0.7
    assert float(L2Regularizer(w).log_prob(params)) == pytest.approx(
        float(GaussianPrior(0.0, (2.0 * w) ** -0.5).log_prob(params))
    )


def test_entropy_prior_is_a_stub() -> None:
    """The M2M entropy prior raises until the module lands."""
    with pytest.raises(NotImplementedError, match="made-to-measure"):
        EntropyPrior().log_prob(jnp.ones(3))
