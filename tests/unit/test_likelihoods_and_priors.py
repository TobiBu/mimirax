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


# --- the made-to-measure entropy prior ---------------------------------------------
#
# Correctness item 1 of the M2M module's plan: the mode is the reference, the
# term is concave, mu scales it linearly, and the gradient matches a finite
# difference. Every claim is checked in both spellings where the two agree, and
# separately where they do not.


def test_entropy_prior_mode_is_the_reference() -> None:
    """The shifted form's gradient vanishes exactly at w = w0, and only there."""
    w0 = jnp.asarray([0.4, 1.0, 2.5, 0.7])
    prior = EntropyPrior(mu=1.5, reference=w0)
    grad = jax.grad(lambda w: prior.log_prob({"weights": w}))(w0)
    assert float(jnp.max(jnp.abs(grad))) == pytest.approx(0.0, abs=1e-14)
    # A displaced point is strictly worse, in both directions.
    at_mode = float(prior.log_prob({"weights": w0}))
    for factor in (0.7, 1.4):
        assert float(prior.log_prob({"weights": factor * w0})) < at_mode


def test_entropy_prior_unshifted_mode_is_the_reference_over_e() -> None:
    """Syer & Tremaine's own spelling has its unconstrained mode at w0 / e.

    Not a defect and not a free choice: ``-sum w log(w / w0)`` is stationary at
    ``log(w / w0) = -1``. Its mode is ``w0`` only on the constant-total-weight
    surface, where the multiplier absorbs the constant -- which is why the
    shifted form is the default and why this test exists to record the
    difference rather than paper over it.
    """
    w0 = jnp.asarray([0.4, 1.0, 2.5, 0.7])
    prior = EntropyPrior(mu=1.0, reference=w0, include_linear_term=False)
    grad = jax.grad(lambda w: prior.log_prob({"weights": w}))(w0 / jnp.e)
    assert float(jnp.max(jnp.abs(grad))) == pytest.approx(0.0, abs=1e-14)
    # At w0 itself the gradient is exactly -mu in every component.
    at_reference = jax.grad(lambda w: prior.log_prob({"weights": w}))(w0)
    assert jnp.allclose(at_reference, -1.0, atol=1e-14)


@pytest.mark.parametrize("include_linear_term", [True, False])
def test_entropy_prior_is_concave(include_linear_term: bool, key) -> None:
    """The Hessian is the negative diagonal -mu / w, so no random direction curves up."""
    w = jnp.asarray([0.3, 0.9, 1.7, 2.2, 0.5])
    prior = EntropyPrior(mu=2.0, reference=0.8, include_linear_term=include_linear_term)
    hessian = jax.hessian(lambda x: prior.log_prob({"weights": x}))(w)
    assert jnp.allclose(hessian, jnp.diag(-2.0 / w), atol=1e-12)
    directions = jax.random.normal(key, (16, w.size))
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
    curvature = jnp.einsum("di,ij,dj->d", directions, hessian, directions)
    assert float(jnp.max(curvature)) < 0.0


def test_entropy_prior_mu_scales_it_linearly() -> None:
    """mu multiplies the whole term, gradient included: it is the only knob."""
    w = jnp.asarray([0.6, 1.3, 0.9])
    one = EntropyPrior(mu=1.0, reference=1.1)
    three = EntropyPrior(mu=3.0, reference=1.1)
    assert float(three.log_prob({"weights": w})) == pytest.approx(
        3.0 * float(one.log_prob({"weights": w})), rel=1e-14
    )
    g_one = jax.grad(lambda x: one.log_prob({"weights": x}))(w)
    g_three = jax.grad(lambda x: three.log_prob({"weights": x}))(w)
    assert jnp.allclose(g_three, 3.0 * g_one, atol=1e-14)


def test_entropy_prior_gradient_matches_a_finite_difference() -> None:
    """AD against a central difference along a random direction.

    The step is measured, not inherited. Relative |AD - FD| for this problem
    (directional derivative 4.0e-1, weights of order 1), central difference,
    float64::

        h     1e-2     1e-3     1e-4     1e-5     1e-6     1e-7     1e-8
        rel   2.1e-5   2.1e-7   2.1e-9   1.6e-10  2.3e-9   1.8e-8   2.7e-8

    Two clean branches meeting at h = 1e-5: the ``O(h^2)`` truncation error
    above it (each decade of h buys two decades of accuracy, exactly as a
    central difference should) and the ``O(eps/h)`` round-off floor below.
    h = 1e-5 is the minimum, the measurement there is 1.6e-10 relative, and the
    tolerance below is 1e-9 -- six times the measured value, which is the margin
    the neighbouring decades' spread justifies and no more.
    """
    w = jnp.asarray([0.75, 1.4, 0.5, 2.1, 1.0])
    prior = EntropyPrior(mu=1.7, reference=jnp.asarray([1.0, 1.0, 0.5, 2.0, 1.2]))

    def value(x):
        return prior.log_prob({"weights": x})

    direction = jnp.asarray([0.3, -0.7, 0.5, 0.2, -0.4])
    direction = direction / jnp.linalg.norm(direction)
    h = 1.0e-5
    fd = float((value(w + h * direction) - value(w - h * direction)) / (2.0 * h))
    ad = float(jnp.sum(jax.grad(value)(w) * direction))
    assert abs(ad - fd) <= 1.0e-9 * abs(fd)


def test_entropy_prior_defaults_to_a_uniform_reference_of_one() -> None:
    """With no reference the mode is the all-ones vector."""
    prior = EntropyPrior(mu=1.0)
    grad = jax.grad(lambda w: prior.log_prob({"weights": w}))(jnp.ones(4))
    assert float(jnp.max(jnp.abs(grad))) == pytest.approx(0.0, abs=1e-14)
    assert float(prior.log_prob({"weights": jnp.ones(4)})) == pytest.approx(4.0)


def test_entropy_prior_reads_a_bare_array_and_names_a_missing_key() -> None:
    """key=None applies it to every leaf; a mapping without the key fails loudly."""
    assert float(EntropyPrior(key=None).log_prob(jnp.ones(3))) == pytest.approx(3.0)
    with pytest.raises(KeyError, match=r"\('mass',\)"):
        EntropyPrior().log_prob({"mass": jnp.ones(3)})


def test_entropy_prior_is_nan_outside_its_domain() -> None:
    """Non-positive weights give nan, not a clipped value.

    The package's rule is that positivity is enforced by a reparameterization,
    not by a guard inside a density; a ``where`` here would hide the mistake it
    is this test's job to expose.
    """
    assert bool(jnp.isnan(EntropyPrior().log_prob({"weights": jnp.asarray([-1.0])})))
