"""Structural conformance of the shipped implementations to the protocols."""

from __future__ import annotations

import jax.numpy as jnp

from mimirax import (
    HMC,
    NUTS,
    AffineTransform,
    Constraint,
    ForceModel,
    ForwardModel,
    FunctionConstraint,
    GaussianLikelihood,
    GaussianPrior,
    IdentityObservable,
    IdentityTransform,
    L2Regularizer,
    Likelihood,
    LogTransform,
    MadeToMeasure,
    MeanFieldVI,
    NelderMead,
    Observable,
    Optimizer,
    Prior,
    ProjectedPositions,
    Reparameterization,
    Sampler,
    SoftplusTransform,
    TotalMassConstraint,
    adam,
)
from mimirax.testing import DirectSumGravity, LinearForwardModel, SoftenedPointMassField


def test_force_model_doubles_satisfy_the_protocol() -> None:
    """Both doubles are ForceModels structurally."""
    field = SoftenedPointMassField(jnp.zeros((1, 3)), jnp.ones((1,)))
    assert isinstance(field, ForceModel)
    assert isinstance(DirectSumGravity(), ForceModel)


def test_forward_and_observable_doubles_satisfy_their_protocols() -> None:
    """The linear model is a ForwardModel; both reference observables are Observables."""
    assert isinstance(LinearForwardModel(jnp.eye(2), jnp.zeros(2)), ForwardModel)
    assert isinstance(IdentityObservable(), Observable)
    assert isinstance(ProjectedPositions(), Observable)


def test_likelihoods_and_priors_satisfy_their_protocols() -> None:
    """Every registered likelihood and prior has log_prob."""
    assert isinstance(GaussianLikelihood(), Likelihood)
    for prior in (GaussianPrior(), L2Regularizer()):
        assert isinstance(prior, Prior)


def test_reparameterizations_and_constraints_satisfy_their_protocols() -> None:
    """Every transform is a Reparameterization; both constraint classes are Constraints."""
    for transform in (
        IdentityTransform(),
        LogTransform(),
        SoftplusTransform(),
        AffineTransform(),
    ):
        assert isinstance(transform, Reparameterization)
    assert isinstance(FunctionConstraint(lambda p: p), Constraint)
    assert isinstance(TotalMassConstraint(1.0), Constraint)


def test_methods_are_optimizers_or_samplers() -> None:
    """The registry's entries all have the minimize or sample shape."""
    for optimizer in (adam(), NelderMead(), MadeToMeasure()):
        assert isinstance(optimizer, Optimizer)
    for sampler in (HMC(), NUTS(), MeanFieldVI()):
        assert isinstance(sampler, Sampler)
