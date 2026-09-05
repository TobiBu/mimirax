"""Composing a forward model, an observable, a likelihood and priors into one objective.

This is the one place the pieces meet. Everything else in the package is
either a pluggable implementation of one protocol or a method that consumes the
scalar functions defined here.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from mimirax._typing import PyTree, Scalar
from mimirax.protocols import ForwardModel, Likelihood, Observable, Prior

__all__ = ["InferenceProblem"]


@dataclass(frozen=True)
class InferenceProblem:
    """A posterior over parameters, assembled from protocol-satisfying parts.

    ``log_posterior(params) = likelihood.log_prob(observable(forward(params)),
    observed) + sum(prior.log_prob(params))``. Every method is differentiable
    in ``params`` provided the parts are, which the protocols require.

    Attributes
    ----------
    forward : ForwardModel
        Parameters to state.
    observable : Observable
        State to data space.
    likelihood : Likelihood
        Data-space comparison.
    observed : Array
        The data.
    priors : tuple[Prior, ...]
        Zero or more priors and regularizers, summed.
    """

    forward: ForwardModel
    observable: Observable
    likelihood: Likelihood
    observed: Array
    priors: tuple[Prior, ...] = ()

    def predict(self, params: PyTree) -> Array:
        """Run the forward model and the observable.

        Parameters
        ----------
        params : PyTree
            The free parameters.

        Returns
        -------
        Array
            The prediction in data space.
        """
        return self.observable(self.forward(params))

    def log_likelihood(self, params: PyTree) -> Scalar:
        """Return the log-likelihood of the data at ``params``.

        Parameters
        ----------
        params : PyTree
            The free parameters.

        Returns
        -------
        Scalar
            ``likelihood.log_prob(predict(params), observed)``.
        """
        return self.likelihood.log_prob(self.predict(params), self.observed)

    def log_prior(self, params: PyTree) -> Scalar:
        """Return the summed log prior density at ``params``.

        Parameters
        ----------
        params : PyTree
            The free parameters.

        Returns
        -------
        Scalar
            Zero when there are no priors.
        """
        total = jnp.zeros(())
        for prior in self.priors:
            total = total + prior.log_prob(params)
        return total

    def log_posterior(self, params: PyTree) -> Scalar:
        """Return the unnormalized log posterior at ``params``.

        Parameters
        ----------
        params : PyTree
            The free parameters.

        Returns
        -------
        Scalar
            ``log_likelihood + log_prior``.
        """
        return self.log_likelihood(params) + self.log_prior(params)

    def negative_log_posterior(self, params: PyTree) -> Scalar:
        """Return ``-log_posterior``, the objective an optimizer minimizes.

        Parameters
        ----------
        params : PyTree
            The free parameters.

        Returns
        -------
        Scalar
            The negative log posterior.
        """
        return -self.log_posterior(params)
