"""Priors and regularizers: a Gaussian, an L2 penalty, and the M2M entropy stub."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from mimirax._typing import PyTree, Scalar

__all__ = ["EntropyPrior", "GaussianPrior", "L2Regularizer"]


def _sum_over_leaves(params: PyTree, fn: Callable[[Array], Scalar]) -> Scalar:
    """Apply ``fn`` to every leaf and sum the scalar results.

    Parameters
    ----------
    params : PyTree
        Any pytree of arrays.
    fn : Callable[[Array], Scalar]
        A function from one array leaf to a scalar.

    Returns
    -------
    Scalar
        The sum.
    """
    total = jnp.zeros(())
    for leaf in jax.tree_util.tree_leaves(params):
        total = total + fn(leaf)
    return total


@dataclass(frozen=True)
class GaussianPrior:
    """Independent Gaussian prior of one mean and width on every parameter.

    The normalization constant is dropped, as in
    :class:`~mimirax.likelihoods.GaussianLikelihood`.

    Attributes
    ----------
    mean : float
        Prior mean.
    sigma : float
        Prior width.
    """

    mean: float = 0.0
    sigma: float = 1.0

    def log_prob(self, params: PyTree) -> Scalar:
        """Return ``-0.5 * sum(((params - mean) / sigma)^2)`` over all leaves.

        Parameters
        ----------
        params : PyTree
            The free parameters.

        Returns
        -------
        Scalar
            The log prior density up to a constant.
        """

        def _leaf(leaf):
            z = (leaf - self.mean) / self.sigma
            return -0.5 * jnp.sum(z * z)

        return _sum_over_leaves(params, _leaf)


@dataclass(frozen=True)
class L2Regularizer:
    """Ridge penalty as a prior: ``log_prob = -weight * sum(params^2)``.

    A Gaussian prior of zero mean and ``sigma = 1 / sqrt(2 weight)``, spelled
    the way an optimizer user thinks of it.

    Attributes
    ----------
    weight : float
        Penalty strength.
    """

    weight: float = 1.0

    def log_prob(self, params: PyTree) -> Scalar:
        """Return the negative weighted sum of squares over all leaves.

        Parameters
        ----------
        params : PyTree
            The free parameters.

        Returns
        -------
        Scalar
            The log prior density up to a constant.
        """
        return _sum_over_leaves(
            params, lambda leaf: -self.weight * jnp.sum(leaf * leaf)
        )


@dataclass(frozen=True)
class EntropyPrior:
    """The made-to-measure entropy prior, ``-mu * sum(w log(w / w0))``. Stub.

    Syer & Tremaine's regularizer keeps particle weights ``w`` close to their
    priors ``w0``. It lands with the M2M module in
    :mod:`mimirax.inference.m2m` (D-026 in the EDDA programme's decisions log).

    Attributes
    ----------
    mu : float
        Regularization strength.
    """

    mu: float = 1.0

    def log_prob(self, params: PyTree) -> Scalar:
        """Not implemented yet.

        Parameters
        ----------
        params : PyTree
            The particle weights.

        Returns
        -------
        Scalar
            Never returns.

        Raises
        ------
        NotImplementedError
            Always, until the M2M module lands.
        """
        raise NotImplementedError(
            "EntropyPrior lands with the made-to-measure module (mimirax.inference.m2m)"
        )
