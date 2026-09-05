"""The Gaussian likelihood and its chi-squared misfit."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from mimirax._typing import Scalar

__all__ = ["GaussianLikelihood", "chi_squared"]


def chi_squared(predicted: Array, observed: Array, sigma: float = 1.0) -> Scalar:
    """Return ``sum(((predicted - observed) / sigma)^2)``.

    Parameters
    ----------
    predicted : Array
        Model prediction.
    observed : Array
        Data, same shape.
    sigma : float
        Homoscedastic noise level.

    Returns
    -------
    Scalar
        The chi-squared misfit.
    """
    residual = (predicted - observed) / sigma
    return jnp.sum(residual * residual)


@dataclass(frozen=True)
class GaussianLikelihood:
    """Independent Gaussian noise of one width on every data point.

    ``log_prob`` is ``-chi_squared / 2``. The normalization constant
    ``-m log(sigma sqrt(2 pi))`` is **dropped**: it does not depend on the
    prediction, so neither optimization nor sampling sees it.

    Attributes
    ----------
    sigma : float
        The noise level.
    """

    sigma: float = 1.0

    def log_prob(self, predicted: Array, observed: Array) -> Scalar:
        """Return the log-likelihood up to its additive constant.

        Parameters
        ----------
        predicted : Array
            Model prediction.
        observed : Array
            Data, same shape.

        Returns
        -------
        Scalar
            ``-0.5 * chi_squared(predicted, observed, sigma)``.
        """
        return -0.5 * chi_squared(predicted, observed, self.sigma)
