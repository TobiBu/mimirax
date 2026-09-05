"""Residuals between prediction and data."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from mimirax._typing import Scalar

__all__ = ["normalized_residuals", "residuals", "rms_residual"]


def residuals(predicted: Array, observed: Array) -> Array:
    """Return ``predicted - observed``.

    Parameters
    ----------
    predicted : Array
        Model prediction.
    observed : Array
        Data, same shape.

    Returns
    -------
    Array
        The signed residuals.
    """
    return predicted - observed


def normalized_residuals(
    predicted: Array, observed: Array, sigma: float = 1.0
) -> Array:
    """Return ``(predicted - observed) / sigma``.

    Parameters
    ----------
    predicted : Array
        Model prediction.
    observed : Array
        Data, same shape.
    sigma : float
        Noise level.

    Returns
    -------
    Array
        Residuals in units of the noise.
    """
    return residuals(predicted, observed) / sigma


def rms_residual(predicted: Array, observed: Array) -> Scalar:
    """Return the root-mean-square residual.

    Parameters
    ----------
    predicted : Array
        Model prediction.
    observed : Array
        Data, same shape.

    Returns
    -------
    Scalar
        ``sqrt(mean((predicted - observed)^2))``.
    """
    r = residuals(predicted, observed)
    return jnp.sqrt(jnp.mean(r * r))
