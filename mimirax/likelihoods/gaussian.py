"""The Gaussian likelihood and its chi-squared misfit.

The width may be a scalar or **one width per data point**. That is not a
convenience: a made-to-measure data vector routinely stacks moments with
incommensurate units -- a binned mass and a binned mass-weighted ``|v|^2`` in
the same vector -- and a scalar ``sigma`` then weights them by accident of
units rather than by how well either is measured. The array form is the
supported way to say what each observable's uncertainty actually is.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from mimirax._typing import Scalar, SigmaLike

__all__ = ["GaussianLikelihood", "chi_squared"]


def chi_squared(predicted: Array, observed: Array, sigma: SigmaLike = 1.0) -> Scalar:
    """Return ``sum(((predicted - observed) / sigma)^2)``.

    Parameters
    ----------
    predicted : Array
        Model prediction.
    observed : Array
        Data, same shape.
    sigma : SigmaLike
        Noise level: a scalar for homoscedastic noise, or an array
        broadcasting against the data for one width per data point.

    Returns
    -------
    Scalar
        The chi-squared misfit.
    """
    residual = (predicted - observed) / sigma
    return jnp.sum(residual * residual)


@dataclass(frozen=True)
class GaussianLikelihood:
    """Independent Gaussian noise on every data point, of one width or of many.

    ``log_prob`` is ``-chi_squared / 2``. The normalization constant
    ``-sum_j log(sigma_j sqrt(2 pi))`` is **dropped**: it does not depend on the
    prediction, so neither optimization nor sampling sees it. It does depend on
    :attr:`sigma`, so this log-density may not be compared across two different
    :attr:`sigma` -- which is a statement about model comparison and not about
    fitting.

    HETEROSCEDASTIC IS THE USUAL CASE HERE.
    A scalar :attr:`sigma` is only meaningful when every entry of the data
    vector is in the same units and measured equally well. A made-to-measure
    vector usually is not: stack a binned mass moment and a binned ``|v|^2``
    moment and their numerical scales differ by whatever ``|v|^2`` happens to
    be, so a scalar width silently weights the fit by unit choice. Pass an
    array of per-point widths instead, one per entry of ``observed``.

    Attributes
    ----------
    sigma : SigmaLike
        The noise level. A scalar applies one width to every point; an
        ``(m,)`` array -- or anything broadcasting against the data -- applies
        one width per point. Must be strictly positive; not checked, because
        :attr:`sigma` may be a traced array.
    """

    sigma: SigmaLike = 1.0

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

    def data_arrays(self) -> dict | None:
        """Return the arrays of the likelihood, for a program to take as arguments.

        A per-point :attr:`sigma` is as large as the data; closed over, it would be
        a constant of the compiled program (``mimirax.shooting``, "data are
        arguments"). A Python-number ``sigma`` stays a literal.

        Returns
        -------
        dict | None
            ``{"sigma": sigma}`` when :attr:`sigma` is an array, else ``None``.
        """
        if isinstance(self.sigma, (jax.Array, np.ndarray)):
            return {"sigma": self.sigma}
        return None

    def with_data_arrays(self, arrays: Any) -> GaussianLikelihood:
        """Return the likelihood with :attr:`sigma` replaced.

        Parameters
        ----------
        arrays : Any
            ``{"sigma": ...}`` as :meth:`data_arrays` returned it (or tracers of
            it), or ``None`` for no change.

        Returns
        -------
        GaussianLikelihood
            The likelihood reading ``arrays``.
        """
        if arrays is None:
            return self
        return dataclasses.replace(self, sigma=arrays["sigma"])
