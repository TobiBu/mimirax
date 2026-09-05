"""Degeneracy diagnostics: which parameter directions the data do not see.

Recovering discrete source positions from field samples has continuous
degeneracies (Paper I section 7 says so in its first paragraph), and the
diagnostics that name them are more useful than the ones that hide them. The
Fisher information's null-ish eigenvectors are the first such diagnostic.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

from mimirax._typing import PyTree, Scalar

__all__ = ["degenerate_directions", "fisher_information", "profile_likelihood"]


def fisher_information(
    negative_log_posterior: Callable[[PyTree], Scalar], params: PyTree
) -> Array:
    """Return the Hessian of the negative log posterior over the flattened parameters.

    At the maximum a posteriori point this is the observed Fisher information;
    elsewhere it is the local curvature and should be read as such.

    Parameters
    ----------
    negative_log_posterior : Callable[[PyTree], Scalar]
        The objective, in the parameters' own pytree structure.
    params : PyTree
        Where to evaluate.

    Returns
    -------
    Array
        ``(d, d)`` symmetric matrix over the flattened parameter vector, in
        the order :func:`mimirax.parameters.ravel` uses.
    """
    flat, unravel = ravel_pytree(params)

    def _flat_objective(x: Array) -> Scalar:
        return negative_log_posterior(unravel(x))

    return jax.hessian(_flat_objective)(flat)


def degenerate_directions(
    fisher: Array, *, rtol: float = 1.0e-8
) -> tuple[Array, Array]:
    """Return the eigenvalues and the eigenvectors below ``rtol`` of the largest.

    Parameters
    ----------
    fisher : Array
        ``(d, d)`` symmetric matrix from :func:`fisher_information`.
    rtol : float
        Eigenvalues below ``rtol * max(eigenvalues)`` count as degenerate.

    Returns
    -------
    tuple[Array, Array]
        All ``(d,)`` eigenvalues in ascending order, and the ``(d, r)`` matrix
        whose columns are the degenerate directions (``r`` may be zero).
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(fisher)
    threshold = rtol * jnp.max(jnp.abs(eigenvalues))
    mask = eigenvalues < threshold
    return eigenvalues, eigenvectors[:, mask]


def profile_likelihood(
    negative_log_posterior: Callable[[PyTree], Scalar],
    params: PyTree,
    direction: Array,
) -> Array:
    """Profile the objective along a direction, re-optimizing the rest. Stub.

    Parameters
    ----------
    negative_log_posterior : Callable[[PyTree], Scalar]
        The objective.
    params : PyTree
        The starting point.
    direction : Array
        A flattened-parameter direction.

    Returns
    -------
    Array
        Never returns.

    Raises
    ------
    NotImplementedError
        Always; needs an optimizer in the loop and is left for a method session.
    """
    raise NotImplementedError("profile_likelihood is a scaffold stub")
