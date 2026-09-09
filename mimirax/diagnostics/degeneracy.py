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

__all__ = [
    "degenerate_directions",
    "effective_parameters",
    "fisher_information",
    "profile_likelihood",
]


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


def effective_parameters(data_curvature: Array, prior_curvature: Array) -> Scalar:
    """Return how many parameters the **data** actually determined.

    ``tr(H_data (H_data + H_prior)^-1)``, the standard effective number of
    degrees of freedom of a regularized fit. In the joint eigenbasis it is
    ``sum_i lambda_i / (lambda_i + pi_i)``: a direction the data constrain
    tightly (``lambda >> pi``) contributes 1, a direction only the prior holds
    up (``lambda << pi``) contributes 0, and the total interpolates.

    WHY THIS AND NOT AN EIGENVALUE COUNT.
    :func:`degenerate_directions` needs an ``rtol`` and returns a *count*, so it
    reports a rank at a threshold and puts a marginal direction wholly on one
    side. This needs no threshold and is the quantity that actually answers "is
    my objective informative enough": on the made-to-measure tracer problem of
    ``reports/M2M_module.md`` -- 64 weights, 10 observables -- it returns
    **10.000**, exactly the number of observables, while
    ``degenerate_directions(rtol=1e-3)`` reports 55 of 64 and the fit reaches a
    chi-squared of 1.4e-08. Ten numbers went in and ten degrees of freedom came
    out; the other 54 are the prior's answer, and no optimizer, step size or
    ``mu`` changes that. More information means more or better observables.

    Read it against the parameter count, not on its own: ``10`` out of 64 says
    the fit is prior-dominated, and ``10`` out of 10 would say it is saturated.

    Parameters
    ----------
    data_curvature : Array
        ``(d, d)`` Hessian of the *negative log likelihood* alone -- the data's
        contribution. Get it as ``fisher_information(problem.negative_log_
        likelihood_only, params)``, or by subtracting the priors' curvature from
        the full :func:`fisher_information`.
    prior_curvature : Array
        ``(d, d)`` Hessian of the negative log prior, same basis and ordering.

    Returns
    -------
    Scalar
        The effective number of data-determined parameters, between 0 and ``d``.
        Not an integer in general.
    """
    total = data_curvature + prior_curvature
    return jnp.trace(data_curvature @ jnp.linalg.solve(total, jnp.eye(total.shape[0])))


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
