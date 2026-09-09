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
    "l_curve_corner",
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


def l_curve_corner(
    regularization: Array, misfit: Array, norm: Array
) -> tuple[Scalar, Array]:
    """Pick a regularization weight at the L-curve's corner.

    The standard Tikhonov criterion, and for made-to-measure it answers "what
    should ``mu`` be" -- a question the module previously answered with "1e-3,
    because it is small". Given a fit repeated over a grid of ``mu``, the corner
    of the log-log curve of misfit against regularizer norm is the point of
    maximum curvature: below it the misfit barely improves while the norm runs
    away, above it the norm barely improves while the misfit degrades.

    Curvature is taken with respect to ``log(regularization)``, so an unevenly
    spaced grid is handled correctly, via

    ``kappa = (rho' eta'' - rho'' eta') / (rho'^2 + eta'^2)^(3/2)``

    with ``eta = log(norm)`` and ``rho = log(misfit)``. The **three** points at
    each end are excluded. Two reasons, and the count is the second one:
    ``jnp.gradient`` falls back to one-sided differences at the ends, which
    routinely produce the largest spurious curvature in the array; and taking it
    twice makes the curvature at index ``i`` depend on the data at ``i +/- 2``,
    so one bad endpoint reaches two points inward and excluding two would not
    contain it.

    MEASURED, against the ``mu`` that actually minimizes the weight error --
    an oracle a real fit does not have -- on three tracer problems over eight
    noise realizations each (``reports/M2M_production_readiness.md``, A4):

    ================================  ==============  ==========  ==========
    criterion                         eff ~ n         eff << n    precise data
    ================================  ==============  ==========  ==========
    this function                     **1.04x**       **1.30x**   **1.02x**
    generalized cross-validation      1.01x           2.15x       1.01x
    discrepancy, chi2 = m - eff       1.72x           2.39x       1.60x
    ``mu = 1e-3``                     2.46x           5.30x       3.18x
    NNLS, the honest mu -> 0 limit    4.70x           13.97x      6.73x
    ================================  ==============  ==========  ==========

    Read as a multiple of the oracle's weight error, so 1.00x is the best
    achievable. GCV -- which needs nothing this package does not already have,
    being ``misfit / (m - effective_parameters)^2`` -- beats this function on
    two of the three and is **2.15x** on the degenerate problem, where this one
    is 1.30x. Neither dominates; both beat ``mu = 1e-3`` on all three, and both
    beat driving ``mu`` to zero, which pins 21 %, 88 % and 51 % of the weights
    at exactly zero on the three problems.

    **No criterion here is universally reliable, and this one is not either.**
    On synthetic Tikhonov problems chosen to be hostile, all three fail, in
    different places:

    ==============================  =========  =========  =============
    synthetic problem               this one   GCV        discrepancy
    ==============================  =========  =========  =============
    hard spectral gap, delta=1e-2   171x       16x        16x
    hard spectral gap, delta=1e-4   5.9x       58x        3312x
    smooth spectrum, s_i ~ 1/i      99x        1.0x       1.0x
    smooth spectrum, s_i ~ 1/i^2    14x        3.5x       3.5x
    ==============================  =========  =========  =============

    So: run it **per problem**, run GCV beside it, and treat a disagreement
    between the two as a warning rather than picking one on faith. The
    curvature array is returned for exactly this reason -- a curve with no
    corner has a flat curvature profile and no peak worth trusting, and a
    caller who only sees the returned ``mu`` cannot tell that apart from a
    sharp corner.

    WHAT TO PASS AS ``norm``. The regularizer's norm, **not** minus its log
    density. For :class:`~mimirax.priors.EntropyPrior` that is
    :meth:`~mimirax.priors.EntropyPrior.divergence`: ``-S`` is negative at the
    prior's own mode, so ``log(-S)`` does not exist and a curve drawn against it
    is a curve of the wrong quantity.

    ``mu`` IS NOT DIMENSIONLESS and the number this returns does not transfer
    between problems: it carries the ratio of the misfit's scale to the norm's,
    so rescaling the data, ``sigma`` or the total weight rescales it. The
    criterion transfers; the value does not. Re-run it per problem.

    Parameters
    ----------
    regularization : Array
        ``(k,)`` regularization weights, strictly positive and strictly
        increasing -- the ``mu`` grid the fit was repeated over. At least nine
        of them.
    misfit : Array
        ``(k,)`` misfit at each ``mu``: the achieved chi-squared, strictly
        positive.
    norm : Array
        ``(k,)`` regularizer norm at each ``mu``, strictly positive.

    Returns
    -------
    tuple[Scalar, Array]
        The ``mu`` at the corner, and the ``(k,)`` curvature array, whose
        three end points on each side are ``nan`` because curvature was not
        evaluated there. Returning the curvature makes a bad corner visible: a
        curve with no corner has no clear peak, and a caller that only sees the
        ``mu`` cannot tell.

    Raises
    ------
    ValueError
        If the three arrays are not the same one-dimensional shape, if there
        are fewer than nine points, or if ``regularization`` is not strictly
        increasing and positive.
    """
    mu = jnp.asarray(regularization)
    chi = jnp.asarray(misfit)
    eta_raw = jnp.asarray(norm)
    if mu.ndim != 1 or chi.shape != mu.shape or eta_raw.shape != mu.shape:
        raise ValueError(
            "regularization, misfit and norm must be one-dimensional and the "
            f"same shape; got {mu.shape}, {chi.shape}, {eta_raw.shape}"
        )
    if mu.shape[0] < 9:
        raise ValueError(
            "an L-curve corner needs at least 9 points, because the three at "
            f"each end are excluded; got {mu.shape[0]}"
        )
    if not bool(jnp.all(mu > 0.0)) or not bool(jnp.all(jnp.diff(mu) > 0.0)):
        raise ValueError(
            "regularization must be strictly positive and strictly increasing"
        )

    s = jnp.log(mu)
    eta = jnp.log(eta_raw)
    rho = jnp.log(chi)
    # ``jnp.gradient`` is typed as returning a list when asked for several
    # axes; ``asarray`` narrows the single-axis case back to an ``Array``.
    d_eta = jnp.asarray(jnp.gradient(eta, s))
    d_rho = jnp.asarray(jnp.gradient(rho, s))
    dd_eta = jnp.asarray(jnp.gradient(d_eta, s))
    dd_rho = jnp.asarray(jnp.gradient(d_rho, s))
    speed = (d_rho * d_rho + d_eta * d_eta) ** 1.5
    curvature = (d_rho * dd_eta - dd_rho * d_eta) / speed
    interior = jnp.abs(curvature).at[:3].set(-jnp.inf).at[-3:].set(-jnp.inf)
    corner = mu[jnp.argmax(interior)]
    reported = curvature.at[:3].set(jnp.nan).at[-3:].set(jnp.nan)
    return corner, reported


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
