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
from jax.scipy.linalg import solve_triangular

from mimirax._typing import PyTree, Scalar, SigmaLike

__all__ = [
    "degenerate_directions",
    "effective_parameters",
    "effective_parameters_from_jacobian",
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

    WHEN THIS IS A FISHER MATRIX, AND WHEN IT IS NOT. For a Gaussian likelihood
    with prediction ``y(theta)``, residual ``r = y - observed`` and noise
    ``sigma``, the Hessian of the negative log likelihood is

    ``J^T C^-1 J + sum_j (r_j / sigma_j^2) d^2 y_j / d theta^2``

    with ``J = dy / dtheta``. The first term is the Fisher information -- the
    Gauss--Newton matrix: positive semi-definite, of rank at most ``min(m, d)``,
    and what a Cramer--Rao forecast or :func:`effective_parameters` means by
    the data's curvature. The second is residual curvature, and it is neither:
    it has no sign and is in general of full rank. So this Hessian **is** the
    Fisher matrix

    * at any point, when the prediction is linear in the parameters -- the
      tracer made-to-measure weights, whose orbits do not depend on them;
    * wherever the residuals vanish -- at the truth of noise-free mock data, or
      at an exact fit.

    At the maximum a posteriori point of noisy data it is the *observed*
    information, which differs from the Fisher matrix by the residual term
    evaluated there. Anywhere else on a nonlinear model it is local curvature
    and **not** a Fisher matrix. Measured in the EDDA Fisher probe
    (``reports/FISHER_fd_vs_ad_probe.md``, section 1.8): in the self-consistent
    made-to-measure construction, where the weights are the masses and move the
    orbits, at uniform weights, with ten observables of 32 weights, the residual
    term fills the null space and :func:`effective_parameters` on this Hessian
    returns **32.000009** -- more than the 32 parameters there are, which only
    an indefinite curvature can give -- where ``J^T C^-1 J`` gives **10.00**.

    For the Fisher matrix away from an exact fit, differentiate the
    *prediction* rather than the objective and use
    :func:`effective_parameters_from_jacobian`. And passed the full negative log
    *posterior*, this includes the priors' curvature too; pass the likelihood
    alone before reading it as the data's information.

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

    WHERE THIS DENSE ROUTE FAILS. It solves against ``H_data + H_prior``, and a
    ``H_data`` built from a Jacobian as ``J^T C^-1 J`` already has the squared
    condition number of ``J``. Its absolute error is about ``eps * d_max``, so
    a direction at the prior's threshold is lost once ``d_max / threshold``
    exceeds ``1 / eps``. Measured in the EDDA Fisher probe
    (``reports/FISHER_fd_vs_ad_probe.md``, section 5.7), on 32 weights:

    * kept in **single precision**, it returns values from **-314 to +257** on
      Jacobians taken at the probe's default steps, and **28.19** on a
      single-precision AD Jacobian that, analysed in double, gives 13.92;
    * in **double precision** it differs from the SVD route by 6.6e-5 to
      **9.7e-4** on the probe's three radial sets -- more than the effects that
      probe set out to measure.

    The single-precision failure is not a slow loss of accuracy. On the unit
    test's version of that set the prior's ``3.2e-6`` lies below the float32
    spacing (``8e-3``) of every diagonal entry of ``J^T C^-1 J``, so
    ``H_data + H_prior`` **is** ``H_data``, bit for bit, and what comes back is
    ``tr(F F^-1)`` of a numerically singular ``F``: rounding, whatever ``mu``.

    It is also only as much a Fisher matrix as ``data_curvature`` is: a Hessian
    taken away from an exact fit of a nonlinear model is not one (see
    :func:`fisher_information`). Given the Jacobian, prefer
    :func:`effective_parameters_from_jacobian`, which has neither problem. This
    function is kept as it is so that existing results reproduce.

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


def effective_parameters_from_jacobian(
    jacobian: Array, prior_curvature: Array, *, sigma: SigmaLike | None = None
) -> Scalar:
    """Return :func:`effective_parameters` from the Jacobian, by an SVD.

    The same quantity, ``tr(F (F + H_prior)^-1)`` with ``F = J^T C^-1 J`` the
    Gauss--Newton matrix -- the Fisher information of a Gaussian likelihood --
    computed without ever forming ``F``. Whiten the Jacobian by the noise and
    by the prior, ``B = C^(-1/2) J L^-T`` with ``H_prior = L L^T``. Then
    ``F + H_prior = L (B^T B + I) L^T``, and the trace is exactly

    ``sum_k s_k^2 / (s_k^2 + 1)``

    over the singular values ``s_k`` of ``B``. A diagonal prior ``p`` makes
    ``L = diag(sqrt(p))`` and the whitening a column scaling, so no ``(d, d)``
    matrix is formed at all. With ``p_i = mu / w_i`` -- the curvature of
    :class:`~mimirax.priors.EntropyPrior` at ``w`` -- the sum is
    ``sum_k d_k / (d_k + mu)`` over the squared singular values ``d_k`` of
    ``C^(-1/2) J diag(sqrt(w))``, the prior-whitened form of
    ``reports/M2M_production_readiness.md`` (A2); at uniform ``w = w0`` it is
    ``sum_k d_k / (d_k + mu / w0)`` over those of ``C^(-1/2) J``. Either way
    ``mu`` is a threshold on a spectrum the observables alone determine.

    WHY AN SVD. Forming ``F`` squares the condition number: its absolute error
    is about ``eps * d_max``, and the dense route loses a direction at the
    threshold once ``d_max / threshold`` exceeds ``1 / eps``. The singular
    values of ``B`` carry an absolute error of about ``eps * s_max`` against a
    threshold at ``s = 1``, so this holds until ``s_max`` reaches ``1 / eps``:
    ``d_max / threshold`` up to ``1 / eps^2``, the square of the dense route's
    range. (Those are norm-wise bounds, i.e. reasoning.) Measured in the EDDA
    Fisher probe (``reports/FISHER_fd_vs_ad_probe.md``, section 5.7) on the
    sensitivity figure's ``radial_wide`` set, 32 weights, ``d_max / threshold``
    of 1.6e13, from a single-precision AD Jacobian: this route in single
    precision gives **13.937** against 13.920 from the same Jacobian analysed in
    double, where :func:`effective_parameters` gives **28.19**. In double
    precision, forward- and reverse-mode Jacobians agree through this route to
    6e-12, while the dense route differs from it by up to 9.7e-4. The margin
    does shrink toward ``1 / eps^2``: on the unit test's circular-orbit version
    of that set, the single-precision error is 0.003 at ``d_max / threshold``
    of 1.6e12 and 0.03 at 1.6e13. The arithmetic is in the precision the
    inputs promote to, so a float64 prior turns a float32 Jacobian's analysis
    into a float64 one.

    WHY A JACOBIAN AND NOT A HESSIAN. ``F`` here is ``J^T C^-1 J`` at whatever
    point ``J`` was taken -- positive semi-definite and of rank at most
    ``min(m, d)``, so the count cannot exceed the number of observables. The
    Hessian of :func:`fisher_information` is that matrix only where the model
    is linear or the residuals vanish; elsewhere its residual curvature is
    counted as information (see there).

    Get ``J`` by differentiating the *prediction* -- ``observable(forward(x))``
    -- over the flattened free parameters: :func:`jax.jacrev` costs ``m``
    reverse passes and is the cheaper when there are more parameters than data,
    as in made-to-measure; :func:`jax.jacfwd` costs ``d`` tangents.

    Parameters
    ----------
    jacobian : Array
        ``(m, d)`` derivative of the prediction with respect to the flattened
        parameters, rows in data order and columns in the prior's order.
        Already whitened, ``C^(-1/2) J``, if ``sigma`` is ``None``.
    prior_curvature : Array
        Hessian of the negative log prior, which must be positive definite.
        Either ``(d,)``, the diagonal of a diagonal prior -- exact, and the form
        to pass for :class:`~mimirax.priors.EntropyPrior`, whose curvature is
        ``mu / w`` -- or ``(d, d)``, a dense one, whitened by its Cholesky
        factor. A direction with no prior curvature has no defined count, and
        a zero diagonal entry, or a dense prior that is not positive definite,
        gives ``nan`` or ``inf``.
    sigma : SigmaLike | None
        Noise width to whiten the rows by: one number, or ``(m,)``, as
        :class:`~mimirax.likelihoods.GaussianLikelihood` takes it. ``None``
        means ``jacobian`` is already whitened; for a correlated noise
        covariance ``C = K K^T``, pass ``K^-1 J`` and ``None``.

    Returns
    -------
    Scalar
        The effective number of data-determined parameters, between 0 and
        ``min(m, d)``. Not an integer in general.

    Raises
    ------
    ValueError
        If ``jacobian`` is not two-dimensional, if ``prior_curvature`` is
        neither ``(d,)`` nor ``(d, d)``, or if ``sigma`` is neither one
        number nor one per row.
    """
    whitened = jnp.asarray(jacobian)
    if whitened.ndim != 2:
        raise ValueError(f"jacobian must be (m, d); got shape {whitened.shape}")
    num_data, num_params = whitened.shape
    if sigma is not None:
        noise = jnp.asarray(sigma)
        if noise.ndim > 1 or noise.size not in (1, num_data):
            raise ValueError(
                f"sigma must be one number or one per row of the ({num_data}, "
                f"{num_params}) jacobian; got shape {noise.shape}"
            )
        whitened = whitened / jnp.reshape(noise, (-1, 1))
    prior = jnp.asarray(prior_curvature)
    if prior.shape == (num_params,):
        whitened = whitened / jnp.sqrt(prior)
    elif prior.shape == (num_params, num_params):
        # B = A L^-T, i.e. B^T = L^-1 A^T: one triangular solve, no inverse.
        factor = jnp.linalg.cholesky(prior)
        whitened = solve_triangular(factor, whitened.T, lower=True).T
    else:
        raise ValueError(
            f"prior_curvature must be ({num_params},) or ({num_params}, "
            f"{num_params}) for a jacobian with {num_params} columns; got shape "
            f"{prior.shape}"
        )
    singular = jnp.linalg.svd(whitened, compute_uv=False)
    squared = singular * singular
    return jnp.sum(squared / (squared + 1.0))


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
