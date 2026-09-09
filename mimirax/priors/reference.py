"""Priors and regularizers: a Gaussian, an L2 penalty, and the M2M entropy prior.

WHERE THE JACOBIAN SITS. Every ``log_prob`` here is a density over the space it
is *handed*, and none of them adds the log-Jacobian of a reparameterization.
That is the scaffold's convention for :class:`GaussianPrior` and it is kept for
:class:`EntropyPrior`: a caller who moves in an unconstrained space and wants a
density over *that* space adds
:func:`mimirax.parameters.log_abs_det_jacobian` itself. The alternative --
folding the Jacobian into the prior -- would make the same prior mean two
different things depending on which optimizer called it. This is friction 1 of
the EDDA programme's D-027 and the choice is made here, once, explicitly:
**priors are densities over the constrained parameters**, and
:meth:`mimirax.inference.MadeToMeasure.minimize` evaluates its objective on
constrained parameters for exactly that reason.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
    """The made-to-measure entropy prior over a positive weight vector.

    Syer & Tremaine's (1996) regularizer keeps the particle weights ``w`` near a
    reference ``w0``. Two spellings of it are in the literature and they do not
    have the same mode, so both are here and the default is stated:

    * ``include_linear_term=True`` (the default, Dehnen 2009's form)::

          S = -sum_i w_i * (log(w_i / w0_i) - 1)

      whose gradient is ``dS/dw_i = -log(w_i / w0_i)``. It vanishes at
      ``w = w0``, so the prior's mode is the reference **unconditionally**.
    * ``include_linear_term=False`` (Syer & Tremaine's own ``S = -sum w log(w /
      w0)``), whose gradient is ``-(log(w_i / w0_i) + 1)`` and therefore
      vanishes at ``w = w0 / e``. Its mode is the reference only *on* the
      constant-total-weight surface that
      :class:`mimirax.parameters.TotalMassConstraint` fixes, where the Lagrange
      multiplier absorbs the constant.

    Both are strictly concave on ``w > 0`` -- the Hessian is the negative
    diagonal ``-mu / w_i`` for either -- so the term can only ever pull a fit
    toward the reference, never create a second optimum. ``log_prob`` returns
    ``mu * S``, the *log density*, so it adds to a likelihood the way every
    other :class:`~mimirax.protocols.Prior` here does; the ``mu S`` term of the
    made-to-measure objective is exactly this, and ``mu`` lives here and nowhere
    else (:class:`~mimirax.inference.MadeToMeasure` deliberately has no second
    copy of it).

    **Domain.** The entropy is defined on ``w > 0`` and this returns ``nan``
    outside it, deliberately: the package's rule is that positivity is enforced
    by a reparameterization (:class:`mimirax.parameters.LogTransform`), not by
    clipping inside a density, and a silent ``where``-guard here would hide a
    parameterization mistake instead of failing on it.

    Attributes
    ----------
    mu : float
        Regularization strength; scales the whole term linearly.
    reference : Array | float | None
        ``w0``. A scalar or an ``(n,)`` array; ``None`` means the uniform
        reference ``1``, so the mode is the all-ones weight vector. A system of
        total weight ``M`` on ``n`` particles wants ``reference=M / n``.
    key : str | None
        Which leaf of a mapping ``params`` holds the weights. With ``None`` the
        prior is applied to every leaf, as :class:`GaussianPrior` is; a
        non-``None`` key (the default ``"weights"``) is what keeps the prior off
        the position and velocity leaves of a made-to-measure parameter dict,
        where a logarithm would be a domain error rather than a regularizer.
    include_linear_term : bool
        Whether to use the ``-1`` shifted form whose mode is the reference; see
        above.
    """

    mu: float = 1.0
    reference: Array | float | None = None
    key: str | None = "weights"
    include_linear_term: bool = True

    def _leaf(self, weights: Array) -> Scalar:
        """Return ``mu * S`` for one weight leaf.

        Parameters
        ----------
        weights : Array
            Strictly positive weights.

        Returns
        -------
        Scalar
            The leaf's contribution to the log prior density.
        """
        w = jnp.asarray(weights)
        reference = jnp.ones_like(w) if self.reference is None else self.reference
        ratio = jnp.log(w / reference)
        if self.include_linear_term:
            ratio = ratio - 1.0
        return -self.mu * jnp.sum(w * ratio)

    def log_prob(self, params: PyTree) -> Scalar:
        """Return ``mu * S`` at ``params``, the entropy prior's log density.

        Parameters
        ----------
        params : PyTree
            The particle weights: a mapping carrying :attr:`key`, or any pytree
            whose leaves are all weights when :attr:`key` is ``None``.

        Returns
        -------
        Scalar
            ``-mu * sum(w * (log(w / w0) - 1))`` with the default
            :attr:`include_linear_term`, and ``-mu * sum(w * log(w / w0))``
            without it. ``nan`` where any weight is non-positive.

        Raises
        ------
        KeyError
            If ``params`` is a mapping and :attr:`key` is not one of its keys.
            The message lists the keys that are there, because the usual cause
            is a parameter dict that spells the weights differently.
        """
        if self.key is not None and isinstance(params, Mapping):
            if self.key not in params:
                raise KeyError(
                    f"EntropyPrior(key={self.key!r}) found no such leaf; params "
                    f"has {tuple(params)}. Pass key=... or key=None to apply the "
                    "prior to every leaf."
                )
            return self._leaf(params[self.key])
        return _sum_over_leaves(params, self._leaf)

    def _divergence_leaf(self, weights: Array) -> Scalar:
        """Return the KL divergence of one weight leaf from the reference.

        Parameters
        ----------
        weights : Array
            Strictly positive weights.

        Returns
        -------
        Scalar
            ``sum_i [w_i log(w_i / w0_i) - w_i + w0_i]``, without ``mu``.
        """
        w = jnp.asarray(weights)
        reference = jnp.ones_like(w) if self.reference is None else self.reference
        return jnp.sum(w * jnp.log(w / reference) - w + reference)

    def divergence(self, params: PyTree) -> Scalar:
        """Return the prior's **norm**: the KL divergence from the reference.

        ``sum_i [w_i log(w_i / w0_i) - w_i + w0_i]``, which is ``>= 0`` and zero
        exactly at ``w = w0``. Note ``mu`` does **not** appear: this is the
        functional ``mu`` weights, not the weighted term.

        WHY THIS EXISTS AND ``-log_prob`` WILL NOT DO.
        Choosing ``mu`` by an L-curve means plotting the misfit against the
        regularizer's norm, and ``-log_prob / mu = -S = sum_i w_i (log(w_i/w0_i)
        - 1)`` is **not a norm**: at the prior's own mode ``w = w0`` it equals
        ``-sum_i w0_i``, which is negative, and its logarithm -- which is what
        an L-curve is drawn in -- does not exist. The two differ by the constant
        ``sum_i w0_i``, so they have the same minimiser and a *different*
        L-curve. An L-curve drawn against ``-S`` is a curve of the wrong
        quantity; this is the right one.

        With ``include_linear_term=False`` the log density is a different
        functional, but this divergence is unchanged -- it is the natural norm
        for either spelling, and it is what
        :func:`mimirax.diagnostics.l_curve_corner` expects.

        Parameters
        ----------
        params : PyTree
            The particle weights, read the same way :meth:`log_prob` reads them.

        Returns
        -------
        Scalar
            The KL divergence from the reference. ``nan`` where any weight is
            non-positive, for the reason :meth:`log_prob` gives.

        Raises
        ------
        KeyError
            If ``params`` is a mapping and :attr:`key` is not one of its keys.
        """
        if self.key is not None and isinstance(params, Mapping):
            if self.key not in params:
                raise KeyError(
                    f"EntropyPrior(key={self.key!r}) found no such leaf; params "
                    f"has {tuple(params)}. Pass key=... or key=None to apply the "
                    "prior to every leaf."
                )
            return self._divergence_leaf(params[self.key])
        return _sum_over_leaves(params, self._divergence_leaf)
