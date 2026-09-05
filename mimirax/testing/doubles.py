"""Analytic test doubles with closed-form gradients.

WHY THESE EXIST. Every core numerical path in mimirax has to be exercised on
CPU with no solver installed, or the suite will not run often enough to be
worth having. These doubles satisfy the protocols in :mod:`mimirax.protocols`
and, unlike a solver, come with their derivatives written out by hand -- so a
test can check ``jax.grad`` against an exact expression rather than against
finite differences whose tolerance has to be argued for.

* :class:`LinearForwardModel` -- ``y = A x + b``; the gradient of a
  sum-of-squares misfit is ``2 A^T (A x + b - y)``.
* :class:`SoftenedPointMassField` -- the softened Newtonian field of fixed
  sources at test-particle positions, with its tidal tensor
  ``d a / d x = -G sum_j m_j [ I / r^3 - 3 d d^T / r^5 ]`` in closed form.
* :class:`DirectSumGravity` -- self-gravity of ``n`` bodies, the honest
  ``O(n^2)`` :class:`~mimirax.protocols.ForceModel` any FMM is checked against.

All are float64-exact to round-off; tests compare at ``atol = 1e-12``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from mimirax._typing import PerParticle, PyTree, Vec3
from mimirax.inference.objective import InferenceProblem
from mimirax.likelihoods import GaussianLikelihood
from mimirax.observables import IdentityObservable

__all__ = [
    "DirectSumGravity",
    "LinearForwardModel",
    "SoftenedPointMassField",
    "make_linear_problem",
]


@dataclass(frozen=True)
class LinearForwardModel:
    """``y = A x + b``: a :class:`~mimirax.protocols.ForwardModel` whose Jacobian is ``A``.

    Attributes
    ----------
    matrix : Array
        ``(m, n)`` design matrix ``A``.
    offset : Array
        ``(m,)`` offset ``b``.
    """

    matrix: Array
    offset: Array

    def __call__(self, params: PyTree) -> Array:
        """Evaluate ``A x + b``.

        Parameters
        ----------
        params : PyTree
            ``(n,)`` parameter vector ``x``.

        Returns
        -------
        Array
            ``(m,)`` prediction.
        """
        return self.matrix @ jnp.asarray(params) + self.offset

    def exact_gradient_sum_of_squares(self, params: Array, observed: Array) -> Array:
        """Return ``d/dx sum((A x + b - y)^2) = 2 A^T (A x + b - y)``.

        Parameters
        ----------
        params : Array
            ``(n,)`` parameter vector.
        observed : Array
            ``(m,)`` data ``y``.

        Returns
        -------
        Array
            ``(n,)`` exact gradient.
        """
        return 2.0 * self.matrix.T @ (self(params) - observed)


@dataclass(frozen=True)
class SoftenedPointMassField:
    """Softened Newtonian field of fixed sources, evaluated at test particles.

    ``a_i = -G sum_j m_j d_ij / (|d_ij|^2 + eps^2)^{3/2}`` with
    ``d_ij = x_i - s_j``. A :class:`~mimirax.protocols.ForceModel` whose
    ``masses`` argument is ignored (the targets are test particles), and the
    analytic stand-in for "accelerations at tracers" in Paper I section 7.

    Attributes
    ----------
    sources : Array
        ``(s, 3)`` source positions.
    source_masses : Array
        ``(s,)`` source masses.
    softening : float
        Plummer softening ``eps``.
    gravitational_constant : float
        ``G``.
    """

    sources: Array
    source_masses: Array
    softening: float = 0.0
    gravitational_constant: float = 1.0

    def _displacements(self, positions: Vec3) -> tuple[Array, Array]:
        """Return the target-minus-source displacements and softened ``r^2``.

        Parameters
        ----------
        positions : Vec3
            ``(n, 3)`` target positions.

        Returns
        -------
        tuple[Array, Array]
            ``(n, s, 3)`` displacements and ``(n, s)`` softened squared
            distances.
        """
        d = positions[:, None, :] - self.sources[None, :, :]
        r2 = jnp.sum(d * d, axis=-1) + self.softening**2
        return d, r2

    def accelerations(
        self,
        positions: Vec3,
        masses: PerParticle,
        *,
        args: object = None,
    ) -> Vec3:
        """Return the field at every target.

        Parameters
        ----------
        positions : Vec3
            ``(n, 3)`` target positions.
        masses : PerParticle
            Ignored: targets are test particles.
        args : object
            Ignored.

        Returns
        -------
        Vec3
            ``(n, 3)`` accelerations.
        """
        del masses, args
        d, r2 = self._displacements(positions)
        inv_r3 = r2 ** (-1.5)
        weighted = self.source_masses[None, :, None] * d * inv_r3[..., None]
        return -self.gravitational_constant * jnp.sum(weighted, axis=1)

    def tidal_tensor(self, positions: Vec3) -> Array:
        """Return ``d a_i / d x_i`` for every target, in closed form.

        Parameters
        ----------
        positions : Vec3
            ``(n, 3)`` target positions.

        Returns
        -------
        Array
            ``(n, 3, 3)`` with ``[i, a, b] = d a_i[a] / d x_i[b]``; equal to
            the diagonal blocks of ``jax.jacfwd(accelerations)``.
        """
        d, r2 = self._displacements(positions)
        inv_r3 = r2 ** (-1.5)
        inv_r5 = r2 ** (-2.5)
        eye = jnp.eye(3)
        iso = eye[None, None, :, :] * inv_r3[..., None, None]
        aniso = 3.0 * d[..., :, None] * d[..., None, :] * inv_r5[..., None, None]
        per_source = self.source_masses[None, :, None, None] * (iso - aniso)
        return -self.gravitational_constant * jnp.sum(per_source, axis=1)


@dataclass(frozen=True)
class DirectSumGravity:
    """Softened self-gravity of ``n`` bodies by direct summation.

    The ``O(n^2)`` reference every hierarchical solver is measured against, and
    the double for a :class:`~mimirax.protocols.ForceModel` that actually uses
    ``masses``. Differentiable in positions and masses; the self-term is masked
    with ``where`` on both the distance and the weight so the gradient carries
    no ``0 * inf``.

    Attributes
    ----------
    softening : float
        Plummer softening ``eps``.
    gravitational_constant : float
        ``G``.
    """

    softening: float = 0.0
    gravitational_constant: float = 1.0

    def accelerations(
        self,
        positions: Vec3,
        masses: PerParticle,
        *,
        args: object = None,
    ) -> Vec3:
        """Return the mutual accelerations.

        Parameters
        ----------
        positions : Vec3
            ``(n, 3)`` positions.
        masses : PerParticle
            ``(n,)`` masses.
        args : object
            Ignored.

        Returns
        -------
        Vec3
            ``(n, 3)`` accelerations.
        """
        del args
        n = positions.shape[0]
        eye = jnp.eye(n, dtype=bool)
        d = positions[:, None, :] - positions[None, :, :]
        r2 = jnp.sum(d * d, axis=-1) + self.softening**2
        r2_safe = jnp.where(eye, 1.0, r2)
        inv_r3 = jnp.where(eye, 0.0, r2_safe ** (-1.5))
        weighted = masses[None, :, None] * d * inv_r3[..., None]
        return -self.gravitational_constant * jnp.sum(weighted, axis=1)


def make_linear_problem(
    key: Array,
    *,
    n_params: int = 4,
    n_data: int = 12,
    noise: float = 0.0,
    sigma: float = 1.0,
) -> tuple[InferenceProblem, Array]:
    """Build a well-posed linear-Gaussian problem with a known truth.

    Parameters
    ----------
    key : Array
        A ``jax.random`` key.
    n_params : int
        Number of free parameters. Static.
    n_data : int
        Number of data points; must exceed ``n_params`` for a unique optimum.
        Static.
    noise : float
        Standard deviation of Gaussian noise added to the synthetic data.
    sigma : float
        The likelihood's assumed noise level.

    Returns
    -------
    tuple[InferenceProblem, Array]
        The problem (linear forward model, identity observable, Gaussian
        likelihood, no priors) and the ``(n_params,)`` true parameters.
    """
    k_matrix, k_offset, k_truth, k_noise = jax.random.split(key, 4)
    matrix = jax.random.normal(k_matrix, (n_data, n_params))
    offset = jax.random.normal(k_offset, (n_data,))
    truth = jax.random.normal(k_truth, (n_params,))
    forward = LinearForwardModel(matrix=matrix, offset=offset)
    observed = forward(truth) + noise * jax.random.normal(k_noise, (n_data,))
    problem = InferenceProblem(
        forward=forward,
        observable=IdentityObservable(),
        likelihood=GaussianLikelihood(sigma=sigma),
        observed=observed,
    )
    return problem, truth
