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
* :class:`VerletSegmentEngine` -- a pure-JAX
  :class:`~mimirax.shooting.SegmentEngine` (velocity Verlet over
  :class:`DirectSumGravity` plus a parameterized field), so multiple shooting is
  testable with no solver installed.

All are float64-exact to round-off; tests compare at ``atol = 1e-12``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from mimirax._typing import PerParticle, PyTree, Vec3
from mimirax.inference.objective import InferenceProblem
from mimirax.likelihoods import GaussianLikelihood
from mimirax.observables import IdentityObservable
from mimirax.shooting import SegmentResult

__all__ = [
    "DirectSumGravity",
    "LinearForwardModel",
    "SoftenedPointMassField",
    "VerletSegmentEngine",
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


@dataclass(frozen=True)
class VerletSegmentEngine:
    """A pure-JAX :class:`~mimirax.shooting.SegmentEngine`: velocity Verlet, no solver.

    The double the multiple-shooting core is tested on without jaccpot. Each
    segment computes its start force, then runs ``segment_steps[segment]``
    steps of the velocity Verlet jaccpot's rollouts write (drift ``x + v dt +
    a dt^2 / 2``, force, kick ``v + (a + a') dt / 2``) in a ``lax.scan``, over
    :class:`DirectSumGravity` plus an optional external field in nornax's
    ``f(x, t, args)`` convention (evaluated at ``t = 0``) whose parameters are
    ``theta``. Differentiated by plain autodiff.

    With :attr:`vjp_only` the segment is a ``jax.custom_vjp`` (its backward is
    the plain VJP), so any forward-mode differentiation through it raises --
    the property jaccpot's checkpointed rollouts have, made available to tests
    that guard a VJP-only path.

    Attributes
    ----------
    segment_steps : tuple[int, ...]
        Steps per segment.
    dt : float
        The timestep.
    field : Callable[[Array, Any, Any], Array] | None
        ``field(positions, t, theta) -> [N, 3]``, or ``None``.
    softening : float
        Plummer softening of the self-gravity.
    gravitational_constant : float
        ``G`` of the self-gravity; ``0`` switches it off.
    vjp_only : bool
        Wrap each segment in a ``custom_vjp``.
    """

    segment_steps: tuple[int, ...]
    dt: float
    field: Callable[[Array, Any, Any], Array] | None = None
    softening: float = 0.1
    gravitational_constant: float = 1.0
    vjp_only: bool = False

    def _acceleration(self, theta: PyTree, positions: Array, masses: Array) -> Array:
        """Return the self-gravity plus the field at ``positions``.

        Parameters
        ----------
        theta : PyTree
            The field's parameters.
        positions : Array
            ``[N, 3]`` positions.
        masses : Array
            ``[N]`` masses.

        Returns
        -------
        Array
            ``[N, 3]`` accelerations.
        """
        acc = jnp.zeros_like(positions)
        if self.gravitational_constant != 0.0:
            gravity = DirectSumGravity(self.softening, self.gravitational_constant)
            acc = gravity.accelerations(positions, masses)
        if self.field is not None:
            acc = acc + self.field(positions, 0.0, theta)
        return acc

    def _segment(
        self,
        theta: PyTree,
        positions: Array,
        velocities: Array,
        masses: Array,
        num_steps: int,
    ) -> tuple[Array, Array]:
        """Run ``num_steps`` velocity-Verlet steps.

        Parameters
        ----------
        theta : PyTree
            The field's parameters.
        positions : Array
            ``[N, 3]`` start positions.
        velocities : Array
            ``[N, 3]`` start velocities.
        masses : Array
            ``[N]`` masses.
        num_steps : int
            Steps. Static.

        Returns
        -------
        tuple[Array, Array]
            The end positions and velocities.
        """
        dt = self.dt

        def step(carry, _):
            x, v, a = carry
            x = x + v * dt + 0.5 * a * dt**2
            a_new = self._acceleration(theta, x, masses)
            v = v + 0.5 * (a + a_new) * dt
            return (x, v, a_new), None

        a0 = self._acceleration(theta, positions, masses)
        (x, v, _), _ = jax.lax.scan(
            step, (positions, velocities, a0), None, length=num_steps
        )
        return x, v

    def __call__(
        self,
        theta: PyTree,
        positions: Array,
        velocities: Array,
        masses: Array,
        segment: int,
    ) -> SegmentResult:
        """Run segment ``segment``.

        Parameters
        ----------
        theta : PyTree
            The field's parameters.
        positions : Array
            ``[N, 3]`` start positions.
        velocities : Array
            ``[N, 3]`` start velocities.
        masses : Array
            ``[N]`` masses.
        segment : int
            Which segment. Static.

        Returns
        -------
        SegmentResult
            The end state; the aux is ``None``.
        """
        num_steps = int(self.segment_steps[segment])
        if not self.vjp_only:
            x, v = self._segment(theta, positions, velocities, masses, num_steps)
            return SegmentResult(x, v, None)

        def run(th, x0, v0, m):
            return self._segment(th, x0, v0, m, num_steps)

        guarded = jax.custom_vjp(run)

        def guarded_fwd(th, x0, v0, m):
            out, pullback = jax.vjp(run, th, x0, v0, m)
            return out, pullback

        def guarded_bwd(pullback, cotangent):
            return pullback(cotangent)

        guarded.defvjp(guarded_fwd, guarded_bwd)
        x, v = guarded(theta, positions, velocities, masses)
        return SegmentResult(x, v, None)
