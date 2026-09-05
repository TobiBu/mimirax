"""Parameter pytrees, reparameterizations, gauge fixing and constraints.

The free parameters of an inference problem are whatever pytree the caller
chose -- a bare ``(n, 3)`` array of positions, a dict of potential parameters,
a nested structure. Nothing here assumes a structure; every function works
through ``jax.tree_util`` or through a flat vector obtained with :func:`ravel`.

Three concerns live here because they are about the *parameter space*, not
about any model:

* **Reparameterizations** -- bijections between an unconstrained space, where
  optimizers and samplers move freely, and the constrained space a forward
  model expects (positive masses, bounded angles). Each is a
  :class:`~mimirax.protocols.Reparameterization` and is registered in
  :data:`REPARAMETERIZATIONS`; ODISSEO's ``Plummer_sphere_reparam`` is the
  ecosystem's existing instance of the pattern.
* **Gauge fixing** -- removing directions the data cannot see. A
  self-gravitating system's centre of mass and net momentum are the standing
  examples: a fit over particle positions that leaves them free wanders along
  a flat direction of the objective.
* **Constraints** -- equalities expressed as a *defect* that is zero when the
  constraint holds (nornax's multiple-shooting ``shooting_defect`` is one), and
  the quadratic penalty that turns a defect into an objective term.

Everything is differentiable in its array arguments; nothing here has a
static-argument requirement under ``jit`` except the ``transforms`` mapping,
whose keys select code paths.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

from mimirax._registry import Registry
from mimirax._typing import PerParticle, PyTree, Scalar, Vec3
from mimirax.protocols import Constraint, Reparameterization

__all__ = [
    "REPARAMETERIZATIONS",
    "AffineTransform",
    "FunctionConstraint",
    "IdentityTransform",
    "LogTransform",
    "SoftplusTransform",
    "TotalMassConstraint",
    "centre_of_mass",
    "constrain",
    "constraint_violation",
    "log_abs_det_jacobian",
    "net_momentum",
    "num_parameters",
    "quadratic_penalty",
    "ravel",
    "remove_centre_of_mass",
    "remove_net_momentum",
    "unconstrain",
]


# --- flat views -----------------------------------------------------------------


def ravel(params: PyTree) -> tuple[Array, Callable[[Array], PyTree]]:
    """Flatten a parameter pytree into one vector, with its inverse.

    Parameters
    ----------
    params : PyTree
        Any pytree of arrays.

    Returns
    -------
    tuple[Array, Callable[[Array], PyTree]]
        The concatenated leaves as a 1-d array, and a function that rebuilds
        the original structure from a vector of the same length.
    """
    flat, unravel = ravel_pytree(params)
    return flat, unravel


def num_parameters(params: PyTree) -> int:
    """Count the scalar degrees of freedom in a parameter pytree.

    Parameters
    ----------
    params : PyTree
        Any pytree of arrays.

    Returns
    -------
    int
        The total number of array elements across all leaves.
    """
    return int(sum(jnp.size(leaf) for leaf in jax.tree_util.tree_leaves(params)))


# --- reparameterizations -----------------------------------------------------------

REPARAMETERIZATIONS: Registry[type] = Registry("reparameterization")


@dataclass(frozen=True)
class IdentityTransform:
    """The trivial reparameterization: constrained equals unconstrained."""

    def forward(self, unconstrained: Array) -> Array:
        """Return ``unconstrained`` unchanged.

        Parameters
        ----------
        unconstrained : Array
            Values in the unconstrained space.

        Returns
        -------
        Array
            The same values.
        """
        return jnp.asarray(unconstrained)

    def inverse(self, constrained: Array) -> Array:
        """Return ``constrained`` unchanged.

        Parameters
        ----------
        constrained : Array
            Values in the constrained space.

        Returns
        -------
        Array
            The same values.
        """
        return jnp.asarray(constrained)

    def log_abs_det_jacobian(self, unconstrained: Array) -> Scalar:
        """Return zero: the identity has unit Jacobian.

        Parameters
        ----------
        unconstrained : Array
            Ignored except for its dtype.

        Returns
        -------
        Scalar
            ``0``.
        """
        return jnp.zeros((), dtype=jnp.asarray(unconstrained).dtype)


@dataclass(frozen=True)
class LogTransform:
    """Positive constrained values: ``constrained = exp(unconstrained)``.

    The natural choice for masses, scale radii and noise levels.
    """

    def forward(self, unconstrained: Array) -> Array:
        """Exponentiate.

        Parameters
        ----------
        unconstrained : Array
            Real values.

        Returns
        -------
        Array
            Strictly positive values, same shape.
        """
        return jnp.exp(unconstrained)

    def inverse(self, constrained: Array) -> Array:
        """Take the logarithm.

        Parameters
        ----------
        constrained : Array
            Strictly positive values.

        Returns
        -------
        Array
            Real values, same shape.
        """
        return jnp.log(constrained)

    def log_abs_det_jacobian(self, unconstrained: Array) -> Scalar:
        """Return ``sum(unconstrained)``, since ``d exp(u) / du = exp(u)``.

        Parameters
        ----------
        unconstrained : Array
            Where to evaluate.

        Returns
        -------
        Scalar
            The summed log-Jacobian.
        """
        return jnp.sum(unconstrained)


@dataclass(frozen=True)
class SoftplusTransform:
    """Positive constrained values with a linear tail: ``softplus(u)``.

    Unlike :class:`LogTransform` this does not blow up for large ``u``, which
    makes it the safer choice under aggressive optimizer steps.
    """

    def forward(self, unconstrained: Array) -> Array:
        """Apply ``log(1 + exp(u))``.

        Parameters
        ----------
        unconstrained : Array
            Real values.

        Returns
        -------
        Array
            Strictly positive values, same shape.
        """
        return jax.nn.softplus(unconstrained)

    def inverse(self, constrained: Array) -> Array:
        """Apply ``log(exp(c) - 1)``.

        Parameters
        ----------
        constrained : Array
            Strictly positive values.

        Returns
        -------
        Array
            Real values, same shape.
        """
        return jnp.log(-jnp.expm1(-constrained)) + constrained

    def log_abs_det_jacobian(self, unconstrained: Array) -> Scalar:
        """Return ``sum(log sigmoid(u))``, the log-derivative of softplus.

        Parameters
        ----------
        unconstrained : Array
            Where to evaluate.

        Returns
        -------
        Scalar
            The summed log-Jacobian.
        """
        return jnp.sum(jax.nn.log_sigmoid(unconstrained))


@dataclass(frozen=True)
class AffineTransform:
    """A fixed rescaling and shift: ``constrained = scale * u + shift``.

    Used to put parameters of very different magnitude on one footing before
    an optimizer sees them.

    Attributes
    ----------
    scale : float
        Multiplicative factor; must be non-zero.
    shift : float
        Additive offset.
    """

    scale: float = 1.0
    shift: float = 0.0

    def forward(self, unconstrained: Array) -> Array:
        """Apply the affine map.

        Parameters
        ----------
        unconstrained : Array
            Real values.

        Returns
        -------
        Array
            ``scale * unconstrained + shift``.
        """
        return self.scale * unconstrained + self.shift

    def inverse(self, constrained: Array) -> Array:
        """Undo the affine map.

        Parameters
        ----------
        constrained : Array
            Real values.

        Returns
        -------
        Array
            ``(constrained - shift) / scale``.
        """
        return (constrained - self.shift) / self.scale

    def log_abs_det_jacobian(self, unconstrained: Array) -> Scalar:
        """Return ``size * log|scale|``.

        Parameters
        ----------
        unconstrained : Array
            Only its size is used.

        Returns
        -------
        Scalar
            The summed log-Jacobian.
        """
        size = jnp.asarray(unconstrained).size
        return jnp.asarray(size * jnp.log(jnp.abs(self.scale)))


REPARAMETERIZATIONS.register("identity", IdentityTransform)
REPARAMETERIZATIONS.register("log", LogTransform)
REPARAMETERIZATIONS.register("softplus", SoftplusTransform)
REPARAMETERIZATIONS.register("affine", AffineTransform)


def constrain(
    unconstrained: Mapping[str, Array],
    transforms: Mapping[str, Reparameterization],
) -> dict[str, Array]:
    """Map a dict of unconstrained parameters to the constrained space.

    Parameters
    ----------
    unconstrained : Mapping[str, Array]
        Parameters keyed by name.
    transforms : Mapping[str, Reparameterization]
        Which reparameterization applies to which key. Keys absent from
        ``transforms`` pass through unchanged.

    Returns
    -------
    dict[str, Array]
        The constrained parameters, same keys.
    """
    return {
        key: transforms[key].forward(value) if key in transforms else value
        for key, value in unconstrained.items()
    }


def unconstrain(
    constrained: Mapping[str, Array],
    transforms: Mapping[str, Reparameterization],
) -> dict[str, Array]:
    """Map a dict of constrained parameters back to the unconstrained space.

    Parameters
    ----------
    constrained : Mapping[str, Array]
        Parameters keyed by name.
    transforms : Mapping[str, Reparameterization]
        Which reparameterization applies to which key. Keys absent from
        ``transforms`` pass through unchanged.

    Returns
    -------
    dict[str, Array]
        The unconstrained parameters, same keys.
    """
    return {
        key: transforms[key].inverse(value) if key in transforms else value
        for key, value in constrained.items()
    }


def log_abs_det_jacobian(
    unconstrained: Mapping[str, Array],
    transforms: Mapping[str, Reparameterization],
) -> Scalar:
    """Sum the log-Jacobians of every transform applied by :func:`constrain`.

    Parameters
    ----------
    unconstrained : Mapping[str, Array]
        Parameters keyed by name, in the unconstrained space.
    transforms : Mapping[str, Reparameterization]
        Which reparameterization applies to which key.

    Returns
    -------
    Scalar
        The total, which a density over the unconstrained space adds to the
        constrained-space log-density.
    """
    total = jnp.zeros(())
    for key, value in unconstrained.items():
        if key in transforms:
            total = total + transforms[key].log_abs_det_jacobian(value)
    return total


# --- gauge fixing ------------------------------------------------------------------


def centre_of_mass(positions: Vec3, masses: PerParticle) -> Array:
    """Return the mass-weighted mean position.

    Parameters
    ----------
    positions : Vec3
        ``(n, 3)`` positions.
    masses : PerParticle
        ``(n,)`` masses.

    Returns
    -------
    Array
        ``(3,)`` centre of mass.
    """
    return jnp.sum(masses[:, None] * positions, axis=0) / jnp.sum(masses)


def remove_centre_of_mass(positions: Vec3, masses: PerParticle) -> Vec3:
    """Translate ``positions`` so their centre of mass is at the origin.

    Parameters
    ----------
    positions : Vec3
        ``(n, 3)`` positions.
    masses : PerParticle
        ``(n,)`` masses.

    Returns
    -------
    Vec3
        The recentred positions.
    """
    return positions - centre_of_mass(positions, masses)[None, :]


def net_momentum(velocities: Vec3, masses: PerParticle) -> Array:
    """Return the total linear momentum.

    Parameters
    ----------
    velocities : Vec3
        ``(n, 3)`` velocities.
    masses : PerParticle
        ``(n,)`` masses.

    Returns
    -------
    Array
        ``(3,)`` momentum.
    """
    return jnp.sum(masses[:, None] * velocities, axis=0)


def remove_net_momentum(velocities: Vec3, masses: PerParticle) -> Vec3:
    """Boost ``velocities`` so the total linear momentum vanishes.

    Parameters
    ----------
    velocities : Vec3
        ``(n, 3)`` velocities.
    masses : PerParticle
        ``(n,)`` masses.

    Returns
    -------
    Vec3
        The velocities in the zero-momentum frame.
    """
    return velocities - (net_momentum(velocities, masses) / jnp.sum(masses))[None, :]


# --- constraints --------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionConstraint:
    """A :class:`~mimirax.protocols.Constraint` wrapping a plain defect function.

    Attributes
    ----------
    defect_fn : Callable[[PyTree], Array]
        Maps parameters to a defect array that is zero when the constraint
        holds.
    """

    defect_fn: Callable[[PyTree], Array]

    def defect(self, params: PyTree) -> Array:
        """Evaluate the wrapped defect function.

        Parameters
        ----------
        params : PyTree
            The free parameters.

        Returns
        -------
        Array
            The defect.
        """
        return self.defect_fn(params)


@dataclass(frozen=True)
class TotalMassConstraint:
    """Fix the sum of a mass vector -- the made-to-measure normalization.

    Attributes
    ----------
    total : float
        The required total mass.
    """

    total: float

    def defect(self, params: PyTree) -> Array:
        """Return ``sum(params) - total`` for a mass-vector parameter.

        Parameters
        ----------
        params : PyTree
            The masses, as an array or a pytree whose leaves are summed.

        Returns
        -------
        Array
            A scalar defect.
        """
        leaves = jax.tree_util.tree_leaves(params)
        total = sum((jnp.sum(leaf) for leaf in leaves), jnp.zeros(()))
        return total - self.total


def quadratic_penalty(
    constraint: Constraint, params: PyTree, weight: float = 1.0
) -> Scalar:
    """Turn a constraint defect into an objective term, ``weight * sum(defect^2)``.

    Parameters
    ----------
    constraint : Constraint
        The constraint.
    params : PyTree
        Where to evaluate it.
    weight : float
        The penalty strength.

    Returns
    -------
    Scalar
        The penalty; add it to a negative log-posterior.
    """
    defect = constraint.defect(params)
    return weight * jnp.sum(defect * defect)


def constraint_violation(constraint: Constraint, params: PyTree) -> Scalar:
    """Return the largest absolute defect component, a convergence diagnostic.

    Parameters
    ----------
    constraint : Constraint
        The constraint.
    params : PyTree
        Where to evaluate it.

    Returns
    -------
    Scalar
        ``max |defect|``.
    """
    return jnp.max(jnp.abs(constraint.defect(params)))
