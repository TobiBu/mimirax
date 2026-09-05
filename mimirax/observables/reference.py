"""Reference observation operators: the trivial ones every pipeline test needs.

Real observables -- kinematic maps, binned densities, stream-track
coordinates -- are the work of the science papers and are added to
:data:`mimirax.observables.OBSERVABLES` as they are written. These two exist so
that every core numerical path has an observable to run through today.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from mimirax._typing import PyTree

__all__ = ["IdentityObservable", "ProjectedPositions"]


@dataclass(frozen=True)
class IdentityObservable:
    """The state itself, as one array: the data *are* the model output."""

    def __call__(self, state: PyTree) -> Array:
        """Return ``state`` as an array.

        Parameters
        ----------
        state : PyTree
            A single array (or array-like) state.

        Returns
        -------
        Array
            ``jnp.asarray(state)``.
        """
        return jnp.asarray(state)


@dataclass(frozen=True)
class ProjectedPositions:
    """Positions projected onto a subset of axes and flattened.

    The simplest mock observable: an ``(n, 3)`` state seen in projection, as an
    imaging survey sees a stellar system. Differentiable in ``state``.

    Attributes
    ----------
    axes : tuple[int, ...]
        Which coordinate axes survive the projection. Static.
    """

    axes: tuple[int, ...] = (0, 1)

    def __call__(self, state: PyTree) -> Array:
        """Project and flatten.

        Parameters
        ----------
        state : PyTree
            ``(n, 3)`` positions.

        Returns
        -------
        Array
            ``(n * len(axes),)`` projected coordinates, particle-major.
        """
        positions = jnp.asarray(state)
        return jnp.take(positions, jnp.asarray(self.axes), axis=1).reshape(-1)
