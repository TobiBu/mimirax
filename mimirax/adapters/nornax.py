"""A nornax rollout as a :class:`~mimirax.protocols.ForwardModel`. ``mimirax[nornax]``. Stub.

This is the integrator half of the made-to-measure module (D-026): initial
conditions and particle weights in, a time series of states out, through
``nornax.block_kdk_rollout`` with any :class:`~mimirax.protocols.ForceModel`
behind nornax's ``MutualForceModel`` protocol. The shape is fixed here; the
body waits for Jaccpot-Dynamics I.
"""

from __future__ import annotations

from dataclasses import dataclass

from mimirax._typing import PyTree
from mimirax.protocols import ForceModel

__all__ = ["NornaxRollout"]


@dataclass(frozen=True)
class NornaxRollout:
    """Integrate a state forward with nornax and return the trajectory. Stub.

    Attributes
    ----------
    force : ForceModel
        The acceleration provider.
    dt : float
        Base time step.
    num_steps : int
        Base steps to take. Static.
    """

    force: ForceModel
    dt: float
    num_steps: int

    def __call__(self, params: PyTree) -> PyTree:
        """Not implemented yet.

        Parameters
        ----------
        params : PyTree
            Initial conditions and weights.

        Returns
        -------
        PyTree
            Never returns.

        Raises
        ------
        NotImplementedError
            Always, until the M2M module lands.
        """
        raise NotImplementedError(
            "NornaxRollout lands with the made-to-measure module (D-026)"
        )
