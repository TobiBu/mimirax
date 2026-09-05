"""An ODISSEO integration as a :class:`~mimirax.protocols.ForwardModel`. ``mimirax[odisseo]``. Stub.

ODISSEO's ``odisseo.integrate(...)`` and its differentiable FMM lane
(``odisseo/differentiable.py``) are the obvious body. Left as a stub because
whether ODISSEO stays on the routing path between jaccpot and nornax is an open
decision (O-4 in the EDDA programme), and an adapter written before that is
settled would be written twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mimirax._typing import PyTree

__all__ = ["OdisseoForwardModel"]


@dataclass(frozen=True)
class OdisseoForwardModel:
    """Run an ODISSEO simulation from parameters. Stub.

    Attributes
    ----------
    config : Any
        An ``odisseo.option_classes.SimulationConfig``.
    params : Any
        An ``odisseo.option_classes.SimulationParams`` template whose fields
        the free parameters overwrite.
    """

    config: Any
    params: Any

    def __call__(self, params: PyTree) -> PyTree:
        """Not implemented yet.

        Parameters
        ----------
        params : PyTree
            The free parameters.

        Returns
        -------
        PyTree
            Never returns.

        Raises
        ------
        NotImplementedError
            Always, until O-4 is settled.
        """
        raise NotImplementedError("OdisseoForwardModel waits on decision O-4")
