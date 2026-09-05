"""Derivative-free optimization. Stub.

Kept in the method registry from day one because the programme's own plan
(the gradient-reliability study ahead of Kessel Run) expects cases where the
gradient through a chaotic rollout is not trustworthy and a derivative-free
method is the honest comparison.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mimirax._typing import PyTree, Scalar
from mimirax.types import FitResult

__all__ = ["NelderMead"]


@dataclass(frozen=True)
class NelderMead:
    """The Nelder-Mead simplex method. Stub.

    Attributes
    ----------
    num_steps : int
        Simplex iterations.
    """

    num_steps: int = 500

    def minimize(
        self,
        objective: Callable[[PyTree], Scalar],
        params: PyTree,
    ) -> FitResult:
        """Not implemented yet.

        Parameters
        ----------
        objective : Callable[[PyTree], Scalar]
            The function to minimize.
        params : PyTree
            The starting point.

        Returns
        -------
        FitResult
            Never returns.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("NelderMead is a scaffold stub")
