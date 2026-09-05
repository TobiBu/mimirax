"""Variational inference. Stub.

A fitted variational family is something one samples from, so the class
satisfies :class:`~mimirax.protocols.Sampler`: ``sample`` fits the family to
``log_density`` and returns draws from it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from jax import Array

from mimirax._typing import PyTree, Scalar
from mimirax.types import SampleResult

__all__ = ["MeanFieldVI"]


@dataclass(frozen=True)
class MeanFieldVI:
    """Mean-field Gaussian variational inference by gradient ascent on the ELBO. Stub.

    Attributes
    ----------
    num_steps : int
        Optimization steps for the variational parameters.
    learning_rate : float
        Step size.
    """

    num_steps: int = 1000
    learning_rate: float = 1.0e-2

    def sample(
        self,
        log_density: Callable[[PyTree], Scalar],
        params: PyTree,
        *,
        key: Array,
        num_samples: int,
    ) -> SampleResult:
        """Not implemented yet.

        Parameters
        ----------
        log_density : Callable[[PyTree], Scalar]
            The unnormalized log target.
        params : PyTree
            Initial mean of the variational family.
        key : Array
            A ``jax.random`` key.
        num_samples : int
            Draws to return from the fitted family.

        Returns
        -------
        SampleResult
            Never returns.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("MeanFieldVI is a scaffold stub")
