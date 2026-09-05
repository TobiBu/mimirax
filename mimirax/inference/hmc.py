"""Hamiltonian Monte Carlo and NUTS. Stubs.

Both are :class:`~mimirax.protocols.Sampler` implementations by signature; the
bodies raise until a method is written against them. Whether that is a
hand-rolled leapfrog integrator or a wrapper over an external library is left
open on purpose -- the registry takes either.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from jax import Array

from mimirax._typing import PyTree, Scalar
from mimirax.types import SampleResult

__all__ = ["HMC", "NUTS"]


@dataclass(frozen=True)
class HMC:
    """Hamiltonian Monte Carlo with a fixed leapfrog trajectory. Stub.

    Attributes
    ----------
    step_size : float
        Leapfrog step.
    num_leapfrog_steps : int
        Steps per proposal.
    """

    step_size: float = 0.1
    num_leapfrog_steps: int = 10

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
            The initial state.
        key : Array
            A ``jax.random`` key.
        num_samples : int
            Samples to draw.

        Returns
        -------
        SampleResult
            Never returns.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("HMC is a scaffold stub")


@dataclass(frozen=True)
class NUTS:
    """The No-U-Turn Sampler. Stub.

    Attributes
    ----------
    step_size : float
        Initial leapfrog step.
    max_tree_depth : int
        Doubling limit.
    """

    step_size: float = 0.1
    max_tree_depth: int = 10

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
            The initial state.
        key : Array
            A ``jax.random`` key.
        num_samples : int
            Samples to draw.

        Returns
        -------
        SampleResult
            Never returns.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("NUTS is a scaffold stub")
