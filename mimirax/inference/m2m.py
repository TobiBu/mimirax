"""Made-to-measure: particle weights adjusted along an orbit integration. Stub.

WHERE THIS LIVES, AND WHY. The EDDA programme's decision D-026 (2026-09-06)
puts the made-to-measure module in mimirax rather than in the integrator
package: its three parts are an orbit integration (nornax's), a force (any
:class:`~mimirax.protocols.ForceModel`), and an observable-residual-driven
weight update with an entropy prior -- and the last is inference. The
integrator enters through :mod:`mimirax.adapters.nornax`, behind the
``mimirax[nornax]`` extra; nothing here imports nornax.

WHAT IT WILL BE. Syer & Tremaine's force-of-change on the weights,
``dw_i/dt = -eps w_i [ sum_j Delta_j(t) K_ij / sigma_j  -  mu d S/d w_i ]``,
with ``Delta_j`` the residual of observable ``j`` against its time-smoothed
model value, ``K_ij`` particle ``i``'s contribution to observable ``j``, and
``S`` the entropy prior of :class:`~mimirax.priors.EntropyPrior`. Driven either
as an ODE alongside the orbits (classic M2M) or as gradient descent on the
time-averaged objective (the differentiable variant this package exists for).

Nothing in this file is implemented; it fixes the name and the signature so
the registry, the docs and the paper plan can refer to it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mimirax._typing import PyTree, Scalar
from mimirax.types import FitResult

__all__ = ["MadeToMeasure"]


@dataclass(frozen=True)
class MadeToMeasure:
    """The made-to-measure weight-adjustment method. Stub.

    Attributes
    ----------
    entropy_weight : float
        ``mu``, the strength of the entropy prior.
    num_steps : int
        Weight-update steps.
    """

    entropy_weight: float = 1.0
    num_steps: int = 100

    def minimize(
        self,
        objective: Callable[[PyTree], Scalar],
        params: PyTree,
    ) -> FitResult:
        """Not implemented yet.

        Parameters
        ----------
        objective : Callable[[PyTree], Scalar]
            The time-averaged negative log posterior over the weights.
        params : PyTree
            The initial particle weights.

        Returns
        -------
        FitResult
            Never returns.

        Raises
        ------
        NotImplementedError
            Always, until Jaccpot-Dynamics I's implementation step.
        """
        raise NotImplementedError(
            "MadeToMeasure lands with Jaccpot-Dynamics I (EDDA programme D-026)"
        )
