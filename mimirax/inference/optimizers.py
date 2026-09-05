"""Gradient-based optimization through optax.

The one inference method the scaffold implements rather than stubs, because
every other method's tests need a working baseline to compare against and
because it is the method Paper I section 7's fit uses.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import optax

from mimirax._typing import PyTree, Scalar
from mimirax.types import FitResult

__all__ = ["OptaxOptimizer", "adam"]


@dataclass(frozen=True)
class OptaxOptimizer:
    """Minimize an objective with any ``optax.GradientTransformation``.

    The loop is a ``lax.scan`` over a static number of steps, so the whole fit
    traces once and runs on device. The trace records the objective *before*
    each update, so ``objective_trace[0]`` is the value at the starting point.

    Attributes
    ----------
    optimizer : optax.GradientTransformation
        The update rule.
    num_steps : int
        How many updates to take. Static.
    """

    optimizer: optax.GradientTransformation
    num_steps: int = 100

    def minimize(
        self,
        objective: Callable[[PyTree], Scalar],
        params: PyTree,
    ) -> FitResult:
        """Run ``num_steps`` updates from ``params``.

        Parameters
        ----------
        objective : Callable[[PyTree], Scalar]
            A differentiable scalar function of the parameters.
        params : PyTree
            The starting point.

        Returns
        -------
        FitResult
            The final parameters and the ``(num_steps,)`` objective trace.
        """
        value_and_grad = jax.value_and_grad(objective)
        opt_state = self.optimizer.init(params)

        def _step(carry, _):
            current, state = carry
            value, grads = value_and_grad(current)
            updates, state = self.optimizer.update(grads, state, current)
            current = optax.apply_updates(current, updates)
            return (current, state), value

        (final, _), trace = jax.lax.scan(
            _step, (params, opt_state), None, length=self.num_steps
        )
        return FitResult(params=final, objective_trace=trace)


def adam(learning_rate: float = 1.0e-2, num_steps: int = 200) -> OptaxOptimizer:
    """Build an Adam optimizer with the given settings.

    Parameters
    ----------
    learning_rate : float
        Adam's step size.
    num_steps : int
        Number of updates.

    Returns
    -------
    OptaxOptimizer
        Ready to ``minimize``.
    """
    return OptaxOptimizer(optax.adam(learning_rate), num_steps=num_steps)
