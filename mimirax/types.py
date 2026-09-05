"""Result containers shared by the inference protocols and their implementations.

Kept in their own module (as ``yggdrax/types.py`` is) so that
:mod:`mimirax.protocols` can reference them without importing any
implementation, and so that an adapter satisfying a protocol structurally can
return them without importing :mod:`mimirax.inference`.
"""

from __future__ import annotations

from typing import NamedTuple

from jax import Array

from mimirax._typing import PyTree

__all__ = ["FitResult", "SampleResult"]


class FitResult(NamedTuple):
    """What an optimizer returns.

    Attributes
    ----------
    params : PyTree
        The final parameters, in the same structure as the initial ones.
    objective_trace : Array
        ``(k,)`` objective value at every step, so convergence can be judged
        from the result alone.
    """

    params: PyTree
    objective_trace: Array


class SampleResult(NamedTuple):
    """What a sampler returns.

    Attributes
    ----------
    samples : PyTree
        Parameter samples with a leading ``k`` axis on every leaf.
    log_density : Array
        ``(k,)`` log target density at each sample.
    """

    samples: PyTree
    log_density: Array
