"""Convergence judgements from an objective trace."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

__all__ = ["has_converged", "relative_change"]


def relative_change(trace: Array, eps: float = 1.0e-300) -> Array:
    """Return ``|x[k] - x[k-1]| / (|x[k-1]| + eps)`` along a trace.

    Parameters
    ----------
    trace : Array
        ``(k,)`` objective values.
    eps : float
        Guard against division by zero.

    Returns
    -------
    Array
        ``(k - 1,)`` relative changes.
    """
    return jnp.abs(jnp.diff(trace)) / (jnp.abs(trace[:-1]) + eps)


def has_converged(trace: Array, *, rtol: float = 1.0e-6, window: int = 5) -> bool:
    """Whether the last ``window`` relative changes are all below ``rtol``.

    A host-side judgement (returns a Python ``bool``), meant for after a fit,
    not inside a traced loop.

    Parameters
    ----------
    trace : Array
        ``(k,)`` objective values, ``k > window``.
    rtol : float
        Relative-change threshold.
    window : int
        How many trailing steps must all be below it.

    Returns
    -------
    bool
        ``True`` when converged by this criterion.
    """
    changes = relative_change(trace)
    if changes.shape[0] < window:
        return False
    return bool(jnp.all(changes[-window:] < rtol))
