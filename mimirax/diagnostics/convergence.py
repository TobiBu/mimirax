"""Convergence judgements from an objective trace."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

__all__ = ["has_converged", "relative_change"]


def relative_change(trace: Array, eps: float | None = None) -> Array:
    """Return ``|x[k] - x[k-1]| / (|x[k-1]| + eps)`` along a trace.

    Parameters
    ----------
    trace : Array
        ``(k,)`` objective values.
    eps : float | None
        Guard against division by zero. ``None`` takes the smallest normal of
        the trace's dtype. A fixed constant cannot serve: ``1e-300`` is exactly
        zero in float32, so on a trace that reached zero the guard underflows
        and leaves ``0 / 0 = nan``.

    Returns
    -------
    Array
        ``(k - 1,)`` relative changes.
    """
    if eps is None:
        eps = float(jnp.finfo(trace.dtype).tiny)
    return jnp.abs(jnp.diff(trace)) / (jnp.abs(trace[:-1]) + eps)


def has_converged(
    trace: Array,
    *,
    rtol: float = 1.0e-6,
    window: int = 5,
    atol: float | None = None,
) -> bool:
    """Whether the last ``window`` steps are all within ``atol + rtol * |x[k-1]|``.

    A host-side judgement (returns a Python ``bool``), meant for after a fit,
    not inside a traced loop.

    The absolute term is what lets a fit that reached the floating-point floor
    count as converged. A purely relative test cannot: once the objective is
    round-off, successive values differ by O(1) in relative terms however still
    they are in absolute ones, so the harder a fit converges the worse its
    verdict. Whether such a fit lands on exactly zero or on 1e-31 is a property
    of the library version, not of the fit.

    Parameters
    ----------
    trace : Array
        ``(k,)`` objective values, ``k > window``.
    rtol : float
        Relative-change threshold, against ``|x[k-1]|``.
    window : int
        How many trailing steps must all be within the threshold.
    atol : float | None
        Absolute-change threshold. ``None`` takes ``eps * max |trace|``, the
        resolution of the trace's own scale: a step below that is not one the
        dtype can represent as progress. It is deliberately tied to the trace
        rather than fixed, so the judgement does not depend on the units the
        objective happens to be in.

    Returns
    -------
    bool
        ``True`` when converged by this criterion.
    """
    changes = jnp.abs(jnp.diff(trace))
    if changes.shape[0] < window:
        return False
    if atol is None:
        atol = float(jnp.finfo(trace.dtype).eps * jnp.max(jnp.abs(trace)))
    previous = jnp.abs(trace[:-1])
    return bool(jnp.all(changes[-window:] <= atol + rtol * previous[-window:]))
