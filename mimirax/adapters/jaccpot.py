"""jaccpot's FMM as a :class:`~mimirax.protocols.ForceModel`. ``mimirax[jaccpot]``.

``FastMultipoleMethod.compute_accelerations(positions, masses)`` already has
the protocol's shape; this adapter only fixes the keyword contract and the
lazy import. **Gradients:** ``compute_accelerations`` builds its tree on the
host on every call and is not traceable by ``jax.grad``. jaccpot's
differentiable seam is its ``differentiable_accelerations`` at frozen topology
(``jaccpot/docs/differentiable_fmm.md``); wiring that in, with the
prepare-once / evaluate-many split it needs, is the adapter's next step and
deliberately not guessed at here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mimirax._typing import PerParticle, Vec3

__all__ = ["JaccpotForceModel"]


@dataclass(frozen=True)
class JaccpotForceModel:
    """Thin wrapper exposing a jaccpot solver through the mimirax protocol.

    Attributes
    ----------
    solver : Any
        A ``jaccpot.FastMultipoleMethod`` (or anything with its
        ``compute_accelerations`` method).
    """

    solver: Any

    def accelerations(
        self,
        positions: Vec3,
        masses: PerParticle,
        *,
        args: object = None,
    ) -> Vec3:
        """Forward to ``solver.compute_accelerations``.

        Parameters
        ----------
        positions : Vec3
            ``(n, 3)`` positions.
        masses : PerParticle
            ``(n,)`` masses.
        args : object
            Ignored; jaccpot's options live on the solver.

        Returns
        -------
        Vec3
            ``(n, 3)`` accelerations.
        """
        del args
        return self.solver.compute_accelerations(positions, masses)

    @classmethod
    def from_preset(cls, preset: str = "balanced", **solver_kwargs: object) -> Any:
        """Build a ``FastMultipoleMethod`` and wrap it.

        Parameters
        ----------
        preset : str
            A jaccpot preset name.
        **solver_kwargs : object
            Forwarded to ``FastMultipoleMethod``.

        Returns
        -------
        Any
            A :class:`JaccpotForceModel`.

        Raises
        ------
        ImportError
            If jaccpot is not installed; the message names the extra.
        """
        try:
            # pyright cannot see jaccpot's `__init__` re-export when the package is
            # installed locally; the import is guarded, so a stale stub is harmless.
            from jaccpot import (
                FastMultipoleMethod,  # pyright: ignore[reportAttributeAccessIssue]
            )
        except ImportError as exc:
            raise ImportError(
                "mimirax.adapters.jaccpot needs jaccpot: install it from GitHub, then "
                "`pip install 'mimirax[jaccpot]'`"
            ) from exc
        return cls(FastMultipoleMethod(preset=preset, **solver_kwargs))
