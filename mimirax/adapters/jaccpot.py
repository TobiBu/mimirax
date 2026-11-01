"""jaccpot's FMM as a :class:`~mimirax.protocols.ForceModel`. ``mimirax[jaccpot]``.

``FastMultipoleMethod.compute_accelerations(positions, masses)`` already has
the protocol's shape; this adapter only fixes the keyword contract and the
lazy import. **Gradients:** ``compute_accelerations`` builds its tree on the
host on every call and is not traceable by ``jax.grad``. jaccpot's
differentiable seam is its ``differentiable_accelerations`` at frozen topology
(``jaccpot/docs/differentiable_fmm.md``); wiring that in, with the
prepare-once / evaluate-many split it needs, is the adapter's next step and
deliberately not guessed at here.

MULTIPLE SHOOTING. :class:`DirectSumSegmentEngine` is a
:class:`~mimirax.shooting.SegmentEngine` on jaccpot's differentiable rollout
(``jaccpot.runtime.grad_rollout.differentiable_rollout``, S9) over its direct
sum: production velocity Verlet, a hand-written checkpointed reverse
(``checkpoint="step"``: a ``custom_vjp``, so forward-mode autodiff through it
fails loudly), the external field's parameters ``theta`` as the rollout's
``external_args``. It is the oracle the fused-lane engine is checked against.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jax import Array

from mimirax._typing import PerParticle, PyTree, Vec3
from mimirax.shooting import SegmentResult

__all__ = ["DirectSumSegmentEngine", "JaccpotForceModel", "rows_field"]


def _grad_rollout() -> tuple[Any, Any]:
    """Import jaccpot's differentiable rollout and direct-sum step, lazily.

    Returns
    -------
    tuple[Any, Any]
        ``(differentiable_rollout, direct_sum_step_fn)``.

    Raises
    ------
    ImportError
        If jaccpot (with ``jaccpot.runtime.grad_rollout``, S9) is not installed;
        the message names the extra.
    """
    try:
        from jaccpot.runtime.grad_rollout import (  # pyright: ignore[reportMissingImports]
            differentiable_rollout,
            direct_sum_step_fn,
        )
    except ImportError as exc:
        raise ImportError(
            "DirectSumSegmentEngine needs jaccpot with jaccpot.runtime.grad_rollout "
            "(S9): install it from GitHub, then `pip install 'mimirax[jaccpot]'`"
        ) from exc
    return differentiable_rollout, direct_sum_step_fn


def rows_field(
    field: Callable[[Array, Any, Any], Array], time: float = 0.0
) -> Callable[[Array, Any], Array]:
    """Adapt a nornax field ``f(x, t, args)`` to jaccpot's ``f(state, args)``.

    jaccpot's rollouts hand the field ``[N, 2, 3]`` rows of position and
    velocity; the adapter reads the position row only (``state[:, 0]``), which
    ``differentiable_rollout`` recognises: it then passes zero velocity rows and
    stores nothing for them. jaccpot's field has no time argument, so the field
    is evaluated at the fixed ``time``: a time-dependent field is out of reach of
    this adapter.

    Parameters
    ----------
    field : Callable[[Array, Any, Any], Array]
        ``field(positions [N, 3], t, args) -> [N, 3]`` acceleration.
    time : float
        The time the field is evaluated at. Static.

    Returns
    -------
    Callable[[Array, Any], Array]
        ``rows(state [N, 2, 3], args) -> [N, 3]``.
    """

    def rows(state: Array, args: Any) -> Array:
        return field(state[:, 0], time, args)

    return rows


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


@dataclass(frozen=True)
class DirectSumSegmentEngine:
    """A :class:`~mimirax.shooting.SegmentEngine` on jaccpot's direct-sum rollout.

    Segment ``k`` is ``differentiable_rollout(direct_sum_step_fn(G=G,
    softening=softening), x, v, masses, dt, segment_steps[k],
    external_fn=rows_field(field), external_args=theta, checkpoint=...)``: the
    production velocity Verlet over the softened ``O(N^2)`` direct sum plus the
    external field, with ``theta`` reaching the field only as the rollout's
    ``external_args`` (jaccpot's ``custom_vjp`` refuses closures over
    differentiated values). Each segment starts by computing its own start force
    (``acceleration=None``), so a node is a position and a velocity, nothing
    else. Pure: no ``prepare``, no ``check`` (the direct sum has no capacity).

    ``G`` and ``softening`` are Python constants of the program; ``dt`` is a
    Python float (a literal, not differentiated).

    Attributes
    ----------
    segment_steps : tuple[int, ...]
        Steps per segment; ``len`` is the number of segments.
    dt : float
        The timestep.
    field : Callable[[Array, Any, Any], Array] | None
        The external field in nornax's ``f(x, t, args)`` convention, evaluated
        at ``t = 0`` (:func:`rows_field`), or ``None`` for self-gravity only
        (``theta`` is then unused).
    G : float
        Gravitational constant of the self-gravity.
    softening : float
        Plummer-equivalent softening length of the self-gravity.
    softening_kernel : str | None
        jaccpot's pair kernel; ``None`` takes its default.
    checkpoint : str | None
        ``differentiable_rollout``'s checkpoint mode: ``"step"`` (default),
        ``"nested"``, ``"offload"``, or ``None`` (a plain scan, which plain
        autodiff -- forward mode included -- goes through).
    """

    segment_steps: tuple[int, ...]
    dt: float
    field: Callable[[Array, Any, Any], Array] | None = None
    G: float = 1.0
    softening: float = 0.0
    softening_kernel: str | None = None
    checkpoint: str | None = "step"

    def __post_init__(self) -> None:
        """Validate the segment lengths.

        Raises
        ------
        ValueError
            If ``segment_steps`` is empty or holds a non-positive count.
        """
        if not self.segment_steps or any(int(s) < 1 for s in self.segment_steps):
            raise ValueError(
                f"segment_steps must be positive step counts, got {self.segment_steps}"
            )

    @property
    def num_segments(self) -> int:
        """Return the number of segments.

        Returns
        -------
        int
            ``len(segment_steps)``.
        """
        return len(self.segment_steps)

    def rollout(
        self,
        theta: PyTree,
        positions: Array,
        velocities: Array,
        masses: Array,
        num_steps: int,
    ) -> SegmentResult:
        """Run ``num_steps`` steps from a state (any horizon, e.g. the single shot).

        Parameters
        ----------
        theta : PyTree
            The field's parameters.
        positions : Array
            ``[N, 3]`` start positions.
        velocities : Array
            ``[N, 3]`` start velocities.
        masses : Array
            ``[N]`` masses.
        num_steps : int
            Steps. Static.

        Returns
        -------
        SegmentResult
            The end state; the aux is the rollout's (``None`` for the direct
            sum).
        """
        differentiable_rollout, direct_sum_step_fn = _grad_rollout()
        step_fn = direct_sum_step_fn(
            G=self.G, softening=self.softening, softening_kernel=self.softening_kernel
        )
        external_fn = None if self.field is None else rows_field(self.field)
        result = differentiable_rollout(
            step_fn,
            positions,
            velocities,
            masses,
            self.dt,
            int(num_steps),
            external_fn=external_fn,
            external_args=None if external_fn is None else theta,
            checkpoint=self.checkpoint,
        )
        return SegmentResult(result.positions, result.velocities, result.aux)

    def __call__(
        self,
        theta: PyTree,
        positions: Array,
        velocities: Array,
        masses: Array,
        segment: int,
    ) -> SegmentResult:
        """Run segment ``segment``.

        Parameters
        ----------
        theta : PyTree
            The field's parameters.
        positions : Array
            ``[N, 3]`` start positions.
        velocities : Array
            ``[N, 3]`` start velocities.
        masses : Array
            ``[N]`` masses.
        segment : int
            Which segment. Static.

        Returns
        -------
        SegmentResult
            The state after ``segment_steps[segment]`` steps.
        """
        return self.rollout(
            theta, positions, velocities, masses, self.segment_steps[segment]
        )
