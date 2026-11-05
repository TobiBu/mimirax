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
fails loudly in the default ``ad_mode``), the external field's parameters
``theta`` as the rollout's ``external_args``. It is the oracle the fused-lane
engine is checked against.

FORWARD MODE. Both engines take an ``ad_mode`` (jaccpot's, main 1bb30620 on):
``"forward"`` traces each segment for ``jax.jvp`` / ``jax.jacfwd`` /
``jax.linearize`` -- a plain scan (``checkpoint`` ``None`` or ``"step"``; the
reverse-pass schedules ``"nested"`` / ``"offload"`` are refused) over the fused
lane's ``custom_jvp`` seams, the primal the production forward bit for bit, first
order only (no ``jax.grad`` of it, no ``jacfwd(jacfwd)``). ``None`` (the default)
follows an enclosing ``jaccpot.runtime.grad_options.ad_mode_override``, then
``JACCPOT_GRAD_AD_MODE``, then reverse: one engine (and its fused plans) serves
the reverse objective and, traced inside ``ad_mode_override("forward")``, the
forward-mode one (``mimirax.inference.parallel``'s ``"jacfwd"`` programs).

:class:`FusedSegmentEngine` is the :class:`~mimirax.shooting.SegmentEngine` on
jaccpot's fused PRODUCTION lane (S10): each segment a gradient run of jaccpot's
``ProductionRunner`` -- ``runner.gradient_plan(...)`` at the segment's node, then
``plan.rollout(...)`` -- differentiable in the field's parameters, the segment's
start state and the masses, rebuilding its tree every step on stopped inputs, its
forward the runner's own trajectory from the node.

THE TWO HALVES OF A FUSED SEGMENT. The fused lane's capacities (walk caps, leaf
capacity, level widths, the far list's length) are STATIC shapes of the compiled
program, settled by a forward pre-pass on the host -- the runner's own run with its
retries and re-plans (jaccpot ``docs/production.md``, "Gradient mode"). So the
engine has both optional capabilities of the seam:

1. :meth:`FusedSegmentEngine.prepare` (host): one pre-pass per segment that has no
   plan yet or whose last verdict failed, at that segment's current node; a no-op
   otherwise (the drivers call it before every step);
2. :meth:`FusedSegmentEngine.__call__` (traced): the segment's gradient run on its
   cached plan, from any start state and parameters;
3. :meth:`FusedSegmentEngine.check` (host): the gradient verdict of every segment
   of the evaluation. ``False`` marks the failed segments for the next
   :meth:`~FusedSegmentEngine.prepare`, and the driver redoes the step.

A plan made at one iterate is reused at the next while its verdict passes, so the
traced objective compiles ONCE across optimizer iterations. A re-plan changes the
program's statics, so the driver must trace its objective again after a
:meth:`~FusedSegmentEngine.prepare` that re-planned (a NEW function object:
``jax.jit`` of the same function reuses that function's trace; the
augmented-Lagrangian driver builds its step again).
:attr:`~FusedSegmentEngine.generation` counts the plans made, and
:meth:`~FusedSegmentEngine.check` refuses the aux of a program traced on a
replaced plan.

ONE PROGRAM PER SEGMENT, ONE DEVICE EACH (S10 session 2). A segment's program
(``mimirax.inference.parallel``) is keyed by
:meth:`~FusedSegmentEngine.segment_key` -- the segment and its plan's id -- so a
re-plan retraces that segment's program alone. A plan's arrays (its traced
``dt`` / force parameters and every runner segment's f_b seed;
:meth:`~FusedSegmentEngine.segment_arrays`) are placed on the segment's device
once per plan and passed to the program as ARGUMENTS (``__call__(...,
arrays=)``): without them the seeds, ``[N]`` each, are constants embedded in the
program. Everything else a ``jaccpot.production_grad.GradientPlan`` holds is host
data or Python statics (the providers hold no device buffer, the segment plans are
ints and dicts, the start state and seeds are numpy), so the pre-pass may run on
any device -- it runs on the default one -- and the program compiles for the
device its arguments are on. The pre-pass's ``forward`` (the runner's handle: the
last accelerations and the provider on the pre-pass's device) is the one device
payload, and it is dropped unless ``keep_forward``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from mimirax._typing import PerParticle, PyTree, Vec3
from mimirax.shooting import SegmentResult

# ``isinstance(x, Tracer)`` is how the runtime asks "am I inside a trace?", and
# there is no public spelling of it: ``jax.core`` is absent from ``jax.__all__``
# and resolves to ``jax._src.core``, while ``jax.extend.core.Tracer`` does not
# exist on the supported jax range (checked against 0.10.2). ``import jax``
# binds the submodule, so the name resolves at import; the suppression is for
# the missing public re-export, not for a missing attribute. jaccpot holds the
# same dependency in its ``_jax_compat`` module; this is the one site here.
Tracer = jax.core.Tracer  # pyright: ignore[reportAttributeAccessIssue]

__all__ = [
    "DirectSumSegmentEngine",
    "FusedSegmentEngine",
    "JaccpotForceModel",
    "rows_field",
]


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
    ad_mode : str | None
        ``differentiable_rollout``'s ``ad_mode``: ``"reverse"``, ``"forward"``
        (``jax.jvp`` / ``jax.jacfwd`` through the segment: a plain scan, so
        ``checkpoint`` must be ``None`` or ``"step"``), or ``None`` (default:
        an enclosing ``ad_mode_override``, then ``JACCPOT_GRAD_AD_MODE``, then
        reverse; the argument is then not passed, as before it existed).
    """

    segment_steps: tuple[int, ...]
    dt: float
    field: Callable[[Array, Any, Any], Array] | None = None
    G: float = 1.0
    softening: float = 0.0
    softening_kernel: str | None = None
    checkpoint: str | None = "step"
    ad_mode: str | None = None

    def __post_init__(self) -> None:
        """Validate the segment lengths and the AD mode.

        Raises
        ------
        ValueError
            If ``segment_steps`` is empty or holds a non-positive count, or
            ``ad_mode`` is unknown or ``"forward"`` with a reverse-pass
            ``checkpoint`` (``"nested"``, ``"offload"``).
        """
        if not self.segment_steps or any(int(s) < 1 for s in self.segment_steps):
            raise ValueError(
                f"segment_steps must be positive step counts, got {self.segment_steps}"
            )
        _check_ad_mode(self.ad_mode, self.checkpoint)

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
            # left out by default: the program (and a jaccpot before the
            # argument) is what it was
            **({} if self.ad_mode is None else {"ad_mode": self.ad_mode}),
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

    def segment_key(self, segment: int) -> tuple:
        """Return the key of segment ``segment`` 's program: its length.

        Segments of equal length are the same function of their inputs, so one
        traced program serves them all (:func:`mimirax.shooting.segment_key`).

        Parameters
        ----------
        segment : int
            The segment.

        Returns
        -------
        tuple
            ``("direct_sum", segment_steps[segment])``.
        """
        return ("direct_sum", int(self.segment_steps[segment]))


# ---------------------------------------------------------------------------
# The fused production lane (S10, track J).
# ---------------------------------------------------------------------------


def _gradient_ok(bits: Any) -> bool:
    """jaccpot's gradient verdict of folded verdict bits (host).

    Parameters
    ----------
    bits : Any
        The ``uint32`` bits of a segment's aux (host or device).

    Returns
    -------
    bool
        ``True`` when no bit of jaccpot's ``GRAD_FAIL_MASK`` is set
        (``jaccpot.runtime.grad_rollout.rollout_ok(..., gradient=True)``, in numpy:
        no device dispatch per segment).
    """
    from jaccpot.runtime.fused_topology import (  # pyright: ignore[reportMissingImports]
        GRAD_FAIL_MASK,
    )

    return (int(np.asarray(bits)) & int(GRAD_FAIL_MASK)) == 0


#: the AD modes an engine takes (jaccpot's ``ad_mode``); ``None`` follows the ambient
AD_MODES = (None, "reverse", "forward")


def _check_ad_mode(ad_mode: str | None, checkpoint: str | None) -> None:
    """Validate an engine's ``ad_mode`` against its ``checkpoint``.

    Parameters
    ----------
    ad_mode : str | None
        One of :data:`AD_MODES`.
    checkpoint : str | None
        The engine's ``differentiable_rollout`` checkpoint mode.

    Raises
    ------
    ValueError
        If ``ad_mode`` is unknown, or ``"forward"`` comes with a reverse-pass
        storage schedule (``"nested"`` / ``"offload"``): a jvp stores no residuals,
        so forward mode is the plain scan (``None``, or ``"step"`` meaning it).
    """
    if ad_mode not in AD_MODES:
        raise ValueError(f"ad_mode must be one of {AD_MODES}, got {ad_mode!r}")
    if ad_mode == "forward" and checkpoint not in (None, "step"):
        raise ValueError(
            f"ad_mode='forward' runs the plain scan: checkpoint must be None or "
            f"'step', got {checkpoint!r} (a reverse-pass storage schedule)"
        )


def _ad_mode_override(ad_mode: str | None) -> Any:
    """jaccpot's ``ad_mode_override(ad_mode)``, or a null context for ``None``.

    Parameters
    ----------
    ad_mode : str | None
        The engine's AD mode.

    Returns
    -------
    Any
        A context manager.
    """
    if ad_mode is None:
        return contextlib.nullcontext()
    from jaccpot.runtime.grad_options import (  # pyright: ignore[reportMissingImports]
        ad_mode_override,
    )

    return ad_mode_override(ad_mode)


def _node_states(nodes: Any, count: int) -> list[tuple[Any, Any]]:
    """The start state of every segment, as ``(positions, velocities)`` pairs.

    Parameters
    ----------
    nodes : Any
        A mapping with ``"positions"`` and ``"velocities"`` stacked over the
        segments (``[S, N, 3]`` each), or a sequence of ``(positions,
        velocities)`` pairs.
    count : int
        The number of segments ``S``.

    Returns
    -------
    list[tuple[Any, Any]]
        ``S`` pairs.

    Raises
    ------
    ValueError
        If there is not one node per segment.
    """
    if isinstance(nodes, Mapping):
        x, v = nodes["positions"], nodes["velocities"]
        pairs = [(x[k], v[k]) for k in range(int(np.shape(x)[0]))]
    else:
        pairs = [(pair[0], pair[1]) for pair in nodes]
    if len(pairs) != count:
        raise ValueError(
            f"prepare needs the start state of every segment ({count}), got "
            f"{len(pairs)} nodes (node 0, the initial condition, included)"
        )
    return pairs


class FusedSegmentEngine:
    """Multiple-shooting segments on jaccpot's fused lane, one gradient plan each.

    Segment ``k`` is ``segment_steps[k]`` leapfrog steps of ``runner``'s production
    lane from the segment's node. :meth:`prepare` makes its plan --
    ``runner.gradient_plan(node_k, masses, dt=dt, num_steps=segment_steps[k],
    external_params=theta)``, the runner's own forward run from the node with its
    retries and re-plans, recorded -- and :meth:`__call__` runs the plan's
    gradient run (``plan.rollout``) from the state it is given. The plans are
    cached and REUSED while :meth:`check` passes; :meth:`prepare` re-plans only the
    segments whose verdict failed, at their current nodes.

    WHAT A SEGMENT IS. A fresh run from the node: its first force is the start
    force at the node (the plan's start kind ``"start"``), not an acceleration
    carried over the segment edge as one long run carries it (``"carried"``). With
    the geometric MAC the two are the same force; under ``mac_type='dehnen_error'``
    the start force's thresholds come from the pre-pass's f_b seed at the node (the
    eager prepare's eq (16b) pass, stopped: a constant of the plan), the carried
    one's from the previous step's f_b -- both forces are in the FMM's error class,
    and they differ by it. A runner whose ``SegmentConfig.steps`` is shorter than a
    shooting segment cuts it into runner segments, chained as the runner chains
    them.

    THE FIELD. ``runner.external_acceleration_fn`` in jaccpot's ``f(state, args)``
    convention (:func:`rows_field` adapts a nornax ``f(x, t, args)``); ``theta`` is
    its ``external_params``, passed to every traced call as an argument (jaccpot's
    hand-written reverses refuse closures over differentiated values). Without a
    field ``theta`` is not used.

    Parameters
    ----------
    runner : Any
        A single-card ``jaccpot.ProductionRunner`` on the fused lane. Its solver's
        engine holds the trace-time state every plan installs while it traces; use
        one runner per engine.
    segment_steps : Sequence[int]
        Steps per shooting segment (static: one program per length and plan).
    dt : float
        The timestep (every segment).
    force_params : Any
        jaccpot ``ForceParams`` for the pre-passes, or ``None`` for the solver's
        (always under ``dehnen_error``).
    checkpoint : Optional[str]
        ``differentiable_rollout``'s checkpoint mode per segment (``"step"``,
        ``"nested"``, ``"offload"``, or ``None`` for a plain scan).
    inner_steps : Optional[int]
        Its ``inner_steps`` (``"nested"``).
    ad_mode : Optional[str]
        The AD mode each traced segment runs in: ``"forward"`` traces
        ``plan.rollout`` inside jaccpot's ``ad_mode_override("forward")`` with
        ``checkpoint=None`` (the plain scan ``GradientPlan.jvp`` / ``jacfwd``
        run), for ``jax.jvp`` / ``jax.jacfwd`` through the segment;
        ``"reverse"`` inside ``ad_mode_override("reverse")``; ``None`` (default)
        installs nothing, so an enclosing override (or ``JACCPOT_GRAD_AD_MODE``)
        decides and the program is what it was before the argument. With
        ``"forward"``, ``checkpoint`` must be ``None`` or ``"step"`` and
        ``inner_steps`` ``None``. The pre-passes (:meth:`prepare`) are forward
        runs either way: one plan serves both modes.
    start_force : str
        Where each segment's start force runs, jaccpot's
        ``GradientPlan.rollout(start_force=...)``: ``"outside"`` (default) beside
        the segment's rollout, the forward the runner's bit for bit; ``"adjoint"``
        inside its hand-written reverse, for less device memory (W at the S9 2e6
        Plummer calibration 883 -> ~406 B/p), step 0's field then added inside the
        program (an ulp off the runner's eager add for a field of more than one
        multiply). Needs a jaccpot with the argument. Not the default: under it
        the segment's forward is the runner's from its node only to round-off (4
        bytes in ``tests/integration/test_fused_segment_engine.py``'s G1, whose
        other gates pass, as do the two-card GPU gates; S10 follow-ups).
    keep_forward : bool
        Keep each plan's pre-pass result (its ``forward``: the end state and the
        runner's handle) instead of only the end state (:attr:`ends`). For tests.

    Attributes
    ----------
    plans : list
        The cached ``jaccpot.production_grad.GradientPlan`` of every segment
        (``None`` before the first :meth:`prepare`).
    ends : list
        Each plan's pre-pass end state ``[N, 2, 3]``: the runner's trajectory from
        the node the plan was made at (the propagated end, :meth:`propagate`).
    generation : int
        Plans made so far: a traced objective must be traced again when it
        changes (a static argument of the jitted objective, or a new function
        object: ``jax.jit`` of the same function reuses its trace).
    failed_segments : tuple
        The segments whose verdict failed at the last :meth:`check`.
    stats : dict
        Host cost: ``prepares``, ``plans`` (pre-passes run), ``prepare_s``,
        ``plan_s`` (per pre-pass, a list), ``checks``, ``check_s``.

    Raises
    ------
    ValueError
        If ``segment_steps`` is empty or holds a length below 1, or
        ``start_force`` is unknown.
    """

    plans: list
    ends: list
    generation: int
    failed_segments: tuple
    stats: dict

    def __init__(
        self,
        runner: Any,
        segment_steps: Sequence[int],
        *,
        dt: float,
        force_params: Any = None,
        checkpoint: Optional[str] = "step",
        inner_steps: Optional[int] = None,
        ad_mode: Optional[str] = None,
        start_force: str = "outside",
        keep_forward: bool = False,
    ) -> None:
        steps = tuple(int(s) for s in segment_steps)
        if not steps or min(steps) < 1:
            raise ValueError(f"segment_steps must be positive, got {steps}")
        _check_ad_mode(ad_mode, checkpoint)
        if ad_mode == "forward" and inner_steps is not None:
            raise ValueError("ad_mode='forward' takes no inner_steps (no 'nested')")
        if start_force not in ("outside", "adjoint"):
            raise ValueError(
                f"start_force must be 'outside' or 'adjoint', got {start_force!r}"
            )
        self.runner = runner
        self.segment_steps = steps
        self.dt = dt
        self.force_params = force_params
        self.checkpoint = checkpoint
        self.inner_steps = inner_steps
        self.start_force = start_force
        self.keep_forward = bool(keep_forward)
        self.ad_mode = ad_mode
        self.plans: list = [None] * len(steps)
        self.ends: list = [None] * len(steps)
        #: each pre-pass's last f_b (``None`` without dehnen_error)
        self._end_f_b: list = [None] * len(steps)
        self.generation = 0
        self.failed_segments: tuple = ()
        #: the id of each segment's plan, also returned in its aux (check reads it)
        self._plan_ids: list = [-1] * len(steps)
        #: segments to plan at the next prepare
        self._stale: set = set(range(len(steps)))
        #: each plan's arrays placed on a device, by (segment, plan id, device)
        self._placed: dict = {}
        # jaccpot's opt-in cache of the pre-pass's freeze program (compiled once per
        # compile key across this engine's pre-passes, not once per pre-pass),
        # where the runner offers it
        self._plan_options: dict = {}
        if "freeze_cache" in inspect.signature(runner.gradient_plan).parameters:
            self._plan_options["freeze_cache"] = {}
        self.stats: dict = {
            "prepares": 0,
            "plans": 0,
            "prepare_s": 0.0,
            "plan_s": [],
            "checks": 0,
            "check_s": 0.0,
        }

    @property
    def num_segments(self) -> int:
        """The number of shooting segments.

        Returns
        -------
        int
            ``len(segment_steps)``.
        """
        return len(self.segment_steps)

    @property
    def stale_segments(self) -> tuple:
        """The segments the next :meth:`prepare` plans.

        Returns
        -------
        tuple
            Sorted segment indices: never planned, or failed at the last
            :meth:`check`.
        """
        return tuple(sorted(self._stale))

    # ------------------------------------------------------------------
    # Host: the pre-passes.
    # ------------------------------------------------------------------

    def _plan_segment(
        self, segment: int, theta: PyTree, positions: Any, velocities: Any, masses: Any
    ) -> None:
        """Make segment ``segment``'s plan at the given node: one pre-pass.

        Parameters
        ----------
        segment : int
            The segment.
        theta : PyTree
            The field's parameters.
        positions : Any
            ``(n, 3)`` node positions.
        velocities : Any
            ``(n, 3)`` node velocities.
        masses : Any
            ``(n,)`` masses.
        """
        field = getattr(self.runner, "external_acceleration_fn", None)
        begun = time.perf_counter()
        plan = self.runner.gradient_plan(
            jnp.stack([jnp.asarray(positions), jnp.asarray(velocities)], axis=1),
            jnp.asarray(masses),
            dt=self.dt,
            num_steps=self.segment_steps[segment],
            force_params=self.force_params,
            external_params=theta if field is not None else None,
            **self._plan_options,
        )
        self.ends[segment] = plan.forward.state
        self._end_f_b[segment] = plan.forward.handle.force_scale
        if not self.keep_forward:
            # the handle holds the runner's last accelerations and provider: the
            # gradient run reads none of it
            plan.forward = None
        self.stats["plan_s"].append(time.perf_counter() - begun)
        self.stats["plans"] += 1
        self.plans[segment] = plan
        self.generation += 1
        self._plan_ids[segment] = self.generation
        # the replaced plan's placed arrays are dead
        for key in [key for key in self._placed if key[0] == segment]:
            del self._placed[key]

    def prepare(self, theta: PyTree, nodes: Any, masses: PerParticle) -> None:
        """Plan the segments without a valid plan, at their current nodes. Host.

        A segment is planned when it has no plan yet or its verdict failed at the
        last :meth:`check`; every other segment keeps its plan (and the traced
        objective its program). Idempotent: with nothing to plan it does no work
        (no host copy, no device call), so a driver may call it before every
        step. A re-plan increments :attr:`generation`.

        Parameters
        ----------
        theta : PyTree
            The field's parameters at the current iterate.
        nodes : Any
            The start state of EVERY segment (node 0, the initial condition,
            included): a mapping with ``"positions"`` and ``"velocities"`` stacked
            ``[S, N, 3]``, or a sequence of ``(positions, velocities)`` pairs.
        masses : PerParticle
            ``(n,)`` masses.
        """
        begun = time.perf_counter()
        stale = self.stale_segments
        if stale:
            pairs = _node_states(nodes, self.num_segments)
            for k in stale:
                self._plan_segment(k, theta, pairs[k][0], pairs[k][1], masses)
        self._stale = set()
        self.stats["prepares"] += 1
        self.stats["prepare_s"] += time.perf_counter() - begun

    def propagate(
        self, theta: PyTree, positions: Vec3, velocities: Vec3, masses: PerParticle
    ) -> dict:
        """Plan every segment along ONE forward run; return the nodes. Host.

        Segment ``k + 1``'s node is segment ``k``'s pre-pass end, so the nodes are
        the propagated ends (every defect zero at ``theta``) and every segment is
        planned at its node, in ``S`` pre-passes -- what :meth:`prepare` would cost.

        Parameters
        ----------
        theta : PyTree
            The field's parameters.
        positions : Vec3
            ``(n, 3)`` initial positions (node 0).
        velocities : Vec3
            ``(n, 3)`` initial velocities.
        masses : PerParticle
            ``(n,)`` masses.

        Returns
        -------
        dict
            ``{"positions": [S, N, 3], "velocities": [S, N, 3]}``: every segment's
            node, node 0 included; the last segment's end is ``ends[-1]``.
        """
        x, v = jnp.asarray(positions), jnp.asarray(velocities)
        xs, vs = [], []
        for k in range(self.num_segments):
            xs.append(x)
            vs.append(v)
            self._plan_segment(k, theta, x, v, masses)
            end = self.ends[k]
            x, v = end[:, 0], end[:, 1]
        self._stale = set()
        return {"positions": jnp.stack(xs), "velocities": jnp.stack(vs)}

    # ------------------------------------------------------------------
    # Traced: one segment's gradient run.
    # ------------------------------------------------------------------

    def segment_key(self, segment: int) -> tuple:
        """Return the key of segment ``segment`` 's program: the segment and its plan.

        A re-plan changes the key, so a driver keying its programs by it traces
        again exactly the re-planned segments (:func:`mimirax.shooting.segment_key`).

        Parameters
        ----------
        segment : int
            The segment.

        Returns
        -------
        tuple
            ``("fused", segment, plan id)`` (plan id ``-1`` before a plan).
        """
        return ("fused", int(segment), int(self._plan_ids[segment]))

    @staticmethod
    def _plan_parameters(plan: Any) -> dict:
        """The plan's parameters the gradient run reads, the field's excepted.

        Parameters
        ----------
        plan : Any
            A ``GradientPlan``.

        Returns
        -------
        dict
            ``{"dt", "force_params"}`` as the runner passed them.
        """
        return {"dt": plan.params["dt"], "force_params": plan.params["force_params"]}

    def segment_arrays(self, segment: int, device: Any = None) -> Any:
        """Return segment ``segment`` 's plan arrays, placed on ``device``. Host.

        The plan's array parameters (``dt`` and the force parameters, when the
        runner traces them; a Python number stays a literal, ``None`` here) and
        every runner segment's f_b seed (``None`` without ``dehnen_error``), for a
        program to take as arguments (``__call__(..., arrays=)``). Placed once per
        plan and device (cached; dropped when the segment is re-planned). They are
        jaccpot's ``GradientPlan.arrays(device)``, where the plan has it (then
        ``boundary`` too, ``None``: a segment is a fresh run from its node).

        Parameters
        ----------
        segment : int
            The segment.
        device : Any
            Where to place them; ``None``: the default device.

        Returns
        -------
        Any
            ``{"params": tuple, "seeds": tuple}`` (and ``"boundary"`` from
            jaccpot's ``GradientPlan.arrays``), or ``None`` before the segment's
            first plan.
        """
        plan = self.plans[segment]
        if plan is None:
            return None
        key = (int(segment), self._plan_ids[segment], device)
        placed = self._placed.get(key)
        if placed is None and hasattr(plan, "arrays"):
            placed = plan.arrays(device)
            self._placed[key] = placed
        elif placed is None:
            # a jaccpot before GradientPlan.arrays: the same data, gathered here
            leaves = jax.tree_util.tree_leaves(self._plan_parameters(plan))
            host = {
                "params": tuple(
                    leaf if isinstance(leaf, (jax.Array, np.ndarray)) else None
                    for leaf in leaves
                ),
                "seeds": tuple(
                    getattr(seg, "seed", None) for seg in getattr(plan, "segments", ())
                ),
            }
            placed = (
                jax.tree_util.tree_map(jnp.asarray, host)
                if device is None
                else jax.device_put(host, device)
            )
            self._placed[key] = placed
        return placed

    def _bind(self, plan: Any, arrays: Any) -> tuple:
        """The plan with ``arrays`` in place of its own, and its parameters. Traced.

        Parameters
        ----------
        plan : Any
            The segment's ``GradientPlan``.
        arrays : Any
            What :meth:`segment_arrays` returned for it (or tracers of it).

        Returns
        -------
        tuple
            ``(plan, params)``: a shallow copy of the plan whose runner segments'
            seeds are ``arrays["seeds"]``, and its ``dt`` / force parameters with
            the array leaves from ``arrays["params"]``.

        Raises
        ------
        ValueError
            If ``arrays`` do not match the plan (taken from another plan).
        """
        own = self._plan_parameters(plan)
        leaves, treedef = jax.tree_util.tree_flatten(own)
        given = arrays["params"]
        segments = tuple(getattr(plan, "segments", ()))
        if len(given) != len(leaves) or len(arrays["seeds"]) != len(segments):
            raise ValueError(
                "segment arrays do not match the segment's plan: take them from "
                "segment_arrays() after the last prepare"
            )
        params = jax.tree_util.tree_unflatten(
            treedef,
            [leaf if new is None else new for leaf, new in zip(leaves, given)],
        )
        if not segments:
            return plan, params
        segments = tuple(
            dataclasses.replace(seg, seed=seed)
            for seg, seed in zip(segments, arrays["seeds"])
        )
        return dataclasses.replace(plan, segments=segments), params

    def __call__(
        self,
        theta: PyTree,
        positions: Vec3,
        velocities: Vec3,
        masses: PerParticle,
        segment: int,
        arrays: Any = None,
    ) -> SegmentResult:
        """Run segment ``segment`` on its plan from the given state. Traced.

        A segment without a plan, called with CONCRETE inputs (outside a trace:
        :meth:`MultipleShootingForwardModel.init_nodes
        <mimirax.shooting.MultipleShootingForwardModel.init_nodes>` run eagerly), is
        planned at those inputs and returns its pre-pass's end -- the runner's
        trajectory from them, which the traced call reproduces bitwise -- with the
        aux ``(f_b, 0, None, plan_id)`` (the pre-pass passed its verdict; its needs
        are not recorded). :meth:`propagate` does the same for every segment.

        Parameters
        ----------
        theta : PyTree
            The field's parameters (differentiated).
        positions : Vec3
            ``(n, 3)`` start positions (differentiated).
        velocities : Vec3
            ``(n, 3)`` start velocities (differentiated).
        masses : PerParticle
            ``(n,)`` masses (differentiated when the caller frees them).
        segment : int
            Which segment (a Python int: each segment is its own program part).
        arrays : Any
            The plan's arrays as :meth:`segment_arrays` returned them (or the
            program's arguments of them): read instead of the plan's own, which
            are otherwise constants of the program. ``None``: the plan's.

        Returns
        -------
        SegmentResult
            ``(positions, velocities, aux)`` after the segment's steps. ``aux`` is
            ``(f_b, bits, needs, plan_id)`` -- jaccpot's folded verdict and the id
            of the plan the program was traced on -- every leaf behind
            ``stop_gradient``: a loss that reads it gets no gradient from it and
            does not trip jaccpot's aux guard. Hand it to :meth:`check`.

        Raises
        ------
        RuntimeError
            If the segment has no plan and is traced (call :meth:`prepare` or
            :meth:`propagate` first).
        """
        segment = int(segment)
        plan = self.plans[segment]
        if plan is None:
            leaves = jax.tree_util.tree_leaves((theta, positions, velocities, masses))
            if any(isinstance(leaf, Tracer) for leaf in leaves):
                raise RuntimeError(
                    f"segment {segment} has no plan: call prepare(theta, nodes, "
                    "masses) or propagate(...) before tracing the objective"
                )
            self._plan_segment(segment, theta, positions, velocities, masses)
            self._stale.discard(segment)
            end = self.ends[segment]
            aux = (
                self._end_f_b[segment],
                jnp.zeros((), jnp.uint32),
                None,
                jnp.asarray(self._plan_ids[segment], jnp.int32),
            )
            return SegmentResult(end[:, 0], end[:, 1], aux)

        # Every input behind a barrier: dt, G and the softening (the plan's
        # arrays, the runner's traced parameters) are constants of the caller's
        # program, and so may be the masses, node 0 or theta (a model's fixed
        # fields). XLA would fold a constant's value into the step and round
        # otherwise than the runner, whose programs take them as arguments (209 of
        # 6144 bytes moved: CPU, N = 512, Plummer). A Python float (a runner
        # without traced_params) stays a literal, as it is there.
        def opaque(tree: Any) -> Any:
            return jax.tree_util.tree_map(
                lambda a: (
                    jax.lax.optimization_barrier(a)
                    if isinstance(a, (jax.Array, np.ndarray))
                    else a
                ),
                tree,
            )

        params = plan.params
        bound: dict = {}
        if arrays is not None and hasattr(plan, "arrays"):
            # jaccpot binds them (GradientPlan.rollout(arrays=)): dt and the force
            # parameters from the arrays (behind the barrier), the seeds read in
            # place of the plan's
            params = {}
            bound["arrays"] = {**arrays, "params": opaque(tuple(arrays["params"]))}
        elif arrays is not None:
            plan, own = self._bind(plan, arrays)
            params = {**params, **own}
        inputs = {
            **opaque(params),
            "positions": opaque(positions),
            "velocities": opaque(velocities),
            "masses": opaque(masses),
            "external_params": (
                opaque(theta) if plan.external_fn is not None else None
            ),
        }
        # the default leaves the argument out (a jaccpot before it)
        start = {} if self.start_force == "outside" else {"start_force": "adjoint"}
        # forward mode: the plain scan of GradientPlan's own jvp / jacfwd, traced
        # with the custom_jvp seams installed (the mode is read at trace time)
        checkpoint = None if self.ad_mode == "forward" else self.checkpoint
        with _ad_mode_override(self.ad_mode):
            out = plan.rollout(
                inputs,
                checkpoint=checkpoint,
                inner_steps=self.inner_steps,
                **start,
                **bound,
            )
        f_b, bits, needs = out.aux
        aux = jax.lax.stop_gradient(
            (f_b, bits, needs, jnp.asarray(self._plan_ids[segment], jnp.int32))
        )
        return SegmentResult(out.positions, out.velocities, aux)

    # ------------------------------------------------------------------
    # Host: the verdict.
    # ------------------------------------------------------------------

    def check(self, auxes: Sequence[Any]) -> bool:
        """The gradient verdict of every segment of an evaluation. Host sync.

        One device-to-host copy of every segment's verdict bits and plan id.

        Parameters
        ----------
        auxes : Sequence[Any]
            The ``aux`` of every segment's :meth:`__call__`, in segment order.

        Returns
        -------
        bool
            ``True`` when every segment's capacities held and its gradient lists
            were valid. ``False`` otherwise: the failed segments are in
            :attr:`failed_segments` and are re-planned by the next
            :meth:`prepare`; the evaluation's values and gradients must be
            discarded and the step redone.

        Raises
        ------
        ValueError
            If there is not one aux per segment.
        RuntimeError
            If an aux comes from a program traced on a plan :meth:`prepare` has
            since replaced (the objective was not traced again after a re-plan).
        """
        begun = time.perf_counter()
        if len(auxes) != self.num_segments:
            raise ValueError(
                f"check needs one aux per segment ({self.num_segments}), got "
                f"{len(auxes)}"
            )
        host = jax.device_get([(aux[1], aux[3]) for aux in auxes])
        stale = [
            k
            for k, (_, plan_id) in enumerate(host)
            if int(plan_id) != self._plan_ids[k]
        ]
        if stale:
            raise RuntimeError(
                f"segments {stale} were evaluated by a program traced on a plan "
                "prepare() has since replaced: trace the objective again after a "
                "prepare that re-planned (FusedSegmentEngine.generation changed)"
            )
        failed = tuple(k for k, (bits, _) in enumerate(host) if not _gradient_ok(bits))
        self.failed_segments = failed
        self._stale |= set(failed)
        self.stats["checks"] += 1
        self.stats["check_s"] += time.perf_counter() - begun
        return not failed
