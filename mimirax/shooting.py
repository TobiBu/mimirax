"""Multiple shooting: a rollout cut into segments joined by free node states.

WHY. A single-shooting fit differentiates one long rollout: its gradient grows
with the Lyapunov time of the system and its memory with the horizon, and the
objective it sees is rough far from the optimum. Multiple shooting cuts the
horizon into ``S`` segments, starts segment ``k > 0`` from a FREE node state
``(x_k, v_k)``, and asks the optimizer to drive the matching defects
``node_{k+1} - end_k`` to zero alongside the data misfit. Each segment is short
(its gradient tame), independent of the others given the nodes (so the segments
can run on separate devices), and the constraint is what makes the chained
segments one trajectory again.

THE SEAM, :class:`SegmentEngine`. What runs a segment is pluggable: the direct
sum in :mod:`mimirax.adapters.jaccpot` (``DirectSumSegmentEngine``), jaccpot's
fused lane (session S10, track J), or the pure-JAX double in
:mod:`mimirax.testing`. An engine is a traceable, differentiable map
``(theta, x, v, masses, segment) -> (x', v', aux)``; ``theta`` (the external
field's parameters) enters the force ONLY as an argument, never closed over,
because jaccpot's checkpointed rollouts are ``custom_vjp`` s that refuse closures
over differentiated values. The ``aux`` is engine-specific and never
differentiated (jaccpot's aux guard refuses a cotangent into it).

Two optional capabilities, found by ``hasattr`` (the convention of
:mod:`mimirax.protocols`; widening the protocol would make ``isinstance`` reject
every engine without them):

* ``prepare(theta, nodes, masses) -> None`` -- HOST, before an objective is
  traced or evaluated. ``nodes`` is ``{"positions": [S, N, 3], "velocities":
  [S, N, 3]}``: the START state of every segment, node 0 included. The fused
  lane plans each segment here (``runner.gradient_plan``: caps, retries, the
  f_b seed); a pure engine has no ``prepare``.
* ``check(auxes) -> bool`` -- HOST, after an evaluation, with the ``S`` segment
  auxes in segment order. ``False`` means the evaluation ran on a truncated force
  (a capacity verdict failed): the caller re-prepares and redoes the step.

:class:`MultipleShootingForwardModel` puts the segments together. It is a
:class:`~mimirax.protocols.ForwardModel` (``model(params)`` is the prediction),
a :class:`~mimirax.protocols.Constraint` (``model.defect(params)`` the stacked
defects), and it offers both from ONE pass of the segments
(:meth:`~MultipleShootingForwardModel.predict_with_defects`), which is what
:class:`~mimirax.inference.InferenceProblem` and
:class:`~mimirax.inference.AugmentedLagrangian` consume.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array

from mimirax._typing import PerParticle, PyTree, Vec3

__all__ = [
    "MultipleShootingForwardModel",
    "SegmentEngine",
    "SegmentResult",
    "ShootingEvaluation",
]

#: what :attr:`MultipleShootingForwardModel.output` accepts
OUTPUTS = ("final", "segment_ends")


class SegmentResult(NamedTuple):
    """The state at the end of one segment, and the engine's aux.

    Attributes
    ----------
    positions : Array
        ``[N, 3]`` positions at the segment's end.
    velocities : Array
        ``[N, 3]`` velocities at the segment's end.
    aux : Any
        Engine-specific and NOT differentiated: ``None`` for the direct sum, the
        fused lane's capacity verdict for jaccpot's provider. Read on the host by
        the engine's optional ``check``.
    """

    positions: Array
    velocities: Array
    aux: Any


@runtime_checkable
class SegmentEngine(Protocol):
    """Runs one segment of a multiple-shooting rollout from a given state.

    Traceable and differentiable in ``theta``, ``positions`` and ``velocities``
    (``masses`` are fixed unless the model frees them). The segment's number of
    steps and its timestep are the engine's static configuration, indexed by the
    static ``segment``. ``theta`` reaches the force only as an argument. Optional
    host-side capabilities ``prepare(theta, nodes, masses)`` and
    ``check(auxes)`` are described in the module docstring; they are not
    protocol members.
    """

    def __call__(
        self,
        theta: PyTree,
        positions: Array,
        velocities: Array,
        masses: Array,
        segment: int,
    ) -> SegmentResult:
        """Run segment ``segment`` from ``(positions, velocities)``.

        Parameters
        ----------
        theta : PyTree
            The external field's parameters (differentiated).
        positions : Array
            ``[N, 3]`` start positions (differentiated).
        velocities : Array
            ``[N, 3]`` start velocities (differentiated).
        masses : Array
            ``[N]`` masses.
        segment : int
            Which segment, ``0 <= segment < S``. Static.

        Returns
        -------
        SegmentResult
            The state at the segment's end and the engine's aux.
        """
        ...


class ShootingEvaluation(NamedTuple):
    """One pass of every segment: the prediction, the defects and the auxes.

    Attributes
    ----------
    prediction : PyTree
        What the model predicts (:attr:`MultipleShootingForwardModel.output`).
    defects : Array
        ``[S-1, N, 6]`` matching defects, ``node_{k+1} - end_k`` with the
        positions in ``[..., :3]`` and the velocities in ``[..., 3:]`` (nornax's
        ``shooting_defect`` convention). ``[0, N, 6]`` for one segment.
    auxes : tuple
        The ``S`` segment auxes, in segment order. Not differentiated.
    """

    prediction: PyTree
    defects: Array
    auxes: tuple


@dataclass(frozen=True)
class MultipleShootingForwardModel:
    """``S`` segments of a rollout joined by free node states.

    Parameters are ``{"theta": <field parameters>, "nodes": {"positions":
    [S-1, N, 3], "velocities": [S-1, N, 3]}}``: node ``k`` (``1 <= k < S``) is
    the start state of segment ``k``; segment 0 starts from the given initial
    condition. With :attr:`free_initial` the initial condition is a node too and
    the node arrays are ``[S, N, 3]``. Build the nodes from a forward run with
    :meth:`init_nodes` (the defects are then zero).

    Attributes
    ----------
    engine : SegmentEngine
        Runs one segment.
    masses : PerParticle
        ``[N]`` particle masses (fixed).
    num_segments : int
        ``S >= 1``. One segment is single shooting through the same engine.
    initial_positions : Vec3 | None
        ``[N, 3]`` start of segment 0; required unless :attr:`free_initial`.
    initial_velocities : Vec3 | None
        ``[N, 3]`` start of segment 0; required unless :attr:`free_initial`.
    free_initial : bool
        Make the initial condition a free node (``nodes`` then hold ``S``
        states).
    output : str
        What the model predicts: ``"final"`` (default) the state at the end of
        the last segment, ``{"positions": [N, 3], "velocities": [N, 3]}``;
        ``"segment_ends"`` the state at every node time after the start, i.e.
        the end of every segment, ``{"positions": [S, N, 3], "velocities": [S, N,
        3]}``. The ENDS, not the node variables: the data then constrain the
        dynamics, and the two agree when the defects vanish.
    """

    engine: SegmentEngine
    masses: PerParticle
    num_segments: int
    initial_positions: Vec3 | None = None
    initial_velocities: Vec3 | None = None
    free_initial: bool = False
    output: str = "final"

    def __post_init__(self) -> None:
        """Validate the static configuration.

        Raises
        ------
        ValueError
            If ``num_segments < 1``, ``output`` is unknown, or the initial
            condition is missing while ``free_initial`` is off.
        """
        if isinstance(self.num_segments, bool) or int(self.num_segments) < 1:
            raise ValueError(f"num_segments must be >= 1, got {self.num_segments!r}")
        if self.output not in OUTPUTS:
            raise ValueError(f"output must be one of {OUTPUTS}, got {self.output!r}")
        if not self.free_initial and (
            self.initial_positions is None or self.initial_velocities is None
        ):
            raise ValueError(
                "initial_positions and initial_velocities are required unless "
                "free_initial=True"
            )

    @property
    def num_nodes(self) -> int:
        """Return how many node states are free parameters.

        Returns
        -------
        int
            ``S - 1``, or ``S`` with :attr:`free_initial`.
        """
        return self.num_segments if self.free_initial else self.num_segments - 1

    def _starts(self, params: PyTree) -> tuple[list[Array], list[Array]]:
        """Return the start state of every segment.

        Parameters
        ----------
        params : PyTree
            ``{"theta": ..., "nodes": {"positions", "velocities"}}``.

        Returns
        -------
        tuple[list[Array], list[Array]]
            ``S`` start positions and ``S`` start velocities, ``[N, 3]`` each.

        Raises
        ------
        ValueError
            If the node arrays do not hold :attr:`num_nodes` states.
        """
        nodes = params["nodes"]
        node_x, node_v = nodes["positions"], nodes["velocities"]
        if node_x.shape[0] != self.num_nodes or node_v.shape[0] != self.num_nodes:
            raise ValueError(
                f"nodes must hold {self.num_nodes} states for {self.num_segments} "
                f"segments (free_initial={self.free_initial}), got "
                f"{node_x.shape[0]} positions and {node_v.shape[0]} velocities"
            )
        xs = [node_x[i] for i in range(self.num_nodes)]
        vs = [node_v[i] for i in range(self.num_nodes)]
        if not self.free_initial:
            xs.insert(0, jnp.asarray(self.initial_positions))
            vs.insert(0, jnp.asarray(self.initial_velocities))
        return xs, vs

    def evaluate(self, params: PyTree) -> ShootingEvaluation:
        """Run every segment once: prediction, defects and auxes.

        Parameters
        ----------
        params : PyTree
            ``{"theta": ..., "nodes": {"positions", "velocities"}}``.

        Returns
        -------
        ShootingEvaluation
            Everything one pass of the segments gives.
        """
        theta = params["theta"]
        xs, vs = self._starts(params)
        ends = [
            self.engine(theta, xs[k], vs[k], self.masses, k)
            for k in range(self.num_segments)
        ]
        n = xs[0].shape[0]
        if self.num_segments > 1:
            defects = jnp.stack(
                [
                    jnp.concatenate(
                        [xs[k + 1] - ends[k].positions, vs[k + 1] - ends[k].velocities],
                        axis=-1,
                    )
                    for k in range(self.num_segments - 1)
                ]
            )
        else:
            defects = jnp.zeros((0, n, 6), dtype=xs[0].dtype)
        if self.output == "final":
            prediction = {
                "positions": ends[-1].positions,
                "velocities": ends[-1].velocities,
            }
        else:
            prediction = {
                "positions": jnp.stack([end.positions for end in ends]),
                "velocities": jnp.stack([end.velocities for end in ends]),
            }
        return ShootingEvaluation(
            prediction=prediction,
            defects=defects,
            auxes=tuple(end.aux for end in ends),
        )

    def predict_with_defects(self, params: PyTree) -> tuple[PyTree, Array]:
        """Return the prediction and the stacked defects from ONE pass.

        Parameters
        ----------
        params : PyTree
            ``{"theta": ..., "nodes": {"positions", "velocities"}}``.

        Returns
        -------
        tuple[PyTree, Array]
            ``(prediction, defects)``, defects ``[S-1, N, 6]``.
        """
        evaluation = self.evaluate(params)
        return evaluation.prediction, evaluation.defects

    def __call__(self, params: PyTree) -> PyTree:
        """Return the prediction (the :class:`~mimirax.protocols.ForwardModel` call).

        Parameters
        ----------
        params : PyTree
            ``{"theta": ..., "nodes": {"positions", "velocities"}}``.

        Returns
        -------
        PyTree
            The prediction; the defects of the same pass are discarded.
        """
        return self.evaluate(params).prediction

    def defect(self, params: PyTree) -> Array:
        """Return the stacked defects (the :class:`~mimirax.protocols.Constraint` call).

        A separate pass of the segments; inside an objective use
        :meth:`predict_with_defects` instead.

        Parameters
        ----------
        params : PyTree
            ``{"theta": ..., "nodes": {"positions", "velocities"}}``.

        Returns
        -------
        Array
            ``[S-1, N, 6]`` defects.
        """
        return self.evaluate(params).defects

    def init_nodes(
        self,
        theta: PyTree,
        positions: Array | None = None,
        velocities: Array | None = None,
    ) -> dict[str, Array]:
        """Build the nodes from a forward run at ``theta``: the defects are zero there.

        Runs segments ``0 .. S-2`` chained, each from the previous one's end, and
        takes the ends as the nodes. Traceable and differentiable in ``theta``
        and the start state: ``theta -> objective(theta, init_nodes(theta))`` is
        the single-shooting objective.

        Parameters
        ----------
        theta : PyTree
            The field's parameters.
        positions : Array | None
            ``[N, 3]`` start of segment 0; ``None`` takes
            :attr:`initial_positions`.
        velocities : Array | None
            ``[N, 3]`` start of segment 0; ``None`` takes
            :attr:`initial_velocities`.

        Returns
        -------
        dict[str, Array]
            ``{"positions": [K, N, 3], "velocities": [K, N, 3]}`` with ``K =``
            :attr:`num_nodes` (the start state first when :attr:`free_initial`).

        Raises
        ------
        ValueError
            If no start state is given and the model holds none.
        """
        x = self.initial_positions if positions is None else positions
        v = self.initial_velocities if velocities is None else velocities
        if x is None or v is None:
            raise ValueError("init_nodes needs a start state")
        x, v = jnp.asarray(x), jnp.asarray(v)
        xs = [x] if self.free_initial else []
        vs = [v] if self.free_initial else []
        for k in range(self.num_segments - 1):
            end = self.engine(theta, x, v, self.masses, k)
            x, v = end.positions, end.velocities
            xs.append(x)
            vs.append(v)
        if not xs:
            empty = jnp.zeros((0,) + x.shape, dtype=x.dtype)
            return {"positions": empty, "velocities": empty}
        return {"positions": jnp.stack(xs), "velocities": jnp.stack(vs)}

    def start_states(self, params: PyTree) -> dict[str, Array]:
        """Return the start state of every segment, node 0 included.

        Parameters
        ----------
        params : PyTree
            ``{"theta": ..., "nodes": {"positions", "velocities"}}``.

        Returns
        -------
        dict[str, Array]
            ``{"positions": [S, N, 3], "velocities": [S, N, 3]}``: the ``nodes``
            an engine's ``prepare`` receives.
        """
        xs, vs = self._starts(params)
        return {"positions": jnp.stack(xs), "velocities": jnp.stack(vs)}

    def prepare(self, params: PyTree) -> None:
        """Delegate to the engine's ``prepare`` (HOST), if it has one.

        Parameters
        ----------
        params : PyTree
            Concrete parameters at which the next evaluation will run.
        """
        prepare = getattr(self.engine, "prepare", None)
        if prepare is None:
            return
        prepare(params["theta"], self.start_states(params), jnp.asarray(self.masses))

    def check(self, auxes: Sequence[Any]) -> bool:
        """Delegate to the engine's ``check`` (HOST), if it has one.

        Parameters
        ----------
        auxes : Sequence[Any]
            The ``S`` segment auxes of an evaluation
            (:attr:`ShootingEvaluation.auxes`).

        Returns
        -------
        bool
            The engine's verdict; ``True`` for an engine without ``check``.
        """
        check = getattr(self.engine, "check", None)
        if check is None:
            return True
        return bool(check(tuple(auxes)))


def stop_aux(aux: Any) -> Any:
    """Return ``aux`` with ``stop_gradient`` on every array leaf.

    For engines whose aux is computed from differentiated values: the aux is
    never differentiated (the module docstring), and an explicit stop keeps a
    plain-autodiff engine from carrying a cotangent path into it.

    Parameters
    ----------
    aux : Any
        Any pytree (``None`` included).

    Returns
    -------
    Any
        The same structure, gradients stopped.
    """
    return jax.tree_util.tree_map(jax.lax.stop_gradient, aux)
