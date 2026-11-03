"""A multiple-shooting objective as one program per segment, one device each.

WHY. The augmented Lagrangian of a multiple-shooting problem is a sum of
per-segment terms plus terms of the parameters alone. With the defects ``c_k =
node_{k+1} - end_k(theta, node_k)``, segment ``k`` contributes ``lambda_k . c_k +
(mu / 2) |c_k|^2`` and, when the likelihood splits over the segments
(:meth:`~mimirax.InferenceProblem.segment_split`), its likelihood term; that
contribution reads ``(theta, node_k, node_{k+1}, lambda_k, mu)`` and the
segment's observations, nothing else. So its value and its gradients in
``theta``, ``node_k`` and ``node_{k+1}`` are ONE program per segment, independent
of the others given the nodes: segment ``k`` runs on ``devices[k % len(devices)]``
and the host sums the contributions (the priors, functions of the whole
parameter vector, are evaluated beside the sum).

HOW. :meth:`ParallelSegmentObjective.dispatch` puts each segment's inputs on its
device (``jax.device_put``) and calls its program; every program is dispatched
before any result is read, so JAX's asynchronous dispatch overlaps them. The
results are copied to the HOME device (``devices[0]``, where the parameters and
the optimizer state live) and :meth:`ParallelSegmentObjective.assemble` -- pure,
traced inside the caller's update program -- sums them into the value, the
gradient (in the parameters' structure), the unconstrained objective and the
stacked defects.

PROGRAMS. One jitted function per segment KEY (``mimirax.shooting.segment_key``:
equal segment lengths of the direct sum share one, the fused lane keys each
segment by its plan), traced once and compiled once per device it runs on. The
masses, the initial condition, the observations and the engine's per-segment
arrays (``segment_arrays``: the fused lane's plan parameters and seeds) are
ARGUMENTS, placed on their segment's device once (the engine's arrays once per
plan), never constants of the program.

ROUND-OFF. Against the single program (the whole objective in one
``jax.value_and_grad``) the split sums the same terms in a different order: the
theta gradient is ``sum_k g_k`` in segment order where autodiff accumulates it in
the reverse pass's order, the multiplier and penalty terms are summed per segment
and then over segments, and a ``"segment_ends"`` likelihood is a sum of
per-segment chi-squares. Each node's gradient is the sum of TWO contributions
(the segment it starts, the defect it closes), which is commutative, so with two
segments and a ``"final"`` output every gradient is bitwise the single
program's.
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from mimirax._typing import PyTree, Scalar
from mimirax.inference.objective import InferenceProblem, ProblemEvaluation
from mimirax.shooting import segment_key

__all__ = ["ParallelSegmentObjective", "SegmentContribution"]


class SegmentContribution(NamedTuple):
    """One segment's share of the augmented Lagrangian and of its gradient.

    Attributes
    ----------
    negative_log_likelihood : Array
        ``-term(end_k, observations_k)``; ``0`` for a segment without data.
    multiplier : Array
        ``sum(lambda_k * c_k)``; ``0`` for the last segment.
    penalty : Array
        ``(mu / 2) sum(c_k * c_k)``; ``0`` for the last segment.
    defect : Array | None
        ``[N, 6]`` ``c_k`` (positions then velocities); ``None`` for the last
        segment.
    grad_theta : PyTree
        d(contribution)/d(theta). ``None`` in a forward-only contribution.
    grad_start : PyTree
        d/d(start state) as ``{"positions", "velocities"}``. ``None`` in a
        forward-only contribution.
    grad_next : PyTree
        d/d(node_{k+1}); ``None`` for the last segment and forward-only.
    """

    negative_log_likelihood: Array
    multiplier: Array
    penalty: Array
    defect: Array | None
    grad_theta: PyTree
    grad_start: PyTree
    grad_next: PyTree


def _reduce_add(values: Sequence[Any]) -> Any:
    """Sum ``values`` left to right, leaf by leaf, WITHOUT a leading zero.

    Parameters
    ----------
    values : Sequence[Any]
        One or more pytrees of the same structure.

    Returns
    -------
    Any
        ``((v0 + v1) + v2) + ...``; ``v0`` itself for one value (a ``0 +`` would
        turn ``-0.0`` into ``0.0``).
    """
    return functools.reduce(
        lambda a, b: jax.tree_util.tree_map(operator.add, a, b), values
    )


class ParallelSegmentObjective:
    """A separable problem's augmented Lagrangian, one program per segment.

    Parameters
    ----------
    problem : InferenceProblem
        A problem whose objective splits over the segments of its forward model
        (:meth:`~mimirax.InferenceProblem.segment_split`).
    devices : Sequence[Any]
        The devices; segment ``k`` runs on ``devices[k % len(devices)]``,
        ``devices[0]`` is the HOME device (parameters, optimizer state, the
        assembly).

    Attributes
    ----------
    problem : InferenceProblem
        The problem.
    devices : tuple
        The devices.
    segment_devices : tuple
        The device of every segment.
    num_traces : int
        Traces of the value-and-gradient programs (one per segment key, plus one
        per key a re-plan or :meth:`refresh` made).
    num_forward_traces : int
        Traces of the forward-only programs (:meth:`dispatch_forward`).

    Raises
    ------
    ValueError
        If ``devices`` is empty or the problem does not split.
    """

    problem: InferenceProblem
    devices: tuple
    segment_devices: tuple
    num_traces: int
    num_forward_traces: int

    def __init__(self, problem: InferenceProblem, devices: Sequence[Any]) -> None:
        devices = tuple(devices)
        if not devices:
            raise ValueError("ParallelSegmentObjective needs at least one device")
        self.problem = problem
        self.split = problem.segment_split()
        self.model = self.split.model
        self.devices = devices
        count = int(self.model.num_segments)
        self.segment_devices = tuple(devices[k % len(devices)] for k in range(count))
        self.num_traces = 0
        self.num_forward_traces = 0
        self._programs: dict[tuple, Callable[..., Any]] = {}
        self._assemble = jax.jit(self.assemble)
        self._assemble_forward = jax.jit(self.assemble_forward)
        # placed ONCE (the engine's per-plan arrays are placed by the engine)
        masses = jnp.asarray(self.model.masses)
        self._masses = {
            device: jax.device_put(masses, device)
            for device in set(self.segment_devices)
        }
        self._initial: Any = None
        if not self.model.free_initial:
            self._initial = jax.device_put(
                {
                    "positions": jnp.asarray(self.model.initial_positions),
                    "velocities": jnp.asarray(self.model.initial_velocities),
                },
                self.segment_devices[0],
            )
        self._observations = tuple(
            None if obs is None else jax.device_put(obs, self.segment_devices[k])
            for k, obs in enumerate(self.split.observations)
        )

    # ------------------------------------------------------------------
    # The layout.
    # ------------------------------------------------------------------

    @property
    def num_segments(self) -> int:
        """Return the number of segments.

        Returns
        -------
        int
            ``S``.
        """
        return int(self.model.num_segments)

    @property
    def home(self) -> Any:
        """Return the HOME device (``devices[0]``).

        Returns
        -------
        Any
            Where the parameters, the optimizer state and the assembly live.
        """
        return self.devices[0]

    def defect_shapes(self, params: PyTree) -> tuple[jax.ShapeDtypeStruct, ...]:
        """Return the shapes of the problem's defects, without tracing a segment.

        Parameters
        ----------
        params : PyTree
            The parameters (their node arrays give ``N`` and the dtype).

        Returns
        -------
        tuple[jax.ShapeDtypeStruct, ...]
            ``([S-1, N, 6],)``: one defect array, the forward model's.
        """
        nodes = params["nodes"]["positions"]
        n = int(nodes.shape[-2])
        return (
            jax.ShapeDtypeStruct((self.num_segments - 1, n, 6), jnp.result_type(nodes)),
        )

    def _key(self, kind: str, segment: int) -> tuple:
        """Return the cache key of segment ``segment`` 's program of ``kind``.

        Parameters
        ----------
        kind : str
            ``"value_and_grad"`` or ``"forward"``.
        segment : int
            The segment.

        Returns
        -------
        tuple
            ``(kind, engine key, has a next node, is observed)``.
        """
        return (
            kind,
            segment_key(self.model.engine, segment),
            segment + 1 < self.num_segments,
            segment in self.split.observed_segments,
        )

    def refresh(self) -> None:
        """Forget every program (they are traced again at their next use). Host.

        What the driver calls after a re-prepare. An engine with ``segment_key``
        needs it not: its keys change with its plans, and a program whose key is
        stale is dropped at the next dispatch.
        """
        if not hasattr(self.model.engine, "segment_key"):
            self._programs.clear()

    def _evict(self, live: set) -> None:
        """Drop the programs of ``live`` 's kinds whose key is not in ``live``.

        Parameters
        ----------
        live : set
            The keys a dispatch used.
        """
        kinds = {key[0] for key in live}
        for key in [k for k in self._programs if k[0] in kinds and k not in live]:
            del self._programs[key]

    # ------------------------------------------------------------------
    # The programs.
    # ------------------------------------------------------------------

    def _segment_value(
        self,
        segment: int,
        has_next: bool,
        observed: bool,
        theta: PyTree,
        start: dict,
        following: Any,
        lam: Any,
        mu: Any,
        masses: Array,
        observations: Any,
        arrays: Any,
    ) -> tuple[Array, tuple]:
        """One segment's contribution, traced: its value and its pieces.

        Parameters
        ----------
        segment : int
            The segment the program is traced for (its key's first segment).
        has_next : bool
            Whether a node follows (a defect).
        observed : bool
            Whether the segment carries a likelihood term.
        theta : PyTree
            The field's parameters.
        start : dict
            ``{"positions", "velocities"}`` start state.
        following : Any
            The next node, or ``None``.
        lam : Any
            ``[N, 6]`` multipliers, or ``None`` (a forward-only program: no
            multiplier or penalty term).
        mu : Any
            The penalty.
        masses : Array
            ``[N]`` masses.
        observations : Any
            The segment's observations, or ``None``.
        arrays : Any
            The engine's arrays for the segment, or ``None``.

        Returns
        -------
        tuple[Array, tuple]
            ``(value, (nll, multiplier, penalty, defect, aux))``.
        """
        engine = self.model.engine
        if arrays is None:
            end = engine(
                theta, start["positions"], start["velocities"], masses, segment
            )
        else:
            end = engine(
                theta,
                start["positions"],
                start["velocities"],
                masses,
                segment,
                arrays=arrays,
            )
        zero = jnp.zeros((), jnp.result_type(end.positions))
        nll = zero
        if observed:
            state = {"positions": end.positions, "velocities": end.velocities}
            nll = -self.split.term(state, observations)
        multiplier, penalty, defect = zero, zero, None
        if has_next:
            defect = jnp.concatenate(
                [
                    following["positions"] - end.positions,
                    following["velocities"] - end.velocities,
                ],
                axis=-1,
            )
            if lam is not None:
                multiplier = jnp.sum(lam * defect)
                penalty = 0.5 * mu * jnp.sum(defect * defect)
        value = nll + multiplier + penalty
        return value, (nll, multiplier, penalty, defect, end.aux)

    def _program(self, kind: str, segment: int) -> Callable[..., Any]:
        """Return (building it on first use) segment ``segment`` 's program.

        Parameters
        ----------
        kind : str
            ``"value_and_grad"`` or ``"forward"``.
        segment : int
            The segment.

        Returns
        -------
        Callable[..., Any]
            The jitted program; ``(theta, start, next, lam, mu, masses,
            observations, arrays) -> (SegmentContribution, aux)``.
        """
        key = self._key(kind, segment)
        program = self._programs.get(key)
        if program is not None:
            return program
        has_next = segment + 1 < self.num_segments
        observed = segment in self.split.observed_segments
        value = functools.partial(self._segment_value, segment, has_next, observed)

        if kind == "value_and_grad":

            def value_and_grad_run(
                theta, start, following, lam, mu, masses, observations, arrays
            ):
                self.num_traces += 1  # runs at trace time only

                def of(theta_, start_, following_):
                    return value(
                        theta_,
                        start_,
                        following_,
                        lam,
                        mu,
                        masses,
                        observations,
                        arrays,
                    )

                (_, pieces), grads = jax.value_and_grad(
                    of, argnums=(0, 1, 2), has_aux=True
                )(theta, start, following)
                nll, multiplier, penalty, defect, aux = pieces
                return (
                    SegmentContribution(nll, multiplier, penalty, defect, *grads),
                    aux,
                )

            program = jax.jit(value_and_grad_run)

        else:

            def forward_run(
                theta, start, following, lam, mu, masses, observations, arrays
            ):
                self.num_forward_traces += 1  # runs at trace time only
                _, pieces = value(
                    theta, start, following, lam, mu, masses, observations, arrays
                )
                nll, multiplier, penalty, defect, aux = pieces
                return (
                    SegmentContribution(
                        nll, multiplier, penalty, defect, None, None, None
                    ),
                    aux,
                )

            program = jax.jit(forward_run)

        self._programs[key] = program
        return program

    # ------------------------------------------------------------------
    # Host: dispatch every segment.
    # ------------------------------------------------------------------

    def _states(self, params: PyTree) -> list[Any]:
        """Return every segment's start state as a pytree (node 0 the placed one).

        Parameters
        ----------
        params : PyTree
            ``{"theta", "nodes"}``.

        Returns
        -------
        list[Any]
            ``S`` start states ``{"positions", "velocities"}``, each where it
            lives (the home device; node 0 on its segment's device).
        """
        nodes = params["nodes"]
        count = self.model.num_nodes
        states = [
            {
                "positions": nodes["positions"][i],
                "velocities": nodes["velocities"][i],
            }
            for i in range(count)
        ]
        if self._initial is not None:
            states.insert(0, self._initial)
        return states

    def _dispatch(
        self, kind: str, params: PyTree, multipliers: Any, mu: Any
    ) -> tuple[tuple, tuple]:
        """Dispatch every segment's program of ``kind``; read nothing. Host.

        Parameters
        ----------
        kind : str
            ``"value_and_grad"`` or ``"forward"``.
        params : PyTree
            ``{"theta", "nodes"}``.
        multipliers : Any
            ``[S-1, N, 6]`` multipliers, or ``None`` (forward).
        mu : Any
            The penalty, or ``None`` (forward).

        Returns
        -------
        tuple[tuple, tuple]
            The ``S`` contributions and the ``S`` auxes, each on its segment's
            device (not yet computed: asynchronous).
        """
        states = self._states(params)
        arrays_of = getattr(self.model.engine, "segment_arrays", None)
        theta_on: dict = {}
        contributions, auxes, live = [], [], set()
        for k, device in enumerate(self.segment_devices):
            has_next = k + 1 < self.num_segments
            if device not in theta_on:
                theta_on[device] = jax.device_put(
                    (params["theta"], None if mu is None else jnp.asarray(mu)), device
                )
            theta, mu_k = theta_on[device]
            lam = None
            if has_next and multipliers is not None:
                lam = jax.device_put(multipliers[k], device)
            following = jax.device_put(states[k + 1], device) if has_next else None
            arrays = None if arrays_of is None else arrays_of(k, device=device)
            program = self._program(kind, k)
            live.add(self._key(kind, k))
            contribution, aux = program(
                theta,
                jax.device_put(states[k], device),
                following,
                lam,
                mu_k,
                self._masses[device],
                self._observations[k],
                arrays,
            )
            contributions.append(contribution)
            auxes.append(aux)
        if hasattr(self.model.engine, "segment_key"):
            self._evict(live)
        return tuple(contributions), tuple(auxes)

    def dispatch(
        self, params: PyTree, multipliers: tuple[Array, ...], mu: Any
    ) -> tuple[tuple, tuple]:
        """Dispatch every segment's value-and-gradient program. Host, asynchronous.

        Parameters
        ----------
        params : PyTree
            ``{"theta", "nodes"}`` (on the home device).
        multipliers : tuple[Array, ...]
            ``(lambda,)``, ``lambda`` ``[S-1, N, 6]`` (the driver's multipliers).
        mu : Any
            The penalty.

        Returns
        -------
        tuple[tuple, tuple]
            The ``S`` :class:`SegmentContribution` s, COPIED to the home device,
            and the ``S`` auxes, on their segment's device (for the engine's
            ``check``).
        """
        lam = multipliers[0] if multipliers else None
        contributions, auxes = self._dispatch("value_and_grad", params, lam, mu)
        return jax.device_put(contributions, self.home), auxes

    def dispatch_forward(self, params: PyTree) -> tuple[tuple, tuple]:
        """Dispatch every segment's forward-only program. Host, asynchronous.

        Parameters
        ----------
        params : PyTree
            ``{"theta", "nodes"}``.

        Returns
        -------
        tuple[tuple, tuple]
            As :meth:`dispatch`, without gradients (and without the multiplier and
            penalty terms, which are zero).
        """
        contributions, auxes = self._dispatch("forward", params, None, None)
        return jax.device_put(contributions, self.home), auxes

    # ------------------------------------------------------------------
    # Traced, on the home device: the sum.
    # ------------------------------------------------------------------

    def _objective(self, params: PyTree, contributions: Sequence[Any]) -> tuple:
        """Return ``f`` and the gradient of the priors (traced).

        Parameters
        ----------
        params : PyTree
            The parameters.
        contributions : Sequence[Any]
            The ``S`` contributions.

        Returns
        -------
        tuple
            ``(f, prior_grad or None)``: ``f = sum_k nll_k - log_prior``.
        """
        f = _reduce_add(
            [
                contributions[k].negative_log_likelihood
                for k in self.split.observed_segments
            ]
        )
        if not self.problem.priors:
            return f, None
        prior, prior_grad = jax.value_and_grad(lambda p: -self.problem.log_prior(p))(
            params
        )
        return f + prior, prior_grad

    def _defects(self, params: PyTree, contributions: Sequence[Any]) -> Array:
        """Stack the defects ``[S-1, N, 6]`` (traced).

        Parameters
        ----------
        params : PyTree
            The parameters (shape and dtype of an empty stack).
        contributions : Sequence[Any]
            The ``S`` contributions.

        Returns
        -------
        Array
            The defects.
        """
        if self.num_segments == 1:
            (shape,) = self.defect_shapes(params)
            return jnp.zeros(shape.shape, shape.dtype)
        return jnp.stack([c.defect for c in contributions[:-1]])

    def assemble(
        self, params: PyTree, contributions: Sequence[SegmentContribution]
    ) -> tuple[Scalar, PyTree, Scalar, Array]:
        """Sum the contributions into the augmented Lagrangian and its gradient.

        Pure: call it inside a jitted program on the home device.

        Parameters
        ----------
        params : PyTree
            ``{"theta", "nodes"}``.
        contributions : Sequence[SegmentContribution]
            What :meth:`dispatch` returned.

        Returns
        -------
        tuple[Scalar, PyTree, Scalar, Array]
            ``(L, dL/dparams, f, defects)``: the gradient in ``params`` '
            structure, ``f`` the unconstrained objective, the defects ``[S-1, N,
            6]``.
        """
        count = self.num_segments
        f, prior_grad = self._objective(params, contributions)
        value = f
        if count > 1:
            multiplier = _reduce_add([c.multiplier for c in contributions[:-1]])
            penalty = _reduce_add([c.penalty for c in contributions[:-1]])
            value = value + multiplier + penalty
        grad_theta = _reduce_add([c.grad_theta for c in contributions])
        offset = 0 if self.model.free_initial else 1
        node_grads = []
        for i in range(self.model.num_nodes):
            k = i + offset  # the segment node i starts
            grad = contributions[k].grad_start
            if k > 0:
                grad = _reduce_add([grad, contributions[k - 1].grad_next])
            node_grads.append(grad)
        if node_grads:
            nodes = {
                name: jnp.stack([g[name] for g in node_grads])
                for name in ("positions", "velocities")
            }
        else:
            nodes = jax.tree_util.tree_map(jnp.zeros_like, params["nodes"])
        grads = {"theta": grad_theta, "nodes": nodes}
        if prior_grad is not None:
            grads = jax.tree_util.tree_map(operator.add, grads, prior_grad)
        return value, grads, f, self._defects(params, contributions)

    def assemble_forward(
        self, params: PyTree, contributions: Sequence[SegmentContribution]
    ) -> tuple[Scalar, Array]:
        """Sum forward-only contributions into ``f`` and the defects. Pure.

        Parameters
        ----------
        params : PyTree
            ``{"theta", "nodes"}``.
        contributions : Sequence[SegmentContribution]
            What :meth:`dispatch_forward` returned.

        Returns
        -------
        tuple[Scalar, Array]
            ``(f, defects)``.
        """
        f, _ = self._objective(params, contributions)
        return f, self._defects(params, contributions)

    # ------------------------------------------------------------------
    # Conveniences (one call: dispatch, then a jitted assembly).
    # ------------------------------------------------------------------

    def value_and_grad(
        self, params: PyTree, multipliers: tuple[Array, ...], mu: Any
    ) -> tuple[Scalar, PyTree, ProblemEvaluation]:
        """Return the augmented Lagrangian, its gradient and the evaluation.

        Parameters
        ----------
        params : PyTree
            ``{"theta", "nodes"}``.
        multipliers : tuple[Array, ...]
            ``(lambda,)``.
        mu : Any
            The penalty.

        Returns
        -------
        tuple[Scalar, PyTree, ProblemEvaluation]
            ``(L, grads, evaluation)``; the evaluation's ``aux`` is the ``S``
            segment auxes.
        """
        contributions, auxes = self.dispatch(params, multipliers, mu)
        value, grads, f, defects = self._assemble(
            jax.device_put(params, self.home), contributions
        )
        return value, grads, ProblemEvaluation(f, (defects,), auxes)

    def evaluate(self, params: PyTree) -> ProblemEvaluation:
        """Return the unconstrained objective, the defects and the auxes.

        Parameters
        ----------
        params : PyTree
            ``{"theta", "nodes"}``.

        Returns
        -------
        ProblemEvaluation
            As :meth:`~mimirax.InferenceProblem.evaluate`.
        """
        contributions, auxes = self.dispatch_forward(params)
        f, defects = self._assemble_forward(
            jax.device_put(params, self.home), contributions
        )
        return ProblemEvaluation(f, (defects,), auxes)
