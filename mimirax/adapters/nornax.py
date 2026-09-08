"""A nornax rollout as a :class:`~mimirax.protocols.ForwardModel`. ``mimirax[nornax]``.

The integrator half of the made-to-measure module (EDDA decision D-026):
initial conditions and particle weights in, a recorded trajectory out, through
``nornax.solvers.leapfrog_kdk.block_kdk_rollout``. nornax is imported *inside*
the call, so this module -- and the protocol bridge below it -- import and are
testable with no solver installed, and a missing nornax fails at the rollout
with a message naming the extra.

THE PROTOCOL MISMATCH, AND WHAT IT COSTS. nornax's integrator wants a
``MutualForceModel``: ``level_accelerations(positions, masses, *, rung, level)``,
the acceleration contributed by pairs whose finer endpoint sits on rung
``level``, applied antisymmetrically. mimirax's
:class:`~mimirax.protocols.ForceModel` gives a *total*
``accelerations(positions, masses)`` and knows nothing about rungs. The two meet
in exactly one place: with every particle on rung 0 and ``k_max = 0``, level 0
*is* the total force, so a mimirax force model can drive a **single-rung**
rollout exactly. :class:`SingleRungMutualForce` is that bridge and it refuses
any other level rather than returning something plausible. A multi-rung rollout
needs a real ``MutualForceModel`` -- nornax's ``MutualDirectSumGravity``, or
jaccpot's ``BlockStepFMM`` -- passed straight through.

WHAT IS DIFFERENTIABLE. ``params["positions"]``, ``params["velocities"]`` and
``params["weights"]`` are all differentiable leaves of the returned trajectory.
The weights reach the *dynamics* only in the self-consistent construction
(:meth:`NornaxRollout.self_consistent`), where they are the masses; in the
tracer construction (:meth:`NornaxRollout.tracer`) they reach the observable
alone and ``d(trajectory)/d(weights)`` is zero on the position and velocity
leaves by construction. What is **not** differentiable, in either: ``rung`` and
the topology, both of which nornax severs from the gradient with
``stop_gradient``. For a tree or FMM backend that is the substantive statement
of the EDDA programme's B4 -- the gradient is the exact fixed-topology gradient
of the numeric path on each segment between rebuilds, and a rebuild cadence of
``rebuild_every`` base steps is where the segments end.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import Array

from mimirax._typing import PerParticle, PyTree, Vec3
from mimirax.protocols import ForceModel

__all__ = ["NornaxRollout", "SingleRungMutualForce"]


def _leapfrog() -> tuple[Any, Any]:
    """Import nornax's rollout and node constructor, lazily.

    Returns
    -------
    tuple[Any, Any]
        ``(block_kdk_rollout, shooting_node)``.

    Raises
    ------
    ImportError
        If nornax is not installed; the message names the extra.
    """
    try:
        from nornax.solvers.leapfrog_kdk import (  # pyright: ignore[reportMissingImports]
            block_kdk_rollout,
            shooting_node,
        )
    except ImportError as exc:
        raise ImportError(
            "mimirax.adapters.nornax needs nornax: install it from GitHub, then "
            "`pip install 'mimirax[nornax]'`"
        ) from exc
    return block_kdk_rollout, shooting_node


def _is_mutual(force: object) -> bool:
    """Say whether ``force`` already speaks nornax's per-level interface.

    Parameters
    ----------
    force : object
        A force model of either flavour.

    Returns
    -------
    bool
        ``True`` when it has ``level_accelerations``, i.e. it is a nornax
        ``MutualForceModel`` and needs no bridge.
    """
    return callable(getattr(force, "level_accelerations", None))


@dataclass(frozen=True)
class SingleRungMutualForce:
    """A mimirax :class:`~mimirax.protocols.ForceModel` behind nornax's per-level interface.

    Valid at ``k_max = 0`` **only**, and it says so by raising. With every
    particle on rung 0 every pair sits on level 0, so level 0's antisymmetric
    contribution is the whole force and the wrapped model's total is exactly
    right. At any other level the honest answer does not exist: a mimirax
    ``ForceModel`` has no notion of which pairs belong to which level, and both
    plausible guesses are wrong -- returning the total at level 0 and zero above
    would kick the whole force with level 0's weight at every sub-step boundary
    (the wrong equations, silently), and returning zero at level 0 would lose
    the force entirely.

    **What conformance can and cannot say about this bridge.** nornax's
    ``check_mutual_force_model`` at ``k_max = 0`` checks shape, finiteness, the
    level partition (trivial with one level) and the *momentum residual* -- and
    the last one is a property of the wrapped model, not of the bridge. A mutual
    model (:class:`mimirax.testing.DirectSumGravity`) passes it to round-off; an
    **external** field (:class:`mimirax.testing.SoftenedPointMassField`, the
    tracer construction's double) does not and must not, because an external
    potential genuinely does not conserve the particles' momentum. That is why
    the tracer construction is single-rung: the antisymmetry the block schedule
    relies on is exactly what an external field lacks. An external field *could*
    be split by ``level = rung_i`` instead of ``max(rung_i, rung_j)`` and driven
    multi-rung, but that is a different bridge for a different force, and
    guessing which one a caller meant from a model that cannot be asked is how
    the wrong equations get integrated.

    Attributes
    ----------
    force : ForceModel
        The mimirax acceleration provider being bridged.
    """

    force: ForceModel

    def level_accelerations(
        self,
        positions: Vec3,
        masses: PerParticle,
        *,
        rung: Array,
        level: int,
        args: object = None,
        topology: object = None,
    ) -> Vec3:
        """Return the wrapped model's total acceleration, as level 0.

        Parameters
        ----------
        positions : Vec3
            ``(n, 3)`` positions.
        masses : PerParticle
            ``(n,)`` masses.
        rung : Array
            Ignored: at ``k_max = 0`` every particle is on rung 0, so the rung
            partition carries no information the total does not already have.
        level : int
            Must be ``0``.
        args : object
            Passed through to the wrapped model's caller-owned channel.
        topology : object
            Accepted and ignored: a mimirax ``ForceModel`` has no frozen
            interaction structure to evaluate against.

        Returns
        -------
        Vec3
            ``(n, 3)`` accelerations.

        Raises
        ------
        ValueError
            If ``level`` is not ``0``; see the class docstring for why there is
            no defensible answer above it.
        """
        del rung, topology
        if int(level) != 0:
            raise ValueError(
                f"SingleRungMutualForce is a k_max=0 bridge and was asked for "
                f"level {level}. A multi-rung rollout needs a real nornax "
                "MutualForceModel (nornax's MutualDirectSumGravity, or jaccpot's "
                "BlockStepFMM); pass one straight to NornaxRollout instead."
            )
        return self.force.accelerations(positions, masses, args=args)


def _record_kinematics(state: Any) -> dict[str, Array]:
    """Record the leaves an observable can act on, at one base-step boundary.

    Parameters
    ----------
    state : Any
        A nornax ``BlockStepState``.

    Returns
    -------
    dict[str, Array]
        ``positions`` and ``velocities``. Deliberately not the whole state: the
        topology can be large and ``rung`` is severed from the gradient, so
        stacking either over the rollout would cost memory for nothing.
    """
    return {"positions": state.positions, "velocities": state.velocities}


@dataclass(frozen=True)
class NornaxRollout:
    """Integrate a state forward with nornax and return the recorded trajectory.

    ``params`` is a mapping with ``"positions"`` ``(n, 3)``, ``"velocities"``
    ``(n, 3)`` and ``"weights"`` ``(n,)``. The returned trajectory is a mapping
    whose leaves all carry a leading ``num_steps`` axis, ready for
    :class:`mimirax.observables.TimeAverage`:

    * ``"positions"``, ``"velocities"`` -- ``(num_steps, n, 3)``, one row per
      base step, from nornax's ``record_fn``;
    * ``"weights"`` -- ``(num_steps, n)``, the weights broadcast along the time
      axis. They are constant along a rollout (nothing in the integrator moves
      a mass), so this materializes nothing the scan had to carry; it is done so
      that the whole trajectory is *one* pytree an averaging observable can
      ``vmap`` over, which keeps ``Observable`` from needing a second argument.

    Use :meth:`self_consistent` or :meth:`tracer` rather than the constructor:
    which of the two a fit is running is the single most consequential thing
    about a made-to-measure problem, and it should be named at the call site.

    The initial state is built with nornax's ``shooting_node``, not by hand.
    That constructor exists precisely to recompute the two leaves that must not
    be free variables -- ``acc`` from the positions and masses, and ``rung``
    from that ``acc`` -- so a fit that moves the weights cannot move the
    integration schedule by moving a derived quantity.

    Attributes
    ----------
    force : Any
        A nornax ``MutualForceModel``, or a mimirax
        :class:`~mimirax.protocols.ForceModel` (bridged by
        :class:`SingleRungMutualForce`, ``k_max = 0`` only).
    dt : float
        Base time step, nornax's ``dt_max``.
    num_steps : int
        Base steps to take, nornax's ``n_base``. Static.
    k_max : int
        Highest block-step rung. ``0`` is the single-rung reduced case. Static.
    weights_are_masses : bool
        Whether ``params["weights"]`` *is* the dynamical mass vector
        (self-consistent M2M) or only enters the observable (tracer M2M).
    masses : Array | None
        The dynamical masses when :attr:`weights_are_masses` is ``False``;
        ``None`` means unit masses, which is what an external field wants since
        it ignores them.
    eta : float
        nornax's rung-assignment accuracy parameter.
    eps : float
        nornax's rung-assignment softening parameter.
    reassign_rungs : bool
        Whether rungs are recomputed at every base-step boundary (production) or
        frozen for the whole rollout. Freeze them to make the map globally
        smooth in the continuous state, which is what a finite-difference
        gradient check needs.

        **This is not a free choice, and the difference is measured.** With it
        on, the rollout integrates a schedule that depends on the weights
        through the acceleration, and nornax severs that dependence with
        ``stop_gradient``. The gradient is therefore exact for the schedule the
        forward pass *realised* -- reverse and forward mode agree on it to
        2.6e-16 -- but the objective is only piecewise smooth in the weights: a
        weight change large enough to move a particle across a rung boundary
        lands on a different map, with a kink between. On a 16-body,
        three-rung test system the production gradient differs from the
        frozen-schedule one by 65 % in norm (cosine 0.953), because the two
        realise different schedules. Neither is wrong; they are gradients of
        different maps. Freeze the schedule when a fit must descend one smooth
        objective, and expect the kinks when it must not.
    checkpoint : bool
        Whether to wrap each base step in ``jax.checkpoint``, bounding the
        retained backward-pass state to the base-step boundaries. Leave it on:
        it does not change the gradient (measured bit-identical, single- and
        multi-rung) and it is what keeps a long rollout's memory linear in
        ``num_steps``.
    checkpoint_substeps : bool
        Whether to additionally remat each sub-step boundary's kick.
        :attr:`checkpoint` bounds memory *across* base steps but still
        materializes all ``2**k_max`` sub-step pair tensors while
        differentiating one base step; this bounds that to one boundary's
        worth. nornax's rollout documents it as needed for deep ``k_max``
        gradients, which otherwise run out of memory -- so it is the knob a
        made-to-measure fit at FMM scale and large ``k_max`` will need, and it
        is exposed here rather than left reachable only by bypassing the
        adapter. Off by default, as in nornax; it changes the memory schedule,
        not the result.
    record : Callable[[Any], PyTree] | None
        nornax's ``record_fn``, which must return a *mapping* so that
        :meth:`__call__` can add the ``"weights"`` leaf beside it; ``None``
        records positions and velocities.
    rebuild_fn : Callable[..., Any] | None
        A tree backend's traceable ``(positions, masses) -> topology`` rebuild,
        e.g. jaccpot's ``BlockStepFMM.rebuild_state``.
    rebuild_every : int
        Base steps per topology rebuild. Requires :attr:`rebuild_fn`.
    base_index : int
        The global base-step index this rollout starts at, which anchors the
        rebuild cadence when a trajectory is run as chained segments.

    Raises
    ------
    ValueError
        If a mimirax ``ForceModel`` is paired with ``k_max > 0``, or if
        ``num_steps`` is not positive.
    """

    force: Any
    dt: float
    num_steps: int
    k_max: int = 0
    weights_are_masses: bool = True
    masses: Array | None = None
    eta: float = 0.1
    eps: float = 1.0
    reassign_rungs: bool = True
    checkpoint: bool = True
    checkpoint_substeps: bool = False
    record: Callable[[Any], PyTree] | None = None
    rebuild_fn: Callable[..., Any] | None = None
    rebuild_every: int = 1
    base_index: int = 0

    def __post_init__(self) -> None:
        """Reject the two combinations that cannot integrate the right equations.

        Raises
        ------
        ValueError
            If the force needs the single-rung bridge but ``k_max > 0``, or if
            ``num_steps`` is not positive.
        """
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1; got {self.num_steps}")
        if self.k_max > 0 and not _is_mutual(self.force):
            raise ValueError(
                f"k_max={self.k_max} needs a nornax MutualForceModel; "
                f"{type(self.force).__name__} is a mimirax ForceModel, which can "
                "only drive a single-rung (k_max=0) rollout -- see "
                "SingleRungMutualForce"
            )

    @classmethod
    def self_consistent(
        cls, force: Any, dt: float, num_steps: int, *, k_max: int = 0, **kwargs: Any
    ) -> NornaxRollout:
        """Build the self-consistent construction: the weights *are* the masses.

        The case the differentiable N-body stack was built for. Because the
        weights are the masses, the orbits themselves depend on the weights, so
        ``d(objective)/d(weights)`` flows through the force evaluations and
        through every step of the integration -- which is the part no classic
        made-to-measure code computes. ``force`` must be a real nornax
        ``MutualForceModel``; a mimirax ``ForceModel`` is rejected above
        ``k_max = 0``.

        Parameters
        ----------
        force : Any
            A nornax ``MutualForceModel``.
        dt : float
            Base time step.
        num_steps : int
            Base steps to take.
        k_max : int
            Highest block-step rung.
        **kwargs : Any
            Any other attribute of :class:`NornaxRollout`.

        Returns
        -------
        NornaxRollout
            Configured with ``weights_are_masses=True``.
        """
        return cls(
            force=force,
            dt=dt,
            num_steps=num_steps,
            k_max=k_max,
            weights_are_masses=True,
            **kwargs,
        )

    @classmethod
    def tracer(
        cls,
        force: Any,
        dt: float,
        num_steps: int,
        *,
        masses: Array | None = None,
        **kwargs: Any,
    ) -> NornaxRollout:
        """Build the classic tracer construction: the weights stay out of the dynamics.

        The orbits are those of test particles in a potential the weights do not
        set, so the trajectory can be computed once and the weight fit runs on
        the recorded kernel -- which is what makes the *classic* force-of-change
        iteration well defined (see
        :meth:`mimirax.inference.MadeToMeasure.force_of_change`). Single-rung by
        construction: an external field is not antisymmetric, so the block
        schedule's momentum-conserving level split does not apply to it.

        Parameters
        ----------
        force : Any
            A mimirax :class:`~mimirax.protocols.ForceModel` -- an analytic
            potential; :class:`mimirax.testing.SoftenedPointMassField` is the
            double -- or a nornax ``MutualForceModel``.
        dt : float
            Base time step.
        num_steps : int
            Base steps to take.
        masses : Array | None
            The dynamical masses. ``None`` means unit masses; an external field
            ignores them, so that is the right default there and the wrong one
            for a self-gravitating system whose masses are not the weights.
        **kwargs : Any
            Any other attribute of :class:`NornaxRollout`.

        Returns
        -------
        NornaxRollout
            Configured with ``weights_are_masses=False`` and ``k_max=0``.
        """
        return cls(
            force=force,
            dt=dt,
            num_steps=num_steps,
            k_max=0,
            weights_are_masses=False,
            masses=masses,
            **kwargs,
        )

    def mutual_force(self) -> Any:
        """Return the force in the form nornax's integrator consumes.

        Returns
        -------
        Any
            ``force`` itself when it already has ``level_accelerations``, and
            :class:`SingleRungMutualForce` around it otherwise.
        """
        if _is_mutual(self.force):
            return self.force
        return SingleRungMutualForce(self.force)

    def dynamical_masses(self, params: PyTree) -> PerParticle:
        """Return the mass vector the integrator advances the state with.

        Parameters
        ----------
        params : PyTree
            The parameter mapping.

        Returns
        -------
        PerParticle
            ``params["weights"]`` in the self-consistent construction, and
            :attr:`masses` (or ones) in the tracer construction.
        """
        weights = jnp.asarray(params["weights"])
        if self.weights_are_masses:
            return weights
        if self.masses is None:
            return jnp.ones_like(weights)
        return jnp.asarray(self.masses)

    def rollout(self, params: PyTree) -> tuple[Any, PyTree]:
        """Run the rollout and return nornax's own ``(final_state, records)``.

        The escape hatch for anything that needs the integrator's state rather
        than an observable's view of it -- a shooting defect between chained
        segments, an energy diagnostic, a topology's pair counts.

        Parameters
        ----------
        params : PyTree
            A mapping with ``"positions"``, ``"velocities"`` and ``"weights"``.

        Returns
        -------
        tuple[Any, PyTree]
            The final ``BlockStepState`` and the stacked per-base-step records.
        """
        block_kdk_rollout, shooting_node = _leapfrog()
        force = self.mutual_force()
        masses = self.dynamical_masses(params)
        node = shooting_node(
            jnp.asarray(params["positions"]),
            jnp.asarray(params["velocities"]),
            masses,
            force,
            k_max=self.k_max,
            dt_max=self.dt,
            eta=self.eta,
            eps=self.eps,
            base_index=self.base_index,
            rebuild_fn=self.rebuild_fn,
        )
        final, records = block_kdk_rollout(
            node,
            self.dt,
            force,
            k_max=self.k_max,
            n_base=self.num_steps,
            eta=self.eta,
            eps=self.eps,
            checkpoint=self.checkpoint,
            checkpoint_substeps=self.checkpoint_substeps,
            reassign_rungs=self.reassign_rungs,
            rebuild_fn=self.rebuild_fn,
            rebuild_every=self.rebuild_every,
            record_fn=self.record if self.record is not None else _record_kinematics,
        )
        return final, records

    def final_state(self, params: PyTree) -> Any:
        """Return only the final ``BlockStepState``.

        Parameters
        ----------
        params : PyTree
            A mapping with ``"positions"``, ``"velocities"`` and ``"weights"``.

        Returns
        -------
        Any
            nornax's ``BlockStepState`` at the end of the rollout.
        """
        return self.rollout(params)[0]

    def __call__(self, params: PyTree) -> PyTree:
        """Integrate and return the recorded trajectory, weights included.

        Parameters
        ----------
        params : PyTree
            A mapping with ``"positions"`` ``(n, 3)``, ``"velocities"``
            ``(n, 3)`` and ``"weights"`` ``(n,)``.

        Returns
        -------
        PyTree
            The records, plus a ``"weights"`` leaf broadcast to
            ``(num_steps, n)`` so every leaf carries the same leading time axis.
        """
        _, records = self.rollout(params)
        weights = jnp.asarray(params["weights"])
        trajectory = dict(records)
        trajectory["weights"] = jnp.broadcast_to(
            weights, (self.num_steps, *weights.shape)
        )
        return trajectory
