"""The contracts mimirax is written against, and nothing else.

WHY PROTOCOLS. The package's reason to exist is to outlive any single solver:
its core must never import ``jaccpot``, ``nornax`` or ``odisseo``. So the core
is written against *structural* interfaces -- anything with the right methods
satisfies them, whether it is jaccpot's FMM, an ODISSEO rollout, an analytic
potential, or the doubles in :mod:`mimirax.testing`. This is how the ecosystem
already works: ``jaccpot/nornax_adapter.py`` satisfies nornax's
``MutualForceModel`` without importing nornax, and nornax's
``forces/base.py`` is the template these follow.

Every protocol here is ``runtime_checkable`` so that ``isinstance`` (and the
optional ``beartype`` hook) can validate an argument structurally. Optional
capabilities are deliberately **not** protocol members: adding one would make
``isinstance`` reject every implementation that predates it. An implementation
advertises an optional capability by having the attribute; a consumer probes
with ``hasattr`` or ``inspect.signature``, never by widening the protocol.

Differentiability is part of every contract below: unless a docstring says
otherwise, an implementation must be traceable by ``jax.grad`` in its array
arguments, because the core numerical paths differentiate through it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from jax import Array

from mimirax._typing import PerParticle, PyTree, Scalar, Vec3
from mimirax.types import FitResult, SampleResult

__all__ = [
    "Constraint",
    "FoldableObservable",
    "ForceModel",
    "ForwardModel",
    "Likelihood",
    "Observable",
    "Optimizer",
    "Prior",
    "Reparameterization",
    "Sampler",
]


@runtime_checkable
class ForceModel(Protocol):
    """A gravitational (or any pairwise) acceleration provider.

    The smallest interface a solver has to offer for mimirax to build a forward
    model on it: accelerations at ``positions`` due to ``masses``. jaccpot's
    ``FastMultipoleMethod.compute_accelerations(positions, masses)`` satisfies it
    through :mod:`mimirax.adapters.jaccpot`; an analytic potential satisfies it
    directly; :class:`mimirax.testing.DirectSumGravity` is the double.

    Differentiable in ``positions`` and ``masses``. A tree or FMM backend is
    differentiable at *fixed topology* only; where its topology is rebuilt is the
    backend's business and must be documented on the adapter.
    """

    def accelerations(
        self,
        positions: Vec3,
        masses: PerParticle,
        *,
        args: object = None,
    ) -> Vec3:
        """Return the acceleration at every particle.

        Parameters
        ----------
        positions : Vec3
            ``(n, 3)`` particle positions.
        masses : PerParticle
            ``(n,)`` particle masses.
        args : object
            Caller-owned channel for backend options; the core never sets it.

        Returns
        -------
        Vec3
            ``(n, 3)`` accelerations.
        """
        ...


@runtime_checkable
class ForwardModel(Protocol):
    """Parameters in, a physical state (or prediction) out.

    This is the seam between *inference* and *physics*: everything above it
    (observables, likelihoods, priors, samplers) is mimirax's; everything below
    it (forces, integrators) is a solver's. A forward model may be a static map
    (positions -> field at tracers, as in Paper I section 7), a rollout
    (initial conditions -> final state, as in ODISSEO), or the identity.

    Must be differentiable in ``params``.
    """

    def __call__(self, params: PyTree) -> PyTree:
        """Map parameters to a state.

        Parameters
        ----------
        params : PyTree
            The free parameters, in whatever structure the caller chose.

        Returns
        -------
        PyTree
            The predicted state an :class:`Observable` can act on.
        """
        ...


@runtime_checkable
class Observable(Protocol):
    """An observation operator: a state in, a data-space array out.

    Kinematic maps, projections, binned densities and stream-track coordinates
    are all observables. Registered in :data:`mimirax.observables.OBSERVABLES`.
    Must be differentiable in ``state``.
    """

    def __call__(self, state: PyTree) -> Array:
        """Evaluate the observable.

        Parameters
        ----------
        state : PyTree
            The output of a :class:`ForwardModel`.

        Returns
        -------
        Array
            The predicted data, shaped like the corresponding observation.
        """
        ...


@runtime_checkable
class FoldableObservable(Protocol):
    """An :class:`Observable` that can also be accumulated one snapshot at a time.

    WHY THIS EXISTS. A time-averaged observable is a *reduction* over a
    trajectory, and the cheap way to evaluate a reduction is to fold it into the
    integration rather than to stack the trajectory and average afterwards.
    Stacking costs ``O(t * n)`` memory in the **forward** pass alone -- for a
    made-to-measure fit over many dynamical times that is the term that decides
    whether the problem fits on a machine, and it has nothing to do with
    autodiff. Folding costs ``O(m)``.

    So this is a *capability*, not a replacement. It is a separate protocol
    because :class:`Observable` is ``runtime_checkable`` and widening it would
    make ``isinstance`` reject every observable that predates this; and it is
    *narrower* than :class:`Observable` because not every observable is
    foldable. A mean folds. Exponential smoothing folds, through the recursion
    ``state <- a * state + value``. A median does not, and neither does anything
    needing the whole sample at once.

    An implementation must satisfy the **fold law**: for any trajectory,

        result(fold(... fold(initial(s_0), value(s_0)) ..., value(s_{t-1})), t)

    equals ``self(stacked_trajectory)``, the plain :class:`Observable` call. The
    two paths are then interchangeable and a caller can choose on memory
    grounds alone. :class:`mimirax.observables.TimeAverage` satisfies both
    protocols and a test pins the law to round-off.

    ``value`` is separated from ``fold`` on purpose: a rollout can then evaluate
    ``value`` inside its own loop -- where the snapshot already is -- and stack
    only the ``(m,)`` results instead of the ``(n, 3)`` state.
    """

    def value(self, snapshot: PyTree) -> Array:
        """Evaluate the observable at one snapshot.

        Parameters
        ----------
        snapshot : PyTree
            A single time slice, with no leading time axis.

        Returns
        -------
        Array
            ``(m,)`` observable values at that snapshot.
        """
        ...

    def initial(self, snapshot: PyTree) -> PyTree:
        """Return the zero accumulator, shaped from a template snapshot.

        Must not depend on the snapshot's *values*, only on its shapes and
        dtypes, so that it can be built with :func:`jax.eval_shape` and carries
        no gradient.

        Parameters
        ----------
        snapshot : PyTree
            A template time slice.

        Returns
        -------
        PyTree
            The accumulator's identity element.
        """
        ...

    def fold(self, state: PyTree, value: Array) -> PyTree:
        """Accumulate one snapshot's value into the running state.

        Called in trajectory order, so an implementation may weight by
        recency. Must be a fixed-shape pytree map, since it runs as a
        ``lax.scan`` carry.

        Parameters
        ----------
        state : PyTree
            The running accumulator.
        value : Array
            ``(m,)`` result of :meth:`value` at the next snapshot.

        Returns
        -------
        PyTree
            The updated accumulator.
        """
        ...

    def result(self, state: PyTree, num_steps: int) -> Array:
        """Finish the reduction, given how many snapshots went into it.

        Parameters
        ----------
        state : PyTree
            The final accumulator.
        num_steps : int
            How many snapshots were folded. Static, so a normalization may
            depend on it.

        Returns
        -------
        Array
            ``(m,)`` reduced observable, equal to the plain
            :class:`Observable` call on the stacked trajectory.
        """
        ...


@runtime_checkable
class Likelihood(Protocol):
    """Compares a prediction with an observation and returns a log-probability.

    A *misfit* is the negative of this; the convention is fixed to log-densities
    so that likelihoods and priors add. Registered in
    :data:`mimirax.likelihoods.LIKELIHOODS`. Must be differentiable in
    ``predicted``.
    """

    def log_prob(self, predicted: Array, observed: Array) -> Scalar:
        """Return the log-likelihood of ``observed`` given ``predicted``.

        Parameters
        ----------
        predicted : Array
            What an :class:`Observable` produced from the model.
        observed : Array
            The data, same shape as ``predicted``.

        Returns
        -------
        Scalar
            The log-likelihood; additive constants may be dropped, and the
            implementation's docstring must say if they are.
        """
        ...


@runtime_checkable
class Prior(Protocol):
    """A log-density over parameters. Regularizers are priors that say so.

    Registered in :data:`mimirax.priors.PRIORS`. Must be differentiable in
    ``params``.
    """

    def log_prob(self, params: PyTree) -> Scalar:
        """Return the log prior density at ``params``.

        Parameters
        ----------
        params : PyTree
            The free parameters.

        Returns
        -------
        Scalar
            The log density, up to an additive constant.
        """
        ...


@runtime_checkable
class Reparameterization(Protocol):
    """A bijection between an unconstrained and a constrained parameter space.

    Optimizers and samplers work in the unconstrained space; the forward model
    sees the constrained one. Implementations live in
    :mod:`mimirax.parameters`.
    """

    def forward(self, unconstrained: Array) -> Array:
        """Map unconstrained values to the constrained space.

        Parameters
        ----------
        unconstrained : Array
            Values in the unconstrained space.

        Returns
        -------
        Array
            The constrained values, same shape.
        """
        ...

    def inverse(self, constrained: Array) -> Array:
        """Map constrained values back to the unconstrained space.

        Parameters
        ----------
        constrained : Array
            Values in the constrained space.

        Returns
        -------
        Array
            The unconstrained values, same shape.
        """
        ...

    def log_abs_det_jacobian(self, unconstrained: Array) -> Scalar:
        """Return ``log abs det J`` of :meth:`forward` at ``unconstrained``.

        Parameters
        ----------
        unconstrained : Array
            Where to evaluate the Jacobian.

        Returns
        -------
        Scalar
            The summed log absolute Jacobian determinant, the term a density
            picks up under the change of variables.
        """
        ...


@runtime_checkable
class Constraint(Protocol):
    """An equality constraint, expressed as a defect that is zero when satisfied.

    nornax's multiple-shooting ``shooting_defect`` is one; a fixed total mass is
    another. Consumed by :func:`mimirax.parameters.quadratic_penalty`.
    """

    def defect(self, params: PyTree) -> Array:
        """Return the constraint defect at ``params``.

        Parameters
        ----------
        params : PyTree
            The free parameters.

        Returns
        -------
        Array
            Any shape; all zeros means the constraint holds.
        """
        ...


@runtime_checkable
class Optimizer(Protocol):
    """Minimizes a scalar objective over a parameter pytree.

    Registered in :data:`mimirax.inference.METHODS`.
    """

    def minimize(
        self,
        objective: Callable[[PyTree], Scalar],
        params: PyTree,
    ) -> FitResult:
        """Minimize ``objective`` starting from ``params``.

        Parameters
        ----------
        objective : Callable[[PyTree], Scalar]
            A differentiable scalar function of the parameters.
        params : PyTree
            The starting point; its structure is preserved in the result.

        Returns
        -------
        FitResult
            Final parameters and the per-step objective trace.
        """
        ...


@runtime_checkable
class Sampler(Protocol):
    """Draws parameter samples from an unnormalized log-density.

    Registered in :data:`mimirax.inference.METHODS`.
    """

    def sample(
        self,
        log_density: Callable[[PyTree], Scalar],
        params: PyTree,
        *,
        key: Array,
        num_samples: int,
    ) -> SampleResult:
        """Draw ``num_samples`` samples of ``log_density`` starting from ``params``.

        Parameters
        ----------
        log_density : Callable[[PyTree], Scalar]
            The unnormalized log target density.
        params : PyTree
            The initial state of the chain.
        key : Array
            A ``jax.random`` key.
        num_samples : int
            How many samples to return. Static.

        Returns
        -------
        SampleResult
            The samples and their log-densities.
        """
        ...
