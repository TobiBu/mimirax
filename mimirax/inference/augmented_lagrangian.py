"""Augmented-Lagrangian minimization of an equality-constrained problem.

WHY THIS SHAPE. Multiple shooting turns a long rollout into segments joined by
free nodes and the equality constraints ``c(p) = 0`` that glue them back (the
matching defects). A quadratic penalty alone needs its weight to go to infinity
to satisfy them, and the problem grows ill-conditioned on the way. The augmented
Lagrangian::

    L(p; lambda, mu) = f(p) + sum lambda . c(p) + (mu / 2) sum |c(p)|^2

reaches ``c = 0`` at a finite ``mu``: an outer loop updates the multipliers
``lambda <- lambda + mu c`` and grows ``mu`` only when the violation stalls;
an inner loop minimizes ``L`` at fixed ``(lambda, mu)`` with a VJP-only optax
rule (Adam by default; ``optax.lbfgs()`` works, its line search included).
Gauss-Newton and Levenberg-Marquardt are not implemented (deferred, S10): the JVPs
they need are now available -- jaccpot's ``ad_mode="forward"`` (main 1bb30620)
gives its rollouts ``custom_jvp`` seams, the segment engines take an ``ad_mode``,
and :class:`~mimirax.inference.parallel.ParallelSegmentObjective` has a
``"jacfwd"`` program for the theta block of the segments' Jacobian -- but the
solver itself (the node block, the damping, the linear solve) is not written.

WHY THE LOOPS ARE IN PYTHON. Both loops run on the host, one jitted optax step
per inner iteration, because a segment engine on jaccpot's fused lane needs the
host around every evaluation: ``prepare`` (per-segment plans, caps, f_b seeds)
before a step and ``check`` (the capacity verdict) after it, re-preparing and
redoing the step when the verdict fails. The step is traced ONCE per plan: the
multipliers and ``mu`` are traced arguments, so a new outer iteration does not
retrace it (:attr:`AugmentedLagrangianResult.num_traces` records it). A failed
check followed by a re-prepare retraces the step on purpose -- a new plan is a
new program -- so an engine's ``prepare`` must not change what an already
traced segment computes unless its ``check`` has just failed.

``f`` is :meth:`InferenceProblem.evaluate`'s unconstrained negative log
posterior and ``c`` its defects, from ONE pass of the forward model;
:attr:`InferenceProblem.penalty_weight` is not used (the driver owns the
penalty).

DATA ARE ARGUMENTS. The problem's data -- the observations, the forward model's
masses and initial condition, a segment engine's per-segment arrays
(:meth:`InferenceProblem.data_arrays`) -- are arguments of the jitted step and
measurement, not constants of them (a constant is embedded in the program: at
``N = 1e5-1e6`` that is the whole data set, in every compiled program).

SEGMENTS IN PARALLEL (``devices=``). By default the step is ONE program: the
whole augmented Lagrangian, its gradient and the optax update. With
:attr:`AugmentedLagrangian.devices` the step of a problem that splits over its
segments (:meth:`InferenceProblem.segment_split`) is one value-and-gradient
program per segment, segment ``k`` on ``devices[k % len(devices)]``, all
dispatched before any is read, then one program on ``devices[0]`` that sums the
contributions and applies the optax update
(:class:`~mimirax.inference.parallel.ParallelSegmentObjective`). The loops, the
``prepare`` / ``check`` / redo protocol and the stopping rules are the same code
in both modes. Two differences: :attr:`AugmentedLagrangianResult.num_traces`
counts the segment programs' traces (one per segment key: equal direct-sum
segments share one), and an optax rule that calls ``value_fn`` (``optax.lbfgs()``
with its default zoom line search) is refused -- a line search needs the whole
objective inside the update program. Adam works; for L-BFGS take
``optax.lbfgs(linesearch=None)`` with :attr:`AugmentedLagrangian.line_search`
``"backtracking"``: an Armijo backtracking on the HOST, each trial point one
parallel forward pass of the segments, its verdict checked (a failed trial is
rejected like a too-long step, and its segments are re-planned at the next
``prepare``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
from jax import Array

from mimirax._typing import PyTree, Scalar
from mimirax.inference.objective import InferenceProblem, ProblemEvaluation
from mimirax.parameters import FunctionConstraint, constraint_violation

__all__ = [
    "AugmentedLagrangian",
    "AugmentedLagrangianResult",
    "AugmentedLagrangianState",
    "augmented_lagrangian_value",
    "defect_violation",
]


class AugmentedLagrangianResult(NamedTuple):
    """What :meth:`AugmentedLagrangian.minimize` returns.

    The first two fields are :class:`~mimirax.types.FitResult` 's, so code that
    reads a fit's ``params`` and ``objective_trace`` reads this one too.

    Attributes
    ----------
    params : PyTree
        The final parameters.
    objective_trace : Array
        ``(k,)`` unconstrained objective ``f`` before every inner step.
    violation_trace : Array
        ``(k,)`` ``max |c|`` before every inner step (absolute).
    lagrangian_trace : Array
        ``(k,)`` augmented Lagrangian before every inner step.
    mu_trace : Array
        ``(k,)`` the penalty ``mu`` of every inner step.
    outer_objective : Array
        ``(j,)`` ``f`` at the end of every outer iteration.
    outer_violation : Array
        ``(j,)`` ``max |c|`` at the end of every outer iteration (absolute).
    outer_mu : Array
        ``(j,)`` the ``mu`` every outer iteration ran with.
    outer_steps : Array
        ``(j,)`` inner steps every outer iteration took.
    multipliers : tuple[Array, ...]
        The final multipliers, one per defect (estimates of the constraint
        forces at the solution).
    converged : bool
        Whether both stopping criteria held.
    num_traces : int
        How many times the inner step was traced (1 unless a failed check
        forced a re-plan).
    num_redos : int
        How many steps were redone after a failed check.
    """

    params: PyTree
    objective_trace: Array
    violation_trace: Array
    lagrangian_trace: Array
    mu_trace: Array
    outer_objective: Array
    outer_violation: Array
    outer_mu: Array
    outer_steps: Array
    multipliers: tuple[Array, ...]
    converged: bool
    num_traces: int
    num_redos: int


class AugmentedLagrangianState(NamedTuple):
    """The driver's whole state at the top of an inner step.

    :meth:`AugmentedLagrangian.minimize` hands one to its ``checkpoint``
    callback before every inner step and takes one as ``resume``: a run resumed
    from a state continues exactly as the run that produced it would have (the
    same steps, bitwise, on the same device and programs). For long fits that
    must outlive a process (a bounded GPU claim, a disconnect).

    Attributes
    ----------
    params : PyTree
        The parameters the next step starts from.
    opt_state : Any
        The inner optimizer's state (re-initialised at every outer iteration).
    multipliers : tuple[Array, ...]
        The multipliers of the running outer iteration.
    mu : float
        Its penalty.
    outer_iteration : int
        Index of the running outer iteration (0-based).
    inner_step : int
        Inner steps already taken in it.
    previous_value : Optional[float]
        ``L`` of its previous inner step (the inner early stop), ``None`` before
        its first.
    previous_violation : float
        The previous outer iteration's violation (``inf`` in the first).
    previous_objective : Optional[float]
        The previous outer iteration's ``f`` (``None`` in the first).
    redos : int
        Steps redone after a failed check so far.
    trace : dict[str, list[float]]
        The inner histories so far (``objective``, ``violation``,
        ``lagrangian``, ``mu``): copies.
    outer : dict[str, list[float]]
        The outer histories so far (``objective``, ``violation``, ``mu``,
        ``steps``): copies.
    """

    params: PyTree
    opt_state: Any
    multipliers: tuple[Array, ...]
    mu: float
    outer_iteration: int
    inner_step: int
    previous_value: Optional[float]
    previous_violation: float
    previous_objective: Optional[float]
    redos: int
    trace: dict[str, list[float]]
    outer: dict[str, list[float]]


def defect_violation(defects: tuple[Array, ...]) -> Scalar:
    """Return ``max |c|`` over every defect, zero when there is none.

    Through :func:`~mimirax.parameters.constraint_violation`, so the number is
    the package's one definition of a violation.

    Parameters
    ----------
    defects : tuple[Array, ...]
        :attr:`~mimirax.inference.objective.ProblemEvaluation.defects`.

    Returns
    -------
    Scalar
        The largest absolute defect component.
    """
    flat = [jnp.ravel(d) for d in defects if d.size]
    if not flat:
        return jnp.zeros(())
    stacked = jnp.concatenate(flat)
    return constraint_violation(FunctionConstraint(lambda _params: stacked), None)


def augmented_lagrangian_value(
    problem: InferenceProblem,
    params: PyTree,
    multipliers: tuple[Array, ...],
    mu: Array | float,
) -> tuple[Scalar, ProblemEvaluation]:
    """Return ``L = f + sum lambda . c + (mu / 2) sum |c|^2`` and the evaluation.

    One pass of the forward model (:meth:`InferenceProblem.evaluate`).

    Parameters
    ----------
    problem : InferenceProblem
        The problem; its defects are the constraints.
    params : PyTree
        Where to evaluate.
    multipliers : tuple[Array, ...]
        One array per defect, shaped like it.
    mu : Array | float
        The penalty parameter.

    Returns
    -------
    tuple[Scalar, ProblemEvaluation]
        The augmented Lagrangian and the evaluation it was built from.

    Raises
    ------
    ValueError
        If the multipliers do not match the defects in number.
    """
    evaluation = problem.evaluate(params)
    if len(multipliers) != len(evaluation.defects):
        raise ValueError(
            f"{len(multipliers)} multipliers for {len(evaluation.defects)} defects"
        )
    value = evaluation.negative_log_posterior
    for lam, defect in zip(multipliers, evaluation.defects):
        value = value + jnp.sum(lam * defect) + 0.5 * mu * jnp.sum(defect * defect)
    return value, evaluation


class _StepOut(NamedTuple):
    """One inner step's outputs; the values are at the step's INPUT parameters.

    Attributes
    ----------
    params : PyTree
        The updated parameters.
    opt_state : Any
        The updated optimizer state.
    lagrangian : Array
        ``L`` before the update.
    objective : Array
        ``f`` before the update.
    violation : Array
        ``max |c|`` before the update.
    aux : Any
        The evaluation's aux, for the host's ``check``.
    """

    params: PyTree
    opt_state: Any
    lagrangian: Array
    objective: Array
    violation: Array
    aux: Any


def _strong(tree: PyTree) -> PyTree:
    """Return ``tree`` with every array leaf strongly typed (same dtype).

    WHY. ``jax.jit`` keys its cache on weak types too, and optax's states flip
    some leaves from weak to strong on their first update (Adam's ``nu`` of a
    scalar parameter, L-BFGS's line-search info), as does a parameter built from
    a Python float. Without this the inner step is retraced on its second call.

    Parameters
    ----------
    tree : PyTree
        Any pytree of arrays (or Python scalars).

    Returns
    -------
    PyTree
        The same values, no leaf weakly typed.
    """
    return jax.tree_util.tree_map(
        lambda leaf: jax.lax.convert_element_type(leaf, jnp.result_type(leaf)), tree
    )


def _host_prepare(forward: Any, params: PyTree) -> None:
    """Call the forward model's ``prepare(params)`` if it has one.

    Parameters
    ----------
    forward : Any
        The problem's forward model.
    params : PyTree
        Concrete parameters.
    """
    prepare = getattr(forward, "prepare", None)
    if prepare is not None:
        prepare(params)


def _host_check(forward: Any, aux: Any) -> bool:
    """Return the forward model's ``check(aux)``, ``True`` without one.

    Parameters
    ----------
    forward : Any
        The problem's forward model.
    aux : Any
        The evaluation's aux.

    Returns
    -------
    bool
        The verdict.
    """
    check = getattr(forward, "check", None)
    return True if check is None else bool(check(aux))


@dataclass(frozen=True)
class AugmentedLagrangian:
    """Augmented-Lagrangian driver: Python outer loop, jitted VJP-only inner steps.

    Outer iteration ``j``: minimize ``L(p; lambda, mu)`` for up to
    :attr:`inner_steps` steps of :attr:`optimizer` (its state re-initialised:
    the objective changed), stopping early when ``L`` changes by less than
    :attr:`inner_rtol` relative; then ``lambda <- lambda + mu c(p)``, and
    ``mu <- min(mu * mu_growth, mu_max)`` when the violation did not shrink below
    :attr:`shrink` times the previous one. Stops when the violation
    ``max |c| / violation_scale <= tol`` AND ``f`` changed by at most
    :attr:`objective_rtol` (relative to ``max(|f|, 1)``) over the outer
    iteration; else after :attr:`max_outer` iterations.

    Attributes
    ----------
    optimizer : optax.GradientTransformation
        The inner rule; Adam (``1e-2``) by default. Rules that take ``value`` /
        ``grad`` / ``value_fn`` (``optax.lbfgs()``) get them.
    inner_steps : int
        Maximum inner steps per outer iteration.
    max_outer : int
        Maximum outer iterations.
    tol : float
        Violation tolerance, relative to :attr:`violation_scale`.
    violation_scale : float
        What the violation is measured against (e.g. the state's magnitude).
    objective_rtol : float
        Outer stagnation tolerance on ``f``.
    inner_rtol : float
        Inner early stop on the relative change of ``L``; ``0`` never stops
        early.
    mu0 : float
        Initial penalty.
    mu_growth : float
        Penalty growth factor.
    mu_max : float
        Penalty cap.
    shrink : float
        The violation must fall below ``shrink`` times the previous outer
        iteration's, or ``mu`` grows.
    max_redo : int
        Re-prepares allowed per step after a failed ``check`` before giving up.
    devices : Optional[Sequence[Any]]
        ``None`` (default): the step is one program (session 1's). A sequence of
        devices: one program per segment, segment ``k`` on ``devices[k %
        len(devices)]``, the parameters, the optimizer state and the sum on
        ``devices[0]`` (module docstring). The problem must split over its
        segments, or :meth:`minimize` raises.
    line_search : Optional[str]
        ``None`` (default): the optimizer's update is the step. ``"backtracking"``
        (with :attr:`devices` only): the update is a DIRECTION, scaled on the host
        by Armijo backtracking (``L(p + a d) <= L(p) + c1 a g.d``, ``a`` halved
        from 1 up to :attr:`line_search_steps` times; no step when none passes),
        every trial a parallel forward pass. For ``optax.lbfgs(linesearch=None)``.
    line_search_steps : int
        Maximum trial points per step.
    line_search_c1 : float
        Armijo's sufficient-decrease constant.
    """

    optimizer: optax.GradientTransformation = field(
        default_factory=lambda: optax.adam(1.0e-2)
    )
    inner_steps: int = 200
    max_outer: int = 20
    tol: float = 1.0e-8
    violation_scale: float = 1.0
    objective_rtol: float = 1.0e-6
    inner_rtol: float = 1.0e-12
    mu0: float = 1.0
    mu_growth: float = 10.0
    mu_max: float = 1.0e12
    shrink: float = 0.25
    max_redo: int = 3
    devices: Optional[Sequence[Any]] = None
    line_search: Optional[str] = None
    line_search_steps: int = 20
    line_search_c1: float = 1.0e-4

    def _compile(
        self, problem: InferenceProblem, counter: list[int]
    ) -> tuple[Callable[..., _StepOut], Callable[..., Any]]:
        """Build the jitted inner step and the jitted measurement (one program each).

        The problem's data (:meth:`InferenceProblem.data_arrays`) are the last
        argument of both.

        Parameters
        ----------
        problem : InferenceProblem
            The problem.
        counter : list[int]
            ``counter[0]`` counts the step's traces.

        Returns
        -------
        tuple[Callable[..., _StepOut], Callable[..., Any]]
            ``step(params, opt_state, multipliers, mu, data)`` and
            ``measure(params, data) -> (f, violation, defects, aux)``.
        """
        rule = optax.with_extra_args_support(self.optimizer)

        def lagrangian(p, multipliers, mu, data):
            bound = problem.with_data_arrays(data)
            value, evaluation = augmented_lagrangian_value(bound, p, multipliers, mu)
            return value, evaluation

        value_and_grad = jax.value_and_grad(lagrangian, has_aux=True)

        def step(p, opt_state, multipliers, mu, data):
            counter[0] += 1  # runs at trace time only
            (value, evaluation), grads = value_and_grad(p, multipliers, mu, data)
            updates, opt_state = rule.update(
                grads,
                opt_state,
                p,
                value=value,
                grad=grads,
                value_fn=lambda q: lagrangian(q, multipliers, mu, data)[0],
            )
            return _StepOut(
                params=_strong(optax.apply_updates(p, updates)),
                opt_state=_strong(opt_state),
                lagrangian=value,
                objective=evaluation.negative_log_posterior,
                violation=defect_violation(evaluation.defects),
                aux=evaluation.aux,
            )

        def measure(p, data):
            evaluation = problem.with_data_arrays(data).evaluate(p)
            return (
                evaluation.negative_log_posterior,
                defect_violation(evaluation.defects),
                evaluation.defects,
                evaluation.aux,
            )

        return jax.jit(step), jax.jit(measure)

    def _evaluator(self, problem: InferenceProblem) -> Any:
        """Return the step / measurement pair for this driver's mode.

        Parameters
        ----------
        problem : InferenceProblem
            The problem.

        Returns
        -------
        Any
            A :class:`_OneProgram` (``devices=None``) or a :class:`_PerSegment`.

        Raises
        ------
        ValueError
            For an unknown :attr:`line_search`, or one without :attr:`devices`.
        """
        if self.line_search not in (None, "backtracking"):
            raise ValueError(
                f"line_search must be None or 'backtracking', got {self.line_search!r}"
            )
        if self.devices is None:
            if self.line_search is not None:
                raise ValueError(
                    "line_search runs in the per-segment mode only (devices=...); in "
                    "one program use the optax rule's own (optax.lbfgs())"
                )
            return _OneProgram(self, problem)
        return _PerSegment(self, problem, self.devices)

    def minimize(
        self,
        problem: InferenceProblem,
        params: PyTree,
        *,
        checkpoint: Optional[Callable[[AugmentedLagrangianState], None]] = None,
        resume: Optional[AugmentedLagrangianState] = None,
    ) -> AugmentedLagrangianResult:
        """Minimize ``problem``'s objective subject to its defects vanishing.

        Parameters
        ----------
        problem : InferenceProblem
            A problem whose forward model offers ``predict_with_defects`` and/or
            that has :attr:`~InferenceProblem.constraints`; without either the
            driver is a plain optax loop.
        params : PyTree
            The starting point (ignored when ``resume`` is given).
        checkpoint : Optional[Callable[[AugmentedLagrangianState], None]]
            Called on the host before every inner step with the driver's whole
            state (opt-in; ``None``: nothing is called). It may save the state,
            or raise to stop the fit there.
        resume : Optional[AugmentedLagrangianState]
            Continue from a state a ``checkpoint`` call received (its arrays may
            have been round-tripped through numpy) instead of starting at
            ``params``.

        Returns
        -------
        AugmentedLagrangianResult
            The final parameters, the inner and outer histories, the
            multipliers.

        Raises
        ------
        RuntimeError
            If a step's ``check`` still fails after :attr:`max_redo`
            re-prepares.
        """
        forward = problem.forward
        evaluator = self._evaluator(problem)
        if resume is None:
            multipliers = tuple(
                jnp.zeros(s.shape, s.dtype) for s in evaluator.defect_shapes(params)
            )
            params = evaluator.place(_strong(params))
            multipliers = evaluator.place(multipliers)
            mu = float(self.mu0)
            redos = 0
            trace: dict[str, list[float]] = {
                "objective": [],
                "violation": [],
                "lagrangian": [],
                "mu": [],
            }
            outer: dict[str, list[float]] = {
                "objective": [],
                "violation": [],
                "mu": [],
                "steps": [],
            }
            previous_violation = math.inf
            previous_objective: float | None = None
            first_outer = 0
        else:
            params = evaluator.place(_strong(resume.params))
            multipliers = evaluator.place(_strong(tuple(resume.multipliers)))
            mu = float(resume.mu)
            redos = int(resume.redos)
            trace = {k: list(v) for k, v in resume.trace.items()}
            outer = {k: list(v) for k, v in resume.outer.items()}
            previous_violation = float(resume.previous_violation)
            previous_objective = resume.previous_objective
            first_outer = int(resume.outer_iteration)
        converged = False

        for outer_iteration in range(first_outer, self.max_outer):
            if resume is not None and outer_iteration == first_outer:
                opt_state = evaluator.place(_strong(resume.opt_state))
                previous_value: float | None = resume.previous_value
                taken = int(resume.inner_step)
            else:
                opt_state = _strong(self.optimizer.init(params))
                previous_value = None
                taken = 0
            mu_array = jnp.asarray(mu, dtype=jnp.result_type(float))
            while taken < self.inner_steps:
                if checkpoint is not None:
                    checkpoint(
                        AugmentedLagrangianState(
                            params=params,
                            opt_state=opt_state,
                            multipliers=multipliers,
                            mu=mu,
                            outer_iteration=outer_iteration,
                            inner_step=taken,
                            previous_value=previous_value,
                            previous_violation=previous_violation,
                            previous_objective=previous_objective,
                            redos=redos,
                            trace={k: list(v) for k, v in trace.items()},
                            outer={k: list(v) for k, v in outer.items()},
                        )
                    )
                _host_prepare(forward, params)
                out = evaluator.step(params, opt_state, multipliers, mu_array)
                attempts = 0
                while not _host_check(forward, out.aux):
                    attempts += 1
                    redos += 1
                    if attempts > self.max_redo:
                        raise RuntimeError(
                            f"the forward model's check failed {attempts} times on "
                            "one step; re-preparing did not fix it"
                        )
                    _host_prepare(forward, params)
                    evaluator.refresh()
                    out = evaluator.step(params, opt_state, multipliers, mu_array)
                out = evaluator.accept(out)
                value = float(out.lagrangian)
                trace["objective"].append(float(out.objective))
                trace["violation"].append(float(out.violation))
                trace["lagrangian"].append(value)
                trace["mu"].append(mu)
                params, opt_state = out.params, out.opt_state
                taken += 1
                if previous_value is not None and abs(
                    value - previous_value
                ) <= self.inner_rtol * max(abs(value), 1.0):
                    break
                previous_value = value

            _host_prepare(forward, params)
            objective, violation, defects, aux = evaluator.measure(params)
            if not _host_check(forward, aux):
                _host_prepare(forward, params)
                evaluator.refresh()
                objective, violation, defects, aux = evaluator.measure(params)
            objective, violation = float(objective), float(violation)
            outer["objective"].append(objective)
            outer["violation"].append(violation)
            outer["mu"].append(mu)
            outer["steps"].append(taken)

            multipliers = _strong(
                tuple(lam + mu * d for lam, d in zip(multipliers, defects))
            )
            stagnant = previous_objective is not None and abs(
                objective - previous_objective
            ) <= self.objective_rtol * max(abs(objective), 1.0)
            if violation <= self.tol * self.violation_scale and stagnant:
                converged = True
                break
            if violation > self.shrink * previous_violation:
                mu = min(mu * self.mu_growth, self.mu_max)
            previous_violation = violation
            previous_objective = objective

        return AugmentedLagrangianResult(
            params=params,
            objective_trace=jnp.asarray(trace["objective"]),
            violation_trace=jnp.asarray(trace["violation"]),
            lagrangian_trace=jnp.asarray(trace["lagrangian"]),
            mu_trace=jnp.asarray(trace["mu"]),
            outer_objective=jnp.asarray(outer["objective"]),
            outer_violation=jnp.asarray(outer["violation"]),
            outer_mu=jnp.asarray(outer["mu"]),
            outer_steps=jnp.asarray(outer["steps"], dtype=jnp.int32),
            multipliers=multipliers,
            converged=converged,
            num_traces=evaluator.num_traces,
            num_redos=redos,
        )


class _OneProgram:
    """The step and the measurement as one program each (session 1's mode).

    Parameters
    ----------
    driver : AugmentedLagrangian
        The driver (its optimizer).
    problem : InferenceProblem
        The problem.
    """

    def __init__(self, driver: AugmentedLagrangian, problem: InferenceProblem) -> None:
        self.driver = driver
        self.problem = problem
        self._counter = [0]
        self._step, self._measure = driver._compile(problem, self._counter)

    @property
    def num_traces(self) -> int:
        """Return how many times the step was traced.

        Returns
        -------
        int
            The count.
        """
        return self._counter[0]

    def defect_shapes(self, params: PyTree) -> tuple:
        """Return the shapes of the defects (traces the problem once, abstractly).

        Parameters
        ----------
        params : PyTree
            The parameters.

        Returns
        -------
        tuple
            One ``ShapeDtypeStruct`` per defect.
        """
        return tuple(jax.eval_shape(self.problem.evaluate, params).defects)

    def place(self, tree: PyTree) -> PyTree:
        """Return ``tree`` unchanged (the default device).

        Parameters
        ----------
        tree : PyTree
            Parameters or multipliers.

        Returns
        -------
        PyTree
            ``tree``.
        """
        return tree

    def refresh(self) -> None:
        """Build the programs again: they are traced again at their next call."""
        self._step, self._measure = self.driver._compile(self.problem, self._counter)

    def step(self, params: PyTree, opt_state: Any, multipliers: Any, mu: Any) -> Any:
        """Run one inner step.

        Parameters
        ----------
        params : PyTree
            The parameters.
        opt_state : Any
            The optimizer state.
        multipliers : Any
            The multipliers.
        mu : Any
            The penalty.

        Returns
        -------
        Any
            A ``_StepOut``.
        """
        data = self.problem.data_arrays()
        return self._step(params, opt_state, multipliers, mu, data)

    def accept(self, out: Any) -> Any:
        """Return the checked step unchanged (the optimizer's update is the step).

        Parameters
        ----------
        out : Any
            The step's ``_StepOut``.

        Returns
        -------
        Any
            ``out``.
        """
        return out

    def measure(self, params: PyTree) -> tuple:
        """Return ``(f, violation, defects, aux)`` at ``params``.

        Parameters
        ----------
        params : PyTree
            The parameters.

        Returns
        -------
        tuple
            The measurement.
        """
        return self._measure(params, self.problem.data_arrays())


def _no_value_fn(_params: PyTree) -> Scalar:
    """Refuse a line search's ``value_fn`` in the per-segment mode (traced).

    Parameters
    ----------
    _params : PyTree
        Unused.

    Returns
    -------
    Scalar
        Never returns.

    Raises
    ------
    NotImplementedError
        Always: the update program runs on one device and does not hold the
        segments.
    """
    raise NotImplementedError(
        "AugmentedLagrangian(devices=...) runs each segment in its own program, so "
        "the optax update cannot evaluate the objective at trial points (value_fn): "
        "use a rule without a line search (Adam, optax.lbfgs(linesearch=None))"
    )


class _PerSegment:
    """The step as one program per segment plus one update program (``devices``).

    Parameters
    ----------
    driver : AugmentedLagrangian
        The driver (its optimizer).
    problem : InferenceProblem
        A problem that splits over its segments.
    devices : Sequence[Any]
        The devices.
    """

    def __init__(
        self,
        driver: AugmentedLagrangian,
        problem: InferenceProblem,
        devices: Sequence[Any],
    ) -> None:
        from mimirax.inference.parallel import ParallelSegmentObjective

        self.objective = ParallelSegmentObjective(problem, devices)
        self.forward = problem.forward
        self.line_search = driver.line_search
        self.line_search_steps = int(driver.line_search_steps)
        self.line_search_c1 = float(driver.line_search_c1)
        #: the last step's base point, direction and slope (line search)
        self._pending: Any = None
        #: trial points evaluated and rejected by the line search
        self.trials = 0
        self.rejected = 0
        rule = optax.with_extra_args_support(driver.optimizer)
        objective = self.objective

        def update(p, opt_state, contributions):
            value, grads, f, defects = objective.assemble(p, contributions)
            updates, opt_state = rule.update(
                grads, opt_state, p, value=value, grad=grads, value_fn=_no_value_fn
            )
            slope = sum(
                jnp.sum(g * u)
                for g, u in zip(
                    jax.tree_util.tree_leaves(grads), jax.tree_util.tree_leaves(updates)
                )
            )
            return (
                _StepOut(
                    params=_strong(optax.apply_updates(p, updates)),
                    opt_state=_strong(opt_state),
                    lagrangian=value,
                    objective=f,
                    violation=defect_violation((defects,)),
                    aux=None,
                ),
                updates,
                slope,
            )

        def measure(p, contributions):
            f, defects = objective.assemble_forward(p, contributions)
            return f, defect_violation((defects,)), (defects,)

        def trial(p, updates, alpha):
            scaled = jax.tree_util.tree_map(lambda u: alpha * u, updates)
            return _strong(optax.apply_updates(p, scaled))

        def lagrangian_at(p, contributions, multipliers, mu):
            f, defects = objective.assemble_forward(p, contributions)
            value = f
            for lam, defect in zip(multipliers, (defects,)):
                value = (
                    value + jnp.sum(lam * defect) + 0.5 * mu * jnp.sum(defect * defect)
                )
            return value

        self._update = jax.jit(update)
        self._measure = jax.jit(measure)
        self._trial = jax.jit(trial)
        self._lagrangian_at = jax.jit(lagrangian_at)

    @property
    def num_traces(self) -> int:
        """Return how many times a segment's value-and-gradient program was traced.

        Returns
        -------
        int
            The count.
        """
        return self.objective.num_traces

    def defect_shapes(self, params: PyTree) -> tuple:
        """Return the shapes of the defects, without tracing a segment.

        Parameters
        ----------
        params : PyTree
            The parameters.

        Returns
        -------
        tuple
            ``([S-1, N, 6],)``.
        """
        return self.objective.defect_shapes(params)

    def place(self, tree: PyTree) -> PyTree:
        """Return ``tree`` on the home device.

        Parameters
        ----------
        tree : PyTree
            Parameters or multipliers.

        Returns
        -------
        PyTree
            ``tree`` on ``devices[0]``.
        """
        return jax.device_put(tree, self.objective.home)

    def refresh(self) -> None:
        """Forget the segment programs an engine without keys may have changed."""
        self.objective.refresh()

    def step(self, params: PyTree, opt_state: Any, multipliers: Any, mu: Any) -> Any:
        """Dispatch every segment, then the update on the home device.

        Parameters
        ----------
        params : PyTree
            The parameters.
        opt_state : Any
            The optimizer state.
        multipliers : Any
            The multipliers.
        mu : Any
            The penalty.

        Returns
        -------
        Any
            A ``_StepOut`` whose ``aux`` is the ``S`` segment auxes (each on its
            segment's device).
        """
        contributions, auxes = self.objective.dispatch(params, multipliers, mu)
        out, updates, slope = self._update(params, opt_state, contributions)
        self._pending = (params, updates, slope, multipliers, mu)
        return out._replace(aux=auxes)

    def accept(self, out: Any) -> Any:
        """Scale the checked step by the host line search, when there is one.

        Armijo backtracking from ``a = 1`` (:attr:`AugmentedLagrangian.line_search`):
        each trial point is one parallel forward pass of the segments, checked by
        the forward model's ``check``; a failed check rejects the trial (the
        engine re-plans those segments at the next ``prepare``). With no trial
        accepted the parameters stay where they were.

        Parameters
        ----------
        out : Any
            The step's ``_StepOut`` (its verdict checked).

        Returns
        -------
        Any
            ``out`` with the accepted parameters.
        """
        if self.line_search is None:
            return out
        params, updates, slope, multipliers, mu = self._pending
        value, slope = float(out.lagrangian), float(slope)
        alpha = 1.0
        for _ in range(self.line_search_steps):
            candidate = self._trial(params, updates, alpha)
            contributions, auxes = self.objective.dispatch_forward(candidate)
            trial_value = float(
                self._lagrangian_at(candidate, contributions, multipliers, mu)
            )
            self.trials += 1
            if (
                _host_check(self.forward, auxes)
                and math.isfinite(trial_value)
                and trial_value <= value + self.line_search_c1 * alpha * slope
            ):
                return out._replace(params=candidate)
            self.rejected += 1
            alpha *= 0.5
        return out._replace(params=params)

    def measure(self, params: PyTree) -> tuple:
        """Return ``(f, violation, defects, aux)`` at ``params``.

        Parameters
        ----------
        params : PyTree
            The parameters.

        Returns
        -------
        tuple
            The measurement; ``aux`` the segment auxes.
        """
        contributions, auxes = self.objective.dispatch_forward(params)
        f, violation, defects = self._measure(params, contributions)
        return f, violation, defects, auxes
