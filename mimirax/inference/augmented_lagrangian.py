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
Gauss-Newton and Levenberg-Marquardt are out: they need JVPs, which jaccpot's
checkpointed rollouts (``custom_vjp`` s) refuse.

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
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

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

    def _compile(
        self, problem: InferenceProblem, counter: list[int]
    ) -> tuple[Callable[..., _StepOut], Callable[..., Any]]:
        """Build the jitted inner step and the jitted measurement.

        Parameters
        ----------
        problem : InferenceProblem
            The problem.
        counter : list[int]
            ``counter[0]`` counts the step's traces.

        Returns
        -------
        tuple[Callable[..., _StepOut], Callable[..., Any]]
            ``step(params, opt_state, multipliers, mu)`` and
            ``measure(params) -> (f, violation, defects, aux)``.
        """
        rule = optax.with_extra_args_support(self.optimizer)

        def lagrangian(p, multipliers, mu):
            value, evaluation = augmented_lagrangian_value(problem, p, multipliers, mu)
            return value, evaluation

        value_and_grad = jax.value_and_grad(lagrangian, has_aux=True)

        def step(p, opt_state, multipliers, mu):
            counter[0] += 1  # runs at trace time only
            (value, evaluation), grads = value_and_grad(p, multipliers, mu)
            updates, opt_state = rule.update(
                grads,
                opt_state,
                p,
                value=value,
                grad=grads,
                value_fn=lambda q: lagrangian(q, multipliers, mu)[0],
            )
            return _StepOut(
                params=_strong(optax.apply_updates(p, updates)),
                opt_state=_strong(opt_state),
                lagrangian=value,
                objective=evaluation.negative_log_posterior,
                violation=defect_violation(evaluation.defects),
                aux=evaluation.aux,
            )

        def measure(p):
            evaluation = problem.evaluate(p)
            return (
                evaluation.negative_log_posterior,
                defect_violation(evaluation.defects),
                evaluation.defects,
                evaluation.aux,
            )

        return jax.jit(step), jax.jit(measure)

    def minimize(
        self, problem: InferenceProblem, params: PyTree
    ) -> AugmentedLagrangianResult:
        """Minimize ``problem``'s objective subject to its defects vanishing.

        Parameters
        ----------
        problem : InferenceProblem
            A problem whose forward model offers ``predict_with_defects`` and/or
            that has :attr:`~InferenceProblem.constraints`; without either the
            driver is a plain optax loop.
        params : PyTree
            The starting point.

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
        counter = [0]
        step, measure = self._compile(problem, counter)
        shapes = jax.eval_shape(problem.evaluate, params).defects
        multipliers = tuple(jnp.zeros(s.shape, s.dtype) for s in shapes)
        params = _strong(params)
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
        converged = False

        for _ in range(self.max_outer):
            opt_state = _strong(self.optimizer.init(params))
            mu_array = jnp.asarray(mu, dtype=jnp.result_type(float))
            previous_value: float | None = None
            taken = 0
            for _ in range(self.inner_steps):
                _host_prepare(forward, params)
                out = step(params, opt_state, multipliers, mu_array)
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
                    step, measure = self._compile(problem, counter)
                    out = step(params, opt_state, multipliers, mu_array)
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
            objective, violation, defects, aux = measure(params)
            if not _host_check(forward, aux):
                _host_prepare(forward, params)
                step, measure = self._compile(problem, counter)
                objective, violation, defects, aux = measure(params)
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
            num_traces=counter[0],
            num_redos=redos,
        )
