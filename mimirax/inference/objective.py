"""Composing a forward model, an observable, a likelihood and priors into one objective.

This is the one place the pieces meet. Everything else in the package is
either a pluggable implementation of one protocol or a method that consumes the
scalar functions defined here.

CONSTRAINTS. Equality constraints enter in two ways. Explicit ones are
:class:`~mimirax.protocols.Constraint` s in :attr:`InferenceProblem.constraints`.
A forward model that computes its own defects in the same pass as its
prediction -- multiple shooting (:class:`mimirax.MultipleShootingForwardModel`)
-- advertises it with a ``predict_with_defects(params) -> (prediction,
defects)`` method (found by ``hasattr``, the convention of
:mod:`mimirax.protocols`), and the problem then uses THOSE defects rather than
running the forward model a second time. :meth:`InferenceProblem.evaluate`
returns the unconstrained objective and every defect from one pass; the
augmented-Lagrangian driver (:mod:`mimirax.inference.augmented_lagrangian`)
consumes it. :meth:`InferenceProblem.log_posterior` adds them as the quadratic
penalty (:func:`~mimirax.parameters.quadratic_penalty`, weight
:attr:`InferenceProblem.penalty_weight`), the baseline any optimizer can use. A
problem with neither runs the code it always ran, bit for bit.

DATA AS ARGUMENTS. :meth:`InferenceProblem.data_arrays` collects what the
objective reads but never differentiates -- the observations, the likelihood's
arrays (a per-point ``sigma``), the forward model's (a multiple-shooting model's
masses and initial condition, its engine's per-segment arrays) -- and
:meth:`InferenceProblem.with_data_arrays` rebuilds the problem around (traced)
replacements. The augmented-Lagrangian driver passes them to its jitted programs
as arguments: closed over, they would be constants embedded in the program.

SEGMENTS. :meth:`InferenceProblem.segment_split` says how the objective of a
multiple-shooting problem splits over the segments (the forward model's
``split_objective`` capability), which
:mod:`mimirax.inference.parallel` needs to run one program per segment.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax.numpy as jnp
from jax import Array

from mimirax._typing import PyTree, Scalar
from mimirax.parameters import FunctionConstraint, quadratic_penalty
from mimirax.protocols import Constraint, ForwardModel, Likelihood, Observable, Prior

__all__ = ["InferenceProblem", "ProblemEvaluation"]


class ProblemEvaluation(NamedTuple):
    """One pass of a problem: the unconstrained objective, the defects, the aux.

    Attributes
    ----------
    negative_log_posterior : Scalar
        ``-(log_likelihood + log_prior)``, WITHOUT any constraint term.
    defects : tuple[Array, ...]
        The forward model's defects first (when it has ``predict_with_defects``),
        then one per entry of :attr:`InferenceProblem.constraints`. Empty for an
        unconstrained problem.
    aux : Any
        The forward model's per-evaluation aux (its ``evaluate(params).auxes``
        when it has one), else ``None``. Not differentiated; read on the host by
        the forward model's optional ``check``.
    """

    negative_log_posterior: Scalar
    defects: tuple[Array, ...]
    aux: Any


@dataclass(frozen=True)
class InferenceProblem:
    """A posterior over parameters, assembled from protocol-satisfying parts.

    ``log_posterior(params) = likelihood.log_prob(observable(forward(params)),
    observed) + sum(prior.log_prob(params))``, minus the quadratic penalty of
    the constraints when there are any (module docstring). Every method is
    differentiable in ``params`` provided the parts are, which the protocols
    require.

    Attributes
    ----------
    forward : ForwardModel
        Parameters to state.
    observable : Observable
        State to data space.
    likelihood : Likelihood
        Data-space comparison.
    observed : Array
        The data.
    priors : tuple[Prior, ...]
        Zero or more priors and regularizers, summed.
    constraints : tuple[Constraint, ...]
        Zero or more equality constraints, beside the forward model's own
        defects (``predict_with_defects``).
    penalty_weight : float
        Weight of the quadratic penalty :meth:`log_posterior` subtracts for the
        defects. The augmented-Lagrangian driver ignores it (it owns its
        penalty).
    """

    forward: ForwardModel
    observable: Observable
    likelihood: Likelihood
    observed: Array
    priors: tuple[Prior, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    penalty_weight: float = 1.0

    @property
    def is_constrained(self) -> bool:
        """Say whether the problem has any defect to satisfy.

        Returns
        -------
        bool
            ``True`` with :attr:`constraints` or a forward model offering
            ``predict_with_defects``.
        """
        return bool(self.constraints) or hasattr(self.forward, "predict_with_defects")

    def evaluate(self, params: PyTree) -> ProblemEvaluation:
        """Return the unconstrained objective and every defect from ONE forward pass.

        The forward model runs once: through its ``evaluate`` (prediction,
        defects and aux) or ``predict_with_defects`` when it has them, else
        through its call. The explicit :attr:`constraints` are evaluated at
        ``params`` beside it.

        Parameters
        ----------
        params : PyTree
            The free parameters.

        Returns
        -------
        ProblemEvaluation
            ``(negative_log_posterior, defects, aux)``.
        """
        aux = None
        defects: tuple[Array, ...] = ()
        evaluate = getattr(self.forward, "evaluate", None)
        if evaluate is not None and hasattr(self.forward, "predict_with_defects"):
            state, forward_defects, aux = evaluate(params)
            defects = (forward_defects,)
        elif hasattr(self.forward, "predict_with_defects"):
            state, forward_defects = getattr(self.forward, "predict_with_defects")(
                params
            )
            defects = (forward_defects,)
        else:
            state = self.forward(params)
        log_likelihood = self.likelihood.log_prob(self.observable(state), self.observed)
        negative = -(log_likelihood + self.log_prior(params))
        defects = defects + tuple(c.defect(params) for c in self.constraints)
        return ProblemEvaluation(
            negative_log_posterior=negative, defects=defects, aux=aux
        )

    def data_arrays(self) -> dict:
        """Return the arrays the objective reads but does not differentiate. Host.

        Returns
        -------
        dict
            ``observed``; ``likelihood`` and ``forward``: their ``data_arrays()``
            when they have the capability, else ``None``.
        """
        likelihood = getattr(self.likelihood, "data_arrays", None)
        forward = getattr(self.forward, "data_arrays", None)
        return {
            "observed": self.observed,
            "likelihood": None if likelihood is None else likelihood(),
            "forward": None if forward is None else forward(),
        }

    def with_data_arrays(self, data: dict) -> InferenceProblem:
        """Return the problem reading ``data`` (e.g. a program's arguments).

        Parameters
        ----------
        data : dict
            As :meth:`data_arrays` returned it, or tracers of it.

        Returns
        -------
        InferenceProblem
            The same problem; its parts with the capability rebuilt around their
            entries.
        """
        likelihood, forward = self.likelihood, self.forward
        if data.get("likelihood") is not None:
            likelihood = getattr(likelihood, "with_data_arrays")(data["likelihood"])
        if data.get("forward") is not None:
            forward = getattr(forward, "with_data_arrays")(data["forward"])
        return dataclasses.replace(
            self, forward=forward, likelihood=likelihood, observed=data["observed"]
        )

    def segment_split(self) -> Any:
        """Return how the objective splits over the forward model's segments.

        Separable: a forward model with the ``split_objective`` capability
        (:meth:`mimirax.MultipleShootingForwardModel.split_objective`) and no
        explicit :attr:`constraints` (a constraint of the whole parameter vector
        couples the segments).

        Returns
        -------
        Any
            The forward model's :class:`~mimirax.shooting.SegmentSplit`.

        Raises
        ------
        ValueError
            If the problem does not split (the message says why).
        """
        if self.constraints:
            raise ValueError(
                "a problem with explicit constraints does not split over segments: "
                "each constraint reads the whole parameter vector"
            )
        split = getattr(self.forward, "split_objective", None)
        if split is None:
            raise ValueError(
                f"{type(self.forward).__name__} has no split_objective: only a "
                "multiple-shooting forward model splits over segments"
            )
        return split(self.observable, self.likelihood, self.observed)

    def penalty(self, defects: tuple[Array, ...]) -> Scalar:
        """Return the quadratic penalty of already-computed defects.

        Parameters
        ----------
        defects : tuple[Array, ...]
            :attr:`ProblemEvaluation.defects`.

        Returns
        -------
        Scalar
            ``penalty_weight * sum(defect^2)`` over every defect, through
            :func:`~mimirax.parameters.quadratic_penalty`.
        """
        total = jnp.zeros(())
        for defect in defects:
            frozen = FunctionConstraint(lambda _params, d=defect: d)
            total = total + quadratic_penalty(frozen, None, self.penalty_weight)
        return total

    def predict(self, params: PyTree) -> Array:
        """Run the forward model and the observable.

        Parameters
        ----------
        params : PyTree
            The free parameters.

        Returns
        -------
        Array
            The prediction in data space.
        """
        return self.observable(self.forward(params))

    def log_likelihood(self, params: PyTree) -> Scalar:
        """Return the log-likelihood of the data at ``params``.

        Parameters
        ----------
        params : PyTree
            The free parameters.

        Returns
        -------
        Scalar
            ``likelihood.log_prob(predict(params), observed)``.
        """
        return self.likelihood.log_prob(self.predict(params), self.observed)

    def log_prior(self, params: PyTree) -> Scalar:
        """Return the summed log prior density at ``params``.

        Parameters
        ----------
        params : PyTree
            The free parameters.

        Returns
        -------
        Scalar
            Zero when there are no priors.
        """
        total = jnp.zeros(())
        for prior in self.priors:
            total = total + prior.log_prob(params)
        return total

    def log_posterior(self, params: PyTree) -> Scalar:
        """Return the unnormalized log posterior at ``params``.

        Parameters
        ----------
        params : PyTree
            The free parameters.

        Returns
        -------
        Scalar
            ``log_likelihood + log_prior``, minus :meth:`penalty` of the
            defects of the same pass for a constrained problem.
        """
        if not self.is_constrained:
            return self.log_likelihood(params) + self.log_prior(params)
        evaluation = self.evaluate(params)
        return -(evaluation.negative_log_posterior + self.penalty(evaluation.defects))

    def negative_log_posterior(self, params: PyTree) -> Scalar:
        """Return ``-log_posterior``, the objective an optimizer minimizes.

        Parameters
        ----------
        params : PyTree
            The free parameters.

        Returns
        -------
        Scalar
            The negative log posterior.
        """
        return -self.log_posterior(params)
