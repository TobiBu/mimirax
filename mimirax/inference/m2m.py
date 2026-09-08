"""Made-to-measure: particle weights adjusted against time-averaged observables.

WHERE THIS LIVES, AND WHY. The EDDA programme's decision D-026 (2026-09-06)
puts the made-to-measure module in mimirax rather than in the integrator
package: its three parts are an orbit integration (nornax's), a force (any
:class:`~mimirax.protocols.ForceModel`), and an observable-residual-driven
weight update with an entropy prior -- and the last is inference. The
integrator enters through :mod:`mimirax.adapters.nornax`, behind the
``mimirax[nornax]`` extra; nothing here imports nornax.

THE OBJECTIVE. Both methods in this module minimize the same scalar,

.. math:: F(w) = \\tfrac{1}{2} \\chi^2\\big(\\bar y(w), Y\\big) - \\mu S(w),

the chi-squared of the **time-averaged** observables against the data plus the
entropy prior of :class:`~mimirax.priors.EntropyPrior`. An
:class:`~mimirax.inference.InferenceProblem` built from
:class:`mimirax.adapters.nornax.NornaxRollout`,
:class:`mimirax.observables.TimeAverage` around
:class:`mimirax.observables.WeightedKernelSum`,
:class:`~mimirax.likelihoods.GaussianLikelihood` and an ``EntropyPrior``
*is* that objective, and ``problem.negative_log_posterior`` is what both
methods below are handed. ``mu`` lives on the prior and has no second copy
here, so the two methods cannot disagree about the regularization they share.

TWO ITERATIONS, ONE STATIONARY POINT. Syer & Tremaine's (1996) force of change
is

.. math:: \\dot w_i = -\\varepsilon\\, w_i \\Big[ \\sum_j \\frac{\\Delta_j K_{ji}}{\\sigma_j} - \\mu \\frac{\\partial S}{\\partial w_i} \\Big]
   = -\\varepsilon\\, w_i \\frac{\\partial F}{\\partial w_i},

so the bracket the classic algorithm assembles by hand is exactly
``dF/dw`` -- which is why this module computes it with :func:`jax.grad` and
does not re-derive it (a test pins the two against each other on the tracer
problem). The two methods then differ only in the metric they descend in:

* :meth:`MadeToMeasure.force_of_change` is the classic update, ``ln w <- ln w -
  eps dF/dw``. That is a *preconditioned* gradient step in ``w`` with
  preconditioner ``diag(w)``, and the multiplicative spelling is what keeps the
  weights positive without a clip -- the historical Euler form ``w <- w (1 -
  eps dF/dw)`` can step straight through zero and differs from this at
  ``O(eps^2)``.
* :meth:`MadeToMeasure.minimize` is the differentiable variant this package
  exists for: plain gradient descent (any optax rule) in ``u = ln w``, i.e. a
  step in ``w`` with preconditioner ``diag(w^2)``, on an objective that is
  differentiated **through the orbit integration**.

Both are stationary exactly where ``dF/dw = 0``, since ``w > 0``. So they must
agree on the answer and are free to disagree on the path -- which is what makes
the classic iteration a usable oracle, and what a measured comparison between
them is testing. When the weights are the masses of a self-gravitating system
the orbits themselves depend on the weights and only the second method's
gradient is the true one; the first is then not merely slower but wrong about
``dF/dw``, which is why :meth:`force_of_change` is documented for the tracer
case.

WHAT IS NOT CLAIMED. Nothing here says a time average over a given window is
long enough, that a chosen ``mu`` is right, or that the method converges at FMM
scale. Those are experiments, and the numbers this module was verified at are
in the EDDA programme's ``reports/M2M_module.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import optax
from jax import Array

from mimirax._typing import PyTree, Scalar
from mimirax.parameters import LogTransform
from mimirax.protocols import Reparameterization
from mimirax.types import FitResult

__all__ = ["MadeToMeasure", "made_to_measure"]


@dataclass(frozen=True)
class MadeToMeasure:
    """The made-to-measure weight-adjustment method, in both of its variants.

    Satisfies :class:`~mimirax.protocols.Optimizer` through :meth:`minimize`;
    :meth:`force_of_change` is the second, classic iteration on the same
    objective and is deliberately *not* the protocol method, so a caller can
    never get the oracle by accident.

    Which leaves are weights is :attr:`keys` (for a mapping ``params``); every
    other leaf is carried through untouched by :meth:`force_of_change` and
    optimized in its own space by :meth:`minimize`. A bare-array ``params`` is
    taken to be the weights themselves.

    Attributes
    ----------
    optimizer : optax.GradientTransformation | None
        The update rule :meth:`minimize` descends with, in the unconstrained
        space; :func:`made_to_measure` builds an Adam one. ``None`` is the
        default and means :meth:`minimize` is unavailable -- no step size is
        guessed on the caller's behalf -- which leaves
        :meth:`force_of_change` usable on its own with :attr:`epsilon`.
    num_steps : int
        Weight-update steps, for either iteration. Static.
    epsilon : float
        The classic force of change's step size, Syer & Tremaine's
        ``epsilon``. Unused by :meth:`minimize`.
    transform : Reparameterization
        The positivity reparameterization applied to the weight leaves.
        :class:`~mimirax.parameters.LogTransform` by default, because gradient
        descent in ``ln w`` is the differentiable counterpart of the classic
        multiplicative update -- see the module docstring.
    keys : tuple[str, ...]
        Which leaves of a mapping ``params`` hold weights.
    refine : optax.GradientTransformation | None
        An optional **second stage**, run from where the first one stopped:
        ``optax.lbfgs()`` is the intended use, and the reason the stage exists
        is that a quasi-Newton rule needs a starting point in the right basin
        more than it needs many steps. ``None`` (the default) runs one stage.
        Whether two stages beat one is a property of the problem and is
        measured, not assumed -- on both of this module's test problems the
        best single-stage fit beats the two-stage one; see
        ``reports/M2M_module.md``.
    refine_steps : int
        Steps for the second stage. Static. Ignored when :attr:`refine` is
        ``None``; a quasi-Newton stage wants tens, not thousands.
    """

    optimizer: optax.GradientTransformation | None = None
    num_steps: int = 100
    epsilon: float = 0.1
    transform: Reparameterization = field(default_factory=LogTransform)
    keys: tuple[str, ...] = ("weights",)
    refine: optax.GradientTransformation | None = None
    refine_steps: int = 0

    def weight_keys(self, params: PyTree) -> tuple[str, ...]:
        """Return which leaves of ``params`` are weights.

        Parameters
        ----------
        params : PyTree
            The parameters.

        Returns
        -------
        tuple[str, ...]
            The subset of :attr:`keys` that ``params`` has, or ``()`` when
            ``params`` is not a mapping and is therefore the weights itself.

        Raises
        ------
        KeyError
            If ``params`` is a mapping and holds none of :attr:`keys`. Silently
            optimizing nothing is the one outcome worse than failing here.
        """
        if not isinstance(params, Mapping):
            return ()
        present = tuple(key for key in self.keys if key in params)
        if not present:
            raise KeyError(
                f"MadeToMeasure(keys={tuple(self.keys)!r}) found no weight leaf; "
                f"params has {tuple(params)}. Pass keys=... to name the weights."
            )
        return present

    def to_unconstrained(self, params: PyTree) -> PyTree:
        """Map the weight leaves into the unconstrained space.

        Parameters
        ----------
        params : PyTree
            Parameters with positive weight leaves.

        Returns
        -------
        PyTree
            The same structure with the weight leaves transformed; a bare array
            is transformed whole.
        """
        keys = self.weight_keys(params)
        if not keys:
            return self.transform.inverse(jnp.asarray(params))
        return {
            key: self.transform.inverse(value) if key in keys else value
            for key, value in params.items()
        }

    def to_constrained(self, unconstrained: PyTree) -> PyTree:
        """Map the weight leaves back into the constrained space.

        Parameters
        ----------
        unconstrained : PyTree
            Parameters in the unconstrained space.

        Returns
        -------
        PyTree
            The same structure with the weight leaves transformed back; a bare
            array is transformed whole.
        """
        keys = self.weight_keys(unconstrained)
        if not keys:
            return self.transform.forward(jnp.asarray(unconstrained))
        return {
            key: self.transform.forward(value) if key in keys else value
            for key, value in unconstrained.items()
        }

    def _descend(
        self,
        rule: optax.GradientTransformation,
        loss: Callable[[PyTree], Scalar],
        start: PyTree,
        num_steps: int,
    ) -> tuple[PyTree, Array]:
        """Run one optax rule for ``num_steps`` in the unconstrained space.

        Supports rules whose ``update`` needs more than a gradient. optax's
        quasi-Newton and line-search rules are
        ``GradientTransformationExtraArgs``: their ``update`` takes ``value``,
        ``grad`` and ``value_fn`` as well, because a line search has to
        *evaluate* the objective at trial points rather than only read its
        gradient. Passing them is what makes ``optax.lbfgs()`` usable here at
        all; without it the call raises a ``TypeError`` naming the three missing
        arguments. They are supplied to every ``ExtraArgs`` rule -- which today
        is all of them, Adam included -- and rules that do not want them ignore
        them.

        When the rule's state already *carries* a cached ``value`` and ``grad``
        (``optax.lbfgs``'s does; Adam's does not),
        :func:`optax.value_and_grad_from_state` reads them instead of
        recomputing. That is not a micro-optimization here: one recomputation
        per step is one extra N-body rollout and its backward pass, which is the
        dominant cost of the whole fit.

        Parameters
        ----------
        rule : optax.GradientTransformation
            The update rule.
        loss : Callable[[PyTree], Scalar]
            The objective, as a function of *unconstrained* parameters.
        start : PyTree
            Starting point, unconstrained.
        num_steps : int
            How many updates to take. Static.

        Returns
        -------
        tuple[PyTree, Array]
            The final unconstrained parameters and the ``(num_steps,)``
            objective trace, recorded before each update.
        """
        opt_state = rule.init(start)
        extra = isinstance(rule, optax.GradientTransformationExtraArgs)
        cached = extra and optax.tree_utils.tree_get(opt_state, "value") is not None
        value_and_grad = (
            optax.value_and_grad_from_state(loss)
            if cached
            else jax.value_and_grad(loss)
        )

        def _step(carry, _):
            current, state = carry
            if cached:
                value, grads = value_and_grad(current, state=state)
            else:
                value, grads = value_and_grad(current)
            kwargs = {"value": value, "grad": grads, "value_fn": loss} if extra else {}
            updates, state = rule.update(grads, state, current, **kwargs)
            current = optax.apply_updates(current, updates)
            return (current, state), value

        (final, _), trace = jax.lax.scan(
            _step, (start, opt_state), None, length=num_steps
        )
        return final, trace

    def minimize(
        self,
        objective: Callable[[PyTree], Scalar],
        params: PyTree,
    ) -> FitResult:
        """Descend ``objective`` in the unconstrained space; the differentiable variant.

        The optimizer moves in ``u``, the objective is always evaluated on the
        *constrained* parameters, and the result comes back constrained and in
        the caller's own structure. The reparameterization's log-Jacobian is
        **not** added: priors here are densities over the constrained
        parameters (see :mod:`mimirax.priors.reference`), so a caller who wants
        a density over ``u`` -- for a sampler, say -- adds
        :func:`mimirax.parameters.log_abs_det_jacobian` to the objective
        itself. Optimizing without it finds the maximum of the constrained
        posterior, which is what a made-to-measure fit is asking for.

        Each stage is one ``lax.scan``, as
        :class:`~mimirax.inference.OptaxOptimizer`'s loop is, so a fit that
        differentiates through an N-body rollout traces once per stage. The
        trace records the objective *before* each update, so
        ``objective_trace[0]`` is the starting value; with a :attr:`refine`
        stage the two traces are concatenated and the trace is
        ``num_steps + refine_steps`` long.

        Any optax rule works, including a **learning-rate schedule**
        (``optax.adam(optax.cosine_decay_schedule(...))``) and the quasi-Newton
        and line-search rules whose ``update`` needs the objective's value and
        the objective itself -- see :meth:`_descend`. Which rule to use is a
        property of the problem and is measured rather than asserted; the
        module's report carries the comparison, and its short version is that
        no rule wins on both of the method's two constructions.

        Parameters
        ----------
        objective : Callable[[PyTree], Scalar]
            The time-averaged negative log posterior, a function of the
            constrained parameters.
        params : PyTree
            The starting point, with positive weight leaves.

        Returns
        -------
        FitResult
            Constrained final parameters in the input's structure, and the
            ``(num_steps,)`` objective trace.

        Raises
        ------
        ValueError
            If :attr:`optimizer` is ``None``; the step size is the caller's
            choice and is not guessed here.
        """
        if self.optimizer is None:
            raise ValueError(
                "MadeToMeasure.minimize needs an optax rule: build one with "
                "made_to_measure(learning_rate=...), or pass optimizer=... . "
                "force_of_change runs without it, on epsilon alone."
            )
        # Bound to a local so the None check above is visible to a type checker
        # inside the scan body, which closes over it.
        rule = self.optimizer

        def loss(unconstrained: PyTree) -> Scalar:
            return objective(self.to_constrained(unconstrained))

        final, trace = self._descend(
            rule, loss, self.to_unconstrained(params), self.num_steps
        )
        if self.refine is not None and self.refine_steps > 0:
            final, refined = self._descend(self.refine, loss, final, self.refine_steps)
            trace = jnp.concatenate([trace, refined])
        return FitResult(params=self.to_constrained(final), objective_trace=trace)

    def force_of_change(
        self,
        objective: Callable[[PyTree], Scalar],
        params: PyTree,
    ) -> FitResult:
        """Run the **classic** Syer & Tremaine force-of-change iteration.

        ``w <- w * exp(-epsilon * dF/dw)``, the weight leaves only; every other
        leaf is returned exactly as it came in. This is the oracle
        :meth:`minimize` is compared against, not a method to prefer: it is
        first-order in one hand-tuned step size, it descends in a fixed metric,
        and -- the substantive point -- it is only the right gradient when the
        orbits do **not** depend on the weights. Use it on the tracer
        construction (:meth:`mimirax.adapters.nornax.NornaxRollout.tracer`).
        On the self-consistent construction ``dF/dw`` here still includes the
        orbits' dependence on the weights, because :func:`jax.grad` cannot be
        told to forget it, so the iteration is *not* the classic algorithm
        there and the classic algorithm is not defined there either.

        Its fixed point is :meth:`minimize`'s: both stop exactly where
        ``dF/dw = 0``. The multiplicative step is what enforces positivity --
        the package's rule is reparameterization, not clipping -- and it is
        identical to a unit-step gradient descent in ``ln w`` scaled by
        ``epsilon``, which is the algebraic reason the two variants share their
        stationary points.

        Parameters
        ----------
        objective : Callable[[PyTree], Scalar]
            The time-averaged negative log posterior, a function of the
            constrained parameters.
        params : PyTree
            The starting point, with positive weight leaves.

        Returns
        -------
        FitResult
            Final parameters in the input's structure, and the
            ``(num_steps,)`` objective trace, recorded before each update.
        """
        keys = self.weight_keys(params)
        value_and_grad = jax.value_and_grad(objective)

        def _update(current, grads):
            if not keys:
                return jnp.asarray(current) * jnp.exp(-self.epsilon * grads)
            return {
                key: (
                    value * jnp.exp(-self.epsilon * grads[key])
                    if key in keys
                    else value
                )
                for key, value in current.items()
            }

        def _step(carry, _):
            value, grads = value_and_grad(carry)
            return _update(carry, grads), value

        final, trace = jax.lax.scan(_step, params, None, length=self.num_steps)
        return FitResult(params=final, objective_trace=trace)


def made_to_measure(
    learning_rate: float | Callable[[Array], Array] = 1.0e-2,
    num_steps: int = 200,
    *,
    epsilon: float = 0.1,
    keys: tuple[str, ...] = ("weights",),
    refine: optax.GradientTransformation | None = None,
    refine_steps: int = 0,
) -> MadeToMeasure:
    """Build a made-to-measure method with an Adam rule, as :func:`~mimirax.adam` does.

    Adam is the default because it is the default everywhere, not because it is
    the best rule for this objective -- on the module's tracer problem plain
    ``optax.sgd(1e-6)`` reaches a gradient norm an order of magnitude smaller,
    and ``optax.lbfgs()`` reaches a lower objective in fifty steps than Adam
    does in ten thousand. Neither transfers to the self-consistent problem,
    where a fixed-step SGD diverges outright. Pass ``optimizer=`` to
    :class:`MadeToMeasure` directly for anything other than Adam, and read the
    measured comparison in ``reports/M2M_module.md`` before choosing.

    Parameters
    ----------
    learning_rate : float | Callable[[Array], Array]
        Adam's step size in the log-weight space, or an ``optax`` schedule --
        ``optax.exponential_decay(0.05, 2000, 0.3)`` measured better on the
        tracer problem than any constant rate tried.
    num_steps : int
        Number of weight updates.
    epsilon : float
        The classic force of change's step size.
    keys : tuple[str, ...]
        Which leaves of a mapping ``params`` hold weights.
    refine : optax.GradientTransformation | None
        An optional second-stage rule, e.g. ``optax.lbfgs()``.
    refine_steps : int
        Steps for the second stage.

    Returns
    -------
    MadeToMeasure
        Ready to :meth:`~MadeToMeasure.minimize` or to
        :meth:`~MadeToMeasure.force_of_change`.
    """
    return MadeToMeasure(
        optimizer=optax.adam(learning_rate),
        num_steps=num_steps,
        epsilon=epsilon,
        keys=keys,
        refine=refine,
        refine_steps=refine_steps,
    )
