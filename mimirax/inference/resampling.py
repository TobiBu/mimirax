"""Uncertainties on made-to-measure weights, by resampling the data.

THE METHOD, AND WHOSE IT IS. This is Algorithm 1 of Bovy, Kawata & Hunt (2018,
`arXiv:1704.03884 <https://arxiv.org/abs/1704.03884>`_), the paper that added
uncertainty quantification to made-to-measure. Their observation is that for a
**linear-Gaussian** model ``Y = K W + delta`` with ``delta ~ N(0, S)``, the
posterior over ``W`` under a *uniform* prior is Gaussian with mean
``V K^T S^-1 Y`` and covariance ``V = (K^T S^-1 K)^-1``, so instead of
computing that covariance one can *sample* it::

    Y~  ~  N(Y, S)
    W~  =  argmin  chi^2(model(W), Y~)

Each ``W~`` is an exact draw from the posterior, because it is a linear
transform of a Gaussian with the right mean and the right covariance. No
Metropolis acceptance, no chain to converge, no step size -- and one
optimization per sample, all of them independent and therefore trivially
parallel.

WHY IT SUITS THIS PROBLEM. Bovy et al. give the reasons and they are the same
reasons a general-purpose sampler is the wrong tool here: the weight space is
high-dimensional (one dimension per particle), the posterior at any given
snapshot is *noisy*, and the weights' uncertainties are strongly correlated.
mimirax's own :class:`~mimirax.inference.HMC` and
:class:`~mimirax.inference.NUTS` would meet all three problems; this method
meets none of them, because it never proposes a step in weight space at all.

WHERE IT IS EXACT, AND WHERE IT IS NOT. This matters more here than in the
original paper, and the module measures it rather than repeating it:

* **Exact** for a linear model with Gaussian noise and no prior. The
  made-to-measure observable *is* linear in the weights
  (:class:`mimirax.observables.WeightedKernelSum`), and in the tracer
  construction the orbits do not depend on them, so the whole objective is
  quadratic and this is the posterior. A test compares the sample covariance
  with ``(K^T S^-1 K)^-1`` directly.
* **Approximate with a prior.** Bovy et al. say so: the method "does not
  properly deal with particle weights for which the penalty term ... has a
  significant effect". Resampling the data does not resample the prior, so the
  draws are too narrow along every direction the prior holds up -- which, on
  this package's own tracer problem, is 54 of 64 directions. Read
  :func:`mimirax.effective_parameters` before trusting a spread from this.
* **Approximate under the positivity constraint**, for the same reason: the
  derivation assumes ``W`` is unconstrained, and a weight pinned near zero has
  a one-sided posterior a Gaussian draw cannot represent.
* **Not a posterior at all** in the self-consistent construction, where the
  weights move the orbits and the model is no longer linear in them.

None of that makes it useless -- it makes it a method with a stated domain,
which is better than a chain whose convergence nobody checked. The two
approximations are quantified in ``tests/unit/test_resampling.py`` and in the
EDDA programme's ``reports/M2M_module.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
from jax import Array

from mimirax._typing import PyTree
from mimirax.inference.objective import InferenceProblem
from mimirax.protocols import Optimizer
from mimirax.types import SampleResult

__all__ = ["DataResampling"]


@dataclass(frozen=True)
class DataResampling:
    """Sample the weight posterior by refitting perturbed data. Algorithm 1.

    Deliberately **not** a :class:`~mimirax.protocols.Sampler`. That protocol
    takes a scalar ``log_density``, and this method cannot work from one: it has
    to reach *inside* the posterior to find the data and perturb it. Claiming
    the protocol would mean accepting a ``log_density`` and ignoring it, so
    :meth:`sample` takes the :class:`~mimirax.inference.InferenceProblem`
    instead and the type system says what the method needs.

    Attributes
    ----------
    optimizer : Optimizer
        What to run per sample. A :class:`~mimirax.inference.MadeToMeasure` is
        the intended choice; its ``num_steps`` multiplies the sample count, so
        this is where the cost lives.
    sequential : bool
        Whether to draw the samples one at a time (:func:`jax.lax.map`, the
        default) or all at once (:func:`jax.vmap`). Each sample is an
        independent optimization, so ``vmap`` is embarrassingly parallel and
        ``num_samples`` times the memory -- which for a fit that differentiates
        through an N-body rollout is the wrong trade by default. Turn it on for
        cheap forward models.
    """

    optimizer: Optimizer
    sequential: bool = True

    def sample(
        self,
        problem: InferenceProblem,
        params: PyTree,
        *,
        key: Array,
        num_samples: int,
    ) -> SampleResult:
        """Draw ``num_samples`` weight vectors by refitting resampled data.

        Each sample perturbs the observations by their own uncertainty,
        ``Y~ ~ N(Y, S)``, refits from ``params``, and keeps the fitted
        parameters. The returned ``log_density`` is the **original** problem's
        log posterior at each sample -- not the perturbed one's -- so it can be
        compared across samples and used to weight them.

        The starting point is ``params`` for every sample, not the previous
        sample: the draws are independent by construction and chaining them
        would introduce a correlation the derivation does not have.

        Parameters
        ----------
        problem : InferenceProblem
            The posterior to sample. Its ``observed``
            is perturbed and its ``likelihood`` must carry a ``sigma``, since
            that is the ``S`` the perturbation is drawn from.
        params : PyTree
            The starting point for every refit -- normally the best fit.
        key : Array
            A ``jax.random`` key.
        num_samples : int
            How many samples to draw. Static.

        Returns
        -------
        SampleResult
            ``samples`` with a leading ``num_samples`` axis on every leaf, and
            ``log_density`` the original posterior at each.

        Raises
        ------
        TypeError
            If the likelihood has no ``sigma``. The algorithm is defined for
            Gaussian uncertainties and there is nothing to perturb by
            otherwise; a caller with a general likelihood needs a general
            sampler.
        """
        sigma = getattr(problem.likelihood, "sigma", None)
        if sigma is None:
            raise TypeError(
                "DataResampling needs a likelihood with a `sigma`: the algorithm "
                "draws perturbed data from N(observed, sigma^2) and is defined "
                "for Gaussian uncertainties only"
            )
        observed = jnp.asarray(problem.observed)
        scale = jnp.asarray(sigma, dtype=observed.dtype)

        def one(sample_key: Array) -> tuple[PyTree, Array]:
            perturbed = observed + scale * jax.random.normal(
                sample_key, observed.shape, dtype=observed.dtype
            )
            resampled = replace(problem, observed=perturbed)
            fit = self.optimizer.minimize(resampled.negative_log_posterior, params)
            return fit.params, problem.log_posterior(fit.params)

        keys = jax.random.split(key, num_samples)
        if self.sequential:
            samples, log_density = jax.lax.map(one, keys)
        else:
            samples, log_density = jax.vmap(one)(keys)
        # `asarray` for the type checker: a newer jax infers the mapped output as
        # `ArrayLike | Any` and `SampleResult.log_density` promises an `Array`.
        return SampleResult(samples=samples, log_density=jnp.asarray(log_density))

    def covariance(self, samples: PyTree, key: str | None = "weights") -> Array:
        """Return the sample covariance of one leaf, the estimate of ``V``.

        Parameters
        ----------
        samples : PyTree
            The ``samples`` field of a :class:`~mimirax.types.SampleResult`.
        key : str | None
            Which leaf to take; ``None`` treats ``samples`` as the array itself.

        Returns
        -------
        Array
            ``(d, d)`` covariance over the ``num_samples`` axis, with the
            ``num_samples - 1`` denominator. Compare it with
            ``(K^T S^-1 K)^-1`` where the model is linear and no prior is
            active; expect it to be *too narrow* otherwise, and by how much the
            prior contributes.
        """
        drawn = samples if key is None else samples[key]
        return jnp.cov(jnp.asarray(drawn), rowvar=False, ddof=1)
