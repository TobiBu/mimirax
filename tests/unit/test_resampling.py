"""Bovy, Kawata & Hunt's Algorithm 1: exact where it claims to be, measured where not.

The method's justification is a theorem about linear-Gaussian models, so the
first test is against the theorem's own answer -- the analytic posterior
covariance `(K^T S^-1 K)^-1` -- and not against a reference implementation or a
tolerance. The remaining tests measure the two places the derivation does not
reach, because a sampler with a stated domain is worth more than one whose
failure mode has to be rediscovered.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

from mimirax import (
    METHODS,
    DataResampling,
    EntropyPrior,
    GaussianLikelihood,
    IdentityObservable,
    IdentityTransform,
    InferenceProblem,
    MadeToMeasure,
    effective_parameters,
    fisher_information,
)
from mimirax.testing import LinearForwardModel

DIMENSION = 6
DATA_POINTS = 20
SIGMA = 0.3


def _linear_problem(key, *, mu: float = 0.0):
    """A linear-Gaussian problem, optionally with an entropy prior of strength mu."""
    k_matrix, k_truth, k_noise = jax.random.split(key, 3)
    matrix = jax.random.normal(k_matrix, (DATA_POINTS, DIMENSION))
    truth = 1.0 + 0.3 * jax.random.normal(k_truth, (DIMENSION,))
    forward = LinearForwardModel(matrix=matrix, offset=jnp.zeros(DATA_POINTS))
    observed = forward(truth) + SIGMA * jax.random.normal(k_noise, (DATA_POINTS,))
    priors = (EntropyPrior(mu=mu, key=None),) if mu else ()
    problem = InferenceProblem(
        forward=forward,
        observable=IdentityObservable(),
        likelihood=GaussianLikelihood(sigma=SIGMA),
        observed=observed,
        priors=priors,
    )
    return problem, matrix, observed


def _sampler(sequential: bool = False) -> DataResampling:
    """Algorithm 1 over an unconstrained quasi-Newton fit, as the derivation assumes."""
    return DataResampling(
        optimizer=MadeToMeasure(
            optimizer=optax.lbfgs(), num_steps=80, transform=IdentityTransform()
        ),
        sequential=sequential,
    )


def test_it_is_registered_and_is_not_a_sampler() -> None:
    """It is in METHODS, and it deliberately does not claim the Sampler protocol.

    :class:`~mimirax.protocols.Sampler` takes a scalar ``log_density``, and this
    method cannot work from one -- it has to reach inside the posterior to find
    the data and perturb it. Accepting a ``log_density`` and ignoring it would
    be worse than not claiming the protocol, so ``sample`` takes the
    ``InferenceProblem``.
    """
    assert METHODS.get("data_resampling") is DataResampling
    assert not hasattr(DataResampling(optimizer=MadeToMeasure()), "log_density")


def test_the_sample_covariance_is_the_analytic_posterior(key) -> None:
    """Algorithm 1 is *exact* for a linear model with Gaussian noise and no prior.

    The theorem: for ``Y = K W + delta``, ``delta ~ N(0, S)``, the posterior
    under a uniform prior is Gaussian with mean ``V K^T S^-1 Y`` and covariance
    ``V = (K^T S^-1 K)^-1``. Drawing ``Y~ ~ N(Y, S)`` and refitting gives a
    draw from exactly that, because the fit is a linear transform of a Gaussian
    with the right mean and variance.

    So this test compares the sample covariance with ``V`` computed in closed
    form -- no reference implementation, no tolerance to argue about beyond
    Monte Carlo error. Measured ``|C - V| / |V|``: 0.229 at 200 samples, 0.085
    at 1000, 0.047 at 4000, 0.028 at 20000 -- the ``1/sqrt(K)`` rate, which is
    itself the evidence that it is converging to ``V`` and not to something
    nearby. The mean lands within 1.3e-03 of ``V K^T S^-1 Y`` at 4000.
    """
    problem, matrix, observed = _linear_problem(key)
    precision = matrix.T @ matrix / SIGMA**2
    covariance = jnp.linalg.inv(precision)
    mean = covariance @ (matrix.T @ observed / SIGMA**2)

    sampler = _sampler()
    start = jnp.ones(DIMENSION)
    few = sampler.sample(
        problem, start, key=jax.random.fold_in(key, 1), num_samples=250
    )
    many = sampler.sample(
        problem, start, key=jax.random.fold_in(key, 1), num_samples=4000
    )

    def error(result):
        got = sampler.covariance(result.samples, key=None)
        return float(jnp.linalg.norm(got - covariance) / jnp.linalg.norm(covariance))

    assert error(many) < 0.10, f"|C - V|/|V| = {error(many):.4f}"
    # And it is converging, not merely close: four times the samples, half the error.
    assert error(many) < 0.6 * error(few)

    drawn_mean = jnp.mean(many.samples, axis=0)
    assert float(jnp.linalg.norm(drawn_mean - mean) / jnp.linalg.norm(mean)) < 1.0e-2
    assert many.samples.shape == (4000, DIMENSION)
    assert many.log_density.shape == (4000,)


def test_a_prior_makes_it_too_narrow_and_effective_parameters_says_by_how_much(
    key,
) -> None:
    """The first caveat, quantified: resampling the data does not resample the prior.

    Bovy et al. state the limitation -- the method "does not properly deal with
    particle weights for which the penalty term ... has a significant effect" --
    and this measures it. Comparing the sampled covariance's trace with the
    Laplace approximation at the mode:

    ==========  =========================  ======================
    ``mu``      tr(sampled) / tr(Laplace)  ``effective_parameters``
    ==========  =========================  ======================
    1e-3        0.9984                     6.0000 of 6
    1e-1        0.9975                     5.9962
    1           0.9899                     5.9623
    10          0.9253                     5.6558
    100         0.6200                     3.9256
    1000        0.1717                     1.0675
    ==========  =========================  ======================

    **The two columns track each other**: the ratio is close to
    ``effective_parameters / d`` throughout, which makes sense -- the sampler
    captures the data's contribution to the variance and nothing else, so the
    fraction it gets right is the fraction of the parameters the data determine.

    That gives a usable rule rather than a warning: run
    :func:`~mimirax.effective_parameters` first, and trust the spread to about
    ``effective_parameters / d``. On this package's made-to-measure tracer
    problem that ratio is 9.9999/64, so the sampled spread there would be
    **wrong by a factor of six** -- which is exactly why the number is worth
    computing before the sampler is run.
    """
    sampler = _sampler()
    start = jnp.ones(DIMENSION)
    ratios = {}
    fractions = {}
    for mu in (1.0e-3, 1.0e3):
        problem, _, _ = _linear_problem(key, mu=mu)
        result = sampler.sample(
            problem, start, key=jax.random.fold_in(key, 1), num_samples=1500
        )
        sampled = sampler.covariance(result.samples, key=None)
        fit = sampler.optimizer.minimize(problem.negative_log_posterior, start)
        laplace = jnp.linalg.inv(
            fisher_information(problem.negative_log_posterior, fit.params)
        )
        ratios[mu] = float(jnp.trace(sampled) / jnp.trace(laplace))
        data_curvature = fisher_information(
            lambda w: -problem.log_likelihood(w), fit.params
        )
        prior_curvature = fisher_information(
            lambda w: -problem.log_prior(w), fit.params
        )
        fractions[mu] = (
            float(effective_parameters(data_curvature, prior_curvature)) / DIMENSION
        )

    # A negligible prior: the sampler is right.
    assert ratios[1.0e-3] > 0.97, ratios[1.0e-3]
    assert fractions[1.0e-3] > 0.99
    # A dominant prior: badly too narrow, and effective_parameters predicts it.
    assert ratios[1.0e3] < 0.30, ratios[1.0e3]
    assert fractions[1.0e3] < 0.30
    assert abs(ratios[1.0e3] - fractions[1.0e3]) < 0.15, (
        f"the rule of thumb missed: ratio {ratios[1.0e3]:.3f} vs "
        f"effective fraction {fractions[1.0e3]:.3f}"
    )


def test_positivity_is_the_second_caveat_and_it_only_bites_where_it_binds(
    key,
) -> None:
    """The derivation assumes unconstrained weights. Measured: it matters at the boundary.

    Bovy et al.: "the algorithm above is based on the assumption that there is
    no constraint on the sign of each of the weight parameters". mimirax's
    default :class:`~mimirax.parameters.LogTransform` is such a constraint. The
    measurement is more specific than the caveat, and in one place worse:

    ==============  ===================  ==============  =====================
    truth, sigma    negative draws       tr/V free       tr/V log-transformed
    ==============  ===================  ==============  =====================
    ``w=1.0, 0.3``  0.00 %               1.0122          1.0122 (identical)
    ``w=0.30, 0.3`` 0.18 %               1.0122          1.0097
    ``w=0.20, 0.3`` 4.0 %                1.0122          **2e+20**
    ``w=0.15, 0.3`` 11.5 %               1.0764          **1e+27, or nan**
    ==============  ===================  ==============  =====================

    Three things, in order of how surprising they are.

    **The parameterization is not the problem; the constraint is.** Where no
    draw wants a negative weight the two samplers agree to *round-off*, because
    each sample is an ``argmin`` and an ``argmin`` does not depend on the
    coordinates it was found in. So a log transform costs nothing until the
    boundary is reached.

    **Where it binds mildly it is a mild bias**, as the caveat says: 0.18 % of
    draws pinned at zero narrows the trace by 0.25 %.

    **Where it binds appreciably the log-parameterised sampler is not
    approximate, it is unusable.** A constrained optimum at ``w = 0`` is
    ``log w -> -inf``, and the covariance inflates by twenty orders of
    magnitude. Bovy et al. describe this case as one the method "does not
    properly deal with"; the honest operational version is that neither variant
    is usable there -- the unconstrained one stays well behaved and returns
    **negative weights**, which are not a distribution function.

    So the diagnostic to run is the fraction of negative draws from the
    unconstrained sampler. Non-negligible, and the answer is a method that
    handles the boundary, not a choice of transform.
    """
    matrix = jax.random.normal(jax.random.fold_in(key, 0), (DATA_POINTS, DIMENSION))
    forward = LinearForwardModel(matrix=matrix, offset=jnp.zeros(DATA_POINTS))
    analytic = jnp.linalg.inv(matrix.T @ matrix / SIGMA**2)
    unconstrained = _sampler()
    positive = DataResampling(
        optimizer=MadeToMeasure(optimizer=optax.lbfgs(), num_steps=120),
        sequential=False,
    )

    def draw(level):
        truth = jnp.full(DIMENSION, level)
        observed = forward(truth) + SIGMA * jax.random.normal(
            jax.random.fold_in(key, 2), (DATA_POINTS,)
        )
        problem = InferenceProblem(
            forward=forward,
            observable=IdentityObservable(),
            likelihood=GaussianLikelihood(sigma=SIGMA),
            observed=observed,
        )
        shared = dict(key=jax.random.fold_in(key, 3), num_samples=2000)
        free = unconstrained.sample(problem, truth, **shared)
        held = positive.sample(problem, truth, **shared)
        return {
            "negative_fraction": float(jnp.mean(free.samples < 0.0)),
            "free_trace": float(
                jnp.trace(unconstrained.covariance(free.samples, key=None))
                / jnp.trace(analytic)
            ),
            "positive_trace": float(
                jnp.trace(positive.covariance(held.samples, key=None))
                / jnp.trace(analytic)
            ),
            "positive_min": float(jnp.min(held.samples)),
        }

    far = draw(1.0)
    assert far["negative_fraction"] == 0.0
    # No draw reaches the boundary, so the transform is irrelevant: an argmin does
    # not care which coordinates found it.
    assert far["positive_trace"] == pytest.approx(far["free_trace"], rel=1e-9)
    assert abs(far["free_trace"] - 1.0) < 0.05

    near = draw(0.15)
    # The unconstrained sampler returns weights that are not a distribution
    # function, which is why the transform is there at all.
    assert near["negative_fraction"] > 0.05
    # And with the transform the covariance is not approximate, it is nonsense.
    # Asserted as "not a usable number" rather than a magnitude, because the
    # overflow lands on a huge value or on nan depending on the draw count.
    assert near["positive_min"] >= 0.0
    assert not 0.5 < near["positive_trace"] < 2.0, near["positive_trace"]


def test_a_non_gaussian_likelihood_is_refused(key) -> None:
    """No `sigma`, nothing to perturb by, so it raises instead of guessing."""

    class Cauchy:
        """A likelihood with no width attribute."""

        def log_prob(self, predicted, observed):
            """Return a Cauchy log-density.

            Parameters
            ----------
            predicted : Array
                Model prediction.
            observed : Array
                The data.

            Returns
            -------
            Array
                The log density.
            """
            return -jnp.sum(jnp.log1p((predicted - observed) ** 2))

    problem, _, _ = _linear_problem(key)
    problem = InferenceProblem(
        forward=problem.forward,
        observable=problem.observable,
        likelihood=Cauchy(),
        observed=problem.observed,
    )
    with pytest.raises(TypeError, match="Gaussian uncertainties"):
        _sampler().sample(problem, jnp.ones(DIMENSION), key=key, num_samples=4)


def test_sequential_and_vectorised_draws_agree(key) -> None:
    """`lax.map` and `vmap` are a memory trade, so they must give the same draws."""
    problem, _, _ = _linear_problem(key)
    start = jnp.ones(DIMENSION)
    kwargs = dict(key=jax.random.fold_in(key, 3), num_samples=32)
    one_at_a_time = _sampler(sequential=True).sample(problem, start, **kwargs)
    all_at_once = _sampler(sequential=False).sample(problem, start, **kwargs)
    assert jnp.allclose(one_at_a_time.samples, all_at_once.samples, atol=1e-10)
    assert jnp.allclose(one_at_a_time.log_density, all_at_once.log_density, atol=1e-10)
