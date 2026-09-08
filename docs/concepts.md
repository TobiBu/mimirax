# Concepts

## Why the name

*Mímir* is the being associated with wisdom in Norse myth. *Mímisbrunnr*, Mímir's well, sits
beneath a root of Yggdrasil, and Odin sacrificed an eye for a drink from it. `mimirax` is the
inference layer that draws on everything below it — the fitting well under the tree. The `-ax`
suffix is the house convention marking a JAX package.

## Scope and non-scope

**In scope.** Everything between a forward model's output and a scientific claim about its
parameters: observation operators, likelihoods and misfits, priors and regularizers, parameter
pytrees with reparameterizations, gauge fixing and constraints, optimizers and samplers, and the
diagnostics that say whether a fit converged and which directions the data never saw.

**Out of scope, by design.** Forces, trees and integrators. The FMM lives in `jaccpot`, the tree
in `yggdrax`, time integration in `nornax`, orchestration of a full simulation in ODISSEO.
`mimirax`'s core imports none of them. It works against the protocols in
`mimirax.protocols`, and solver-specific glue lives in `mimirax.adapters` behind optional extras.

## Depend on protocols, not on solvers

The package's reason to exist is to outlive any single solver. Every core numerical path is
written against a structural interface — `ForceModel`, `ForwardModel`, `Observable`,
`Likelihood`, `Prior`, `Reparameterization`, `Constraint`, `Optimizer`, `Sampler` — and is tested
on analytic doubles (`mimirax.testing`) whose gradients are known in closed form. The doubles
run on CPU with nothing installed but JAX and optax.

An adapter (`mimirax/adapters/*.py`) imports its solver and satisfies a protocol. The dependency
direction is the ecosystem's rule: the adapter imports the solver; the solver never imports
`mimirax`; the core imports neither.

## Extensibility is the primary design axis

Observables, likelihoods, priors, reparameterizations and inference methods are registries:

```python
from mimirax import OBSERVABLES

class KinematicMap:
    def __call__(self, state):
        ...

OBSERVABLES.register("kinematic_map", KinematicMap)
```

Adding a new observable or a new sampler is one class satisfying the protocol and one
`register(...)` call. No core module is edited.

## Made to measure

`mimirax` exists because made-to-measure is inference, not integration (decision D-026 of the
EDDA programme). The method fits particle *weights* so that observables **averaged along the
orbits** match data, regularized by an entropy prior. Every quantity it fits is linear in the
weights,

```
y_j = sum_i w_i K_j(z_i),
```

which is `WeightedKernelSum`; the time average is `TimeAverage`; the entropy prior is
`EntropyPrior`; the orbits come from `mimirax.adapters.nornax.NornaxRollout`. Nothing new was
added to the seams to make this fit — an `InferenceProblem` over those four parts *is* the
made-to-measure objective, `chi²` of the time-averaged observables plus `mu S`.

**Two constructions, and the difference matters.** In the *tracer* construction
(`NornaxRollout.tracer`) the weights do not enter the dynamics: the orbits are those of test
particles in a potential the weights do not set, which is classic made-to-measure. In the
*self-consistent* construction (`NornaxRollout.self_consistent`) the weights **are** the masses,
so the orbits themselves depend on what is being fitted — and the gradient of the objective runs
through every force evaluation of the integration. That second case is what the differentiable
N-body stack was built for. The term it adds is **68 %** of the gradient's norm after a fraction
of a dynamical time, measured; the classic force-of-change bracket cannot contain it, because it
assumes the orbits are fixed. Prior work computes the analogous term for a handful of
*external-potential* parameters by finite differencing the orbit integration (Bovy, Kawata & Hunt
2018), which costs one integration per parameter and is entirely practical for one or two; it is
not, when the parameters are the *N* particle masses.

**Two iterations, one stationary point.** `MadeToMeasure.minimize` is the differentiable variant:
any `optax` rule descending in `ln w`, differentiated through the rollout. `MadeToMeasure.
force_of_change` is Syer & Tremaine's classic update `w <- w exp(-eps dF/dw)`, kept as the oracle
the first is measured against. The bracket the classic algorithm assembles by hand is exactly
`dF/dw`, so both use the same autodiff gradient and both are stationary where it vanishes; they
differ only in the metric they descend in. A test pins the autodiff gradient against the
hand-written bracket, and another measures the two iterations against each other.

**What the weights are.** Not stars. A made-to-measure particle is a Monte Carlo sample of a
distribution function, and its weight is the mass carried on that orbit — so the method fits an
*orbit distribution*, and `params["weights"]` is that. In the tracer construction the weights are
the tracer population's only; in the self-consistent one they are simultaneously the orbit
occupation and the source of the gravity, which is what "self-consistent" means and also what the
construction cannot yet separate (a luminous weight and a dynamical mass fitted independently is
not expressible — see the module's report).

**What is fitted.** The weights, and nothing else unless asked: `MadeToMeasure.also_fit` opts into
the initial conditions. That default is the method's definition, not a convenience.

**How informative the objective is, as a number.** `effective_parameters(data_curvature,
prior_curvature)` returns `tr(H_data (H_data + H_prior)^-1)`, the effective degrees of freedom the
*data* determined. On the module's tracer problem — 64 weights, 10 observables — it returns
**9.9999**, while the fit reaches a `chi²` of `1.4e-8`. Ten numbers in, ten degrees of freedom
out; the other 54 are the prior's. No optimizer, step size or `mu` moves that, so "make the
objective more informative" means more or better observables and nothing else.

**The conditioning is the practical problem.** The objective is *strictly convex* in the weights,
so the minimum is unique — but the Fisher condition number is 2.6e7, and every Adam rate and
schedule tried plateaus 2 % short of it while LBFGS reaches it and the classic force of change
reaches it to `3e-8`. At this conditioning the optimizer choice is not a speed question, so
`MadeToMeasure` requires one rather than picking.

**What the module does not claim.** That a given time-average window is long enough, that a given
`mu` is well chosen, or that any of this converges at FMM scale. Those are Jaccpot-Dynamics I's
experiments. What *is* measured lives in the tests' docstrings and in the module's report:
gradients through the rollout against a closed form with no autodiff in it, against forward-mode
autodiff, and against finite differences with the `|AD - FD|` versus `h` curves quoted; recovery
residuals; and the degeneracy of each test problem read off the Fisher information rather than
tuned away.

## What is implemented and what is a stub

Implemented and tested: parameters (`ravel`, four reparameterizations, gauge fixing,
constraints), the reference observables, the made-to-measure observables (`WeightedKernelSum`,
`TimeAverage`, `GaussianRadialBins`), the Gaussian likelihood, the Gaussian, L2 and entropy
priors, `InferenceProblem`, `OptaxOptimizer`, `MadeToMeasure`, the residual / convergence /
Fisher diagnostics, the `nornax` rollout adapter, and the three test doubles.

Stubs that fix a name and a signature and raise `NotImplementedError`: `HMC`, `NUTS`,
`MeanFieldVI`, `NelderMead`, `profile_likelihood`, and the ODISSEO adapter.
