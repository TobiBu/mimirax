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
N-body stack was built for and what no classic made-to-measure code computes.

**Two iterations, one stationary point.** `MadeToMeasure.minimize` is the differentiable variant:
any `optax` rule descending in `ln w`, differentiated through the rollout. `MadeToMeasure.
force_of_change` is Syer & Tremaine's classic update `w <- w exp(-eps dF/dw)`, kept as the oracle
the first is measured against. The bracket the classic algorithm assembles by hand is exactly
`dF/dw`, so both use the same autodiff gradient and both are stationary where it vanishes; they
differ only in the metric they descend in. A test pins the autodiff gradient against the
hand-written bracket, and another measures the two iterations against each other.

**What the module does not claim.** That a given time-average window is long enough, that a given
`mu` is well chosen, or that any of this converges at FMM scale. Those are Jaccpot-Dynamics I's
experiments. What *is* measured lives in the tests' docstrings and in the module's report:
gradients through the rollout against finite differences with the measured `|AD - FD|` versus `h`
curve quoted, recovery residuals, and the degeneracy of each test problem read off the Fisher
information rather than tuned away. On the recovery problems the fit is essentially perfect and
the weights are *not* recovered, because ten observables cannot determine sixty-four weights —
that is the reported result, and the diagnostics that say so ship with it.

## What is implemented and what is a stub

Implemented and tested: parameters (`ravel`, four reparameterizations, gauge fixing,
constraints), the reference observables, the made-to-measure observables (`WeightedKernelSum`,
`TimeAverage`, `GaussianRadialBins`), the Gaussian likelihood, the Gaussian, L2 and entropy
priors, `InferenceProblem`, `OptaxOptimizer`, `MadeToMeasure`, the residual / convergence /
Fisher diagnostics, the `nornax` rollout adapter, and the three test doubles.

Stubs that fix a name and a signature and raise `NotImplementedError`: `HMC`, `NUTS`,
`MeanFieldVI`, `NelderMead`, `profile_likelihood`, and the ODISSEO adapter.
