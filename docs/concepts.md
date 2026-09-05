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

## What is implemented and what is a stub

Implemented and tested: parameters (`ravel`, four reparameterizations, gauge fixing,
constraints), the reference observables, the Gaussian likelihood, the Gaussian and L2 priors,
`InferenceProblem`, `OptaxOptimizer`, the residual / convergence / Fisher diagnostics, and the
three test doubles.

Stubs that fix a name and a signature and raise `NotImplementedError`: `HMC`, `NUTS`,
`MeanFieldVI`, `NelderMead`, `MadeToMeasure`, `EntropyPrior`, `profile_likelihood`, and the
`nornax` and ODISSEO adapters. The made-to-measure module is born here by decision D-026 of the
EDDA programme and is filled in by the Jaccpot-Dynamics I implementation work.
