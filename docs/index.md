# mimirax

`mimirax` is the inference and reconstruction layer of the MIDGARD stack: it fits, samples and
diagnoses the parameters of differentiable *N*-body forward models. It owns observation
operators, likelihoods, priors, parameter handling and inference methods, and it consumes forces
and integrators through protocols rather than importing any particular solver.

The pipeline is:

```text
parameters (any pytree)
    -> reparameterization / gauge fixing     (parameters)
    -> forward model                          (a ForceModel or rollout behind a protocol)
    -> observable                             (observables: state -> data space)
    -> likelihood + priors                    (likelihoods, priors)
    -> optimizer / sampler                    (inference)
    -> residuals, traces, degeneracies        (diagnostics)
```

## Installation

```bash
pip install -e ".[dev]"      # library + quality tooling; no solver needed
pip install -e ".[docs]"     # to build this documentation
```

Solver glue is behind extras — `mimirax[jaccpot]`, `mimirax[nornax]`, `mimirax[odisseo]` — and
each needs the sibling package installed from GitHub first; see the README.

## Quick start

```python
import jax
import jax.numpy as jnp
import optax

from mimirax import GaussianLikelihood, IdentityObservable, InferenceProblem, OptaxOptimizer
from mimirax.testing import SoftenedPointMassField

tracers = 3.0 * jax.random.normal(jax.random.PRNGKey(0), (12, 3))

def forward(source):
    field = SoftenedPointMassField(source, jnp.ones(1), softening=0.2)
    return field.accelerations(tracers, jnp.ones(12)).reshape(-1)

observed = forward(jnp.asarray([[0.4, -0.3, 0.2]]))
problem = InferenceProblem(forward, IdentityObservable(), GaussianLikelihood(0.01), observed)
fit = OptaxOptimizer(optax.adam(0.02), num_steps=1500).minimize(
    problem.negative_log_posterior, jnp.zeros((1, 3))
)
```

```{toctree}
:maxdepth: 2
:caption: Documentation

concepts
api
```
