# mimirax

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![isort](https://img.shields.io/badge/imports-isort-1674b1.svg)
![pytest](https://img.shields.io/badge/tests-pytest-0a9edc.svg)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)

<p align="center">
  <img src="./mimirax.svg" alt="mimirax Logo" width="420" />
</p>

`mimirax` is the inference and reconstruction layer of the MIDGARD stack: a JAX-native package
that fits, samples and diagnoses the parameters of differentiable *N*-body forward models. It
owns observation operators, likelihoods, priors, parameter handling and inference methods, and
it reaches forces and integrators only through protocols — never by importing a solver.

**Status: scaffold.** The structure, contracts, registries and test doubles are in place and
tested; most inference methods are stubs that fix a name and a signature. See
[Scope and non-scope](#scope-and-non-scope) and `docs/concepts.md` for what is real.

## Why the name

*Mímir* is the being associated with wisdom in Norse myth. *Mímisbrunnr*, Mímir's well, sits
beneath a root of Yggdrasil, and Odin sacrificed an eye for a drink from it. `mimirax` is the
inference layer that draws on everything below it — the fitting well under the tree. The `-ax`
suffix is the house convention marking a JAX package, as in `yggdrax` and `nornax`.

## Scope and non-scope

**In scope:** everything between a forward model's output and a claim about its parameters —
observables, likelihoods and misfits, priors and regularizers, parameter pytrees with
reparameterizations, gauge fixing and constraints, optimizers and samplers, and fit diagnostics.

**Out of scope, by design:** solvers and integrators. The Fast Multipole Method lives in
[`jaccpot`](https://github.com/TobiBu/jaccpot), the tree in
[`yggdrax`](https://github.com/TobiBu/yggdrax), time integration in
[`nornax`](https://github.com/TobiBu/nornax), whole-simulation orchestration in
[ODISSEO](https://github.com/vepe99/Odisseo). `mimirax`'s core imports none of them; the
solver-specific glue is in `mimirax/adapters/`, behind optional extras.

## Features

- Protocols for the seams — `ForceModel`, `ForwardModel`, `Observable`, `Likelihood`, `Prior`,
  `Reparameterization`, `Constraint`, `Optimizer`, `Sampler` — all `runtime_checkable`
- Registries for observables, likelihoods, priors, reparameterizations and inference methods:
  adding one is a class plus a `register(...)` call, no core edits
- Parameter handling: `ravel`, log / softplus / affine reparameterizations with exact
  log-Jacobians, centre-of-mass and momentum gauge fixing, constraint defects and penalties
- `InferenceProblem`: forward model + observable + likelihood + priors -> one differentiable
  log posterior
- `OptaxOptimizer`: any `optax` update rule in a `lax.scan`, jittable end to end
- Diagnostics: residuals, convergence from the objective trace, Fisher information and its
  degenerate directions
- Analytic test doubles with closed-form gradients (`mimirax.testing`): a linear model, a
  softened point-mass field with its tidal tensor, direct-sum self-gravity
- Stubs with fixed signatures for HMC, NUTS, mean-field VI, Nelder–Mead, made-to-measure, the
  entropy prior, and the `nornax` / ODISSEO adapters

## Installation

Install from source:

```bash
pip install -e .
```

Install with development tooling:

```bash
pip install -e ".[dev]"
```

The core needs only JAX and optax and its whole test suite runs on CPU. The solver adapters are
extras, and none of the solvers is on PyPI — install the sibling from GitHub first, then the
extra:

```bash
pip install git+https://github.com/TobiBu/yggdrax.git
pip install git+https://github.com/TobiBu/jaccpot.git
pip install -e ".[jaccpot]"
```

## Quick Start

Recover one softened point mass's position from accelerations sampled at twelve tracers — the
Paper I §7 problem in miniature, on the analytic double:

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
print(fit.params, fit.objective_trace[-1])
```

Swap `forward` for a jaccpot FMM through `mimirax.adapters.jaccpot.JaccpotForceModel` and
nothing above the forward model changes.

## Extending

```python
from mimirax import OBSERVABLES, PRIORS, METHODS

OBSERVABLES.register("kinematic_map", KinematicMap)   # a class with __call__(state) -> Array
PRIORS.register("smoothness", SmoothnessPrior)        # a class with log_prob(params) -> Scalar
METHODS.register("my_sampler", MySampler)             # sample(log_density, params, *, key, num_samples)
```

Each registry refuses a silent overwrite (`overwrite=True` to replace) and lists what exists in
the error for a typo.

## Development

Run quality gates locally:

```bash
black --check .
isort --check-only .
pydoclint --config pyproject.toml mimirax/
flake8 --select=F821,F822 --builtins=n,m,t,k,d .
pyright mimirax
pytest
```

Or run pre-commit hooks:

```bash
pre-commit run --all-files
```

Coverage is enforced in CI via `pytest-cov` (80 %):

```bash
pytest --cov=mimirax --cov-report=term-missing
```

## Documentation

Hosted documentation: **https://tobias-buck.de/mimirax/** (published to GitHub Pages on every
push to `main` by the `Docs` workflow).

To build it locally:

```bash
pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
```

## Runtime Type Checking

Enable package-wide runtime checks (`jaxtyping` + `beartype`) at import time:

```bash
export MIMIRAX_RUNTIME_TYPECHECK=1
```

## Project Structure

- `mimirax/protocols.py`: the contracts everything is written against
- `mimirax/parameters.py`: pytrees, reparameterizations, gauge fixing, constraints
- `mimirax/observables`, `mimirax/likelihoods`, `mimirax/priors`: pluggable registries
- `mimirax/inference`: `InferenceProblem`, the optax optimizer, method stubs including M2M
- `mimirax/adapters`: solver glue behind extras (`jaccpot`, `nornax`, `odisseo`)
- `mimirax/diagnostics`: residuals, convergence, degeneracy
- `mimirax/testing`: analytic doubles with exact gradients
- `tests/unit`, `tests/integration`: the suite, all CPU, no solver required

## CI

GitHub Actions runs:

- formatter and lint checks (`black`, `isort`, `pydoclint`, `flake8` undefined names)
- `pyright` static type check
- unit and integration tests with the coverage threshold, plus a runtime-typecheck smoke
- the Sphinx build; deployment to GitHub Pages on `main`
- release build and PyPI publish on version tags

Workflow files: `.github/workflows/ci.yml`, `docs.yml`, `release.yml`.

## License

MIT.
