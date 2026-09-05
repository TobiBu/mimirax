# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The core has no solver dependency and the whole test suite runs on CPU. For the optional
solver adapters, install the sibling package first — none of them is on PyPI:

```bash
pip install -e ../jaccpot --no-deps   # or: pip install git+https://github.com/TobiBu/jaccpot.git
pip install -e ../nornax
pip install -e ".[jaccpot,nornax]"
```

## Local quality checks

Run these before opening a pull request:

```bash
black --check .
isort --check-only .
pydoclint --config pyproject.toml mimirax/
flake8 --select=F821,F822 --builtins=n,m,t,k,d .
pyright mimirax
pytest
```

`black`, `isort` and `pydoclint` are pinned in `pyproject.toml` to exactly the
`.pre-commit-config.yaml` hook revs. Keep them equal: a floating `>=` lets a local run and the
hook disagree.

## Pre-commit

Install and enable hooks once:

```bash
pip install pre-commit
pre-commit install
```

Run all hooks on demand:

```bash
pre-commit run --all-files
```

## Testing and coverage

CI enforces coverage through `pytest-cov` (80 %). Every core numerical path must run on the
analytic test doubles in `mimirax/testing/` — a test that needs a GPU or an installed solver
will not be run often enough to be useful.

```bash
pytest --cov=mimirax --cov-report=term-missing
```

Runtime type checks (`jaxtyping` + `beartype`) are available via import-hook
instrumentation. To enable during debugging:

```bash
export MIMIRAX_RUNTIME_TYPECHECK=1
```

## Extending

Observables, likelihoods, priors and inference methods are registries
(`mimirax.observables.OBSERVABLES`, `.likelihoods.LIKELIHOODS`, `.priors.PRIORS`,
`.inference.METHODS`). A new one is a class satisfying the protocol in `mimirax/protocols.py`
plus one `register(...)` call — no core module needs editing. Solver glue goes in
`mimirax/adapters/`, behind an optional extra, and imports the solver; the core never does.

## Pull requests

- Keep changes focused and scoped.
- Include tests for behaviour changes; numerical tests run in float64 against the doubles.
- Update README and docs when user-facing APIs change.
