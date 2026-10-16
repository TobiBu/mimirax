# studies

Scripts that **measure something and draw it**. Not library code, not tests.

A study answers a question that wants a figure or a table rather than an assertion:
"how much does this observable actually constrain?", "where does this method stop
working?". Each one is self-contained, takes no arguments, fixes its seed, writes
its numbers to `results/*.json` and renders from that JSON — never from values
retyped into a plotting call.

`studies/` is deliberately **outside the `mimirax` package**: it is excluded from
the wheel (`[tool.setuptools.packages.find] include = ["mimirax*"]`), from pyright
(`include = ["mimirax"]`) and from coverage (`source = ["mimirax"]`), and it is the
only place in this repository that imports `matplotlib`. It is still formatted and
documented to the same standard — black, isort, pydoclint and flake8 run over it
like everything else.

Install what a study needs with the `studies` extra:

```bash
pip install -e ".[studies]"
```

## The rule these follow

**The script is the deliverable; the image is its output.** A figure whose numbers
cannot be regenerated is a figure that cannot be defended, and figures outlive the
sessions that make them — they end up on slides, in grants, in other people's
talks. So: one script, one seed, one JSON, and every annotation formatted from that
JSON.

## What is here

| Study | Question | Outputs |
|---|---|---|
| `sensitivity_figure.py` | What do the data actually constrain, and which datum says so? | `results/sensitivity_figure.json`, `sensitivity_information_{light,dark}.{pdf,png}`, `sensitivity_information_map_{light,dark}.{pdf,png}` |
| `fisher_fd_vs_ad.py` | Does a finite-difference Fisher matrix overstate what the data constrain? | `results/fisher_fd_vs_ad.json`, `fisher_fd_vs_ad.{pdf,png}` |

### `sensitivity_figure.py`

Draws the made-to-measure tracer problem of
`tests/integration/test_m2m_rollout.py::test_the_weights_are_recovered_when_the_observables_determine_them`
two ways: the eigenvalue spectrum the observables determine with the prior's
threshold across it, and the per-pixel Fisher information over the LOSVD grid.

Two things worth knowing before editing it:

- **`eigvalsh` is wrong here.** On the radial problem the data curvature is
  numerically indefinite and `eigvalsh` returns 10 negative eigenvalues (min
  −1.05e-10). The eigenvalues are taken as squared singular values of the whitened
  Jacobian instead — exact for a Gauss-Newton curvature, and
  `reports/M2M_production_readiness.md` A2's prescription for the same reason.
- **The LOSVD kernel is defined locally**, not imported. D-027 keeps domain
  observables in the domain packages; the integration test carries its own copy for
  the same reason, and this is that copy.

### `fisher_fd_vs_ad.py`

Builds the Fisher matrix `Jᵀ C⁻¹ J` from AD and from careful finite differences (forward,
central, Richardson) at a fixed step ladder and at the default steps, in single and double
precision, and compares each against the double-precision AD reference: the Jacobian error, the
spectrum eigenvalue by eigenvalue, `effective_parameters`, the degenerate subspace and the
cross/quadratic split of the error. Three arms: the sensitivity figure's tracer problem (linear in
the weights, so finite differences' best case), the self-consistent construction where the
weights are the masses, and three source masses of an external field (`P = 3`).

Three things worth knowing before editing it:

- **The two precisions run in separate processes.** `jax_enable_x64` is process-wide, so the
  script launches itself twice as a worker, with the precision passed in
  `FISHER_PROBE_PRECISION` and set before `jax.numpy` is imported. Every observable, Jacobian
  and spectrum a worker produces has its dtype asserted. Do not import `sensitivity_figure.py`
  from it: that module turns x64 on at import.
- **All analysis is in double precision**, on Jacobians cast after they were measured, so rows
  differ only in the derivative method and the arithmetic it ran in. The `native` columns are what
  a pipeline kept in one precision throughout would get.
- **At uniform weights the shipped Hessian route is not a Fisher matrix** in the
  self-consistent case: the residual curvature is non-zero there. The script uses Gauss–Newton
  `Jᵀ C⁻¹ J` throughout.
