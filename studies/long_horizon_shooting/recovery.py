"""Recovery from perturbed starts: single against multiple shooting, against the horizon.

For a horizon of ``S`` snapshots (``common.problems``: the full phase-space
state every ``SEGMENT_STEPS`` steps, noise ``SIGMA``; the same data for both
methods), ``K`` starts ``theta_0 = theta* + RADIUS u_k`` with ``u_k`` a random
unit vector (seed ``k``). Both methods run the SAME driver,
:class:`~mimirax.AugmentedLagrangian` with ``optax.lbfgs()`` inner steps and
the same stopping rule and budget (``INNER_STEPS`` per outer iteration,
``MAX_OUTER`` outer iterations):

``single``
    params ``{"theta"}``, :class:`common.SeriesModel` (one trajectory); no
    defects, so the driver is a plain L-BFGS loop restarted every outer
    iteration.
``multiple``
    params ``{"theta", "nodes"}``, ``S`` segments, ``output="segment_ends"``;
    nodes initialised from a forward run at the START ``theta_0``
    (``init_nodes``: defects zero); penalty ``mu0 = 1 / sigma^2`` (a defect
    weighs what a datum does), violation tolerance ``TOL`` (absolute, the
    state is O(1)).
``multiple_data``
    the same, nodes initialised from the DATA (node ``k`` = the observed state
    at its time, noise included; :func:`data_nodes`): the classical
    multiple-shooting start (Bock 1981; Baake et al. 1992), available here
    because the whole phase-space state is observed.

Success: the final ``theta`` within ``SUCCESS_TOL`` of the truth in every
component (``1e-3``: a 0.1 % estimate of the halo's mass and scale, 100x closer
than the start). The single-shot objective at the result, minus its value at the
truth, is recorded beside it but is NOT part of the test: on a long horizon the
global minimum is far narrower than any practical tolerance (its Fisher width is
``1e-9 .. 1e-15`` at 16 .. 31 Lyapunov times, ``roughness.py``), so a good
estimate can sit on a slope of ``f`` many orders above the noise. Nor is there an
"oracle" fit from the truth: on long horizons single shooting leaves the basin
from the truth itself (``--radius 0``).

Cost: segment passes, counted on the host (``common.CountingEngine``), minus
the forward-only measurement the driver makes after every outer iteration; one
pass is one segment's value and gradient (line-search trial points included).
A single-shooting evaluation is ``S`` passes, as is a multiple-shooting one: so
the counts compare equal work. NOT wall time (shared cards); wall time is
recorded for information only.

One JSON per (method, S, start) under ``$LONG_HORIZON_RUNS/recovery/``;
existing files are skipped, so chunks resume. Arguments (all optional):
``--method single,multiple --segments 2,4,8 --starts 0-9 --radius 0.1``. GPU:
``JAX_PLATFORMS=cuda,cpu`` (``cpu`` too: the pass counter is a host callback).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as c  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from mimirax import AugmentedLagrangian  # noqa: E402

RADIUS = 0.1
K = 10
SEGMENTS = (2, 4, 5, 6, 8)
INNER_STEPS = 200
MAX_OUTER = 20
TOL = 1.0e-8
SUCCESS_TOL = 1.0e-3


def driver() -> AugmentedLagrangian:
    """The one driver both methods run.

    Returns
    -------
    AugmentedLagrangian
        L-BFGS inner steps, ``mu0 = 1 / sigma^2``.
    """
    return AugmentedLagrangian(
        optimizer=optax.lbfgs(),
        inner_steps=INNER_STEPS,
        max_outer=MAX_OUTER,
        tol=TOL,
        mu0=1.0 / c.SIGMA**2,
    )


def start_theta(k: int, radius: float) -> np.ndarray:
    """Start ``k``: the truth plus ``radius`` along a random unit vector.

    Parameters
    ----------
    k : int
        Start index (the seed).
    radius : float
        Distance from the truth in ``(log M, log r_s)``.

    Returns
    -------
    np.ndarray
        ``[P]``.
    """
    u = np.random.default_rng(1000 + k).normal(size=len(c.NAMES))
    return c.theta_array(c.TRUTH) + radius * u / np.linalg.norm(u)


def fit(
    problems: c.Problems,
    method: str,
    theta0: np.ndarray,
    partial: Path | None = None,
) -> dict[str, Any]:
    """Run one fit and return its record.

    Parameters
    ----------
    problems : c.Problems
        The data and both problems at one horizon.
    method : str
        ``"single"``, ``"multiple"`` or ``"multiple_data"``.
    theta0 : np.ndarray
        Start.
    partial : Path | None
        Where the fit's progress is written while it runs (removed when it
        ends): a fit stopped by a time limit leaves its last ``theta`` and pass
        count there.

    Returns
    -------
    dict[str, Any]
        Final ``theta``, the driver's histories, the pass count.

    Raises
    ------
    ValueError
        For an unknown ``method``.
    """
    theta = c.theta_dict(theta0)
    if method == "single":
        problem, params = problems.single, {"theta": theta}
    elif method == "multiple":
        nodes = jax.jit(problems.multiple_model.init_nodes)(theta)
        problem, params = problems.multiple, {"theta": theta, "nodes": nodes}
    elif method == "multiple_data":
        problem, params = problems.multiple, {
            "theta": theta,
            "nodes": data_nodes(problems),
        }
    else:
        raise ValueError(f"unknown method {method!r}")
    jax.effects_barrier()
    problems.counter.passes = 0
    if partial is not None:
        problems.log.reset(partial, method=method, theta0=np.asarray(theta0).tolist())
    start = time.time()
    result = driver().minimize(problem, params)
    jax.effects_barrier()
    problems.log.reset(None)
    if partial is not None:
        partial.unlink(missing_ok=True)
    wall = time.time() - start
    outer = len(result.outer_steps)
    passes = problems.counter.passes - outer * problems.num_segments
    final = c.theta_array(result.params["theta"])
    return {
        "theta": final.tolist(),
        "converged": bool(result.converged),
        "outer_iterations": outer,
        "inner_steps": int(len(result.objective_trace)),
        "outer_steps": np.asarray(result.outer_steps).tolist(),
        "segment_passes": int(passes),
        "evaluations": passes / problems.num_segments,
        "outer_objective": np.asarray(result.outer_objective).tolist(),
        "outer_violation": np.asarray(result.outer_violation).tolist(),
        "outer_mu": np.asarray(result.outer_mu).tolist(),
        "num_traces": int(result.num_traces),
        "wall_s_information_only": wall,
    }


def data_nodes(problems: c.Problems) -> dict[str, Any]:
    """Nodes from the data: node ``k`` is the observed state at its time.

    Node ``k`` (``1 <= k < S``) starts segment ``k`` at the time of snapshot
    ``k - 1`` (the end of segment ``k - 1``), so it is set to that snapshot's
    noisy observation. Uses the data only, never the truth.

    Parameters
    ----------
    problems : c.Problems
        The data.

    Returns
    -------
    dict[str, Any]
        ``{"positions": [S-1, N, 3], "velocities": [S-1, N, 3]}``.
    """
    s = problems.num_segments
    n = problems.system.positions.shape[0]
    snapshots = problems.observed.reshape(s, n, 6)[: s - 1]
    return {"positions": snapshots[..., :3], "velocities": snapshots[..., 3:]}


def single_objective(problems: c.Problems, theta: np.ndarray) -> float:
    """The single-shooting objective at ``theta`` (the success test's yardstick).

    Parameters
    ----------
    problems : c.Problems
        The data.
    theta : np.ndarray
        ``[P]``.

    Returns
    -------
    float
        ``-log posterior`` of the single-shooting problem.
    """
    value = jax.jit(lambda p: problems.single.evaluate(p).negative_log_posterior)
    return float(value({"theta": c.theta_dict(theta)}))


def reference(problems: c.Problems, root: Path) -> float:
    """The single-shot objective at the truth; cached per horizon.

    Parameters
    ----------
    problems : c.Problems
        The data at this horizon.
    root : Path
        Result directory.

    Returns
    -------
    float
        ``f_S(theta*)``.
    """
    path = root / f"reference_S{problems.num_segments:02d}.json"
    if path.exists():
        return float(c.read_json(path)["objective_at_truth"])
    value = single_objective(problems, c.theta_array(c.TRUTH))
    c.write_json(
        path,
        {
            "segments": problems.num_segments,
            "objective_at_truth": value,
            "provenance": c.provenance(),
        },
    )
    return value


def parse_range(text: str) -> list[int]:
    """Parse ``"0-9"`` or ``"1,3,5"``.

    Parameters
    ----------
    text : str
        The range.

    Returns
    -------
    list[int]
        The integers.
    """
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    """Run every requested (method, S, start) without a result file.

    Returns
    -------
    int
        Exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="single,multiple_data,multiple")
    parser.add_argument("--segments", default=",".join(map(str, SEGMENTS)))
    parser.add_argument("--starts", default=f"0-{K - 1}")
    parser.add_argument("--radius", type=float, default=RADIUS)
    args = parser.parse_args()
    root = c.RUNS / "recovery"
    system = c.build_system()
    for s in parse_range(args.segments):
        problems = c.problems(system, s)
        at_truth = reference(problems, root)
        for method in args.method.split(","):
            for k in parse_range(args.starts):
                path = root / f"{method}_S{s:02d}_r{args.radius:g}_k{k:02d}.json"
                if path.exists():
                    continue
                theta0 = start_theta(k, args.radius)
                partial = root / "partial" / path.name
                partial.parent.mkdir(parents=True, exist_ok=True)
                record = fit(problems, method, theta0, partial)
                final = np.asarray(record["theta"])
                error = final - c.theta_array(c.TRUTH)
                objective = single_objective(problems, final)
                df = objective - at_truth
                record.update(
                    method=method,
                    segments=s,
                    start=k,
                    radius=args.radius,
                    theta0=theta0.tolist(),
                    error=error.tolist(),
                    objective=objective,
                    objective_at_truth=at_truth,
                    objective_above_truth=df,
                    success=bool(np.max(np.abs(error)) <= SUCCESS_TOL),
                    provenance=c.provenance(),
                )
                c.write_json(path, record)
                print(
                    f"{method:8s} S {s:2d} k {k:2d}: |err| {np.max(np.abs(error)):.2e}"
                    f" df {df:.3g} success {record['success']} passes "
                    f"{record['segment_passes']} ({record['inner_steps']} inner, "
                    f"{record['outer_iterations']} outer) "
                    f"{record['wall_s_information_only']:.0f} s",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
