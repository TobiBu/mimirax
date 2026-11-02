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
existing files are skipped, so chunks resume. A fit is itself resumable: every
``SAVE_EVERY_S`` seconds (and when the process's ``--deadline-min`` passes) the
driver's whole state -- parameters, the L-BFGS state, multipliers, ``mu``, the
loop counters and histories (:class:`~mimirax.AugmentedLagrangianState`) -- and
the pass count go to ``partial/<name>.state.{npz,json}``; the next process that
takes the fit continues from there, bitwise as if never stopped. The deadline
is a property of the PROCESS (a bounded GPU claim), not of the fit: a suspended
fit is finished later by the driver's own stopping rule or its budget.
``--queue FILE`` runs a priority list of ``method S start`` lines, one fit at a
time, taking a lock per fit so several workers share the list. Arguments (all optional):
``--method single,multiple --segments 2,4,8 --starts 0-9 --radius 0.1``. GPU:
``JAX_PLATFORMS=cuda,cpu`` (``cpu`` too: the pass counter is a host callback).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as c  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from mimirax import AugmentedLagrangian, AugmentedLagrangianState  # noqa: E402

RADIUS = 0.1
K = 10
SEGMENTS = (2, 4, 5, 6, 8)
INNER_STEPS = 200
MAX_OUTER = 20
TOL = 1.0e-8
SUCCESS_TOL = 1.0e-3
#: seconds between saved driver states of a running fit
SAVE_EVERY_S = 300.0
#: worker exit codes: queue done, suspended at the deadline, only locked fits left
EXIT_DONE, EXIT_SUSPENDED, EXIT_LOCKED = 0, 10, 11


class FitSuspended(Exception):
    """A fit's state was saved at its process's deadline; another process resumes it."""


def _state_paths(state_path: Path) -> tuple[Path, Path]:
    """The array and scalar files of a saved driver state.

    Parameters
    ----------
    state_path : Path
        ``partial/<name>.state`` (no suffix).

    Returns
    -------
    tuple[Path, Path]
        ``(<...>.npz, <...>.json)``.
    """
    return state_path.with_suffix(".state.npz"), state_path.with_suffix(".state.json")


def save_state(
    state_path: Path,
    state: AugmentedLagrangianState,
    problems: c.Problems,
    resumes: int,
) -> None:
    """Write the driver's state and the fit's bookkeeping (atomically, arrays first).

    Parameters
    ----------
    state_path : Path
        ``partial/<name>.state``.
    state : AugmentedLagrangianState
        What the driver's ``checkpoint`` got.
    problems : c.Problems
        Supplies the pass counter and the progress log.
    resumes : int
        How many times the fit has been resumed so far.
    """
    import time as _time

    jax.effects_barrier()  # every pass of the steps so far counted
    npz, meta = _state_paths(state_path)
    leaves = jax.tree_util.tree_leaves(
        (state.params, state.opt_state, state.multipliers)
    )
    tmp = npz.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, *[np.asarray(leaf) for leaf in leaves])
    tmp.replace(npz)
    c.write_json(
        meta,
        {
            "mu": state.mu,
            "outer_iteration": state.outer_iteration,
            "inner_step": state.inner_step,
            "previous_value": state.previous_value,
            "previous_violation": state.previous_violation,
            "previous_objective": state.previous_objective,
            "redos": state.redos,
            "trace": state.trace,
            "outer": state.outer,
            "leaves": len(leaves),
            "segment_passes": problems.counter.passes,
            "log_steps": problems.log.steps,
            "log_history": problems.log.history,
            "elapsed_s": _time.time() - problems.log.start,
            "resumes": resumes,
            "saved_at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": socket.gethostname(),
            "provenance": c.provenance(),
        },
    )


def load_state(
    state_path: Path, template: AugmentedLagrangianState
) -> tuple[AugmentedLagrangianState, dict[str, Any]] | None:
    """Read a saved driver state, its arrays shaped like ``template`` 's.

    Parameters
    ----------
    state_path : Path
        ``partial/<name>.state``.
    template : AugmentedLagrangianState
        A fresh state of the same fit (supplies the pytree structure).

    Returns
    -------
    tuple[AugmentedLagrangianState, dict[str, Any]] | None
        The state and the scalar record, or ``None`` when nothing was saved.

    Raises
    ------
    ValueError
        If the saved arrays do not match the template's structure.
    """
    npz, meta = _state_paths(state_path)
    if not (npz.exists() and meta.exists()):
        return None
    record = c.read_json(meta)
    leaves, treedef = jax.tree_util.tree_flatten(
        (template.params, template.opt_state, template.multipliers)
    )
    with np.load(npz) as data:
        saved = [jax.numpy.asarray(data[f"arr_{i}"]) for i in range(len(data.files))]
    if len(saved) != len(leaves) or any(
        a.shape != b.shape or a.dtype != b.dtype for a, b in zip(saved, leaves)
    ):
        raise ValueError(f"{npz}: saved arrays do not match the fit's structure")
    params, opt_state, multipliers = jax.tree_util.tree_unflatten(treedef, saved)
    state = AugmentedLagrangianState(
        params=params,
        opt_state=opt_state,
        multipliers=tuple(multipliers),
        mu=float(record["mu"]),
        outer_iteration=int(record["outer_iteration"]),
        inner_step=int(record["inner_step"]),
        previous_value=record["previous_value"],
        previous_violation=float(record["previous_violation"]),
        previous_objective=record["previous_objective"],
        redos=int(record["redos"]),
        trace=record["trace"],
        outer=record["outer"],
    )
    return state, record


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
    cap_s: float = 0.0,
    state_path: Path | None = None,
    deadline: float = 0.0,
) -> dict[str, Any]:
    """Run one fit (or resume it from its saved state) and return its record.

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
    cap_s : float
        Wall-time cap of the fit in seconds (``0``: none; needs ``partial``). A
        capped fit returns its last progress record, ``status="time cap"``:
        a measurement budget for the cards, NOT the driver's stopping rule.
    state_path : Path | None
        Where the driver's state is saved every :data:`SAVE_EVERY_S` seconds
        and at ``deadline`` (``partial/<name>.state``; needs ``partial``); a
        state found there is resumed. Removed when the fit ends.
    deadline : float
        Epoch seconds (``0``: none) past which the fit's state is saved and
        :class:`FitSuspended` raised (the process's claim ends; the fit does
        not).

    Returns
    -------
    dict[str, Any]
        Final ``theta``, the driver's histories, the pass count.

    Raises
    ------
    ValueError
        For an unknown ``method``.
    FitSuspended
        When ``deadline`` passed; the state is on disk.
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
        problems.log.reset(
            partial, method=method, theta0=np.asarray(theta0).tolist(), cap_s=cap_s
        )
    start = time.time()
    resume, resumes = None, 0
    if state_path is not None:
        loaded = load_state(state_path, _template(problem, params))
        if loaded is not None:
            resume, record = loaded
            resumes = int(record["resumes"]) + 1
            problems.counter.passes = int(record["segment_passes"])
            problems.log.restore(
                record["log_steps"], record["log_history"], record["elapsed_s"]
            )
            start = time.time() - float(record["elapsed_s"])
            print(
                f"resuming {state_path.name}: outer {resume.outer_iteration} inner "
                f"{resume.inner_step}, {problems.counter.passes} passes, "
                f"resume {resumes}",
                flush=True,
            )
    last_save = [time.time()]

    def checkpoint(state: AugmentedLagrangianState) -> None:
        now = time.time()
        late = bool(deadline) and now >= deadline
        if state_path is None or not (late or now - last_save[0] >= SAVE_EVERY_S):
            return
        save_state(state_path, state, problems, resumes)
        last_save[0] = now
        if late:
            raise FitSuspended(f"{state_path.name} saved at the deadline")

    try:
        result = driver().minimize(
            problem,
            params,
            checkpoint=None if state_path is None else checkpoint,
            resume=resume,
        )
    except FitSuspended:
        problems.log.reset(None)
        raise
    except c.FitTimeCap:
        jax.effects_barrier()
        assert partial is not None
        last = c.read_json(partial)
        problems.log.reset(None)
        partial.unlink(missing_ok=True)
        return {
            "theta": last["theta"],
            "converged": False,
            "status": "time cap",
            "time_cap_s": cap_s,
            "outer_iterations": None,
            "inner_steps": last["inner_steps_so_far"],
            "segment_passes": int(last["segment_passes_so_far"]),
            "evaluations": last["segment_passes_so_far"] / problems.num_segments,
            "history_steps_passes_maxerror": last["history"],
            "wall_s_information_only": time.time() - start,
        }
    jax.effects_barrier()
    history = list(problems.log.history)
    problems.log.reset(None)
    if partial is not None:
        partial.unlink(missing_ok=True)
    if state_path is not None:
        for path in _state_paths(state_path):
            path.unlink(missing_ok=True)
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
        "num_redos": int(result.num_redos),
        "status": "converged" if result.converged else "budget",
        "history_steps_passes_maxerror": history,
        "inner_objective": np.asarray(result.objective_trace).tolist(),
        "inner_violation": np.asarray(result.violation_trace).tolist(),
        "inner_mu": np.asarray(result.mu_trace).tolist(),
        "budget": {"inner_steps": INNER_STEPS, "max_outer": MAX_OUTER, "tol": TOL},
        "resumes": resumes,
        "wall_s_information_only": wall,
    }


def _template(problem: Any, params: dict[str, Any]) -> AugmentedLagrangianState:
    """A fresh driver state of this fit: the pytree structure a saved one must match.

    Parameters
    ----------
    problem : Any
        The fit's :class:`~mimirax.InferenceProblem`.
    params : dict[str, Any]
        Its starting parameters.

    Returns
    -------
    AugmentedLagrangianState
        Zero multipliers, the optimizer's initial state.
    """
    shapes = jax.eval_shape(problem.evaluate, params).defects
    return AugmentedLagrangianState(
        params=params,
        opt_state=driver().optimizer.init(params),
        multipliers=tuple(jax.numpy.zeros(d.shape, d.dtype) for d in shapes),
        mu=0.0,
        outer_iteration=0,
        inner_step=0,
        previous_value=None,
        previous_violation=float("inf"),
        previous_objective=None,
        redos=0,
        trace={},
        outer={},
    )


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


def result_path(root: Path, method: str, s: int, radius: float, k: int) -> Path:
    """The result file of one fit.

    Parameters
    ----------
    root : Path
        ``$LONG_HORIZON_RUNS/recovery``.
    method : str
        The method.
    s : int
        Segments.
    radius : float
        Start radius.
    k : int
        Start index.

    Returns
    -------
    Path
        ``<method>_S<ss>_r<radius>_k<kk>.json``.
    """
    return root / f"{method}_S{s:02d}_r{radius:g}_k{k:02d}.json"


def run_one(
    problems: c.Problems,
    root: Path,
    method: str,
    k: int,
    radius: float,
    cap_min: float = 0.0,
    deadline: float = 0.0,
) -> dict[str, Any]:
    """Run (or resume) one fit and write its result file.

    Parameters
    ----------
    problems : c.Problems
        The data at this horizon.
    root : Path
        ``$LONG_HORIZON_RUNS/recovery``.
    method : str
        The method.
    k : int
        Start index.
    radius : float
        Start radius.
    cap_min : float
        Wall-time cap per fit in minutes (``0``: none).
    deadline : float
        The process's deadline (epoch seconds, ``0``: none).

    Returns
    -------
    dict[str, Any]
        The record written.
    """
    s = problems.num_segments
    at_truth = reference(problems, root)
    path = result_path(root, method, s, radius, k)
    theta0 = start_theta(k, radius)
    partial = root / "partial" / path.name
    partial.parent.mkdir(parents=True, exist_ok=True)
    record = fit(
        problems,
        method,
        theta0,
        partial,
        60.0 * cap_min,
        state_path=partial.with_suffix(""),
        deadline=deadline,
    )
    final = np.asarray(record["theta"])
    error = final - c.theta_array(c.TRUTH)
    objective = single_objective(problems, final)
    df = objective - at_truth
    record.update(
        method=method,
        segments=s,
        start=k,
        radius=radius,
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
        f" df {df:.3g} success {record['success']} {record['status']} passes "
        f"{record['segment_passes']} ({record['inner_steps']} inner, "
        f"{record['outer_iterations']} outer, {record.get('resumes', 0)} resumes) "
        f"{record['wall_s_information_only']:.0f} s",
        flush=True,
    )
    return record


def read_queue(path: Path) -> list[tuple[str, int, int]]:
    """Read a priority list: one ``method S start`` per line, ``#`` comments.

    Parameters
    ----------
    path : Path
        The list.

    Returns
    -------
    list[tuple[str, int, int]]
        ``(method, S, k)`` in order.
    """
    tasks = []
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            method, s, k = line.split()
            tasks.append((method, int(s), int(k)))
    return tasks


def _lock_holder_alive(lock: Path) -> bool:
    """Whether the process that wrote ``lock`` is still running (same host).

    Parameters
    ----------
    lock : Path
        A lock file (``{"pid", "host"}``).

    Returns
    -------
    bool
        ``True`` while the holder runs (or on another host: not checkable).
    """
    try:
        holder = json.loads(lock.read_text())
    except (OSError, ValueError):
        return False
    if holder.get("host") != socket.gethostname():
        return True
    try:
        os.kill(int(holder["pid"]), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def take_lock(lock: Path) -> bool:
    """Take a fit's lock (a stale one, whose holder died, is broken first).

    Parameters
    ----------
    lock : Path
        ``locks/<name>.lock``.

    Returns
    -------
    bool
        Whether this process now holds it.
    """
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists() and not _lock_holder_alive(lock):
        lock.unlink(missing_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as handle:
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, handle)
    return True


def queue_state(root: Path, tasks: list[tuple[str, int, int]], radius: float) -> str:
    """``"runnable"``, ``"locked"`` (only fits other workers hold) or ``"done"``.

    Parameters
    ----------
    root : Path
        ``$LONG_HORIZON_RUNS/recovery``.
    tasks : list[tuple[str, int, int]]
        The queue.
    radius : float
        Start radius.

    Returns
    -------
    str
        The queue's state.
    """
    locked = False
    for method, s, k in tasks:
        path = result_path(root, method, s, radius, k)
        if path.exists():
            continue
        lock = root / "locks" / (path.stem + ".lock")
        if lock.exists() and _lock_holder_alive(lock):
            locked = True
            continue
        return "runnable"
    return "locked" if locked else "done"


def work_queue(
    queue: Path, radius: float, cap_min: float, deadline_min: float, peek: bool
) -> int:
    """Run the queue's fits in order, one at a time, until it is done or the deadline.

    Parameters
    ----------
    queue : Path
        The priority list (:func:`read_queue`).
    radius : float
        Start radius.
    cap_min : float
        Wall-time cap per fit (``0``: none).
    deadline_min : float
        This process's wall-time budget in minutes (``0``: none): a fit running
        at the deadline is saved and resumed by the next worker.
    peek : bool
        Only report the queue's state (exit code), run nothing.

    Returns
    -------
    int
        :data:`EXIT_DONE`, :data:`EXIT_SUSPENDED` or :data:`EXIT_LOCKED`.
    """
    root = c.RUNS / "recovery"
    tasks = read_queue(queue)
    codes = {"done": EXIT_DONE, "locked": EXIT_LOCKED, "runnable": EXIT_SUSPENDED}
    if peek:
        state = queue_state(root, tasks, radius)
        print(state, flush=True)
        return codes[state]
    deadline = time.time() + 60.0 * deadline_min if deadline_min else 0.0
    system = c.build_system()
    built: dict[int, c.Problems] = {}
    for method, s, k in tasks:
        path = result_path(root, method, s, radius, k)
        if path.exists():
            continue
        if deadline and time.time() > deadline - 60.0:
            return EXIT_SUSPENDED
        lock = root / "locks" / (path.stem + ".lock")
        if not take_lock(lock):
            continue
        try:
            if path.exists():  # finished by another worker meanwhile
                continue
            if s not in built:
                built.clear()  # one horizon's programs at a time
                built[s] = c.problems(system, s)
            run_one(built[s], root, method, k, radius, cap_min, deadline)
        except FitSuspended as stop:
            print(f"suspended: {stop}", flush=True)
            return EXIT_SUSPENDED
        finally:
            lock.unlink(missing_ok=True)
    state = queue_state(root, tasks, radius)
    return (
        EXIT_LOCKED
        if state == "locked"
        else (EXIT_DONE if state == "done" else EXIT_SUSPENDED)
    )


def main() -> int:
    """Run every requested (method, S, start) without a result file.

    Returns
    -------
    int
        Exit status (:data:`EXIT_DONE`, or with ``--queue`` also
        :data:`EXIT_SUSPENDED` / :data:`EXIT_LOCKED`).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="single,multiple_data,multiple")
    parser.add_argument("--segments", default=",".join(map(str, SEGMENTS)))
    parser.add_argument("--starts", default=f"0-{K - 1}")
    parser.add_argument("--radius", type=float, default=RADIUS)
    parser.add_argument(
        "--cap-min", type=float, default=0.0, help="wall-time cap per fit, minutes"
    )
    parser.add_argument("--queue", type=Path, help="priority list: method S start")
    parser.add_argument(
        "--deadline-min",
        type=float,
        default=0.0,
        help="this process's budget, minutes: the running fit is saved, not cut",
    )
    parser.add_argument("--peek", action="store_true", help="report the queue only")
    args = parser.parse_args()
    if args.queue is not None:
        return work_queue(
            args.queue, args.radius, args.cap_min, args.deadline_min, args.peek
        )
    root = c.RUNS / "recovery"
    system = c.build_system()
    deadline = time.time() + 60.0 * args.deadline_min if args.deadline_min else 0.0
    for s in parse_range(args.segments):
        problems = c.problems(system, s)
        for method in args.method.split(","):
            for k in parse_range(args.starts):
                if result_path(root, method, s, args.radius, k).exists():
                    continue
                try:
                    run_one(
                        problems, root, method, k, args.radius, args.cap_min, deadline
                    )
                except FitSuspended as stop:
                    print(f"suspended: {stop}", flush=True)
                    return EXIT_SUSPENDED
    return 0


if __name__ == "__main__":
    sys.exit(main())
