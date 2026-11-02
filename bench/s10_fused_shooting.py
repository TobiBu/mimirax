"""Multiple shooting on jaccpot's fused production lane, on one GPU (S10 session 2, track F).

usage: python bench/s10_fused_shooting.py --task TASK --n N [--segments S] [--steps L]
       [--dt DT] [--softening EPS] [--sigma SIGMA] [--offset DLOGM,DLOGRS] [--out OUT.json]

THE SYSTEM. A self-gravitating disc (total mass ``--disc-mass``, 1e-2 by default)
of ``N`` particles on near-circular orbits at R in [0.5, 3], ``|z| ~ 0.05``, in
nornax's analytic galaxy potential: a Miyamoto-Nagai disc field (M 1, a 1, b
0.1, fixed) plus an NFW halo whose ``(log M, log r_s)`` is the unknown ``theta``
(truth ``(log 5, log 3)``) -- the N = 64 construction of mimirax's
``tests/integration/test_shooting_galaxy_potentials.py``, scaled. Self-gravity on
jaccpot's fused lane, ``FastMultipoleMethod(preset="galaxy", softening=eps)``
(``dehnen_error`` 1e-4, leaf 64, p6, float32 arrays, ``JAX_ENABLE_X64=1``), the
field through the runner's ``external_acceleration_fn`` (mimirax ``rows_field``)
with ``theta`` its ``external_params``. Softening ``0.015 (N / 1e5)^(-1/3)`` by
default (about half the mean interparticle spacing in the disc's slab), ``dt``
0.01 (an orbit at R = 1 is about 9 time units), ``S`` segments of ``L`` steps
(``FusedSegmentEngine``, one gradient plan per segment, ``checkpoint="step"``;
``--start-force adjoint``: each segment's start force inside the rollout's adjoint,
jaccpot's ``GradientPlan.rollout(start_force="adjoint")``, S10 follow-ups).

TASKS (one per process; every task writes its JSON record to ``--out``):

* ``prepass``: ``FusedSegmentEngine.propagate`` at the truth (``S`` pre-passes),
  again at the same nodes (warm), at the start ``theta`` (warm, other nodes), with
  and without ``gradient_plan(freeze_cache=...)``: backend compiles and their
  seconds per pre-pass (``jax.monitoring``), the pre-pass's wall seconds. Local
  cards are shared: the seconds are a cost STRUCTURE, not a timing.
* ``memory``: per segment, the value-and-gradient program of ``sum(w . x_L) +
  sum(w . v_L)`` in ``(theta, x_k, v_k)`` (``memory_analysis()``; run once: the
  allocator's peak and the verdict) for every ``L`` in ``--steps-list`` (W and r
  fitted per particle); then the augmented Lagrangian's jitted step (the whole
  iteration: every segment's gradient, the optimizer's update) for Adam and
  L-BFGS: its temporaries, arguments (parameters, optimizer state, multipliers),
  and the closure constants (masses, node 0, the observations) in its HLO.
* ``fd``: the augmented Lagrangian's gradient in ``theta`` and along a node
  direction against central differences at several ``h`` (float32), at the
  perturbed start; the verdict of every evaluation.
* ``al``: the recovery: observe the final state with noise ``--sigma`` at the
  truth, start at ``truth + --offset`` with nodes propagated there, run
  ``AugmentedLagrangian`` (``--optimizer``); theta's error against the truth and
  against the noise (the Fisher matrix from central differences of the
  prediction in ``theta``), outer / inner iterations, traces, redos, the peak.
* ``timing``: ``--reps`` warm calls of each program the HoreKa job times: the
  per-segment value-and-gradient, a pre-pass with and without ``freeze_cache``,
  one augmented-Lagrangian step (= one inner iteration).

CONTENTION. As jaccpot's ``bench/grad_rollout_timing.py``: ``nvidia-smi``'s other
compute processes on the card before and after; a contended row's seconds are not
to be quoted. Timings belong to the HoreKa job
(``bench/slurm/s10_shooting_timing.sbatch``).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Optional

import numpy as np

TASKS = ("prepass", "memory", "fd", "al", "timing", "parity")
#: (log M, log r_s) of the NFW halo
TRUTH = (math.log(5.0), math.log(3.0))
_ENV_KEYS = (
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "CUDA_VISIBLE_DEVICES",
    "XLA_FLAGS",
    "JAX_PLATFORMS",
    "JAX_ENABLE_X64",
)


def _args(argv: Optional[list] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--segments", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50, help="L, steps per segment")
    ap.add_argument(
        "--steps-list", default=None, help="memory: several L (comma separated)"
    )
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--softening", type=float, default=None)
    ap.add_argument("--disc-mass", type=float, default=1e-2)
    ap.add_argument(
        "--system",
        default="disc",
        choices=("disc", "plummer"),
        help="plummer: S9's clipped Plummer (W calibration against S9's rows)",
    )
    ap.add_argument(
        "--cell-min-level",
        type=lambda s: s if s == "auto" else int(s),
        default=None,
        help="override the preset's TreeConfig.cell_min_level (8); 'auto': jaccpot's "
        "scale-aware pick (S10 follow-ups)",
    )
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=1e-3, help="observation noise")
    ap.add_argument(
        "--offset", default="0.1,-0.05", help="start theta - truth (log M, log r_s)"
    )
    ap.add_argument("--optimizer", default="lbfgs", choices=("lbfgs", "adam"))
    ap.add_argument(
        "--devices",
        type=int,
        default=None,
        help="per-segment mode on the first K devices (default: one program)",
    )
    ap.add_argument("--lr", type=float, default=1e-3, help="adam's learning rate")
    ap.add_argument("--mu0", type=float, default=100.0)
    ap.add_argument("--tol", type=float, default=1e-5, help="AL violation, relative")
    ap.add_argument("--objective-rtol", type=float, default=1e-6)
    ap.add_argument("--inner-steps", type=int, default=50)
    ap.add_argument("--inner-rtol", type=float, default=1e-9)
    ap.add_argument("--max-outer", type=int, default=12)
    ap.add_argument("--fd-h", default="0.1,0.03,0.01,0.003,0.001,0.0003")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-al-memory", action="store_true", help="memory: segments only")
    ap.add_argument(
        "--segment-variant",
        default="engine",
        choices=("engine", "direct"),
        help="memory: the engine's segment, or one rollout with its start force inside",
    )
    ap.add_argument(
        "--start-force",
        default="outside",
        choices=("outside", "adjoint"),
        help="the engine's start_force: jaccpot GradientPlan.rollout(start_force=)",
    )
    ap.add_argument(
        "--only",
        default="seggrad",
        choices=("prepass", "seggrad", "alstep"),
        help="timing",
    )
    ap.add_argument(
        "--no-freeze-cache", action="store_true", help="timing: pre-pass without it"
    )
    ap.add_argument("--out", default=None)
    return ap.parse_args(argv)


# --------------------------------------------------------------------------
# Bookkeeping: compiles, the allocator, the card.
# --------------------------------------------------------------------------


class Compiles:
    """Backend compiles (count, seconds) from ``jax.monitoring``, cumulative."""

    def __init__(self) -> None:
        import jax
        from jax._src import dispatch

        self.count = 0
        self.seconds = 0.0
        self.traces = 0
        event, trace_event = dispatch.BACKEND_COMPILE_EVENT, dispatch.JAXPR_TRACE_EVENT

        self.names: list = []

        def listener(name: str, duration: float, **kw: Any) -> None:
            if name == event:
                self.count += 1
                self.seconds += float(duration)
                self.names.append((str(kw.get("fun_name")), round(float(duration), 3)))
            elif name == trace_event:
                self.traces += 1

        jax.monitoring.register_event_duration_secs_listener(listener)
        self.cache_hits = 0

        def hits(name: str, **_: Any) -> None:
            if name == "/jax/compilation_cache/cache_hits":
                self.cache_hits += 1

        jax.monitoring.register_event_listener(hits)

    def mark(self) -> tuple:
        return (self.count, self.seconds, self.traces, self.cache_hits, len(self.names))

    def since(self, mark: tuple) -> dict:
        return {
            "compiles": self.count - mark[0],
            "compile_s": round(self.seconds - mark[1], 3),
            "jaxpr_traces": self.traces - mark[2],
            "persistent_cache_hits": self.cache_hits - mark[3],
            "programs": (
                self.names[mark[4] :] if self.count - mark[0] <= 40 else "(> 40)"
            ),
        }


def _alloc() -> dict:
    import jax

    stats = jax.devices()[0].memory_stats() or {}
    return {
        k: stats.get(k)
        for k in (
            "bytes_in_use",
            "peak_bytes_in_use",
            "bytes_limit",
            "largest_alloc_size",
        )
    }


def _card_state() -> dict:
    """Other compute processes on this process's card, and its utilisation."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd:
        return {"other_processes": "unknown (no CUDA_VISIBLE_DEVICES)"}
    card = cvd.split(",")[0]
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                card,
                "--query-gpu=uuid,utilization.gpu,memory.used",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        uuid = gpu.split(",")[0].strip()
        procs = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid,used_memory",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            .stdout.strip()
            .splitlines()
        )
        others = [
            p for p in procs if p.startswith(uuid) and f", {os.getpid()}," not in p
        ]
        return {"utilization": gpu, "other_processes": others or "none"}
    except Exception as exc:  # noqa: BLE001
        return {"other_processes": f"nvidia-smi failed: {exc}"}


def _heads() -> dict:
    """The checkouts' HEADs (mimirax, jaccpot, nornax), as imported."""
    out = {}
    for name in ("mimirax", "jaccpot", "nornax"):
        try:
            module = __import__(name)
            root = os.path.dirname(os.path.dirname(os.path.abspath(module.__file__)))
            out[name] = subprocess.run(
                ["git", "-C", root, "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            out[name] = f"unknown: {exc}"
    return out


def _bpp(nbytes: Any, n: int) -> Any:
    return None if nbytes is None else round(float(nbytes) / n, 2)


def _ma(compiled: Any, n: int) -> dict:
    ma = compiled.memory_analysis()
    out = {}
    for key in (
        "temp_size_in_bytes",
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "generated_code_size_in_bytes",
        "host_temp_size_in_bytes",
    ):
        value = getattr(ma, key, None)
        if value is not None:
            out[key.replace("_size_in_bytes", "_bytes")] = int(value)
            out[key.replace("_size_in_bytes", "_B_per_particle")] = _bpp(value, n)
    return out


def _digest(*arrays: Any) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------
# The system.
# --------------------------------------------------------------------------


def _potentials() -> Any:
    from nornax import potentials

    return potentials


_GALAXY: Any = None


def galaxy_field(x: Any, t: Any, theta: Any) -> Any:
    """The disc field (fixed) plus the NFW halo whose ``(log M, log r_s)`` is ``theta``."""
    global _GALAXY
    import jax.numpy as jnp

    potentials = _potentials()
    if _GALAXY is None:
        _GALAXY = potentials.CompositePotential(
            disc=potentials.MiyamotoNagai(), halo=potentials.NFW()
        )
    disc = potentials.MiyamotoNagaiParams(M=1.0, a=1.0, b=0.1)
    halo = potentials.NFWParams(
        M=jnp.exp(theta["log_halo_mass"]), r_s=jnp.exp(theta["log_halo_scale"])
    )
    return _GALAXY(x, t, {"disc": disc, "halo": halo})


_ROWS: Any = None


def rows() -> Any:
    """``rows_field(galaxy_field)``: ONE function object (the runner's compile key)."""
    global _ROWS
    if _ROWS is None:
        from mimirax.adapters.jaccpot import rows_field

        _ROWS = rows_field(galaxy_field)
    return _ROWS


def theta_of(values: Any, dtype: Any = None) -> dict:
    import jax.numpy as jnp

    dtype = jnp.float32 if dtype is None else dtype
    return {
        "log_halo_mass": jnp.asarray(values[0], dtype),
        "log_halo_scale": jnp.asarray(values[1], dtype),
    }


def theta_values(theta: Any) -> list:
    return [float(theta["log_halo_mass"]), float(theta["log_halo_scale"])]


def system(args: argparse.Namespace) -> tuple:
    """``(x, v, m)`` of ``--system``: the disc, or S9's clipped Plummer (calibration)."""
    if args.system == "disc":
        return disc(args.n, args.seed, args.disc_mass)
    import jax.numpy as jnp

    # jaccpot bench/grad_rollout_memory.py's _plummer: clipped at 20 scale radii,
    # velocities 0.1 N(0, 1), total mass 1 -- the S9 memory rows' system
    rng = np.random.default_rng(args.seed)
    u = rng.uniform(0.0, 1.0, args.n)
    r = np.minimum(1.0 / np.sqrt(u ** (-2.0 / 3.0) - 1.0), 20.0)
    mu = rng.uniform(-1.0, 1.0, args.n)
    phi = rng.uniform(0.0, 2.0 * np.pi, args.n)
    st = np.sqrt(1.0 - mu * mu)
    x = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    v = 0.1 * rng.normal(size=(args.n, 3))
    m = np.full((args.n,), 1.0 / args.n)
    return tuple(jnp.asarray(a_, jnp.float32) for a_ in (x, v, m))


def disc(n: int, seed: int, disc_mass: float) -> tuple:
    """``(x, v, m)``, float32: near-circular orbits in the true potential (float64 field)."""
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    radius = rng.uniform(0.5, 3.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    z = 0.05 * rng.normal(size=n)
    x = np.stack([radius * np.cos(phi), radius * np.sin(phi), z], -1)
    a = np.asarray(galaxy_field(jnp.asarray(x), 0.0, theta_of(TRUTH, jnp.float64)))
    radial = -(a[:, 0] * np.cos(phi) + a[:, 1] * np.sin(phi))
    speed = np.sqrt(radius * radial)
    v = np.stack([-speed * np.sin(phi), speed * np.cos(phi), np.zeros(n)], -1)
    m = np.full((n,), disc_mass / n)
    return tuple(jnp.asarray(a_, jnp.float32) for a_ in (x, v, m))


def softening(args: argparse.Namespace) -> float:
    if args.softening is not None:
        return float(args.softening)
    return 0.015 * (args.n / 1e5) ** (-1.0 / 3.0)


def make_engine(
    args: argparse.Namespace, steps: Optional[int] = None, freeze_cache: bool = True
) -> Any:
    """A galaxy-preset runner with the field, and its ``FusedSegmentEngine``."""
    from jaccpot import FastMultipoleMethod
    from jaccpot.production import ProductionRunner, SegmentConfig

    from mimirax.adapters.jaccpot import FusedSegmentEngine

    length = int(args.steps if steps is None else steps)
    kw: dict = {}
    if args.cell_min_level is not None:
        # the preset's tree with another floor on the leaves' Morton level
        import dataclasses

        base = FastMultipoleMethod(preset="galaxy").advanced
        kw["advanced"] = dataclasses.replace(
            base,
            tree=dataclasses.replace(base.tree, cell_min_level=args.cell_min_level),
        )
    solver = FastMultipoleMethod(preset="galaxy", softening=softening(args), **kw)
    runner = ProductionRunner(
        solver,
        segment=SegmentConfig(steps=length, diagnostics_every=0),
        external_acceleration_fn=rows(),
    )
    start = {} if args.start_force == "outside" else {"start_force": args.start_force}
    engine = FusedSegmentEngine(runner, (length,) * args.segments, dt=args.dt, **start)
    if not freeze_cache:
        engine._plan_options = {}
    return engine


def count_plans(engine: Any, compiles: Compiles, log: list) -> None:
    """Record every pre-pass's compiles and seconds (wraps ``runner.gradient_plan``)."""
    runner = engine.runner
    inner = runner.gradient_plan

    def recorded(*a: Any, **kw: Any) -> Any:
        mark = compiles.mark()
        begun = time.perf_counter()
        plan = inner(*a, **kw)
        row = compiles.since(mark)
        row["wall_s"] = round(time.perf_counter() - begun, 3)
        row["freeze_cache"] = "freeze_cache" in kw
        row["kinds"] = [s.kind for s in plan.segments]
        row["retries"] = [s.retries for s in plan.segments]
        row["leaf_capacity"] = plan.segments[0].plan.get("leaf_capacity")
        row["caps"] = {
            k: v
            for k, v in plan.segments[0].plan.get("walk_caps", {}).items()
            if isinstance(v, (int, np.integer))
        }
        log.append(row)
        return plan

    runner.gradient_plan = recorded


def _phase_space(state: Any) -> Any:
    """The observable: the final phase-space state, in float64 (the misfit's sums)."""
    import jax.numpy as jnp

    return (
        jnp.concatenate([state["positions"], state["velocities"]], -1)
        .astype(jnp.float64)
        .reshape(-1)
    )


def build_problem(
    args: argparse.Namespace, engine: Any, x: Any, v: Any, m: Any, observed: Any
) -> Any:
    from mimirax import (
        GaussianLikelihood,
        InferenceProblem,
        MultipleShootingForwardModel,
    )

    model = MultipleShootingForwardModel(engine, m, args.segments, x, v)
    problem = InferenceProblem(
        model, _phase_space, GaussianLikelihood(sigma=args.sigma), observed
    )
    return model, problem


def free_nodes(nodes: dict) -> dict:
    return {"positions": nodes["positions"][1:], "velocities": nodes["velocities"][1:]}


def observe(args: argparse.Namespace, engine: Any) -> Any:
    """The final state at the truth (the last pre-pass's end) plus noise, float64."""
    import jax.numpy as jnp

    end = np.asarray(engine.ends[-1], np.float64)
    clean = np.concatenate([end[:, 0], end[:, 1]], -1).reshape(-1)
    rng = np.random.default_rng(args.seed + 1000)
    return jnp.asarray(clean + args.sigma * rng.normal(size=clean.shape)), clean


def devices_of(args: argparse.Namespace) -> Any:
    """``None`` (one program) or the first ``--devices`` devices (per-segment mode)."""
    import jax

    if args.devices is None:
        return None
    return jax.devices()[: args.devices]


def make_driver(args: argparse.Namespace, scale: float) -> Any:
    """The augmented Lagrangian: one program, or per segment with ``--devices K``.

    L-BFGS in one program takes optax's zoom line search (inside the step); per
    segment the update is a direction and the host backtracks (track P's mode).
    """
    import optax

    from mimirax.inference.augmented_lagrangian import AugmentedLagrangian

    devices = devices_of(args)
    if args.optimizer == "lbfgs":
        rule = optax.lbfgs() if devices is None else optax.lbfgs(linesearch=None)
    else:
        rule = optax.adam(args.lr)
    return AugmentedLagrangian(
        optimizer=rule,
        inner_steps=args.inner_steps,
        max_outer=args.max_outer,
        tol=args.tol,
        violation_scale=scale,
        objective_rtol=args.objective_rtol,
        inner_rtol=args.inner_rtol,
        mu0=args.mu0,
        devices=devices,
        line_search=(
            "backtracking"
            if devices is not None and args.optimizer == "lbfgs"
            else None
        ),
    )


def setup(args: argparse.Namespace, rec: dict, compiles: Compiles) -> dict:
    """The engine planned at the truth, the observations, the start (nodes planned there)."""
    import jax.numpy as jnp

    x, v, m = system(args)
    engine = make_engine(args)
    log: list = []
    count_plans(engine, compiles, log)
    truth = theta_of(TRUTH)
    t0 = time.perf_counter()
    truth_nodes = engine.propagate(truth, x, v, m)
    rec["prepass_truth_s"] = round(time.perf_counter() - t0, 2)
    observed, clean = observe(args, engine)
    offset = [float(s) for s in args.offset.split(",")]
    start = theta_of([TRUTH[0] + offset[0], TRUTH[1] + offset[1]])
    t0 = time.perf_counter()
    start_nodes = engine.propagate(start, x, v, m)
    rec["prepass_start_s"] = round(time.perf_counter() - t0, 2)
    rec["prepasses"] = log
    scale = float(jnp.max(jnp.abs(jnp.concatenate([x, v], -1))))
    rec["violation_scale"] = scale
    return dict(
        x=x,
        v=v,
        m=m,
        engine=engine,
        truth=truth,
        truth_nodes=truth_nodes,
        observed=observed,
        clean=clean,
        start=start,
        start_nodes=start_nodes,
        scale=scale,
        log=log,
    )


# --------------------------------------------------------------------------
# Tasks.
# --------------------------------------------------------------------------


def task_prepass(args: argparse.Namespace, rec: dict, compiles: Compiles) -> None:
    x, v, m = system(args)
    truth = theta_of(TRUTH)
    offset = [float(s) for s in args.offset.split(",")]
    start = theta_of([TRUTH[0] + offset[0], TRUTH[1] + offset[1]])
    for freeze in (True, False):
        engine = make_engine(args, freeze_cache=freeze)
        log: list = []
        count_plans(engine, compiles, log)
        for label, theta in (
            ("truth_cold", truth),
            ("truth_warm", truth),
            ("start_warm", start),
        ):
            mark = compiles.mark()
            t0 = time.perf_counter()
            engine.propagate(theta, x, v, m)
            row = compiles.since(mark)
            row["wall_s"] = round(time.perf_counter() - t0, 2)
            row["per_segment"] = log[-args.segments :]
            rec.setdefault("prepass", {})[f"{label}_freeze{int(freeze)}"] = row
            print(
                f"pre-pass {label} freeze_cache={freeze}: {row['compiles']} compiles "
                f"({row['compile_s']} s), {row['wall_s']} s; per segment "
                f"{[(r['compiles'], r['wall_s']) for r in row['per_segment']]}",
                flush=True,
            )
        del engine
        gc.collect()


def _direct_segment(engine: Any, k: int) -> Any:
    """Segment ``k`` as ONE ``differentiable_rollout`` that computes its own start force.

    The runner's gradient plan (``GradientPlan.rollout``) evaluates a ``"start"``
    segment's start force OUTSIDE the rollout's hand-written reverse (under
    ``jax.checkpoint``, the field added eagerly so the forward is the runner's
    bit for bit); S9 session 2 (D, 374228f5) moved the plain rollout's start force
    INSIDE the adjoint, as unit 0 of its one reverse loop (W 775 -> 450 B/p at
    8e6). This variant passes ``acceleration=None`` with the plan's seed, the
    plan's provider and statics: the W the engine would have with that lever. Its
    forward is not the runner's bitwise (the field is added inside the program).
    """
    from jaccpot.production import _trace_state_kept
    from jaccpot.production_grad import _install_plan, _under_plan
    from jaccpot.runtime.grad_rollout import differentiable_rollout

    plan = engine.plans[k]
    (seg,) = plan.segments
    assert seg.kind == "start", seg.kind

    def run(theta: Any, x: Any, v: Any, m: Any) -> Any:
        import jax.numpy as jnp

        in_chain = [True]
        with _trace_state_kept(plan.engine):
            _install_plan(plan.engine, seg.plan)
            step_fn = _under_plan(
                plan.engine, seg.plan, seg.provider.differentiable_step, in_chain
            )
            out = differentiable_rollout(
                step_fn,
                x,
                v,
                m,
                plan.params["dt"],
                seg.steps,
                params=plan.params["force_params"],
                aux=(jnp.asarray(seg.seed), None, None),
                external_fn=plan.external_fn,
                external_args=theta,
                checkpoint="step",
            )
        in_chain[0] = False
        return out

    return run


def _segment_program(engine: Any, k: int, variant: str = "engine") -> Any:
    import jax
    import jax.numpy as jnp

    direct = _direct_segment(engine, k) if variant == "direct" else None

    def loss(theta: Any, x: Any, v: Any, m: Any, w: Any) -> Any:
        if direct is not None:
            out = direct(theta, x, v, m)
            value = jnp.sum(w * out.positions.astype(jnp.float64)) + jnp.sum(
                w * out.velocities.astype(jnp.float64)
            )
            return value, out.aux[1]
        end = engine(theta, x, v, m, k)
        value = jnp.sum(w * end.positions.astype(jnp.float64)) + jnp.sum(
            w * end.velocities.astype(jnp.float64)
        )
        return value, end.aux[1]

    return jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))


def segment_memory(
    args: argparse.Namespace, rec: dict, compiles: Compiles, steps: int
) -> dict:
    """The per-segment value-and-gradient programs of an engine with ``L = steps``."""
    import jax
    import jax.numpy as jnp

    x, v, m = system(args)
    engine = make_engine(args, steps=steps)
    log: list = []
    count_plans(engine, compiles, log)
    truth = theta_of(TRUTH)
    t0 = time.perf_counter()
    nodes = engine.propagate(truth, x, v, m)
    out: dict = {
        "steps": steps,
        "prepass_s": round(time.perf_counter() - t0, 2),
        "segments": [],
        "prepasses": log,
    }
    w = jnp.asarray(np.random.default_rng(5).normal(size=(args.n, 3)))
    for k in range(args.segments):
        mark = compiles.mark()
        t0 = time.perf_counter()
        compiled = (
            _segment_program(engine, k, args.segment_variant)
            .lower(truth, nodes["positions"][k], nodes["velocities"][k], m, w)
            .compile()
        )
        row = {
            "segment": k,
            "compile_s": round(time.perf_counter() - t0, 2),
            **compiles.since(mark),
        }
        row.update(_ma(compiled, args.n))
        before = _alloc()
        (value, bits), grads = compiled(
            truth, nodes["positions"][k], nodes["velocities"][k], m, w
        )
        jax.block_until_ready(grads)
        after = _alloc()
        row["bits"] = int(bits)
        row["finite"] = bool(
            all(
                bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(grads)
            )
        )
        row["grad_theta"] = theta_values(grads[0])
        row["bytes_in_use_before"] = before["bytes_in_use"]
        row["peak_after"] = after["peak_bytes_in_use"]
        row["peak_set_here"] = (after["peak_bytes_in_use"] or 0) > (
            before["peak_bytes_in_use"] or 0
        )
        row["model_peak"] = (
            (before["bytes_in_use"] or 0) + row["temp_bytes"] + row["output_bytes"]
        )
        row["peak_over_model"] = (
            round(row["peak_after"] / row["model_peak"], 4)
            if row["peak_after"]
            else None
        )
        out["segments"].append(row)
        print(
            f"L={steps} segment {k}: temp {row['temp_B_per_particle']} B/p, args "
            f"{row['argument_B_per_particle']}, out {row['output_B_per_particle']}; peak "
            f"{_bpp(row['peak_after'], args.n)} B/p (model {_bpp(row['model_peak'], args.n)}, "
            f"set here {row['peak_set_here']}); bits {row['bits']:#x}",
            flush=True,
        )
        del compiled, grads
        gc.collect()
    del engine
    gc.collect()
    return out


def constants_step(args: argparse.Namespace, driver: Any, problem: Any) -> Any:
    """Session 1's inner step: masses, node 0 and the observations CLOSED OVER.

    The body of ``AugmentedLagrangian._compile``'s step before track P made the
    data arguments: what the constant version costs in compile time and
    executable size. Called with the same ``data`` argument (unused) so the two
    lower alike.
    """
    import jax
    import optax

    from mimirax.inference.augmented_lagrangian import (
        _StepOut,
        _strong,
        augmented_lagrangian_value,
        defect_violation,
    )

    rule = optax.with_extra_args_support(driver.optimizer)

    def lagrangian(p: Any, multipliers: Any, mu: Any) -> Any:
        return augmented_lagrangian_value(problem, p, multipliers, mu)

    def step(p: Any, opt_state: Any, multipliers: Any, mu: Any, data: Any) -> Any:
        del data
        (value, evaluation), grads = jax.value_and_grad(lagrangian, has_aux=True)(
            p, multipliers, mu
        )
        updates, opt_state = rule.update(
            grads,
            opt_state,
            p,
            value=value,
            grad=grads,
            value_fn=lambda q: lagrangian(q, multipliers, mu)[0],
        )
        return _StepOut(
            params=_strong(optax.apply_updates(p, updates)),
            opt_state=_strong(opt_state),
            lagrangian=value,
            objective=evaluation.negative_log_posterior,
            violation=defect_violation(evaluation.defects),
            aux=evaluation.aux,
        )

    return jax.jit(step)


def _al_step_memory(
    args: argparse.Namespace,
    rec: dict,
    compiles: Compiles,
    s: dict,
    optimizer: str,
    data_args: bool = False,
) -> dict:
    import jax
    import jax.numpy as jnp

    from mimirax.inference.augmented_lagrangian import _strong

    args_opt = argparse.Namespace(
        **{**vars(args), "optimizer": optimizer, "devices": None}
    )
    model, problem = build_problem(
        args_opt, s["engine"], s["x"], s["v"], s["m"], s["observed"]
    )
    driver = make_driver(args_opt, s["scale"])
    counter = [0]
    step, _measure = driver._compile(problem, counter)
    data = problem.data_arrays()
    if not data_args:
        step = constants_step(args_opt, driver, problem)
    params = _strong({"theta": s["start"], "nodes": free_nodes(s["start_nodes"])})
    opt_state = _strong(driver.optimizer.init(params))
    shapes = jax.eval_shape(problem.evaluate, params).defects
    multipliers = tuple(jnp.zeros(d.shape, d.dtype) for d in shapes)
    mu = jnp.asarray(args.mu0, jnp.float64)
    mark = compiles.mark()
    t0 = time.perf_counter()
    lowered = step.lower(params, opt_state, multipliers, mu, data)
    text = lowered.as_text()
    compiled = lowered.compile()
    row = {
        "optimizer": optimizer,
        "data_args": data_args,
        "compile_s": round(time.perf_counter() - t0, 2),
        **compiles.since(mark),
    }
    row.update(_ma(compiled, args.n))
    nbytes = lambda tree: int(
        sum(a.size * a.dtype.itemsize for a in jax.tree_util.tree_leaves(tree))
    )  # noqa: E731
    row["params_bytes"] = nbytes(params)
    row["opt_state_bytes"] = nbytes(opt_state)
    row["multipliers_bytes"] = nbytes(multipliers)
    for key in ("params", "opt_state", "multipliers"):
        row[f"{key}_B_per_particle"] = _bpp(row[f"{key}_bytes"], args.n)
    row["hlo_text_MB"] = round(len(text) / 1e6, 2)
    # the closure constants: masses, node 0, the observations (large dense constants)
    row["hlo_large_constants"] = text.count("dense_resource") + text.count('dense<"0x')
    before = _alloc()
    out = compiled(params, opt_state, multipliers, mu, data)
    jax.block_until_ready(out.params)
    after = _alloc()
    row["bits"] = [int(a[1]) for a in out.aux]
    row["lagrangian"] = float(out.lagrangian)
    row["bytes_in_use_before"] = before["bytes_in_use"]
    row["peak_after"] = after["peak_bytes_in_use"]
    row["peak_set_here"] = (after["peak_bytes_in_use"] or 0) > (
        before["peak_bytes_in_use"] or 0
    )
    row["model_peak"] = (
        (before["bytes_in_use"] or 0) + row["temp_bytes"] + row["output_bytes"]
    )
    print(
        f"AL step ({optimizer}, data {'arguments' if data_args else 'constants'}): temp {row['temp_B_per_particle']} B/p, args "
        f"{row['argument_B_per_particle']} (params {row['params_B_per_particle']}, opt "
        f"{row['opt_state_B_per_particle']}, multipliers {row['multipliers_B_per_particle']}), out "
        f"{row['output_B_per_particle']}, code {row.get('generated_code_B_per_particle')}; peak "
        f"{_bpp(row['peak_after'], args.n)} B/p (model {_bpp(row['model_peak'], args.n)}); "
        f"compile {row['compile_s']} s; bits {row['bits']}",
        flush=True,
    )
    del compiled, out
    gc.collect()
    return row


def task_memory(args: argparse.Namespace, rec: dict, compiles: Compiles) -> None:
    lengths = [int(s) for s in (args.steps_list or str(args.steps)).split(",")]
    rec["segment_programs"] = [segment_memory(args, rec, compiles, L) for L in lengths]
    if len(lengths) >= 2:
        # temp = W + L r per particle: the two-point fit over the first segment
        rows_ = [
            (r["steps"], r["segments"][0]["temp_bytes"] / args.n)
            for r in rec["segment_programs"]
        ]
        (l1, t1), (l2, t2) = rows_[0], rows_[-1]
        r = (t2 - t1) / (l2 - l1)
        rec["fit"] = {
            "r_B_per_particle_per_step": round(r, 3),
            "W_B_per_particle": round(t1 - l1 * r, 2),
        }
        print(
            f"fit: W {rec['fit']['W_B_per_particle']} B/p, r {rec['fit']['r_B_per_particle_per_step']} B/p/step",
            flush=True,
        )
    if args.steps in lengths and not args.no_al_memory:
        s = setup(args, rec, compiles)
        rec["al_step"] = [
            _al_step_memory(args, rec, compiles, s, opt, data_args)
            for opt, data_args in (("adam", True), ("adam", False), ("lbfgs", True))
        ]


def _lagrangian_fn(args: argparse.Namespace, s: dict) -> tuple:
    """The jitted augmented Lagrangian at fixed multipliers (zero) and mu, with the auxes."""
    import jax
    import jax.numpy as jnp

    from mimirax.inference.augmented_lagrangian import augmented_lagrangian_value

    model, problem = build_problem(
        args, s["engine"], s["x"], s["v"], s["m"], s["observed"]
    )
    shapes = jax.eval_shape(
        problem.evaluate, {"theta": s["start"], "nodes": free_nodes(s["start_nodes"])}
    ).defects
    rng = np.random.default_rng(17)
    # nonzero multipliers so the multiplier term is exercised
    lam = tuple(jnp.asarray(rng.normal(size=d.shape) * 1e2, d.dtype) for d in shapes)
    mu = jnp.asarray(args.mu0, jnp.float64)

    def value(p: Any) -> Any:
        val, evaluation = augmented_lagrangian_value(problem, p, lam, mu)
        return val, tuple(a[1] for a in evaluation.aux)

    return (
        model,
        problem,
        jax.jit(value),
        jax.jit(jax.value_and_grad(value, has_aux=True)),
    )


def task_fd(args: argparse.Namespace, rec: dict, compiles: Compiles) -> None:
    import jax
    import jax.numpy as jnp

    s = setup(args, rec, compiles)
    model, problem, value, value_and_grad = _lagrangian_fn(args, s)
    p0 = {"theta": s["start"], "nodes": free_nodes(s["start_nodes"])}
    # the nodes moved off the propagated ends (a nonzero defect), so the defect
    # terms carry gradient into both segments
    d = jnp.asarray(
        np.random.default_rng(23).normal(size=p0["nodes"]["positions"].shape),
        jnp.float32,
    )
    p0 = {
        "theta": p0["theta"],
        "nodes": {
            "positions": p0["nodes"]["positions"] + 1e-3 * d,
            "velocities": p0["nodes"]["velocities"],
        },
    }
    mark = compiles.mark()
    (val0, bits0), grads = value_and_grad(p0)
    rec["gradient"] = {
        "value": float(val0),
        "bits": [int(b) for b in bits0],
        **compiles.since(mark),
    }
    from jaccpot.runtime.fused_topology import GRAD_FAIL_MASK

    rec["gradient"]["verdict_ok"] = all(
        (int(b) & int(GRAD_FAIL_MASK)) == 0 for b in bits0
    )
    exact = {
        "log_halo_mass": float(grads["theta"]["log_halo_mass"]),
        "log_halo_scale": float(grads["theta"]["log_halo_scale"]),
        "node_positions_along_d": float(
            jnp.sum(grads["nodes"]["positions"].astype(jnp.float64) * d)
        ),
    }
    rec["exact"] = exact
    print(
        f"value {float(val0):.9e}, bits {rec['gradient']['bits']}, exact {exact}",
        flush=True,
    )

    def moved(which: str, h: float) -> dict:
        if which == "node_positions_along_d":
            return {
                "theta": p0["theta"],
                "nodes": {**p0["nodes"], "positions": p0["nodes"]["positions"] + h * d},
            }
        th = dict(p0["theta"])
        th[which] = th[which] + jnp.asarray(h, jnp.float32)
        return {"theta": th, "nodes": p0["nodes"]}

    hs = [float(h) for h in args.fd_h.split(",")]
    rec["fd"] = {}
    for which in exact:
        rows_ = []
        for h in hs:
            scale_h = h if which != "node_positions_along_d" else h * 1e-2
            plus, bp = value(moved(which, scale_h))
            minus, bm = value(moved(which, -scale_h))
            # the step actually taken in float32 (theta's ulp)
            if which != "node_positions_along_d":
                taken = float(moved(which, scale_h)["theta"][which]) - float(
                    moved(which, -scale_h)["theta"][which]
                )
            else:
                taken = 2 * scale_h
            fd = (float(plus) - float(minus)) / taken
            err = abs(fd - exact[which]) / abs(exact[which])
            rows_.append(
                {
                    "h": scale_h,
                    "fd": fd,
                    "rel_err": err,
                    "bits": [int(b) for b in bp] + [int(b) for b in bm],
                }
            )
            print(
                f"  {which} h={scale_h:.1e}: fd {fd:.6e} vs {exact[which]:.6e}: rel {err:.2e}",
                flush=True,
            )
        rec["fd"][which] = rows_
    # the objective's round-off noise: the same point, nudged by one ulp in theta
    th = dict(p0["theta"])
    th["log_halo_mass"] = jnp.nextafter(
        th["log_halo_mass"], jnp.asarray(10.0, jnp.float32)
    )
    nudged, _ = value({"theta": th, "nodes": p0["nodes"]})
    again, _ = value(p0)
    rec["noise"] = {
        "value_a_vs_a": float(again) - float(val0),
        "value_one_ulp_theta": float(nudged) - float(val0),
        "expected_one_ulp": exact["log_halo_mass"]
        * (float(th["log_halo_mass"]) - float(p0["theta"]["log_halo_mass"])),
    }
    print(f"noise: {rec['noise']}", flush=True)


def fisher(args: argparse.Namespace, s: dict, h: float = 1e-2) -> dict:
    """The Fisher matrix of theta from central differences of the prediction."""
    import jax

    engine = s["engine"]
    model, _ = build_problem(args, engine, s["x"], s["v"], s["m"], s["observed"])
    # single shooting on the same engine: the nodes propagated at each theta
    jac = []
    for i in range(2):
        cols = []
        for sign in (1.0, -1.0):
            th = list(TRUTH)
            th[i] += sign * h
            engine.propagate(theta_of(th), s["x"], s["v"], s["m"])
            end = np.asarray(engine.ends[-1], np.float64)
            cols.append(np.concatenate([end[:, 0], end[:, 1]], -1).reshape(-1))
        jac.append((cols[0] - cols[1]) / (2 * h))
    jac = np.stack(jac, 1)
    info = jac.T @ jac / args.sigma**2
    cov = np.linalg.inv(info)
    del jax
    return {
        "fisher": info.tolist(),
        "std": np.sqrt(np.diag(cov)).tolist(),
        "cov": cov.tolist(),
        "h": h,
    }


def task_al(args: argparse.Namespace, rec: dict, compiles: Compiles) -> None:
    import jax

    s = setup(args, rec, compiles)
    engine = s["engine"]
    model, problem = build_problem(args, engine, s["x"], s["v"], s["m"], s["observed"])
    driver = make_driver(args, s["scale"])
    params = {"theta": s["start"], "nodes": free_nodes(s["start_nodes"])}
    plans_before = engine.stats["plans"]
    mark = compiles.mark()
    t0 = time.perf_counter()
    # progress: one line per verdict (every step, redo, line-search trial and
    # measurement checks the segments once)
    inner_check = engine.check
    progress: list = []

    def check(auxes: Any) -> bool:
        ok = inner_check(auxes)
        progress.append((round(time.perf_counter() - t0, 1), bool(ok)))
        if len(progress) % 10 == 1 or not ok:
            print(
                f"  check {len(progress)} at {progress[-1][0]} s: ok {ok}, "
                f"plans {engine.stats['plans']}, compiles {compiles.count}",
                flush=True,
            )
        return ok

    engine.check = check
    devs = devices_of(args) or jax.devices()[:1]
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    rec["layout"] = [
        {
            "segment": k,
            "device": str(devs[k % len(devs)]),
            "physical_card": cvd[devs[k % len(devs)].id] if cvd != [""] else None,
        }
        for k in range(args.segments)
    ]
    result = driver.minimize(problem, params)
    engine.check = inner_check
    rec["checks_timeline"] = progress
    rec["al_s"] = round(time.perf_counter() - t0, 1)
    rec["al_compiles"] = compiles.since(mark)
    rec["after_al"] = _alloc()
    est = theta_values(result.params["theta"])
    rec["result"] = {
        "converged": bool(result.converged),
        "theta": est,
        "error": [e - t for e, t in zip(est, TRUTH)],
        "start_error": [float(o) for o in args.offset.split(",")],
        "outer": int(len(result.outer_steps)),
        "inner": int(len(result.objective_trace)),
        "outer_steps": [int(k) for k in result.outer_steps],
        "outer_violation": [float(k) for k in result.outer_violation],
        "outer_violation_relative": [
            float(k) / s["scale"] for k in result.outer_violation
        ],
        "outer_objective": [float(k) for k in result.outer_objective],
        "outer_mu": [float(k) for k in result.outer_mu],
        "num_traces": int(result.num_traces),
        "num_redos": int(result.num_redos),
        "replans": int(engine.stats["plans"] - plans_before),
        "checks": int(engine.stats["checks"]),
        "objective_trace_tail": [float(k) for k in result.objective_trace[-10:]],
        "violation_trace_tail": [float(k) for k in result.violation_trace[-10:]],
    }
    import jax

    rec["result"]["peak_per_device"] = {
        str(d): (d.memory_stats() or {}).get("peak_bytes_in_use")
        for d in (devices_of(args) or jax.devices()[:1])
    }
    rec["result"]["line_search_trials"] = None
    # the objective at the truth (its trajectory: defects zero): the noise's chi^2 / 2
    clean, observed = np.asarray(s["clean"]), np.asarray(s["observed"])
    rec["result"]["objective_at_truth"] = float(
        0.5 * np.sum(((clean - observed) / args.sigma) ** 2)
    )
    rec["result"]["fisher"] = fisher(args, s)
    std = rec["result"]["fisher"]["std"]
    rec["result"]["error_in_sigma"] = [
        e / sd for e, sd in zip(rec["result"]["error"], std)
    ]
    print(json.dumps(rec["result"], indent=1), flush=True)


def task_parity(args: argparse.Namespace, rec: dict, compiles: Compiles) -> None:
    """The per-segment mode's summed value and gradient against the one program.

    At the start (nodes moved off the propagated ends by 1e-3, nonzero
    multipliers), ``AugmentedLagrangian``'s one-program Lagrangian (the data as
    arguments) on device 0 against ``ParallelSegmentObjective`` on
    ``--devices K`` (segment k on device k mod K): the value and every gradient
    leaf, max abs difference relative to the leaf's max abs, and the bitwise
    count. Records which segment ran on which device (and physical card).
    """
    import jax
    import jax.numpy as jnp

    from mimirax.inference.augmented_lagrangian import augmented_lagrangian_value
    from mimirax.inference.parallel import ParallelSegmentObjective

    s = setup(args, rec, compiles)
    model, problem = build_problem(
        args, s["engine"], s["x"], s["v"], s["m"], s["observed"]
    )
    p0 = {"theta": s["start"], "nodes": free_nodes(s["start_nodes"])}
    d = jnp.asarray(
        np.random.default_rng(23).normal(size=p0["nodes"]["positions"].shape),
        jnp.float32,
    )
    p0["nodes"] = {**p0["nodes"], "positions": p0["nodes"]["positions"] + 1e-3 * d}
    rng = np.random.default_rng(17)
    shape = (args.segments - 1, args.n, 6)
    lam = (jnp.asarray(rng.normal(size=shape) * 1e2, jnp.float32),)
    mu = jnp.asarray(args.mu0, jnp.float64)

    def one(p: Any, data: Any) -> Any:
        bound = problem.with_data_arrays(data)
        value, evaluation = augmented_lagrangian_value(bound, p, lam, mu)
        return value, tuple(a[1] for a in evaluation.aux)

    (v1, bits1), g1 = jax.jit(jax.value_and_grad(one, has_aux=True))(
        p0, problem.data_arrays()
    )
    devices = devices_of(args) or jax.devices()[:1]
    objective = ParallelSegmentObjective(problem, devices)
    contributions, auxes = objective.dispatch(p0, lam, mu)
    v2, g2, _f, _defects = jax.jit(objective.assemble)(p0, contributions)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    rec["layout"] = [
        {
            "segment": k,
            "device": str(dev),
            "physical_card": cvd[dev.id] if dev.id < len(cvd) and cvd != [""] else None,
        }
        for k, dev in enumerate(objective.segment_devices)
    ]
    rows = {
        "value": {
            "one": float(v1),
            "split": float(v2),
            "diff": float(v2) - float(v1),
            "relative": abs(float(v2) - float(v1)) / abs(float(v1)),
        }
    }
    leaves1, tree = jax.tree_util.tree_flatten_with_path(g1)
    leaves2 = jax.tree_util.tree_leaves(g2)
    for (path, a), b in zip(leaves1, leaves2):
        a, b = np.asarray(a), np.asarray(jax.device_get(b))
        name = jax.tree_util.keystr(path)
        rows[name] = {
            "max_abs_diff_rel": float(
                np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300)
            ),
            "elements_differ": int(np.sum(a != b)),
            "size": int(a.size),
        }
    rec["parity"] = rows
    rec["bits_one"] = [int(b) for b in bits1]
    rec["bits_split"] = [int(a[1]) for a in auxes]
    print(json.dumps({"layout": rec["layout"], "parity": rows}, indent=1), flush=True)


def _reps(fn: Any, reps: int) -> dict:
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    q25, q75 = np.percentile(ts, [25, 75])
    return {
        "s": ts,
        "min_s": float(np.min(ts)),
        "median_s": float(np.median(ts)),
        "iqr_s": float(q75 - q25),
    }


def task_timing(args: argparse.Namespace, rec: dict, compiles: Compiles) -> None:
    """One program kind (``--only``), ``--reps`` warm calls after a cold one.

    * ``prepass``: ``propagate`` (``S`` pre-passes) alternating the truth's and the
      start's ``theta`` (other nodes each time: a re-plan's cost), with
      ``freeze_cache`` unless ``--no-freeze-cache``;
    * ``seggrad``: each segment's value-and-gradient program at the truth's nodes;
    * ``alstep``: one augmented-Lagrangian inner step (every segment's gradient and
      the optimizer's update; L-BFGS's line search included) at the start.
    """
    import jax
    import jax.numpy as jnp

    from mimirax.inference.augmented_lagrangian import _strong

    only = args.only
    timing: dict = {}
    rec["timing"] = timing
    if only == "prepass":
        x, v, m = system(args)
        engine = make_engine(args, freeze_cache=not args.no_freeze_cache)
        log: list = []
        count_plans(engine, compiles, log)
        truth = theta_of(TRUTH)
        offset = [float(s_) for s_ in args.offset.split(",")]
        start = theta_of([TRUTH[0] + offset[0], TRUTH[1] + offset[1]])
        mark = compiles.mark()
        t0 = time.perf_counter()
        engine.propagate(truth, x, v, m)
        timing["cold"] = {"s": time.perf_counter() - t0, **compiles.since(mark)}
        thetas = [start, truth]
        state = {"i": 0}

        def once() -> None:
            engine.propagate(thetas[state["i"] % 2], x, v, m)
            state["i"] += 1

        mark = compiles.mark()
        timing["warm"] = {**_reps(once, args.reps), **compiles.since(mark)}
        timing["warm"]["per_segment_median_s"] = (
            timing["warm"]["median_s"] / args.segments
        )
        timing["freeze_cache"] = not args.no_freeze_cache
        rec["prepasses"] = log
    elif only == "seggrad":
        x, v, m = system(args)
        engine = make_engine(args)
        truth = theta_of(TRUTH)
        nodes = engine.propagate(truth, x, v, m)
        w = jnp.asarray(np.random.default_rng(5).normal(size=(args.n, 3)))
        for k in range(args.segments):
            call = (truth, nodes["positions"][k], nodes["velocities"][k], m, w)
            t0 = time.perf_counter()
            compiled = _segment_program(engine, k).lower(*call).compile()
            compile_s = time.perf_counter() - t0
            out = compiled(*call)
            jax.block_until_ready(out)
            bits = int(out[0][1])
            del out
            row = _reps(lambda: jax.block_until_ready(compiled(*call)), args.reps)
            row.update(compile_s=compile_s, bits=bits, **_ma(compiled, args.n))
            row["per_step_ms"] = 1e3 * row["median_s"] / args.steps
            timing[f"segment_{k}"] = row
            del compiled
            gc.collect()
    else:
        # one inner iteration as the driver runs it: prepare (a no-op on valid
        # plans), the step, the verdict, the accept (per segment: the host line
        # search's trial forwards)
        from mimirax.inference.augmented_lagrangian import _host_check, _host_prepare

        s = setup(args, rec, compiles)
        engine = s["engine"]
        model, problem = build_problem(
            args, engine, s["x"], s["v"], s["m"], s["observed"]
        )
        driver = make_driver(args, s["scale"])
        ev = driver._evaluator(problem)
        params = ev.place(
            _strong({"theta": s["start"], "nodes": free_nodes(s["start_nodes"])})
        )
        opt_state = _strong(driver.optimizer.init(params))
        lam = ev.place(
            tuple(jnp.zeros(d.shape, d.dtype) for d in ev.defect_shapes(params))
        )
        mu = jnp.asarray(args.mu0, jnp.float64)
        verdicts: list = []

        def iteration() -> None:
            _host_prepare(model, params)
            out = ev.step(params, opt_state, lam, mu)
            verdicts.append(_host_check(model, out.aux))
            out = ev.accept(out)
            jax.block_until_ready(out.params)

        mark = compiles.mark()
        t0 = time.perf_counter()
        iteration()
        cold = {"s": time.perf_counter() - t0, **compiles.since(mark)}
        trials0 = getattr(ev, "trials", 0)
        row = _reps(iteration, args.reps)
        row.update(
            cold=cold,
            verdicts=verdicts,
            optimizer=args.optimizer,
            devices=args.devices,
            num_traces=ev.num_traces,
            line_search_trials_per_iteration=(getattr(ev, "trials", 0) - trials0)
            / args.reps,
        )
        row["peak_per_device"] = [
            {"device": str(d), **(d.memory_stats() or {})}
            for d in (devices_of(args) or jax.devices()[:1])
        ]
        for dev in row["peak_per_device"]:
            dev.update({k: dev.get(k) for k in ("peak_bytes_in_use", "bytes_in_use")})
            for key in [
                k
                for k in dev
                if k
                not in ("device", "peak_bytes_in_use", "bytes_in_use", "bytes_limit")
            ]:
                del dev[key]
        timing["al_iteration"] = row
    for key, row in timing.items():
        if isinstance(row, dict) and "median_s" in row:
            print(
                f"{only} {key}: median {row['median_s']:.3f} s (min {row['min_s']:.3f}, IQR {row['iqr_s']:.3f})",
                flush=True,
            )


def main(argv: Optional[list] = None) -> int:
    args = _args(argv)
    import jax

    jax.config.update("jax_enable_x64", True)
    compiles = Compiles()
    rec: dict = {
        "config": {**vars(args), "softening_used": softening(args)},
        "env": {k: os.environ.get(k) for k in _ENV_KEYS},
        "device": str(jax.devices()[0]),
        "jax": jax.__version__,
        "card_before": _card_state(),
        "heads": _heads(),
    }
    print(
        f"{args.task}: N {args.n}, {args.segments} x {args.steps} steps, dt {args.dt}, "
        f"eps {softening(args):.4g} on {rec['device']}",
        flush=True,
    )
    t0 = time.perf_counter()
    {
        "prepass": task_prepass,
        "memory": task_memory,
        "fd": task_fd,
        "al": task_al,
        "timing": task_timing,
        "parity": task_parity,
    }[args.task](args, rec, compiles)
    rec["wall_s"] = round(time.perf_counter() - t0, 1)
    rec["compiles_total"] = {
        "count": compiles.count,
        "seconds": round(compiles.seconds, 1),
    }
    rec["final_alloc"] = _alloc()
    rec["card_after"] = _card_state()
    rec["contended"] = (
        rec["card_before"].get("other_processes") != "none"
        or rec["card_after"].get("other_processes") != "none"
    )
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(rec, fh, indent=1, default=str)
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
