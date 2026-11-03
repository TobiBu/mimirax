"""The system, the forward models and the bookkeeping shared by the study's scripts.

THE SYSTEM. A self-gravitating stellar system -- a Plummer sphere of ``N``
equal-mass stars, total mass ``M_STARS`` and scale ``A_STARS`` -- sitting at the
centre of nornax's analytic galaxy: session 1's Miyamoto-Nagai disc field
(fixed) plus an NFW halo whose ``(log M, log r_s)`` are the unknowns ``theta``.
The stars' own gravity and the galaxy's are of the same order inside the system
(the virial ``W`` of the galaxy is 1.3x the stars' own), so the halo enters
every star's orbit at order one, and the stars' mutual encounters make the
motion chaotic. The velocities are the Plummer distribution function's, scaled
so the system starts in virial balance in the TOTAL potential. Why this system
and not a cluster on an orbit through the galaxy, or a self-gravitating disc:
``README.md`` (measured in ``lyapunov.py``).

THE ENGINE. Session 1's :class:`~mimirax.adapters.jaccpot.DirectSumSegmentEngine`:
jaccpot's ``differentiable_rollout`` (velocity Verlet) over the softened ``O(N^2)``
direct sum (jaccpot's default ``ferrers3`` kernel) plus the galaxy through
:func:`~mimirax.adapters.jaccpot.rows_field`, ``checkpoint="step"`` for gradients
and ``checkpoint=None`` for forward-mode tangents. Float64 throughout.

THE DATA. The full phase-space state at the end of every segment -- a time
series of ``S`` snapshots, ``SEGMENT_STEPS`` steps apart -- with Gaussian noise
``SIGMA``. Single and multiple shooting see the same snapshots: single shooting
through :class:`SeriesModel` (one chained rollout, params ``{"theta"}``),
multiple shooting through :class:`~mimirax.MultipleShootingForwardModel` with
``output="segment_ends"``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
if os.environ.get("LONG_HORIZON_DEVICE"):  # pick one of several visible devices
    jax.config.update(
        "jax_default_device", jax.devices()[int(os.environ["LONG_HORIZON_DEVICE"])]
    )

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from nornax import potentials  # noqa: E402

from mimirax import (  # noqa: E402
    GaussianLikelihood,
    InferenceProblem,
    MultipleShootingForwardModel,
)
from mimirax.adapters.jaccpot import DirectSumSegmentEngine  # noqa: E402
from mimirax.shooting import SegmentResult  # noqa: E402

# --- the galaxy (session 1's, tests/integration/test_shooting_galaxy_potentials.py) ---
DISC = potentials.MiyamotoNagaiParams(M=1.0, a=1.0, b=0.1)
GALAXY = potentials.CompositePotential(
    disc=potentials.MiyamotoNagai(), halo=potentials.NFW()
)
TRUTH = {
    "log_halo_mass": jnp.log(jnp.asarray(5.0)),
    "log_halo_scale": jnp.log(jnp.asarray(3.0)),
}
NAMES = tuple(TRUTH)

# --- the stars ---------------------------------------------------------------------
N_STARS = 1024
M_STARS = 0.2
A_STARS = 0.3
#: softening (Plummer-equivalent length) in units of A_STARS
SOFTENING_OVER_A = 0.05
DT = 0.02
SEED = 0

# --- the data ----------------------------------------------------------------------
#: steps between snapshots = steps per multiple-shooting segment
SEGMENT_STEPS = 160
SIGMA = 1.0e-3
#: the longest horizon, in segments (the noise is drawn for it)
MAX_SEGMENTS = 16

#: where the per-configuration result files go (one JSON per configuration)
RUNS = Path(
    os.environ.get(
        "LONG_HORIZON_RUNS",
        str(Path(__file__).resolve().parents[1] / "results" / "long_horizon_runs"),
    )
)
RESULTS = Path(__file__).resolve().parents[1] / "results"


def galaxy_field(x: Any, t: Any, theta: Any) -> Any:
    """The disc field (fixed) plus the NFW halo whose parameters are ``theta``.

    Parameters
    ----------
    x : Any
        ``[N, 3]`` positions.
    t : Any
        Time (ignored: the galaxy is static).
    theta : Any
        ``{"log_halo_mass", "log_halo_scale"}``.

    Returns
    -------
    Any
        ``[N, 3]`` accelerations.
    """
    halo = potentials.NFWParams(
        M=jnp.exp(theta["log_halo_mass"]), r_s=jnp.exp(theta["log_halo_scale"])
    )
    return GALAXY(x, t, {"disc": DISC, "halo": halo})


def galaxy_potential(x: Any, theta: Any) -> Any:
    """The galaxy's potential per position.

    Parameters
    ----------
    x : Any
        ``[N, 3]`` positions.
    theta : Any
        The halo's parameters.

    Returns
    -------
    Any
        ``[N]`` potentials.
    """
    halo = potentials.NFWParams(
        M=jnp.exp(theta["log_halo_mass"]), r_s=jnp.exp(theta["log_halo_scale"])
    )
    return GALAXY.potential(x, 0.0, {"disc": DISC, "halo": halo})


def plummer(
    n: int, mass: float, scale: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a Plummer sphere (Aarseth, Henon & Wielen 1974), ``G = 1``, centred.

    Radii beyond ``20 scale`` are redrawn one by one.

    Parameters
    ----------
    n : int
        Particles.
    mass : float
        Total mass.
    scale : float
        Plummer scale.
    rng : np.random.Generator
        The random stream.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Positions ``[n, 3]``, velocities ``[n, 3]`` (self-consistent, isotropic)
        and masses ``[n]``, centre of mass at rest at the origin.
    """
    r = np.empty(n)
    for i in range(n):
        while True:
            u = rng.uniform(1.0e-12, 1.0)
            r[i] = scale / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
            if r[i] < 20.0 * scale:
                break

    def directions() -> np.ndarray:
        cos_t = rng.uniform(-1.0, 1.0, n)
        phi = rng.uniform(0.0, 2.0 * np.pi, n)
        sin_t = np.sqrt(1.0 - cos_t**2)
        return np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], -1)

    x = r[:, None] * directions()
    q = np.empty(n)
    for i in range(n):
        while True:
            a, b = rng.uniform(0.0, 1.0, 2)
            if 0.1 * b < a * a * (1.0 - a * a) ** 3.5:
                q[i] = a
                break
    speed = q * np.sqrt(2.0 * mass / np.sqrt(r**2 + scale**2))
    v = speed[:, None] * directions()
    x -= x.mean(0)
    v -= v.mean(0)
    return x, v, np.full(n, mass / n)


def softening(scale: float = A_STARS, over_a: float = SOFTENING_OVER_A) -> float:
    """The softening length.

    Parameters
    ----------
    scale : float
        The Plummer scale.
    over_a : float
        Softening in units of the scale.

    Returns
    -------
    float
        ``over_a * scale``.
    """
    return float(over_a * scale)


@dataclass(frozen=True)
class System:
    """An initial condition and the numbers that set its time scales.

    Attributes
    ----------
    positions : Any
        ``[N, 3]``.
    velocities : Any
        ``[N, 3]``.
    masses : Any
        ``[N]``.
    softening : float
        The softening length.
    half_mass_radius : float
        Median 3-D radius at t = 0.
    rms_speed : float
        ``sqrt(<v^2>)`` at t = 0.
    crossing_time : float
        ``half_mass_radius / rms_speed``.
    virial_scale : float
        The factor the Plummer velocities were scaled by.
    """

    positions: Any
    velocities: Any
    masses: Any
    softening: float
    half_mass_radius: float
    rms_speed: float
    crossing_time: float
    virial_scale: float


def build_system(
    n: int = N_STARS,
    over_a: float = SOFTENING_OVER_A,
    seed: int = SEED,
    mass: float = M_STARS,
    scale: float = A_STARS,
) -> System:
    """The stars at the galaxy's centre, in virial balance in the total potential.

    The Plummer velocities are multiplied by ``s = sqrt(W_total / 2K)``, with
    ``W = -sum m x . a`` over the stars' own softened force plus the galaxy's at
    the truth, so that ``2K = W`` at t = 0.

    Parameters
    ----------
    n : int
        Stars.
    over_a : float
        Softening in units of ``scale``.
    seed : int
        Seed of the sample.
    mass : float
        Total stellar mass.
    scale : float
        Plummer scale.

    Returns
    -------
    System
        The initial condition and its scales.
    """
    from jaccpot.runtime.grad_rollout import (  # pyright: ignore[reportMissingImports]
        direct_sum_step_fn,
    )

    x, v, m = plummer(n, mass, scale, np.random.default_rng(seed))
    x, v, m = jnp.asarray(x), jnp.asarray(v), jnp.asarray(m)
    eps = softening(scale, over_a)
    a_self, _ = direct_sum_step_fn(softening=eps)(x, m)
    a_total = a_self + galaxy_field(x, 0.0, TRUTH)
    w_total = -jnp.sum(m[:, None] * x * a_total)
    two_k = jnp.sum(m[:, None] * v * v)
    s = float(jnp.sqrt(w_total / two_k))
    v = v * s
    r_half = float(jnp.median(jnp.linalg.norm(x, axis=1)))
    v_rms = float(jnp.sqrt(jnp.mean(jnp.sum(v * v, axis=1))))
    return System(x, v, m, eps, r_half, v_rms, r_half / v_rms, s)


def energy(system: System, x: Any, v: Any, theta: Any = None) -> float:
    """Total energy: kinetic, the stars' softened pair energy, the galaxy's.

    Parameters
    ----------
    system : System
        Masses and softening.
    x : Any
        ``[N, 3]`` positions.
    v : Any
        ``[N, 3]`` velocities.
    theta : Any
        The halo (``None``: the truth).

    Returns
    -------
    float
        ``E`` (the self-pair constant included; it does not move).
    """
    from jaccpot.runtime.reference import (  # pyright: ignore[reportMissingImports]
        compute_gravitational_potential,
    )

    theta = TRUTH if theta is None else theta
    m = system.masses
    phi = compute_gravitational_potential(x, m, x, softening=system.softening)
    kinetic = 0.5 * jnp.sum(m * jnp.sum(v * v, axis=1))
    return float(
        kinetic + 0.5 * jnp.sum(m * phi) + jnp.sum(m * galaxy_potential(x, theta))
    )


def engine(
    segment_steps: tuple[int, ...],
    system: System,
    checkpoint: str | None = "step",
    dt: float = DT,
) -> DirectSumSegmentEngine:
    """Session 1's direct-sum segment engine on this system.

    Parameters
    ----------
    segment_steps : tuple[int, ...]
        Steps per segment.
    system : System
        Supplies the softening.
    checkpoint : str | None
        ``"step"`` for gradients, ``None`` for forward-mode tangents.
    dt : float
        Timestep.

    Returns
    -------
    DirectSumSegmentEngine
        The engine.
    """
    return DirectSumSegmentEngine(
        segment_steps=tuple(segment_steps),
        dt=dt,
        field=galaxy_field,
        softening=system.softening,
        checkpoint=checkpoint,
    )


class CountingEngine:
    """A segment engine that counts its segment passes on the host.

    Every traced call of the wrapped engine carries a ``jax.debug.callback``
    that increments :attr:`passes` when the compiled program runs it -- once per
    segment per evaluation, the line search's trial points included (the callback
    sits outside the rollout's ``custom_vjp``, in the primal pass, so a
    value-and-gradient evaluation counts once). Pure otherwise: the wrapped
    engine's results are returned unchanged.

    Parameters
    ----------
    inner : DirectSumSegmentEngine
        The engine to count.
    """

    def __init__(self, inner: DirectSumSegmentEngine) -> None:
        self.inner = inner
        self.passes = 0

    def _tick(self) -> None:
        """Count one segment pass (host)."""
        self.passes += 1

    @property
    def num_segments(self) -> int:
        """Return the wrapped engine's segment count.

        Returns
        -------
        int
            ``len(segment_steps)``.
        """
        return self.inner.num_segments

    def __call__(
        self, theta: Any, positions: Any, velocities: Any, masses: Any, segment: int
    ) -> SegmentResult:
        """Run the wrapped engine's segment and count it.

        Parameters
        ----------
        theta : Any
            The halo.
        positions : Any
            ``[N, 3]``.
        velocities : Any
            ``[N, 3]``.
        masses : Any
            ``[N]``.
        segment : int
            Which segment. Static.

        Returns
        -------
        SegmentResult
            The wrapped engine's result.
        """
        jax.debug.callback(self._tick)
        return self.inner(theta, positions, velocities, masses, segment)


class FitTimeCap(Exception):
    """Raised by :meth:`ProgressLog.record` when a fit exceeds its wall-time cap."""


class ProgressLog:
    """Host-side progress of one fit, written while it runs.

    The augmented-Lagrangian driver calls the forward model's ``prepare(params)``
    on the host, with concrete parameters, before every inner step; the models
    below forward that to :meth:`record`, which keeps the step count and the
    latest ``theta`` and every :attr:`every` steps writes them (with the segment
    passes so far) to :attr:`path` -- so a fit stopped by a time limit still
    leaves where it got to. Inactive while :attr:`path` is ``None``.

    Parameters
    ----------
    counter : CountingEngine
        Supplies the pass count.
    every : int
        Steps between writes.
    """

    def __init__(self, counter: "CountingEngine", every: int = 25) -> None:
        self.counter = counter
        self.every = every
        self.path: Path | None = None
        self.steps = 0
        self.start = 0.0
        self.meta: dict[str, Any] = {}
        #: ``[inner steps, segment passes, max |theta - theta*|]`` at every write
        self.history: list[list[float]] = []

    def restore(self, steps: int, history: list[list[float]], elapsed: float) -> None:
        """Continue a resumed fit's log: its step count, history and wall time so far.

        Parameters
        ----------
        steps : int
            Inner steps the fit had taken.
        history : list[list[float]]
            Its :attr:`history` so far.
        elapsed : float
            Wall seconds it had run (information only).
        """
        import time

        self.steps, self.history = int(steps), [list(h) for h in history]
        self.start = time.time() - float(elapsed)

    def reset(self, path: Path | None, **meta: Any) -> None:
        """Start (or, with ``path=None``, stop) logging a fit.

        Parameters
        ----------
        path : Path | None
            Where to write the partial record.
        **meta : Any
            Copied into every record; ``cap_s`` (popped) is a wall-time cap in
            seconds: past it the next write raises :class:`FitTimeCap`.
        """
        import time

        self.path, self.steps, self.start, self.meta = path, 0, time.time(), meta
        self.cap_s = float(meta.pop("cap_s", 0.0) or 0.0)
        if path is not None:
            self.history: list[list[float]] = []

    def record(self, theta: Any) -> None:
        """Count one inner step; write the partial record every :attr:`every`.

        Parameters
        ----------
        theta : Any
            The step's (concrete) ``theta``.

        Raises
        ------
        FitTimeCap
            When the fit has run past its wall-time cap.
        """
        import time

        if self.path is None:
            return
        self.steps += 1
        if self.steps % self.every:
            return
        values = theta_array(theta)
        error = values - theta_array(TRUTH)
        self.history.append(
            [self.steps, self.counter.passes, float(np.max(np.abs(error)))]
        )
        write_json(
            self.path,
            {
                **self.meta,
                "inner_steps_so_far": self.steps,
                "segment_passes_so_far": self.counter.passes,
                "theta": values.tolist(),
                "error": error.tolist(),
                "elapsed_s_information_only": time.time() - self.start,
                "history": self.history,
            },
        )
        if self.cap_s and time.time() - self.start > self.cap_s:
            raise FitTimeCap(f"{self.steps} inner steps in {self.cap_s:.0f} s")


@dataclass(frozen=True)
class LoggedShootingModel(MultipleShootingForwardModel):
    """:class:`~mimirax.MultipleShootingForwardModel` reporting to a :class:`ProgressLog`.

    Attributes
    ----------
    log : ProgressLog | None
        Gets every ``prepare``'s ``theta``.
    """

    log: ProgressLog | None = None

    def prepare(self, params: Any) -> None:
        """Delegate to the base class, then record the step.

        Parameters
        ----------
        params : Any
            Concrete parameters of the next step.
        """
        super().prepare(params)
        if self.log is not None:
            self.log.record(params["theta"])


@dataclass(frozen=True)
class SeriesModel:
    """Single shooting with the time-series data: one chained rollout, params ``{"theta"}``.

    Runs the ``S`` segments of a :class:`~mimirax.MultipleShootingForwardModel`
    back to back, each from the previous one's end (so ONE trajectory: a
    segment's start force is the previous segment's end force at the same
    positions, bit for bit -- session 1's gate), and predicts every segment end:
    exactly what the multiple-shooting model with ``output="segment_ends"``
    predicts once its defects vanish. No ``predict_with_defects``, so the
    augmented-Lagrangian driver runs it as a plain optax loop.

    Attributes
    ----------
    model : MultipleShootingForwardModel
        Supplies the engine, masses, initial condition and ``S``.
    log : ProgressLog | None
        Gets every ``prepare``'s ``theta``.
    """

    model: MultipleShootingForwardModel
    log: ProgressLog | None = None

    def prepare(self, params: Any) -> None:
        """Record the step (host; the driver calls it before every inner step).

        Parameters
        ----------
        params : Any
            Concrete ``{"theta": ...}``.
        """
        if self.log is not None:
            self.log.record(params["theta"])

    def __call__(self, params: Any) -> dict[str, Any]:
        """Return every segment end along the one trajectory at ``params["theta"]``.

        Parameters
        ----------
        params : Any
            ``{"theta": ...}``.

        Returns
        -------
        dict[str, Any]
            ``{"positions": [S, N, 3], "velocities": [S, N, 3]}``.
        """
        m = self.model
        theta = params["theta"]
        x, v = jnp.asarray(m.initial_positions), jnp.asarray(m.initial_velocities)
        xs, vs = [], []
        for k in range(m.num_segments):
            end = m.engine(theta, x, v, m.masses, k)
            x, v = end.positions, end.velocities
            xs.append(x)
            vs.append(v)
        return {"positions": jnp.stack(xs), "velocities": jnp.stack(vs)}


def phase_space(state: dict[str, Any]) -> Any:
    """Flatten a predicted state into the data vector.

    Parameters
    ----------
    state : dict[str, Any]
        ``{"positions", "velocities"}``, ``[..., N, 3]`` each.

    Returns
    -------
    Any
        The flat data vector, positions and velocities interleaved per star.
    """
    return jnp.concatenate([state["positions"], state["velocities"]], -1).reshape(-1)


@dataclass
class Problems:
    """The single- and multiple-shooting problems on one data set.

    Attributes
    ----------
    system : System
        The initial condition.
    num_segments : int
        ``S`` (the number of snapshots).
    single : InferenceProblem
        Params ``{"theta"}``, :class:`SeriesModel`.
    multiple : InferenceProblem
        Params ``{"theta", "nodes"}``, ``S`` segments.
    multiple_model : MultipleShootingForwardModel
        The multiple-shooting model (for ``init_nodes``).
    counter : CountingEngine
        Counts segment passes of both problems.
    observed : Any
        The noisy data.
    clean : Any
        The noise-free data at the truth.
    log : ProgressLog
        Progress of the fit running on either problem.
    """

    system: System
    num_segments: int
    single: InferenceProblem
    multiple: InferenceProblem
    multiple_model: MultipleShootingForwardModel
    counter: CountingEngine
    observed: Any
    clean: Any
    log: ProgressLog


def problems(
    system: System,
    num_segments: int,
    segment_steps: int = SEGMENT_STEPS,
    sigma: float = SIGMA,
    noise_seed: int = 11,
) -> Problems:
    """Build the data and both problems for a horizon of ``num_segments`` segments.

    The noise is drawn for the LONGEST horizon the study uses once and sliced,
    so a shorter horizon's data are a prefix of a longer one's: the horizons
    differ only in how much of the same time series they see.

    Parameters
    ----------
    system : System
        The initial condition.
    num_segments : int
        ``S``.
    segment_steps : int
        Steps per segment (between snapshots).
    sigma : float
        Noise standard deviation on every phase-space coordinate.
    noise_seed : int
        Seed of the noise.

    Returns
    -------
    Problems
        Both problems on the same observations.
    """
    counter = CountingEngine(engine((segment_steps,) * num_segments, system))
    log = ProgressLog(counter)
    model = LoggedShootingModel(
        counter,  # pyright: ignore[reportArgumentType]
        system.masses,
        num_segments,
        system.positions,
        system.velocities,
        output="segment_ends",
        log=log,
    )
    series = SeriesModel(model, log=log)
    clean = jax.jit(lambda th: phase_space(series({"theta": th})))(TRUTH)
    per_snapshot = clean.size // num_segments
    noise = jax.random.normal(
        jax.random.PRNGKey(noise_seed), (MAX_SEGMENTS * per_snapshot,), clean.dtype
    )[: clean.size]
    observed = clean + sigma * noise
    likelihood = GaussianLikelihood(sigma=sigma)
    single = InferenceProblem(series, phase_space, likelihood, observed)
    multiple = InferenceProblem(model, phase_space, likelihood, observed)
    return Problems(
        system, num_segments, single, multiple, model, counter, observed, clean, log
    )


def theta_array(theta: Any) -> np.ndarray:
    """Return ``theta`` as a numpy vector in :data:`NAMES` order.

    Parameters
    ----------
    theta : Any
        A dict of scalars.

    Returns
    -------
    np.ndarray
        ``[P]``.
    """
    return np.asarray([float(theta[k]) for k in NAMES])


def theta_dict(values: Any) -> dict[str, Any]:
    """Return a ``theta`` dict from a vector in :data:`NAMES` order.

    Parameters
    ----------
    values : Any
        ``[P]``.

    Returns
    -------
    dict[str, Any]
        float64 scalars (traceable: ``values`` may be a tracer).
    """
    return {k: jnp.asarray(values[i], jnp.float64) for i, k in enumerate(NAMES)}


def write_json(path: Path, record: dict[str, Any]) -> None:
    """Write ``record`` atomically (a crashed chunk leaves no half file).

    Parameters
    ----------
    path : Path
        Destination.
    record : dict[str, Any]
        JSON-serialisable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")  # concurrent writers: own temp
    tmp.write_text(json.dumps(record, indent=1, allow_nan=True) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """Read a result file.

    Parameters
    ----------
    path : Path
        Source.

    Returns
    -------
    dict[str, Any]
        The record.
    """
    return json.loads(path.read_text())


def code_versions() -> dict[str, str]:
    """The git commit of every package the fits run (``+dirty`` with local changes).

    Returns
    -------
    dict[str, str]
        ``{"mimirax": ..., "jaccpot": ..., "nornax": ...}``; ``"unknown"`` when a
        package is not in a git checkout.
    """
    import importlib
    import subprocess

    out = {}
    for name in ("mimirax", "jaccpot", "nornax"):
        try:
            folder = Path(importlib.import_module(name).__file__ or "").parent
            head = subprocess.run(
                ["git", "-C", str(folder), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(folder), "status", "--porcelain", "-uno"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            out[name] = head + ("+dirty" if dirty else "")
        except (OSError, subprocess.CalledProcessError, ImportError):
            out[name] = "unknown"
    return out


def provenance() -> dict[str, Any]:
    """The configuration every result file records.

    Returns
    -------
    dict[str, Any]
        Versions, the device, the system's constants.
    """
    import platform

    return {
        "code": code_versions(),
        "jax": jax.__version__,
        "device": str(jax.devices()[0]),
        "x64": bool(jax.config.read("jax_enable_x64")),
        "python": platform.python_version(),
        "n_stars": N_STARS,
        "m_stars": M_STARS,
        "a_stars": A_STARS,
        "softening_over_a": SOFTENING_OVER_A,
        "dt": DT,
        "segment_steps": SEGMENT_STEPS,
        "sigma": SIGMA,
        "truth": {k: float(v) for k, v in TRUTH.items()},
        "disc": DISC._asdict(),
    }
