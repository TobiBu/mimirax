"""How chaotic is the system, and does the halo couple into the chaos?

For each configuration, one trajectory at the truth and four tangents carried
along it with ``jax.jvp`` through the engine's ``checkpoint=None`` rollout
(the checkpointed modes are ``custom_vjp`` s and refuse forward mode), in chunks
of ``RENORM`` steps:

* a random phase-space tangent, renormalised after every chunk (Benettin et al.
  1980): the largest Lyapunov exponent ``lambda`` is the mean log growth per
  unit time over the SECOND half of the run (the first half lets the tangent
  align with the most unstable direction);
* the tangent of the state along each ``theta`` direction, ``dX(t)/dtheta_i``,
  started at zero and never renormalised: how far a change of the halo has
  moved the state by time ``t``. Exponential growth at the rate ``lambda`` says
  the halo couples into the chaos; ``t^2`` growth says it acts through a
  regular (e.g. centre-of-mass) motion.

The tangent norm is dimensionless, ``sqrt(sum (dx / r_h)^2 + (dv / v_rms)^2)``
(lambda does not depend on that choice; the early transient does).

Configurations (one JSON each under ``$LONG_HORIZON_RUNS/lyapunov/``; existing
files are skipped, so the script resumes):

* ``candidate_*``: the three candidate systems of the README at ``N = 1024``;
* ``grid_N*_eps*_seed*``: the chosen (embedded) system over ``N`` and the
  softening, three seeds;
* ``dt_*``: the chosen configuration at four timesteps, with the energy drift.

Run: ``python studies/long_horizon_shooting/lyapunov.py`` (a GPU makes it a few
minutes; set ``JAX_PLATFORMS=cuda,cpu``).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as c  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

RENORM = 10
#: run length of the grid, in crossing times
RUN_CROSSINGS = 40.0
GRID_N = (256, 512, 1024, 2048)
GRID_EPS = (0.025, 0.05, 0.1, 0.2)
GRID_SEEDS = (0, 1, 2)
DT_LADDER = (0.005, 0.01, 0.02, 0.04)


def measure(
    system: c.System,
    steps: int,
    dt: float,
    r_scale: float,
    v_scale: float,
) -> dict[str, Any]:
    """Run the trajectory with its tangents; return the growth histories.

    Parameters
    ----------
    system : c.System
        The initial condition.
    steps : int
        Steps (rounded down to a multiple of ``RENORM``).
    dt : float
        Timestep.
    r_scale : float
        Length unit of the tangent norm.
    v_scale : float
        Velocity unit of the tangent norm.

    Returns
    -------
    dict[str, Any]
        ``lambda`` (per unit time), the running estimate, the log norms of the
        ``theta`` tangents over time, the energy drift and the wall time.
    """
    engine = c.engine((RENORM,), system, checkpoint=None, dt=dt)
    m = system.masses

    def run(theta: Any, x: Any, v: Any) -> tuple[Any, Any]:
        end = engine.rollout(theta, x, v, m, RENORM)
        return end.positions, end.velocities

    @jax.jit
    def chunk(theta: Any, x: Any, v: Any, tangents: Any) -> tuple[Any, ...]:
        # one jvp per (dtheta, dx, dv) tangent; the primal is the same in each
        results = [jax.jvp(run, (theta, x, v), t) for t in tangents]
        x1, v1 = results[0][0]
        return x1, v1, [r[1] for r in results]

    def norm(dx: Any, dv: Any) -> float:
        return float(
            jnp.sqrt(jnp.sum((dx / r_scale) ** 2) + jnp.sum((dv / v_scale) ** 2))
        )

    zero = jax.tree_util.tree_map(jnp.zeros_like, c.TRUTH)
    x0, v0 = system.positions, system.velocities
    dx = jax.random.normal(jax.random.PRNGKey(1), x0.shape) * r_scale
    dv = jax.random.normal(jax.random.PRNGKey(2), v0.shape) * v_scale
    s = norm(dx, dv)
    tangents = [(zero, dx / s, dv / s)]
    for name in c.NAMES:
        unit = {k: jnp.asarray(1.0 if k == name else 0.0) for k in c.NAMES}
        tangents.append((unit, jnp.zeros_like(x0), jnp.zeros_like(v0)))
    x, v = x0, v0
    logs, sens = [], {k: [] for k in c.NAMES}
    start = time.time()
    for _ in range(steps // RENORM):
        x1, v1, outs = chunk(c.TRUTH, x, v, tangents)
        g = norm(*outs[0])
        logs.append(float(np.log(g)))
        tangents[0] = (zero, outs[0][0] / g, outs[0][1] / g)
        for i, name in enumerate(c.NAMES):
            tx, tv = outs[i + 1]
            sens[name].append(float(np.log(norm(tx, tv))))
            tangents[i + 1] = (tangents[i + 1][0], tx, tv)
        x, v = x1, v1
    jax.block_until_ready(x)
    wall = time.time() - start
    logs_a = np.asarray(logs)
    t = dt * RENORM * np.arange(1, len(logs_a) + 1)
    cum = np.cumsum(logs_a)
    half = len(logs_a) // 2
    lam = (cum[-1] - cum[half]) / (t[-1] - t[half])
    # growth rate of each theta tangent over the second half (a straight-line fit)
    rates = {
        k: float(np.polyfit(t[half:], np.asarray(s_)[half:], 1)[0])
        for k, s_ in sens.items()
    }
    e0, e1 = c.energy(system, x0, v0), c.energy(system, x, v)
    return {
        "lambda": float(lam),
        "time": t.tolist(),
        "lambda_running": (cum / t).tolist(),
        "log_theta_tangent": sens,
        "theta_tangent_rate_second_half": rates,
        "energy_drift": float((e1 - e0) / abs(e0)),
        "wall_s": wall,
    }


def summarise(record: dict[str, Any], system: c.System, dt: float) -> dict[str, Any]:
    """Add the exponent in crossing-time and step units.

    Parameters
    ----------
    record : dict[str, Any]
        :func:`measure`'s output.
    system : c.System
        Supplies the crossing time.
    dt : float
        Timestep.

    Returns
    -------
    dict[str, Any]
        ``record`` with ``lambda_t_cross``, ``lambda_dt``, ``lyapunov_time``,
        ``steps_per_lyapunov_time`` and the system's scales.
    """
    lam = record["lambda"]
    record.update(
        lambda_t_cross=lam * system.crossing_time,
        lambda_dt=lam * dt,
        lyapunov_time=1.0 / lam,
        steps_per_lyapunov_time=1.0 / (lam * dt),
        crossing_time=system.crossing_time,
        half_mass_radius=system.half_mass_radius,
        rms_speed=system.rms_speed,
        softening=system.softening,
        dt=dt,
        n=int(system.masses.shape[0]),
    )
    return record


# --- the two rejected candidates ----------------------------------------------------


def orbiting_cluster(n: int, seed: int = 0) -> tuple[c.System, float]:
    """A light Plummer cluster on a circular orbit at R = 2 through the galaxy.

    Mass 0.01, scale 0.05 (Jacobi radius ~0.28), softening 0.05 of the scale.

    Parameters
    ----------
    n : int
        Stars.
    seed : int
        Seed.

    Returns
    -------
    tuple[c.System, float]
        The system (its crossing time from the self-consistent Plummer
        velocities) and the timestep (0.02 crossing times).
    """
    mass, scale, radius = 0.01, 0.05, 2.0
    x, v, m = c.plummer(n, mass, scale, np.random.default_rng(seed))
    pull = c.galaxy_field(jnp.asarray([[radius, 0.0, 0.0]]), 0.0, c.TRUTH)
    v_circ = float(jnp.sqrt(-pull[0, 0] * radius))
    r_half = float(np.median(np.linalg.norm(x, axis=1)))
    v_rms = float(np.sqrt(np.mean(np.sum(v * v, axis=1))))
    x[:, 0] += radius
    v[:, 1] += v_circ
    system = c.System(
        jnp.asarray(x),
        jnp.asarray(v),
        jnp.asarray(m),
        0.05 * scale,
        r_half,
        v_rms,
        r_half / v_rms,
        1.0,
    )
    return system, 0.02 * system.crossing_time


def self_gravitating_disc(n: int, seed: int = 0) -> tuple[c.System, float]:
    """Session 1's disc, ten times heavier: mass 0.1 on circular orbits, R in [0.5, 2].

    Parameters
    ----------
    n : int
        Stars.
    seed : int
        Seed.

    Returns
    -------
    tuple[c.System, float]
        The system (crossing time: the orbital time at R = 1 over 2 pi) and
        ``dt = 0.02``.
    """
    k_r, k_phi, k_z = jax.random.split(jax.random.PRNGKey(seed), 3)
    radius = jax.random.uniform(k_r, (n,), minval=0.5, maxval=2.0)
    phi = jax.random.uniform(k_phi, (n,), maxval=2.0 * jnp.pi)
    z = 0.05 * jax.random.normal(k_z, (n,))
    x = jnp.stack([radius * jnp.cos(phi), radius * jnp.sin(phi), z], -1)
    a = c.galaxy_field(x, 0.0, c.TRUTH)
    radial = -(a[:, 0] * jnp.cos(phi) + a[:, 1] * jnp.sin(phi))
    speed = jnp.sqrt(radius * radial)
    v = jnp.stack([-speed * jnp.sin(phi), speed * jnp.cos(phi), jnp.zeros(n)], -1)
    pull = c.galaxy_field(jnp.asarray([[1.0, 0.0, 0.0]]), 0.0, c.TRUTH)
    omega = float(jnp.sqrt(-pull[0, 0]))
    system = c.System(
        x, v, jnp.full((n,), 0.1 / n), 0.02, 1.0, float(omega), 1.0 / omega, 1.0
    )
    return system, 0.02


def configurations() -> list[tuple[str, Any]]:
    """Every configuration, as ``(key, thunk)`` pairs; a thunk returns the record.

    Returns
    -------
    list[tuple[str, Any]]
        In run order.
    """
    out: list[tuple[str, Any]] = []

    def chosen(n: int, eps: float, seed: int, dt: float, crossings: float) -> Any:
        def thunk() -> dict[str, Any]:
            system = c.build_system(n=n, over_a=eps, seed=seed)
            steps = int(crossings * system.crossing_time / dt)
            record = measure(
                system, steps, dt, system.half_mass_radius, system.rms_speed
            )
            record.update(over_a=eps, seed=seed, virial_scale=system.virial_scale)
            return summarise(record, system, dt)

        return thunk

    def candidate(builder: Any, crossings: float) -> Any:
        def thunk() -> dict[str, Any]:
            system, dt = builder(c.N_STARS)
            steps = int(crossings * system.crossing_time / dt)
            record = measure(
                system, steps, dt, system.half_mass_radius, system.rms_speed
            )
            return summarise(record, system, dt)

        return thunk

    out.append(
        (
            "candidate_embedded",
            chosen(c.N_STARS, c.SOFTENING_OVER_A, 0, c.DT, RUN_CROSSINGS),
        )
    )
    out.append(
        ("candidate_orbiting_cluster", candidate(orbiting_cluster, RUN_CROSSINGS))
    )
    out.append(("candidate_disc", candidate(self_gravitating_disc, 4.0 * np.pi)))
    for n in GRID_N:
        for eps in GRID_EPS:
            dt = c.DT * min(1.0, eps / c.SOFTENING_OVER_A)
            for seed in GRID_SEEDS:
                key = f"grid_N{n}_eps{eps}_seed{seed}"
                out.append((key, chosen(n, eps, seed, dt, RUN_CROSSINGS)))
    for dt in DT_LADDER:
        out.append((f"dt_{dt}", chosen(c.N_STARS, c.SOFTENING_OVER_A, 0, dt, 20.0)))
    return out


def main() -> int:
    """Run every configuration that has no result file yet.

    Returns
    -------
    int
        Exit status.
    """
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    root = c.RUNS / "lyapunov"
    for key, thunk in configurations():
        path = root / f"{key}.json"
        if path.exists() or not key.startswith(only):
            continue
        record = thunk()
        record.update(key=key, provenance=c.provenance())
        c.write_json(path, record)
        print(
            f"{key}: lambda {record['lambda']:.4f} /time = "
            f"{record['lambda_t_cross']:.3f} /t_cr, {record['steps_per_lyapunov_time']:.0f}"
            f" steps per Lyapunov time; theta-tangent rates "
            f"{[round(r, 3) for r in record['theta_tangent_rate_second_half'].values()]};"
            f" dE/E {record['energy_drift']:.1e}; {record['wall_s']:.1f} s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
