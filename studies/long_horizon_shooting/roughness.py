"""How rough is the objective in theta, against the horizon?

Two objectives on the same data (``common.problems``: the full phase-space state
every ``SEGMENT_STEPS`` steps, noise ``SIGMA``), at every horizon of
``S = 1 .. MAX_SEGMENTS`` snapshots at once:

``single``
    The single-shooting negative log-likelihood ``f_S(theta) = sum_{k<S} |X_k(theta)
    - D_k|^2 / 2 sigma^2`` along ONE trajectory from the initial condition. One
    forward run to the longest horizon gives every ``f_S`` (the data of a shorter
    horizon are a prefix of the longer one's).
``multiple``
    The multiple-shooting augmented Lagrangian at fixed multipliers (zero) and
    penalty ``MU`` (``1 / sigma^2``: a defect weighs what a datum does), with the
    NODES HELD AT THE TRUTH'S PROPAGATED STATES: ``L_S(theta) = sum_{k<S}
    |end_k(theta, n_k) - D_k|^2 / 2 sigma^2 + MU / 2 sum_{k<S-1} |n_{k+1} -
    end_k(theta, n_k)|^2`` with ``n_k`` the noise-free truth at node ``k``. Why
    these nodes: at nodes propagated at ``theta`` itself (``init_nodes(theta)``)
    every defect vanishes and ``L_S`` IS ``f_S`` -- the scan would repeat
    ``single``; optimised nodes (the profile ``min_nodes L``) need an inner
    optimisation per scan point. Held at the truth, the scan is the slice of the
    landscape the multiple-shooting optimizer works in once its nodes are near the
    trajectory: each segment is ``SEGMENT_STEPS`` long, whatever ``S`` is.

Measured along three unit directions in ``theta`` (log halo mass, log halo scale,
and one fixed random direction) through the truth:

* the scan itself on a uniform window ``|delta| <= 0.1`` (401 points), a fine
  window ``|delta| <= 0.004`` (321 points) and a log ladder ``|delta| = 1e-8 ..
  1e-1`` (4 per decade);
* the number of interior local minima in each window (a lower bound once the
  minima are finer than the grid);
* the basin half-width: walking out from the scan's minimum on the merged grid,
  the first offset at which the objective stops increasing (the smaller side);
* the exact gradient (``jax.jacfwd`` through ``checkpoint=None``: two
  parameters, one forward pass with two tangents) at offsets ``1e-4, 1e-3,
  1e-2`` along the direction, and the Gauss-Newton (Fisher) matrix ``J^T J /
  sigma^2`` of the data at the truth, from the same tangents.

One NPZ (the scans) and one JSON (the statistics) per (objective, direction)
under ``$LONG_HORIZON_RUNS/roughness/``; existing files are skipped. GPU:
``JAX_PLATFORMS=cuda,cpu python studies/long_horizon_shooting/roughness.py``.
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

MU = 1.0 / c.SIGMA**2
COARSE = np.linspace(-0.1, 0.1, 401)
FINE = np.linspace(-0.004, 0.004, 321)
LADDER = 10.0 ** np.arange(-8.0, -0.99, 0.25)
GRADIENT_OFFSETS = (1.0e-4, 1.0e-3, 1.0e-2)
#: theta points per batched call (``multiple`` runs MAX_SEGMENTS segments per point)
BATCH = {"single": 8, "multiple": 2}


def directions() -> dict[str, np.ndarray]:
    """The three unit directions in ``theta`` (in :data:`common.NAMES` order).

    Returns
    -------
    dict[str, np.ndarray]
        ``log_halo_mass``, ``log_halo_scale`` and ``random`` (seed 7).
    """
    rng = np.random.default_rng(7)
    u = rng.normal(size=len(c.NAMES))
    out = {name: np.eye(len(c.NAMES))[i] for i, name in enumerate(c.NAMES)}
    out["random"] = u / np.linalg.norm(u)
    return out


class Objectives:
    """Both objectives at every horizon, as jitted functions of ``theta``.

    Parameters
    ----------
    system : c.System
        The initial condition.
    """

    def __init__(self, system: c.System) -> None:
        s_max = c.MAX_SEGMENTS
        self.problems = c.problems(system, s_max)
        observed = self.problems.observed.reshape(s_max, -1)
        engine = c.engine((c.SEGMENT_STEPS,), system, checkpoint=None)
        m = system.masses

        def segment(theta: Any, x: Any, v: Any) -> tuple[Any, Any]:
            end = engine.rollout(theta, x, v, m, c.SEGMENT_STEPS)
            return end.positions, end.velocities

        def chain(theta: Any) -> Any:
            def body(carry: Any, _: Any) -> tuple[Any, Any]:
                x, v = segment(theta, *carry)
                return (x, v), jnp.concatenate([x, v], -1).reshape(-1)

            _, ends = jax.lax.scan(
                body, (system.positions, system.velocities), None, length=s_max
            )
            return ends  # [S_max, 6N]

        truth_ends = jax.jit(chain)(c.TRUTH)
        n = system.positions.shape[0]
        nodes = jnp.concatenate(
            [
                jnp.concatenate([system.positions, system.velocities], -1)[None],
                truth_ends[:-1].reshape(s_max - 1, n, 6),
            ]
        )  # [S_max, N, 6]: the start of every segment, noise-free

        def from_nodes(theta: Any) -> Any:
            ends = jax.vmap(lambda node: segment(theta, node[:, :3], node[:, 3:]))(
                nodes
            )
            return jnp.concatenate(ends, -1).reshape(s_max, -1)  # [S_max, 6N]

        def single(theta: Any) -> Any:
            r = chain(theta) - observed
            return jnp.cumsum(0.5 * jnp.sum(r * r, axis=1) / c.SIGMA**2)

        def multiple(theta: Any) -> Any:
            ends = from_nodes(theta)
            r = ends - observed
            data = 0.5 * jnp.sum(r * r, axis=1) / c.SIGMA**2
            defect = nodes[1:].reshape(s_max - 1, -1) - ends[:-1]
            penalty = 0.5 * MU * jnp.sum(defect * defect, axis=1)
            # L_S: data of segments < S, defects 0 .. S-2
            return jnp.cumsum(data) + jnp.concatenate(
                [jnp.zeros(1), jnp.cumsum(penalty)]
            )

        self.prediction = {"single": chain, "multiple": from_nodes}
        self.value = {"single": single, "multiple": multiple}
        self.batched = {
            k: jax.jit(jax.vmap(lambda vec, f=f: f(c.theta_dict(vec))))
            for k, f in self.value.items()
        }
        self.jacobian = {
            k: jax.jit(jax.jacfwd(lambda vec, f=f: f(c.theta_dict(vec))))
            for k, f in self.value.items()
        }
        self.data_jacobian = {
            k: jax.jit(jax.jacfwd(lambda vec, f=f: f(c.theta_dict(vec))))
            for k, f in self.prediction.items()
        }

    def scan(self, kind: str, points: np.ndarray) -> np.ndarray:
        """Evaluate one objective at many ``theta`` vectors, every horizon.

        Parameters
        ----------
        kind : str
            ``"single"`` or ``"multiple"``.
        points : np.ndarray
            ``[M, P]`` theta vectors.

        Returns
        -------
        np.ndarray
            ``[M, S_max]``: row ``i``, column ``S - 1`` is the objective of
            horizon ``S`` at ``points[i]``.
        """
        out = []
        batch = BATCH[kind]
        for i in range(0, len(points), batch):
            chunk = points[i : i + batch]
            pad = batch - len(chunk)
            if pad:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, 0)])
            values = np.asarray(self.batched[kind](jnp.asarray(chunk)))
            out.append(values[: batch - pad])
        return np.concatenate(out)


def local_minima(values: np.ndarray) -> int:
    """Count strict interior local minima of a 1-D scan.

    Parameters
    ----------
    values : np.ndarray
        ``[M]``.

    Returns
    -------
    int
        The count.
    """
    inner = values[1:-1]
    return int(np.sum((inner < values[:-2]) & (inner < values[2:])))


def basin_half_width(offsets: np.ndarray, values: np.ndarray) -> float:
    """Walk out from the scan's minimum while the objective increases.

    Parameters
    ----------
    offsets : np.ndarray
        ``[M]`` sorted offsets.
    values : np.ndarray
        ``[M]`` the objective there.

    Returns
    -------
    float
        The smaller of the two sides' distance from the minimum to the first
        point where the objective decreases (the window edge if it never does).
    """
    i0 = int(np.argmin(values))
    right = i0
    while right + 1 < len(values) and values[right + 1] > values[right]:
        right += 1
    left = i0
    while left > 0 and values[left - 1] > values[left]:
        left -= 1
    return float(min(offsets[right] - offsets[i0], offsets[i0] - offsets[left]))


def merged_offsets() -> np.ndarray:
    """Every scan offset once, sorted.

    The three grids share points up to round-off (``linspace`` gives
    ``5.000000000000004e-4`` where the ladder gives ``5e-4``); rounded to 1e-12
    before the union, so no two scan points are a round-off apart -- a pair that
    is reads as a spurious local extremum (the objective's own round-off between
    them) and stopped the basin walk at the first one.

    Returns
    -------
    np.ndarray
        The sorted offsets.
    """
    return np.unique(
        np.round(np.concatenate([COARSE, FINE, LADDER, -LADDER, [0.0]]), 12)
    )


def scan_statistics(offsets: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    """Minima, basin widths and the minimum's place, per horizon, from one scan.

    Offsets a round-off apart (scans written before :func:`merged_offsets`
    rounded them) are merged, keeping the first.

    Parameters
    ----------
    offsets : np.ndarray
        ``[M]`` sorted scan offsets.
    values : np.ndarray
        ``[M, S_max]`` the objective at every horizon.

    Returns
    -------
    dict[str, Any]
        ``segments``, ``minima_coarse``, ``minima_fine``, ``basin_half_width``,
        ``argmin_coarse``, ``value_at_truth``.
    """
    keys = np.round(offsets, 12)
    keys, first = np.unique(keys, return_index=True)
    values = values[first]
    at = {float(o): i for i, o in enumerate(keys)}

    def rows(grid: np.ndarray) -> np.ndarray:
        return values[[at[round(float(o), 12)] for o in grid]]

    coarse, fine = rows(COARSE), rows(FINE)
    horizons = range(values.shape[1])
    return {
        "segments": [s + 1 for s in horizons],
        "minima_coarse": [local_minima(coarse[:, s]) for s in horizons],
        "minima_fine": [local_minima(fine[:, s]) for s in horizons],
        "basin_half_width": [basin_half_width(keys, values[:, s]) for s in horizons],
        "argmin_coarse": [float(COARSE[np.argmin(coarse[:, s])]) for s in horizons],
        "value_at_truth": values[at[0.0]].tolist(),
    }


def restat() -> None:
    """Recompute :func:`scan_statistics` of every saved scan into its JSON (CPU)."""
    for npz in sorted((c.RUNS / "roughness").glob("*.npz")):
        path = npz.with_suffix(".json")
        if not path.exists():
            continue
        record = c.read_json(path)
        saved = np.load(npz)
        record.update(scan_statistics(saved["offsets"], saved["values"]))
        c.write_json(path, record)
        print(f"{path.name}: basin {[f'{w:.1e}' for w in record['basin_half_width']]}")


def run(kind: str, name: str, direction: np.ndarray, obj: Objectives) -> None:
    """Scan one objective along one direction and write its files.

    Parameters
    ----------
    kind : str
        ``"single"`` or ``"multiple"``.
    name : str
        The direction's name.
    direction : np.ndarray
        Unit vector.
    obj : Objectives
        The jitted objectives.
    """
    root = c.RUNS / "roughness"
    path = root / f"{kind}_{name}.json"
    if path.exists():
        return
    truth = c.theta_array(c.TRUTH)
    start = time.time()
    merged = merged_offsets()
    values = obj.scan(kind, truth[None] + merged[:, None] * direction[None])
    stats: dict[str, Any] = {
        "kind": kind,
        "direction": name,
        "direction_vector": direction.tolist(),
        **scan_statistics(merged, values),
    }
    gradients = {}
    for offset in GRADIENT_OFFSETS:
        jac = np.asarray(obj.jacobian[kind](jnp.asarray(truth + offset * direction)))
        gradients[f"{offset:g}"] = {
            "norm": np.linalg.norm(jac, axis=1).tolist(),
            "along_direction": (jac @ direction).tolist(),
        }
    stats["gradient"] = gradients
    if name == "random":  # the Fisher matrix does not depend on the direction
        jd = np.asarray(obj.data_jacobian[kind](jnp.asarray(truth)))  # [S, 6N, P]
        fisher = np.cumsum(np.einsum("sdp,sdq->spq", jd, jd), axis=0) / c.SIGMA**2
        stats["fisher"] = fisher.tolist()
        stats["fisher_forecast_sigma"] = [
            np.sqrt(np.diag(np.linalg.inv(f))).tolist() for f in fisher
        ]
        stats["fisher_eigenvalues"] = [np.linalg.eigvalsh(f).tolist() for f in fisher]
    stats["wall_s"] = time.time() - start
    stats["provenance"] = c.provenance()
    np.savez_compressed(
        root / f"{kind}_{name}.npz",
        offsets=merged,
        values=values,
        horizons=np.arange(1, c.MAX_SEGMENTS + 1),
    )
    c.write_json(path, stats)
    print(
        f"{kind} {name}: minima (coarse) {stats['minima_coarse']}\n"
        f"  basin {[f'{w:.1e}' for w in stats['basin_half_width']]}\n"
        f"  |grad| at 1e-3 {[f'{g:.1e}' for g in gradients['0.001']['norm']]}; "
        f"{stats['wall_s']:.0f} s",
        flush=True,
    )


def main() -> int:
    """Run every (objective, direction) without a result file.

    Returns
    -------
    int
        Exit status.
    """
    (c.RUNS / "roughness").mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["restat"]:
        restat()
        return 0
    kinds = sys.argv[1].split(",") if len(sys.argv) > 1 else ["single", "multiple"]
    obj = Objectives(c.build_system())
    for kind in kinds:
        for name, direction in directions().items():
            run(kind, name, direction, obj)
    return 0


if __name__ == "__main__":
    sys.exit(main())
