"""Collect the study's result files into one JSON and draw the figure from it.

Reads ``$LONG_HORIZON_RUNS/{lyapunov,roughness,recovery}/`` (whatever exists),
writes ``studies/results/long_horizon_shooting.json`` with every number the
README quotes, reads it back and draws ``long_horizon_shooting.{pdf,png}``
(also beside the run files) from the JSON alone. CPU, seconds::

    python studies/long_horizon_shooting/summary.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as c  # noqa: E402
import numpy as np  # noqa: E402

# Okabe-Ito, as in the other studies; the methods also differ in marker and line style.
METHOD_STYLE = {
    "single": dict(colour="#D55E00", marker="o", line="-", name="single shooting"),
    "multiple_data": dict(
        colour="#0072B2", marker="s", line="--", name="multiple, nodes from the data"
    ),
    "multiple": dict(
        colour="#009E73",
        marker="^",
        line=":",
        name="multiple, nodes from a forward run",
    ),
}
#: the roughness scans have one multiple-shooting objective (nodes at the truth)
SCAN_STYLE = {
    "single": METHOD_STYLE["single"],
    "multiple": dict(
        colour="#0072B2", marker="s", line="--", name="multiple (nodes at the truth)"
    ),
}
#: recovery.py's success tolerance (max |theta - theta*|), applied to every record
SUCCESS_TOL = 1.0e-3
EPS_COLOUR = {0.025: "#E69F00", 0.05: "#009E73", 0.1: "#0072B2", 0.2: "#CC79A7"}


def _load(folder: str) -> list[dict[str, Any]]:
    """Every JSON record in one result folder.

    Parameters
    ----------
    folder : str
        ``lyapunov``, ``roughness`` or ``recovery``.

    Returns
    -------
    list[dict[str, Any]]
        The records, file name under ``"_file"``.
    """
    root = c.RUNS / folder
    out = []
    for path in sorted(root.glob("*.json")) if root.exists() else []:
        record = json.loads(path.read_text())
        record["_file"] = path.name
        out.append(record)
    return out


def lyapunov_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Tabulate the candidates, the (N, softening) grid and the timestep ladder.

    Parameters
    ----------
    records : list[dict[str, Any]]
        The lyapunov folder.

    Returns
    -------
    dict[str, Any]
        ``candidates``, ``grid`` (mean and spread over seeds), ``dt``, ``chosen``.
    """
    keep = (
        "lambda",
        "lambda_t_cross",
        "lambda_dt",
        "steps_per_lyapunov_time",
        "crossing_time",
        "dt",
        "softening",
        "energy_drift",
        "theta_tangent_rate_second_half",
        "half_mass_radius",
        "rms_speed",
    )
    out: dict[str, Any] = {"candidates": {}, "grid": {}, "dt": {}}
    grid: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = r["key"]
        if key.startswith("candidate_"):
            entry = {k: r[k] for k in keep}
            lam = r["lambda"]
            t = np.asarray(r["time"]) * lam
            log_tangent = np.asarray(r["log_theta_tangent"][c.NAMES[0]])
            # how far above (or below) exp(lambda t) the halo's tangent sits
            entry["coupling_log_tangent_minus_lambda_t"] = {
                f"{x:g}": float(log_tangent[i] - t[i])
                for x in (1, 2, 4, 8, 12, 16)
                if (i := int(np.searchsorted(t, x))) < len(t)
            }
            entry["log_tangent_at"] = {
                f"{x:g}": float(log_tangent[i])
                for x in (1, 2, 4, 8, 12, 16)
                if (i := int(np.searchsorted(t, x))) < len(t)
            }
            out["candidates"][key.removeprefix("candidate_")] = entry
        elif key.startswith("grid_"):
            grid[(r["n"], r["over_a"])].append(r)
        elif key.startswith("dt_"):
            out["dt"][f"{r['dt']:g}"] = {k: r[k] for k in keep}
    for (n, eps), rows in sorted(grid.items()):
        lam_tc = np.asarray([r["lambda_t_cross"] for r in rows])
        steps = np.asarray([r["steps_per_lyapunov_time"] for r in rows])
        out["grid"][f"N{n}_eps{eps:g}"] = {
            "n": n,
            "softening_over_a": eps,
            "seeds": len(rows),
            "lambda_t_cross_mean": float(lam_tc.mean()),
            "lambda_t_cross_std": float(lam_tc.std(ddof=1)) if len(rows) > 1 else 0.0,
            "lambda_mean": float(np.mean([r["lambda"] for r in rows])),
            "steps_per_lyapunov_time_mean": float(steps.mean()),
            "dt": rows[0]["dt"],
            "energy_drift_max": float(max(abs(r["energy_drift"]) for r in rows)),
        }
    # lambda t_cross = C N^alpha (eps / a)^beta over the grid's eps <= 0.1 runs (at
    # 0.2 a the second-half estimate has not settled: theta tangents grow slower)
    rows = [r for (n, eps), rs in grid.items() if eps <= 0.1 for r in rs]
    if len(rows) > 3:
        design = np.asarray([[1.0, np.log(r["n"]), np.log(r["over_a"])] for r in rows])
        target = np.log([r["lambda_t_cross"] for r in rows])
        coef = np.linalg.lstsq(design, target, rcond=None)[0]
        out["fit"] = {
            "C": float(np.exp(coef[0])),
            "alpha_N": float(coef[1]),
            "beta_softening": float(coef[2]),
            "rms_log_residual": float(np.sqrt(np.mean((design @ coef - target) ** 2))),
            "runs": len(rows),
        }
    chosen = out["grid"].get(f"N{c.N_STARS}_eps{c.SOFTENING_OVER_A:g}")
    if chosen is not None:
        lam = chosen["lambda_mean"]
        out["chosen"] = {
            "lambda": lam,
            "lyapunov_time": 1.0 / lam,
            "steps_per_lyapunov_time": 1.0 / (lam * c.DT),
            "segment_in_lyapunov_times": c.SEGMENT_STEPS * c.DT * lam,
            "lambda_t_cross": chosen["lambda_t_cross_mean"],
        }
    return out


def roughness_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep the per-horizon statistics of every (objective, direction).

    Parameters
    ----------
    records : list[dict[str, Any]]
        The roughness folder.

    Returns
    -------
    dict[str, Any]
        Keyed ``<kind>_<direction>``.
    """
    out = {}
    for r in records:
        entry = {
            k: r[k]
            for k in (
                "segments",
                "minima_coarse",
                "minima_fine",
                "basin_half_width",
                "argmin_coarse",
                "gradient",
            )
        }
        for k in ("fisher_forecast_sigma", "fisher_eigenvalues"):
            if k in r:
                entry[k] = r[k]
        out[f"{r['kind']}_{r['direction']}"] = entry
    return out


def recovery_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Success rate and cost per (method, horizon, radius); the objective at the truth.

    Parameters
    ----------
    records : list[dict[str, Any]]
        The recovery folder.

    Returns
    -------
    dict[str, Any]
        ``configurations`` and ``objective_at_truth``.
    """
    groups: dict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
    reference = {}
    for r in records:
        if r["_file"].startswith("reference_"):
            reference[f"S{r['segments']:02d}"] = r["objective_at_truth"]
            continue
        groups[(r["method"], r["segments"], r["radius"])].append(r)
    # fits stopped by a process time limit: their last progress record
    for path in sorted((c.RUNS / "recovery" / "partial").glob("*.json")):
        if ".state" in path.name:
            continue  # a saved driver state, not a progress record
        if (c.RUNS / "recovery" / path.name).exists():
            continue  # finished after all
        p = json.loads(path.read_text())
        name = path.stem  # <method>_S<ss>_r<radius>_k<kk>
        method = name.split("_S")[0]
        s_ = int(name.split("_S")[1][:2])
        radius = float(name.split("_r")[1].split("_k")[0])
        groups[(method, s_, radius)].append(
            {
                "error": p["error"],
                "segment_passes": p["segment_passes_so_far"],
                "inner_steps": p["inner_steps_so_far"],
                "outer_iterations": None,
                "converged": False,
                "status": "time limit",
                "objective_above_truth": None,
                "wall_s_information_only": p["elapsed_s_information_only"],
                "history_steps_passes_maxerror": p.get("history", []),
                "start": int(name.split("_k")[1]),
            }
        )
    configs = {}
    for (method, s, radius), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["start"])
        passes = np.asarray([r["segment_passes"] for r in rows], float)
        err = np.asarray([np.max(np.abs(r["error"])) for r in rows])
        ok = [bool(e <= SUCCESS_TOL) for e in err]
        status = [
            r.get("status") or ("converged" if r["converged"] else "budget")
            for r in rows
        ]
        # passes after which theta stayed within the tolerance to the end (the
        # cost of the answer, before the driver's own stopping rule); fits
        # shorter than one history interval (25 steps) count their total
        to_tol = []
        for r, good in zip(rows, ok):
            history = r.get("history_steps_passes_maxerror") or []
            if not good or (not history and r.get("status") == "time limit"):
                # failed, or stopped before progress histories were recorded
                to_tol.append(None)
                continue
            since = r["segment_passes"]
            for h in reversed(history):
                if h[2] > SUCCESS_TOL:
                    break
                since = h[1]
            to_tol.append(since)
        reached = [t for t in to_tol if t is not None]
        finished = [st in ("converged", "budget") for st in status]
        stop = passes[np.asarray(finished, bool)] if any(finished) else np.zeros(0)
        configs[f"{method}_S{s:02d}_r{radius:g}"] = {
            "method": method,
            "segments": s,
            "radius": radius,
            "starts": len(rows),
            "start_indices": [r["start"] for r in rows],
            "successes": int(sum(ok)),
            "success_rate": float(np.mean(ok)),
            "status_per_start": status,
            "segment_passes_per_start": passes.tolist(),
            "segment_passes_median": float(np.median(passes)),
            "segment_passes_min": float(passes.min()),
            "segment_passes_max": float(passes.max()),
            "passes_to_tolerance_per_start": to_tol,
            "passes_to_tolerance_median": (
                float(np.median(reached)) if reached else None
            ),
            "passes_to_tolerance_min": float(min(reached)) if reached else None,
            "passes_to_tolerance_max": float(max(reached)) if reached else None,
            # the driver's own reading: fits that ended by its stopping rule
            # ("converged") or its full budget ("budget"), no cap, no oracle
            "finished": int(sum(finished)),
            "converged": int(sum(st == "converged" for st in status)),
            "budget_exhausted": int(sum(st == "budget" for st in status)),
            "capped_or_unfinished": int(len(rows) - sum(finished)),
            "passes_to_stop_median": float(np.median(stop)) if stop.size else None,
            "passes_to_stop_min": float(stop.min()) if stop.size else None,
            "passes_to_stop_max": float(stop.max()) if stop.size else None,
            "final_violation_per_start": [
                (r.get("outer_violation") or [None])[-1] for r in rows
            ],
            "outer_iterations_per_start": [r.get("outer_iterations") for r in rows],
            "evaluations_median": float(np.median(passes) / s),
            "inner_steps_median": float(np.median([r["inner_steps"] for r in rows])),
            "error_max_abs_median": float(np.median(err)),
            "error_max_abs_per_start": err.tolist(),
            "objective_above_truth_per_start": [
                r["objective_above_truth"] for r in rows
            ],
            "wall_s_median_information_only": float(
                np.median([r["wall_s_information_only"] for r in rows])
            ),
        }
    return {"configurations": configs, "objective_at_truth": reference}


def figure(record: dict[str, Any]) -> Any:
    """Draw the study's six panels from the JSON record.

    Parameters
    ----------
    record : dict[str, Any]
        The summary, read back from disk.

    Returns
    -------
    Any
        The matplotlib figure.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    lam = record["lyapunov"].get("chosen", {}).get("lambda", float("nan"))
    seg_lt = c.SEGMENT_STEPS * c.DT * lam  # one segment in Lyapunov times
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.6))
    fig.patch.set_facecolor("#FFFFFF")
    for ax in axes.flat:
        ax.grid(True, color="#E0E0E0", linewidth=0.6)
        ax.set_axisbelow(True)

    # (a) lambda vs N per softening
    ax = axes[0, 0]
    by_eps: dict[float, list[tuple[int, float, float]]] = defaultdict(list)
    for row in record["lyapunov"]["grid"].values():
        by_eps[row["softening_over_a"]].append(
            (row["n"], row["lambda_t_cross_mean"], row["lambda_t_cross_std"])
        )
    for eps, rows in sorted(by_eps.items()):
        rows.sort()
        n, m, s = (np.asarray(v) for v in zip(*rows))
        ax.errorbar(
            n,
            m,
            yerr=s,
            marker="o",
            markersize=6,
            linewidth=2.0,
            capsize=3,
            color=EPS_COLOUR.get(eps, "#555555"),
            label=f"softening {eps:g} a",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\lambda\, t_{\rm cross}$")
    ax.set_title("(a) largest Lyapunov exponent", loc="left")
    ax.legend(frameon=False, fontsize=9)

    rough = record["roughness"]
    # (b) gradient norm at a fixed offset vs horizon
    ax = axes[0, 1]
    for kind, st in SCAN_STYLE.items():
        entry = rough.get(f"{kind}_random")
        if entry is None:
            continue
        t = np.asarray(entry["segments"]) * seg_lt
        g = np.asarray(entry["gradient"]["0.001"]["norm"])
        ax.semilogy(
            t,
            g,
            st["line"],
            marker=st["marker"],
            color=st["colour"],
            linewidth=2.0,
            markersize=6,
            label=st["name"],
        )
    if "single_random" in rough:
        t = np.asarray(rough["single_random"]["segments"]) * seg_lt
        g0 = rough["single_random"]["gradient"]["0.001"]["norm"][0]
        ax.semilogy(
            t,
            g0 * np.exp(t - t[0]),
            ":",
            color="#555555",
            linewidth=1.4,
            label=r"$\propto e^{\lambda T}$",
        )
    ax.set_xlabel(r"horizon $T$ [Lyapunov times]")
    ax.set_ylabel(r"$|\nabla_\theta|$ at $|\delta\theta| = 10^{-3}$ (random dir.)")
    ax.set_title("(b) gradient at a fixed offset", loc="left")
    ax.legend(frameon=False, fontsize=9)

    # (c) basin half-width vs horizon
    ax = axes[0, 2]
    for kind, st in SCAN_STYLE.items():
        for direction, alpha in (
            ("random", 1.0),
            (c.NAMES[0], 0.45),
            (c.NAMES[1], 0.45),
        ):
            entry = rough.get(f"{kind}_{direction}")
            if entry is None:
                continue
            t = np.asarray(entry["segments"]) * seg_lt
            ax.semilogy(
                t,
                entry["basin_half_width"],
                st["line"],
                marker=st["marker"],
                color=st["colour"],
                alpha=alpha,
                linewidth=2.0 if alpha == 1 else 1.2,
                markersize=6 if alpha == 1 else 4,
                label=st["name"] if direction == "random" else None,
            )
    ax.set_xlabel(r"horizon $T$ [Lyapunov times]")
    ax.set_ylabel(r"basin half-width in $\theta$")
    ax.set_title("(c) width of the true minimum's basin", loc="left")
    ax.legend(
        frameon=False,
        fontsize=9,
        title="faint: log M, log r_s directions",
        title_fontsize=8,
    )

    # (d) local minima in |delta| <= 0.1
    ax = axes[1, 0]
    for kind, st in SCAN_STYLE.items():
        entry = rough.get(f"{kind}_random")
        if entry is None:
            continue
        t = np.asarray(entry["segments"]) * seg_lt
        ax.plot(
            t,
            entry["minima_coarse"],
            st["line"],
            marker=st["marker"],
            color=st["colour"],
            linewidth=2.0,
            markersize=6,
            label=st["name"],
        )
    ax.set_xlabel(r"horizon $T$ [Lyapunov times]")
    ax.set_ylabel(r"local minima in $|\delta\theta| \leq 0.1$ (401 pts)")
    ax.set_title("(d) spurious minima along a random direction", loc="left")
    ax.legend(frameon=False, fontsize=9)

    configs = record["recovery"]["configurations"]
    # (e) success rate
    ax = axes[1, 1]
    for kind, st in METHOD_STYLE.items():
        rows = sorted(
            (v["segments"], v["success_rate"], v["starts"])
            for v in configs.values()
            if v["method"] == kind and v["radius"] == 0.1
        )
        if not rows:
            continue
        s, rate, starts = (np.asarray(v) for v in zip(*rows))
        ax.plot(
            s * seg_lt,
            rate,
            st["line"],
            marker=st["marker"],
            color=st["colour"],
            linewidth=2.0,
            markersize=7,
            label=st["name"],
        )
        for x, y, k in zip(s * seg_lt, rate, starts):
            ax.annotate(
                f"{round(y * k)}/{k}",
                (x, y),
                textcoords="offset points",
                xytext=(5, -11 if kind == "single" else 4),
                fontsize=7,
                color="#555555",
            )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"horizon $T$ [Lyapunov times]")
    ax.set_ylabel("success rate (starts at $|\\delta\\theta| = 0.1$)")
    ax.set_title("(e) recovery from perturbed starts (successes/starts)", loc="left")
    ax.legend(frameon=False, fontsize=9)

    # (f) cost of each successful fit, two readings: passes until theta stayed
    # within tolerance (from the history: an oracle stop, filled) and passes to
    # the driver's own stop -- its rule or its full budget (hollow)
    ax = axes[1, 2]
    offset = {"single": -0.12, "multiple_data": 0.0, "multiple": 0.12}
    for kind, st in METHOD_STYLE.items():
        xs, ys, xs_stop, ys_stop = [], [], [], []
        for v in configs.values():
            if v["method"] != kind or v["radius"] != 0.1:
                continue
            x = v["segments"] * seg_lt + offset[kind]
            for t, ok, passes, status in zip(
                v["passes_to_tolerance_per_start"],
                [e <= SUCCESS_TOL for e in v["error_max_abs_per_start"]],
                v["segment_passes_per_start"],
                v["status_per_start"],
            ):
                if t is not None:
                    xs.append(x)
                    ys.append(t)
                if ok and status in ("converged", "budget"):
                    xs_stop.append(x)
                    ys_stop.append(passes)
        if xs:
            ax.scatter(
                xs,
                ys,
                marker=st["marker"],
                s=36,
                color=st["colour"],
                label=st["name"] + " (to tolerance)",
                alpha=0.85,
                edgecolors="#FFFFFF",
                linewidths=0.8,
            )
        if xs_stop:
            ax.scatter(
                xs_stop,
                ys_stop,
                marker=st["marker"],
                s=44,
                facecolors="none",
                edgecolors=st["colour"],
                linewidths=1.2,
                label="  ... to the driver's stop",
            )
    ax.set_yscale("log")
    ax.set_xlabel(r"horizon $T$ [Lyapunov times]")
    ax.set_ylabel("segment passes per successful fit")
    ax.set_title("(f) cost: to tolerance (filled), to the driver's stop", loc="left")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Single vs multiple shooting on a chaotic N = {c.N_STARS} system "
        f"(one segment = {seg_lt:.2f} Lyapunov times = {c.SEGMENT_STEPS} steps)",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def main() -> int:
    """Collect, write, read back, draw.

    Returns
    -------
    int
        Exit status.
    """
    record = {
        "provenance": c.provenance(),
        "runs": str(c.RUNS),
        "lyapunov": lyapunov_summary(_load("lyapunov")),
        "roughness": roughness_summary(_load("roughness")),
        "recovery": recovery_summary(_load("recovery")),
    }
    c.RESULTS.mkdir(parents=True, exist_ok=True)
    payload = c.RESULTS / "long_horizon_shooting.json"
    payload.write_text(json.dumps(record, indent=1) + "\n")
    record = json.loads(payload.read_text())
    fig = figure(record)
    for suffix in ("pdf", "png"):
        fig.savefig(c.RESULTS / f"long_horizon_shooting.{suffix}", dpi=200)
        fig.savefig(c.RUNS / f"long_horizon_shooting.{suffix}", dpi=200)
    print(f"wrote {payload} and the figure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
