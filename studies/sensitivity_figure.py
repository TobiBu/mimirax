"""Generate the information-content figures from the made-to-measure problem.

The programme's whole case is that a differentiable model can say what the data
do and do not constrain, *before* the fit is run. These figures are that claim,
drawn from the tracer problem asserted by
``tests/integration/test_m2m_rollout.py::
test_the_weights_are_recovered_when_the_observables_determine_them``.

Two figures, one computation:

``sensitivity_information``
    The spectrum the observables determine, with the prior's threshold drawn
    across it, beside the consequence: the same 32 orbits recovered under two
    instruments. The under-determined one reaches a chi-squared nine orders of
    magnitude smaller and a weight error 31 times larger.
``sensitivity_information_map``
    Which datum carries information about which parameter -- the LOSVD grid,
    and the per-pixel Fisher contribution to two named orbit weights.

Every number drawn is written to ``studies/results/sensitivity_figure.json``
first and read back from it, so nothing is hand-placed. Run with no arguments::

    python studies/sensitivity_figure.py

NOTE ON THE SPECTRUM. ``eigvalsh`` of the data curvature returns **negative**
eigenvalues on the radial problem (10 of 32, min -1.05e-10): the Hessian is
numerically indefinite at that spectrum. The eigenvalues are therefore taken as
the squared singular values of the *whitened Jacobian* instead, which is
``reports/M2M_production_readiness.md`` A2's prescription and is exact for a
Gauss-Newton curvature. This is recorded here so it is not rediscovered.
"""

from __future__ import annotations

import json
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mimirax import (  # noqa: E402
    EntropyPrior,
    GaussianLikelihood,
    GaussianRadialBins,
    InferenceProblem,
    MadeToMeasure,
    TimeAverage,
    WeightedKernelSum,
)
from mimirax.adapters.nornax import NornaxRollout  # noqa: E402
from mimirax.diagnostics import effective_parameters, fisher_information  # noqa: E402
from mimirax.testing import SoftenedPointMassField  # noqa: E402

SEED = 0
NUM_ORBITS = 32
MU = 1.0e-8
SOFTENING = 0.2
DT = 0.12
NUM_STEPS = 48
# Orbit radii span this range; the radial *bins* are centred on a different one.
# Conflating the two changes the fit outcome -- they are separate on purpose.
ORBIT_RMIN, ORBIT_RMAX = 0.5, 2.0
BIN_CENTRES = (0.4, 2.0)
NUM_BINS = 5
BIN_WIDTH = 0.3
# The LOSVD grid, copied from the integration test so that the figure and the
# test describe the same system. Not the same numbers as the orbit radii.
NUM_RADII, NUM_SPEEDS = 8, 8
LOSVD_RADII = (0.35, 1.85)
LOSVD_SPEEDS = (-0.9, 0.9)
LOSVD_RADIAL_WIDTH, LOSVD_VELOCITY_WIDTH = 0.25, 0.18
FIT_STEPS = 1500

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Okabe-Ito. Distinguishable under every common colour-vision deficiency, and
# separable in greyscale; the claim is never carried by hue alone -- each
# observable set also has its own marker and line style.
BLUE, VERMILLION = "#0072B2", "#D55E00"
SETS = {
    "radial": dict(colour=BLUE, marker="o", line="-", name="5 radial bins x 2 moments"),
    "losvd": dict(
        colour=VERMILLION, marker="s", line="--", name="LOSVD, 8 radii x 8 speeds"
    ),
}
THEMES = {
    "light": dict(fg="#101010", muted="#505050", bg="#FFFFFF", grid="#C8C8C8"),
    "dark": dict(fg="#F0F0F0", muted="#A0A0A0", bg="#0B0D12", grid="#3A3F4A"),
}


def tangential_orbits(key: jax.Array, n: int) -> tuple[jax.Array, jax.Array]:
    """Draw ``n`` near-circular tracer orbits in a softened point mass.

    Mirrors the construction asserted by the integration test, so the figure and
    the test describe the same system.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    n : int
        Number of orbits.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Positions ``(n, 3)`` and velocities ``(n, 3)``.
    """
    k_r, k_dir, k_v, k_f = jax.random.split(key, 4)
    u = jax.random.uniform(k_r, (n,))
    radius = ORBIT_RMIN * (ORBIT_RMAX / ORBIT_RMIN) ** u
    direction = jax.random.normal(k_dir, (n, 3))
    direction = direction / jnp.linalg.norm(direction, axis=1, keepdims=True)
    raw = jax.random.normal(k_v, (n, 3))
    tangent = raw - jnp.sum(raw * direction, axis=1, keepdims=True) * direction
    tangent = tangent / jnp.linalg.norm(tangent, axis=1, keepdims=True)
    circular = jnp.sqrt(1.0 / jnp.sqrt(radius**2 + SOFTENING**2))
    fraction = jax.random.uniform(k_f, (n,), minval=0.75, maxval=1.0)
    return radius[:, None] * direction, (fraction * circular)[:, None] * tangent


def external_field() -> SoftenedPointMassField:
    """Return the potential the tracer orbits move in: one softened point mass.

    Returns
    -------
    SoftenedPointMassField
        The field, as the integration test builds it.
    """
    return SoftenedPointMassField(
        sources=jnp.zeros((1, 3)),
        source_masses=jnp.asarray([1.0]),
        softening=SOFTENING,
    )


def losvd_kernel() -> Callable[..., jax.Array]:
    """Return projected-radius bins crossed with line-of-sight velocity bins.

    Defined here rather than imported because ``mimirax`` deliberately ships no
    such kernel: [D-027] keeps domain observables in the domain packages, and
    the integration test carries its own copy for the same reason. This one is
    that copy, so the figure describes the system the test asserts.

    Returns
    -------
    Callable[..., jax.Array]
        Maps a rollout state to per-particle kernel values ``(n, 64)``, ordered
        radius-major so that a ``(8, 8)`` reshape is ``[radius, velocity]``.
    """
    radii = jnp.linspace(*LOSVD_RADII, NUM_RADII)
    speeds = jnp.linspace(*LOSVD_SPEEDS, NUM_SPEEDS)

    def kernel(state):
        positions = jnp.asarray(state["positions"])
        projected = jnp.linalg.norm(positions[:, :2], axis=-1)
        line_of_sight = jnp.asarray(state["velocities"])[:, 2]
        radial = jnp.exp(
            -0.5 * ((projected[:, None] - radii[None, :]) / LOSVD_RADIAL_WIDTH) ** 2
        )
        velocity = jnp.exp(
            -0.5
            * ((line_of_sight[:, None] - speeds[None, :]) / LOSVD_VELOCITY_WIDTH) ** 2
        )
        return (radial[:, :, None] * velocity[:, None, :]).reshape(
            positions.shape[0], -1
        )

    return kernel


def observable_sets() -> dict[str, Any]:
    """Return the two observation operators compared by the figures.

    Returns
    -------
    dict[str, Any]
        Keyed by set label; each value is a kernel.
    """
    return {
        "radial": GaussianRadialBins(
            centres=jnp.linspace(*BIN_CENTRES, NUM_BINS),
            width=BIN_WIDTH,
            moments=("mass", "v2"),
        ),
        "losvd": losvd_kernel(),
    }


def measure() -> dict[str, Any]:
    """Run both fits and compute every number the figures show.

    Returns
    -------
    dict[str, Any]
        JSON-serialisable record of the measurement.
    """
    key = jax.random.PRNGKey(SEED)
    k_system, k_weights = jax.random.split(key)
    positions, velocities = tangential_orbits(k_system, NUM_ORBITS)
    truth = 1.0 + (jax.random.uniform(k_weights, (NUM_ORBITS,)) - 0.5)
    truth = truth / jnp.sum(truth)
    reference = float(jnp.mean(truth))
    truth_params = {
        "positions": positions,
        "velocities": velocities,
        "weights": truth,
    }
    start = {**truth_params, "weights": jnp.full((NUM_ORBITS,), reference)}
    rollout = NornaxRollout.tracer(
        external_field(), dt=DT, num_steps=NUM_STEPS, reassign_rungs=False
    )
    threshold = MU / reference
    radii = np.asarray(jnp.linalg.norm(positions, axis=1))

    record: dict[str, Any] = {
        "provenance": {
            "seed": SEED,
            "num_orbits": NUM_ORBITS,
            "mu": MU,
            "softening": SOFTENING,
            "dt": DT,
            "num_steps": NUM_STEPS,
            "fit_steps": FIT_STEPS,
            "optimizer": "optax.lbfgs",
            "threshold_mu_over_w0": threshold,
            "reference_weight": reference,
            "jax": jax.__version__,
        },
        "orbit_radii": radii.tolist(),
        "truth_weights": np.asarray(truth).tolist(),
        "sets": {},
    }

    for label, kernel in observable_sets().items():
        observable = TimeAverage(WeightedKernelSum(kernel))
        observed = observable(rollout(truth_params))
        sigma = 0.02 * jnp.maximum(
            jnp.abs(observed), 1.0e-3 * jnp.max(jnp.abs(observed))
        )
        problem = InferenceProblem(
            forward=rollout,
            observable=observable,
            likelihood=GaussianLikelihood(sigma=sigma),
            observed=observed,
            priors=(EntropyPrior(mu=MU, reference=reference),),
        )
        fit = MadeToMeasure(optimizer=optax.lbfgs(), num_steps=FIT_STEPS).minimize(
            problem.negative_log_posterior, start
        )
        weights = fit.params["weights"]

        data_curvature = fisher_information(
            lambda w: -problem.log_likelihood({**start, "weights": w}),
            start["weights"],
        )
        prior_curvature = fisher_information(
            lambda w: -problem.log_prior({**start, "weights": w}), start["weights"]
        )
        effective = float(effective_parameters(data_curvature, prior_curvature))

        # The whitened Jacobian. Its squared singular values are the data
        # curvature's eigenvalues exactly, without eigvalsh's negatives, and
        # its rows are the per-datum information this figure maps.
        jacobian = jax.jacfwd(lambda w: observable(rollout({**start, "weights": w})))(
            start["weights"]
        )
        jacobian = jacobian.reshape(-1, NUM_ORBITS)
        whitened = jacobian / jnp.reshape(sigma, (-1, 1))
        singular = jnp.linalg.svd(whitened, compute_uv=False)
        spectrum = np.zeros(NUM_ORBITS)
        spectrum[: singular.size] = np.asarray(singular) ** 2
        spectrum = np.sort(spectrum)[::-1]

        record["sets"][label] = {
            "name": SETS[label]["name"],
            "m": int(observed.size),
            "effective": effective,
            "effective_from_spectrum": float(np.sum(spectrum / (spectrum + threshold))),
            "chi_squared": -2.0 * float(problem.log_likelihood(fit.params)),
            "weight_error": float(
                jnp.linalg.norm(weights - truth) / jnp.linalg.norm(truth)
            ),
            "spectrum": spectrum.tolist(),
            "num_above_threshold": int(np.sum(spectrum > threshold)),
            "recovered_weights": np.asarray(weights).tolist(),
            "per_datum_information": (np.asarray(whitened) ** 2).tolist(),
            "observed": np.asarray(observed).reshape(-1).tolist(),
        }

    record["sets"]["losvd"]["grid"] = {
        "shape": [NUM_RADII, NUM_SPEEDS],
        "radial_centres": np.asarray(jnp.linspace(*LOSVD_RADII, NUM_RADII)).tolist(),
        "velocity_centres": np.asarray(
            jnp.linspace(*LOSVD_SPEEDS, NUM_SPEEDS)
        ).tolist(),
    }
    # Deterministic choice, stated so it cannot be mistaken for cherry-picking:
    # the innermost and outermost orbit by radius.
    record["mapped_orbits"] = [int(np.argmin(radii)), int(np.argmax(radii))]
    return record


def _caption(figure: plt.Figure, theme: dict[str, str], text: str) -> None:
    """Lay a wrapped caption along the bottom of a figure.

    Wrapped explicitly rather than left to the renderer, which clips it.

    Parameters
    ----------
    figure : plt.Figure
        Figure to caption.
    theme : dict[str, str]
        One entry of :data:`THEMES`.
    text : str
        Caption text; newlines are honoured, long lines are wrapped.
    """
    lines: list[str] = []
    for paragraph in text.strip().split("\n"):
        lines.extend(textwrap.wrap(paragraph.strip(), width=168) or [""])
    figure.text(
        0.006,
        0.012,
        "\n".join(lines),
        fontsize=8.2,
        color=theme["muted"],
        ha="left",
        va="bottom",
        linespacing=1.45,
    )


def _style(axis: plt.Axes, theme: dict[str, str]) -> None:
    """Apply a theme's foreground colours to one axis.

    Parameters
    ----------
    axis : plt.Axes
        The axis to restyle.
    theme : dict[str, str]
        One entry of :data:`THEMES`.
    """
    axis.set_facecolor("none")
    for spine in axis.spines.values():
        spine.set_color(theme["muted"])
    axis.tick_params(colors=theme["fg"], labelsize=12)
    axis.xaxis.label.set_color(theme["fg"])
    axis.yaxis.label.set_color(theme["fg"])
    axis.title.set_color(theme["fg"])


def figure_one(record: dict[str, Any], theme_name: str) -> plt.Figure:
    """Draw the spectrum and its consequence.

    Parameters
    ----------
    record : dict[str, Any]
        Output of :func:`measure`.
    theme_name : str
        Key of :data:`THEMES`.

    Returns
    -------
    plt.Figure
        The finished figure.
    """
    theme = THEMES[theme_name]
    threshold = record["provenance"]["threshold_mu_over_w0"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    figure.patch.set_facecolor(theme["bg"])

    for label, meta in SETS.items():
        entry = record["sets"][label]
        spectrum = np.asarray(entry["spectrum"])
        positive = spectrum > 0.0
        index = np.arange(1, NUM_ORBITS + 1)
        left.semilogy(
            index[positive],
            spectrum[positive],
            meta["line"],
            marker=meta["marker"],
            color=meta["colour"],
            markersize=6,
            linewidth=2.0,
            label=f"{meta['name']}  (m = {entry['m']})",
        )

    # The directions the data do not constrain are **exactly** zero, not small.
    # Drawing them on the log axis would give them a value they do not have, so
    # they are marked at the floor and counted instead.
    floor = 1.0e-9
    radial = np.asarray(record["sets"]["radial"]["spectrum"])
    zeros = np.flatnonzero(radial == 0.0) + 1
    if zeros.size:
        left.plot(
            zeros,
            np.full(zeros.size, floor),
            marker="x",
            linestyle="none",
            color=SETS["radial"]["colour"],
            markersize=7,
            markeredgewidth=1.6,
        )
        left.annotate(
            f"{zeros.size} directions are **exactly** zero".replace("**", ""),
            xy=(float(zeros.mean()), floor),
            xytext=(float(zeros.mean()), floor * 10**1.5),
            ha="center",
            fontsize=11,
            color=SETS["radial"]["colour"],
            arrowprops=dict(
                arrowstyle="->", color=SETS["radial"]["colour"], linewidth=1.4
            ),
        )

    left.axhline(threshold, color=theme["fg"], linewidth=1.6, linestyle=":")
    left.text(
        NUM_ORBITS + 0.5,
        threshold * 2.0,
        f"prior threshold  $\\mu/w_0$ = {threshold:.1e}",
        ha="right",
        va="bottom",
        fontsize=11,
        color=theme["fg"],
    )
    for label, meta in SETS.items():
        entry = record["sets"][label]
        above = entry["num_above_threshold"]
        left.annotate(
            f"{above} of {NUM_ORBITS}\nabove the cut",
            xy=(above, np.asarray(entry["spectrum"])[above - 1]),
            xytext=(above - 1.5, 10.0 ** (-4.6 if label == "radial" else -2.2)),
            ha="right",
            fontsize=11,
            color=meta["colour"],
            arrowprops=dict(arrowstyle="->", color=meta["colour"], linewidth=1.4),
        )
    left.set_ylim(floor / 6.0, 10.0**8)
    left.set_xlim(0, NUM_ORBITS + 1)
    left.set_xlabel("eigenvalue index $k$  (of 32 orbit weights)")
    left.set_ylabel("data-curvature eigenvalue $d_k$")
    left.set_title(
        "The instrument sets the spectrum.\nThe prior only sets where it is cut.",
        fontsize=13.5,
        loc="left",
        color=theme["fg"],
    )
    left.legend(
        fontsize=10,
        facecolor="none",
        edgecolor=theme["muted"],
        labelcolor=theme["fg"],
        loc="upper right",
    )
    left.grid(True, which="major", color=theme["grid"], linewidth=0.6, alpha=0.5)
    _style(left, theme)

    truth = np.asarray(record["truth_weights"])
    for label, meta in SETS.items():
        entry = record["sets"][label]
        right.plot(
            truth,
            np.asarray(entry["recovered_weights"]),
            meta["marker"],
            color=meta["colour"],
            markersize=8,
            alpha=0.85,
            linestyle="none",
            label=(
                f"{meta['name']}\n"
                f"  effective = {entry['effective']:.2f} of {NUM_ORBITS}"
                f"   $\\chi^2$ = {entry['chi_squared']:.1e}\n"
                f"  weight error = {entry['weight_error']:.3f}"
            ),
        )
    limits = [float(truth.min()) * 0.75, float(truth.max()) * 1.25]
    right.plot(limits, limits, "-", color=theme["muted"], linewidth=1.2, zorder=0)
    right.set_xlim(limits)
    right.set_ylim(limits)
    right.set_xlabel("true orbit weight")
    right.set_ylabel("recovered orbit weight")
    right.set_title(
        "The better fit is the worse answer.\nThe fit is not what fails; the observations are.",
        fontsize=13.5,
        loc="left",
        color=theme["fg"],
    )
    right.legend(
        fontsize=9.5,
        facecolor="none",
        edgecolor=theme["muted"],
        labelcolor=theme["fg"],
        loc="upper left",
    )
    right.grid(True, color=theme["grid"], linewidth=0.6, alpha=0.5)
    _style(right, theme)

    figure.tight_layout(rect=(0.0, 0.155, 1.0, 1.0))
    _caption(
        figure,
        theme,
        "Mock tracer orbits in a softened point mass -- not a galaxy, not an instrument, not "
        f"data. N = {NUM_ORBITS} orbits, softening = {SOFTENING}, {NUM_STEPS} steps of "
        f"dt = {DT}, entropy prior mu = {MU:.0e}, LBFGS for {FIT_STEPS} steps. Eigenvalues are "
        "the squared singular values of the whitened Jacobian, which is exact here and avoids "
        "the negative eigenvalues eigvalsh returns at this spectrum.\n"
        "'effective' is effective_parameters, a local Gaussian-limit quantity, not a posterior. "
        "Both panels show both instruments: cropping either to the LOSVD alone turns the point "
        "into a capability claim and inverts it.",
    )
    return figure


def figure_two(record: dict[str, Any], theme_name: str) -> plt.Figure:
    """Draw the per-datum information map over the LOSVD grid.

    Parameters
    ----------
    record : dict[str, Any]
        Output of :func:`measure`.
    theme_name : str
        Key of :data:`THEMES`.

    Returns
    -------
    plt.Figure
        The finished figure.
    """
    theme = THEMES[theme_name]
    entry = record["sets"]["losvd"]
    shape = tuple(entry["grid"]["shape"])
    radial = np.asarray(entry["grid"]["radial_centres"])
    speeds = np.asarray(entry["grid"]["velocity_centres"])
    extent = (speeds[0], speeds[-1], radial[0], radial[-1])
    information = np.asarray(entry["per_datum_information"])
    observed = np.asarray(entry["observed"]).reshape(shape)
    radii = np.asarray(record["orbit_radii"])

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.6))
    figure.patch.set_facecolor(theme["bg"])

    first = axes[0].imshow(
        observed, origin="lower", aspect="auto", extent=extent, cmap="cividis"
    )
    axes[0].set_title(
        "The observation\nLOSVD, 8 radii x 8 speeds (m = 64)",
        fontsize=12.5,
        loc="left",
        color=theme["fg"],
        pad=14,
    )
    figure.colorbar(first, ax=axes[0]).set_label(
        "time-averaged weighted counts", color=theme["fg"], fontsize=10
    )

    for position, orbit in enumerate(record["mapped_orbits"]):
        axis = axes[position + 1]
        panel = information[:, orbit].reshape(shape)
        image = axis.imshow(
            panel, origin="lower", aspect="auto", extent=extent, cmap="cividis"
        )
        axis.set_title(
            f"Which pixels know about orbit {orbit}\n"
            f"radius = {radii[orbit]:.2f}  "
            f"(total $H_{{kk}}$ = {panel.sum():.2e})",
            fontsize=12.5,
            loc="left",
            color=theme["fg"],
            pad=14,
        )
        figure.colorbar(image, ax=axis).set_label(
            "Fisher information per pixel", color=theme["fg"], fontsize=10
        )

    for axis in axes:
        axis.set_xlabel("line-of-sight velocity")
        axis.set_ylabel("projected radius")
        _style(axis, theme)
    for bar in figure.axes[len(axes) :]:
        bar.tick_params(colors=theme["fg"], labelsize=9)
        bar.yaxis.label.set_color(theme["fg"])
        for spine in bar.spines.values():
            spine.set_color(theme["muted"])

    totals = [float(information[:, orbit].sum()) for orbit in record["mapped_orbits"]]
    figure.tight_layout(rect=(0.0, 0.205, 1.0, 1.0))
    _caption(
        figure,
        theme,
        "Per-pixel contribution to the Fisher information about one orbit weight, "
        "I_jk = (d d_j / d w_k)^2 / sigma_j^2, so that the pixels of a panel sum to that "
        "orbit's H_kk. This is the diagonal term: it says what each pixel contributes on its "
        "own, and does not show how pixels trade off against each other.\n"
        "The two orbits are the innermost and outermost by radius, chosen by index rather than "
        f"by eye. The information panels have independent colour scales -- the outer orbit "
        f"carries {totals[1] / totals[0]:.0f}x the total of the inner one -- so compare the "
        "colour-bar numbers, not the brightness.\n"
        "Mock tracer orbits in a softened point mass -- not a galaxy, not an instrument, not "
        f"data. N = {NUM_ORBITS} orbits, softening = {SOFTENING}, {NUM_STEPS} steps of "
        f"dt = {DT}. cividis is used because it is monotonic in greyscale and safe under "
        "common colour-vision deficiencies.",
    )
    return figure


def main() -> int:
    """Measure, write the JSON, and render both figures in both themes.

    Returns
    -------
    int
        Process exit status.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    record = measure()
    payload = RESULTS / "sensitivity_figure.json"
    payload.write_text(json.dumps(record, indent=1) + "\n")

    # Read back, so the figures cannot see anything the JSON does not carry.
    record = json.loads(payload.read_text())
    for name, draw in (
        ("sensitivity_information", figure_one),
        ("sensitivity_information_map", figure_two),
    ):
        for theme_name in THEMES:
            figure = draw(record, theme_name)
            for suffix in ("pdf", "png"):
                figure.savefig(
                    RESULTS / f"{name}_{theme_name}.{suffix}",
                    dpi=200,
                    facecolor=figure.get_facecolor(),
                )
            plt.close(figure)

    for label, entry in record["sets"].items():
        print(
            f"{label:7s} m={entry['m']:3d}  effective={entry['effective']:.4f}  "
            f"spectrum={entry['effective_from_spectrum']:.4f}  "
            f"chi2={entry['chi_squared']:.2e}  error={entry['weight_error']:.4f}  "
            f"above cut={entry['num_above_threshold']}"
        )
    print(f"wrote {payload} and 8 figure files to {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
