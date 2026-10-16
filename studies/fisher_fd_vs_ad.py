"""Does a finite-difference Fisher matrix overstate what the data constrain?

The EDDA programme's ``RATIONALE.md`` (section 2.8) argued, from reasoning
alone, that a Fisher matrix built from a noisy Jacobian ``J + E`` is lifted at
the bottom of its spectrum: if ``E`` is unbiased,
``E[(J+E)^T C^-1 (J+E)] = F + E[E^T C^-1 E]`` and the added term is positive
semi-definite. The bottom of the spectrum is what ``effective_parameters`` and
the sensitivity figure report, so the claim is that finite differences
*overstate* the information exactly where the answer matters. This study
measures it, in three arms:

``A`` -- tracer, linear in the weights (finite differences' best case)
    The sensitivity figure's three instruments (all ``m = 64`` against ``n = 32``
    weights) and the old ``m = 10`` radial control. The observable is a fixed
    matrix times ``w``, so a forward or central difference has **zero truncation
    error** here and only cancellation error: it improves monotonically with
    ``h``. Nothing found on this arm alone may be generalised.
``B`` -- self-consistent, the weights are the masses and move the orbits
    32 bodies under ``MutualDirectSumGravity``, the 100-column LOSVD, its
    ``m = 10`` radial control and (an addition to the prompt) the ``m = 64``
    wide radial set, at seeds 1, 2 and 3. Truncation error is real here.
``C`` -- three source masses, ``P = 3`` against ``m = 64``
    The forward-mode regime: few parameters, many data, where a finite
    difference is the *fair* competitor. Arm A's tracer orbits in a
    three-centre field whose masses are the parameters. No prior, so the
    quantities are the spectrum, the condition number and the Cramer--Rao
    forecast errors ``sqrt(diag F^-1)``.

Every Fisher matrix is evaluated **at a point**; no fit is run anywhere.

THE METHODS. The reference for every arm and both precisions is the
**double-precision** ``jax.jacfwd`` Jacobian; ``jax.jacrev`` beside it is the AD
noise floor. Measured against it: AD in single precision, and three finite
differences in both -- forward (``P + 1`` runs), central (``2P``) and
Richardson-extrapolated central at ``h`` and ``h/2`` (``4P``) -- at a fixed
ladder ``h = 1e-1 ... 1e-9`` relative to each parameter's scale, and at the
default a user would pick without an oracle: ``sqrt(eps)``, ``eps^(1/3)`` and
``eps^(1/5)`` times the scale. The ladder and the defaults were fixed before
the sweep ran and are not tuned. Every step is made exactly representable
(``step = (x + h) - x``), and every difference is batched with ``vmap``, as
``jacfwd`` batches its tangents.

THE TWO PRECISIONS RUN IN SEPARATE PROCESSES. ``jax_enable_x64`` is
process-wide: a single-precision arm run in a process where anything has turned
it on silently computes in double and measures nothing. So this script, run
with no arguments, is a *driver* that launches itself twice as a worker -- the
precision is passed in the ``FISHER_PROBE_PRECISION`` environment variable and
set before ``jax.numpy`` is imported -- and each worker asserts the dtype of
every observable, Jacobian and spectrum it produces. The float64 worker builds
the systems (random draws, mock data, noise model) and hands them to the
float32 worker as arrays, cast once at entry; so both precisions describe the
same system, and the float32 arm's error includes representing its inputs in
single precision, which a single-precision user cannot avoid either. The float64
worker measures that input-representation floor separately.

THE ANALYSIS IS IN DOUBLE PRECISION, FOR EVERY METHOD. The driver casts each
measured Jacobian to float64 and does all the linear algebra there, so the only
thing that differs between two rows is the Jacobian -- the derivative method and
the arithmetic it was computed in -- and not the eigen-solver. Each worker also
reports what a pipeline kept entirely in its own precision gives
(``*_native``), as a separate column.

SPECTRA ARE SQUARED SINGULAR VALUES OF THE WHITENED JACOBIAN, NEVER
``eigvalsh``. On these problems ``eigvalsh`` returns negative eigenvalues at the
bottom of the spectrum (``studies/sensitivity_figure.py``), which here would be
read as the effect under test. The shipped dense route
(``effective_parameters`` on ``J^T C^-1 J``) and the shipped
``degenerate_directions`` are reported beside the SVD route, not instead of it.

THE MECHANISM. ``F_method - F_ref`` splits into a cross term
``J^T C^-1 E + E^T C^-1 J`` and a quadratic term ``E^T C^-1 E``. In the
reference's eigenbasis the Rayleigh quotient of ``F_method`` along direction
``v_k`` is exactly ``d_k + c_k + q_k``, with ``c_k = 2 (A v_k).(Delta v_k)`` and
``q_k = |Delta v_k|^2`` (``A``, ``Delta`` the whitened Jacobian and error). The
section 2.8 argument holds only where ``q`` dominates the bottom of the
spectrum; where ``c`` does, the error is systematic and its sign is open.

Outputs: ``studies/results/fisher_fd_vs_ad.json`` (every number quoted) and
``fisher_fd_vs_ad.{pdf,png}``, rendered from the JSON after reading it back.
Run with no arguments::

    python studies/fisher_fd_vs_ad.py
"""

from __future__ import annotations

import os

# The precision is chosen per process, before jax.numpy exists. ``None`` means
# this process is the driver, which only analyses and draws, in double.
_PRECISION_VARIABLE = "FISHER_PROBE_PRECISION"
_SCRATCH_VARIABLE = "FISHER_PROBE_SCRATCH"
PRECISION = os.environ.get(_PRECISION_VARIABLE)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", PRECISION != "float32")

import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from mimirax import (  # noqa: E402
    EntropyPrior,
    GaussianLikelihood,
    GaussianRadialBins,
    InferenceProblem,
    TimeAverage,
    WeightedKernelSum,
)
from mimirax.adapters.nornax import NornaxRollout  # noqa: E402
from mimirax.diagnostics import (  # noqa: E402
    degenerate_directions,
    effective_parameters,
    fisher_information,
)
from mimirax.testing import SoftenedPointMassField  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
COMMITTED_SENSITIVITY = RESULTS / "sensitivity_figure.json"

# --- fixed before the sweep ran; never tuned afterwards ----------------------------

LADDER = tuple(10.0 ** (-k) for k in range(1, 10))
FD_METHODS = ("forward", "central", "richardson")
# The step a user picks without an oracle: the textbook balance of truncation
# against cancellation for a first-, second- and fourth-order formula.
DEFAULT_EXPONENT = {"forward": 1.0 / 2.0, "central": 1.0 / 3.0, "richardson": 1.0 / 5.0}
TIMING_REPEATS = 7

# --- the systems ----------------------------------------------------------------

SOFTENING = 0.2
# Arm A, copied from studies/sensitivity_figure.py so that the two describe the
# same system; Phase 0 checks this copy against that script's committed JSON.
A_SEED = 0
A_NUM = 32
A_MU = 1.0e-8
A_DT, A_STEPS = 0.12, 48
ORBIT_RMIN, ORBIT_RMAX = 0.5, 2.0
RADIAL_CENTRES = (0.4, 2.0)
LOSVD_RADII = (0.35, 1.85)
LOSVD_SPEEDS = (-0.9, 0.9)
LOSVD_RADIAL_WIDTH, LOSVD_VELOCITY_WIDTH = 0.25, 0.18
# Arm B, copied from tests/integration/test_m2m_rollout.py ::
# test_the_weights_are_recovered_in_the_self_consistent_construction.
B_SEEDS = (1, 2, 3)
B_NUM = 32
B_MU = 1.0e-6
B_DT, B_STEPS = 0.02, 24
# Arm C. Section 5.2 of the literature note that names the three-offset-mass
# field does not pin it, so it is chosen here, before any result, and fixed:
# three unequal masses (total 1, so Arm A's orbits stay bound) on a triangle of
# radius 0.25, offset slightly out of the plane so no symmetry survives. Three
# non-collinear fixed centres are generically non-integrable; that is not
# measured here.
C_SOURCES = (
    (0.25, 0.0, 0.0),
    (-0.125, 0.2165, 0.05),
    (-0.125, -0.2165, -0.05),
)
C_MASSES = (0.5, 0.3, 0.2)
# Pre-registered as the right panel's case: the set with the most directions
# below the prior threshold, i.e. the most room for a lift to show.
FIGURE_CASE = "A/radial_wide"

CASES: tuple[dict[str, Any], ...] = (
    *(
        dict(name=f"A/{label}", arm="A", set=label, seed=A_SEED)
        for label in ("radial_wide", "radial_narrow", "losvd", "radial_control")
    ),
    *(
        dict(name=f"B/{label}/seed{seed}", arm="B", set=label, seed=seed)
        for seed in B_SEEDS
        for label in ("losvd", "radial_control", "radial_wide")
    ),
    dict(name="C/losvd", arm="C", set="losvd", seed=A_SEED),
)


def tangential_orbits(key: jax.Array, n: int) -> tuple[jax.Array, jax.Array]:
    """Draw ``n`` near-circular tracer orbits in a softened point mass.

    Verbatim from ``studies/sensitivity_figure.py``.

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


def plummer_like(key: jax.Array, n: int) -> tuple[jax.Array, jax.Array]:
    """Draw a small centrally concentrated cloud on mildly bound orbits.

    Verbatim from ``tests/integration/test_m2m_rollout.py::_plummer_like``,
    including its reuse of one key for the radii and the directions.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    n : int
        Number of bodies.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Positions ``(n, 3)`` and velocities ``(n, 3)``.
    """
    k_pos, k_vel = jax.random.split(key)
    radius = jax.random.uniform(k_pos, (n,), minval=0.3, maxval=2.0)
    direction = jax.random.normal(k_pos, (n, 3))
    direction = direction / jnp.linalg.norm(direction, axis=1, keepdims=True)
    return radius[:, None] * direction, 0.3 * jax.random.normal(k_vel, (n, 3))


def losvd_kernel(num_radii: int, num_speeds: int) -> Callable[..., jax.Array]:
    """Return projected-radius bins crossed with line-of-sight velocity bins.

    The kernel of the sensitivity figure and the integration test, with the
    grid size as the only free choice (8 x 8 there, 10 x 10 in the
    self-consistent test). Local for the reason D-027 gives.

    Parameters
    ----------
    num_radii : int
        Number of projected-radius bins.
    num_speeds : int
        Number of line-of-sight velocity bins.

    Returns
    -------
    Callable[..., jax.Array]
        Maps a state to per-particle kernel values ``(n, num_radii * num_speeds)``.
    """

    def kernel(state):
        radii = jnp.linspace(*LOSVD_RADII, num_radii)
        speeds = jnp.linspace(*LOSVD_SPEEDS, num_speeds)
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


def observable_for(case: dict[str, Any]) -> TimeAverage:
    """Return the time-averaged observable a case is measured with.

    Parameters
    ----------
    case : dict[str, Any]
        One entry of :data:`CASES`.

    Returns
    -------
    TimeAverage
        The observation operator.

    Raises
    ------
    ValueError
        If the case names an unknown observable set.
    """
    label = case["set"]
    if label == "radial_wide":
        kernel = GaussianRadialBins(
            centres=jnp.linspace(*RADIAL_CENTRES, 32), width=0.6, moments=("mass", "v2")
        )
    elif label == "radial_narrow":
        kernel = GaussianRadialBins(
            centres=jnp.linspace(*RADIAL_CENTRES, 32), width=0.3, moments=("mass", "v2")
        )
    elif label == "radial_control":
        kernel = GaussianRadialBins(
            centres=jnp.linspace(*RADIAL_CENTRES, 5), width=0.3, moments=("mass", "v2")
        )
    elif label == "losvd":
        size = 10 if case["arm"] == "B" else 8
        kernel = losvd_kernel(size, size)
    else:
        raise ValueError(f"unknown observable set {label!r}")
    return TimeAverage(WeightedKernelSum(kernel))


def point_mass() -> SoftenedPointMassField:
    """Return Arm A's potential: one unit softened point mass at the origin.

    Returns
    -------
    SoftenedPointMassField
        The field.
    """
    return SoftenedPointMassField(
        sources=jnp.zeros((1, 3)), source_masses=jnp.asarray([1.0]), softening=SOFTENING
    )


def three_centres(masses: jax.Array) -> SoftenedPointMassField:
    """Return Arm C's potential, with the three source masses as an argument.

    Parameters
    ----------
    masses : jax.Array
        ``(3,)`` source masses: the parameters of Arm C.

    Returns
    -------
    SoftenedPointMassField
        The field; ``source_masses`` is a traced array field, so no library
        change is needed to differentiate through it.
    """
    return SoftenedPointMassField(
        sources=jnp.asarray(C_SOURCES, dtype=masses.dtype),
        source_masses=masses,
        softening=SOFTENING,
    )


def self_consistent_rollout(reassign_rungs: bool = False) -> NornaxRollout:
    """Return Arm B's rollout: 32 mutually attracting bodies, single rung.

    Parameters
    ----------
    reassign_rungs : bool
        The test leaves this at nornax's default; at ``k_max = 0`` there is one
        rung and it changes nothing, which Phase 0 checks bit for bit.

    Returns
    -------
    NornaxRollout
        The self-consistent construction.
    """
    from nornax.forces.mutual_direct import MutualDirectSumGravity

    return NornaxRollout.self_consistent(
        MutualDirectSumGravity(G=1.0, softening=SOFTENING, k_max=0),
        dt=B_DT,
        num_steps=B_STEPS,
        k_max=0,
        reassign_rungs=reassign_rungs,
    )


def forward_function(
    case: dict[str, Any], inputs: dict[str, jax.Array]
) -> Callable[[jax.Array], jax.Array]:
    """Return the map from a case's parameters to its observables.

    Parameters
    ----------
    case : dict[str, Any]
        One entry of :data:`CASES`.
    inputs : dict[str, jax.Array]
        The case's system, in the worker's precision.

    Returns
    -------
    Callable[[jax.Array], jax.Array]
        ``x (P,) -> y (m,)``: weights for Arms A and B, source masses for C.
    """
    observable = observable_for(case)
    positions, velocities = inputs["positions"], inputs["velocities"]
    if case["arm"] == "A":
        rollout = NornaxRollout.tracer(
            point_mass(), dt=A_DT, num_steps=A_STEPS, reassign_rungs=False
        )

        def over_weights(w):
            state = {"positions": positions, "velocities": velocities, "weights": w}
            return observable(rollout(state))

        return over_weights
    if case["arm"] == "B":
        rollout = self_consistent_rollout()

        def over_masses(w):
            state = {"positions": positions, "velocities": velocities, "weights": w}
            return observable(rollout(state))

        return over_masses
    weights = inputs["weights"]

    def over_sources(masses):
        rollout = NornaxRollout.tracer(
            three_centres(masses), dt=A_DT, num_steps=A_STEPS, reassign_rungs=False
        )
        state = {"positions": positions, "velocities": velocities, "weights": weights}
        return observable(rollout(state))

    return over_sources


# --- the Jacobian methods --------------------------------------------------------


def central_difference(
    f: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """Return a central-difference Jacobian, ``2P`` runs batched with ``vmap``.

    Parameters
    ----------
    f : Callable[[jax.Array], jax.Array]
        ``x (P,) -> y (m,)``.

    Returns
    -------
    Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
        ``(x, h_rel, scale) -> J (m, P)``. The steps are made exactly
        representable, ``up = (x + h) - x`` and ``down = x - (x - h)``, and
        the quotient divides by the step actually taken.
    """

    def jacobian(x, h_rel, scale):
        h = h_rel * scale
        up = (x + h) - x
        down = x - (x - h)
        plus = jax.vmap(f)(x[None, :] + jnp.diag(up))
        minus = jax.vmap(f)(x[None, :] - jnp.diag(down))
        return ((plus - minus) / (up + down)[:, None]).T

    return jacobian


def forward_difference(
    f: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """Return a forward-difference Jacobian, ``P + 1`` runs batched with ``vmap``.

    Parameters
    ----------
    f : Callable[[jax.Array], jax.Array]
        ``x (P,) -> y (m,)``.

    Returns
    -------
    Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
        ``(x, h_rel, scale) -> J (m, P)``, with exactly representable steps.
    """

    def jacobian(x, h_rel, scale):
        step = (x + h_rel * scale) - x
        base = f(x)
        moved = jax.vmap(f)(x[None, :] + jnp.diag(step))
        return ((moved - base[None, :]) / step[:, None]).T

    return jacobian


def richardson_difference(
    f: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """Return a Richardson-extrapolated central difference, ``4P`` runs.

    ``(4 D(h/2) - D(h)) / 3`` cancels a central difference's ``O(h^2)`` term,
    leaving ``O(h^4)``: the "careful finite difference" a referee asks for.

    Parameters
    ----------
    f : Callable[[jax.Array], jax.Array]
        ``x (P,) -> y (m,)``.

    Returns
    -------
    Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
        ``(x, h_rel, scale) -> J (m, P)``.
    """
    central = central_difference(f)

    def jacobian(x, h_rel, scale):
        coarse = central(x, h_rel, scale)
        fine = central(x, 0.5 * h_rel, scale)
        return (4.0 * fine - coarse) / 3.0

    return jacobian


FD_BUILDERS = {
    "forward": forward_difference,
    "central": central_difference,
    "richardson": richardson_difference,
}


def compile_and_time(fn: Callable[..., jax.Array], *args: Any) -> tuple[Any, dict]:
    """Compile ``fn`` ahead of time and time it, compilation separately.

    Parameters
    ----------
    fn : Callable[..., jax.Array]
        The function to jit.
    *args : Any
        Arguments to lower against and to run with.

    Returns
    -------
    tuple[Any, dict]
        The compiled executable, and ``{"compile_s", "run_s"}``: lowering plus
        compilation, and the median of :data:`TIMING_REPEATS` runs after one
        warm-up.
    """
    start = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    compile_s = time.perf_counter() - start
    compiled(*args).block_until_ready()
    runs = []
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        compiled(*args).block_until_ready()
        runs.append(time.perf_counter() - start)
    return compiled, {"compile_s": compile_s, "run_s": statistics.median(runs)}


# --- the systems, built once in double precision -------------------------------------


def noise_model(observed: jax.Array) -> jax.Array:
    """Return the per-observable sigma both source scripts use: 2 % relative.

    Parameters
    ----------
    observed : jax.Array
        ``(m,)`` noise-free mock data.

    Returns
    -------
    jax.Array
        ``(m,)`` standard deviations, floored at 1e-3 of the largest datum.
    """
    return 0.02 * jnp.maximum(jnp.abs(observed), 1.0e-3 * jnp.max(jnp.abs(observed)))


def build_systems() -> dict[str, dict[str, np.ndarray]]:
    """Draw every case's system and mock data, in double precision.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Per case: ``positions``, ``velocities``, the evaluation point ``x``,
        the step ``scale``, ``sigma``, and for Arms A and B the ``truth``,
        ``reference`` weight and prior ``threshold`` ``mu / w0``.
    """
    systems: dict[str, dict[str, np.ndarray]] = {}
    key = jax.random.PRNGKey(A_SEED)
    k_system, k_weights = jax.random.split(key)
    a_positions, a_velocities = tangential_orbits(k_system, A_NUM)
    a_truth = 1.0 + (jax.random.uniform(k_weights, (A_NUM,)) - 0.5)
    a_truth = a_truth / jnp.sum(a_truth)
    a_reference = float(jnp.mean(a_truth))
    b_draws = {}
    for seed in B_SEEDS:
        k_system, k_weights, _ = jax.random.split(jax.random.PRNGKey(seed), 3)
        positions, velocities = plummer_like(k_system, B_NUM)
        truth = 0.5 + jax.random.uniform(k_weights, (B_NUM,))
        b_draws[seed] = (positions, velocities, truth, float(jnp.mean(truth)))

    for case in CASES:
        if case["arm"] == "A":
            positions, velocities, truth, reference = (
                a_positions,
                a_velocities,
                a_truth,
                a_reference,
            )
            mu = A_MU
        elif case["arm"] == "B":
            positions, velocities, truth, reference = b_draws[case["seed"]]
            mu = B_MU
        else:
            masses = jnp.asarray(C_MASSES)
            inputs = {
                "positions": a_positions,
                "velocities": a_velocities,
                "weights": a_truth,
            }
            observed = forward_function(case, inputs)(masses)
            systems[case["name"]] = {
                **{k: np.asarray(v) for k, v in inputs.items()},
                "x": np.asarray(masses),
                "scale": np.asarray(masses),
                "sigma": np.asarray(noise_model(observed)),
            }
            continue
        inputs = {"positions": positions, "velocities": velocities}
        observed = forward_function(case, inputs)(truth)
        w0 = jnp.full((truth.size,), reference)
        systems[case["name"]] = {
            "positions": np.asarray(positions),
            "velocities": np.asarray(velocities),
            "truth": np.asarray(truth),
            "x": np.asarray(w0),
            "scale": np.asarray(w0),
            "sigma": np.asarray(noise_model(observed)),
            "reference": np.asarray(reference),
            "mu": np.asarray(mu),
            "threshold": np.asarray(mu / reference),
        }
    return systems


def problem_for(
    case: dict[str, Any], system: dict[str, jax.Array]
) -> InferenceProblem | None:
    """Return the shipped :class:`InferenceProblem` of an Arm A or B case.

    Parameters
    ----------
    case : dict[str, Any]
        One entry of :data:`CASES`.
    system : dict[str, jax.Array]
        That case's system, as :func:`build_systems` made it.

    Returns
    -------
    InferenceProblem | None
        The problem, or ``None`` for Arm C, which has no prior.
    """
    if case["arm"] == "C":
        return None
    if case["arm"] == "A":
        rollout = NornaxRollout.tracer(
            point_mass(), dt=A_DT, num_steps=A_STEPS, reassign_rungs=False
        )
    else:
        rollout = self_consistent_rollout()
    truth = {
        "positions": system["positions"],
        "velocities": system["velocities"],
        "weights": system["truth"],
    }
    observable = observable_for(case)
    return InferenceProblem(
        forward=rollout,
        observable=observable,
        likelihood=GaussianLikelihood(sigma=system["sigma"]),
        observed=observable(rollout(truth)),
        priors=(
            EntropyPrior(mu=float(system["mu"]), reference=float(system["reference"])),
        ),
    )


# --- the worker: one precision, one process --------------------------------------


def _key(case_name: str, method: str, h: str) -> str:
    """Return the archive key of one Jacobian.

    Parameters
    ----------
    case_name : str
        A case's name.
    method : str
        The method label.
    h : str
        The step label, or ``"none"`` for AD.

    Returns
    -------
    str
        A key safe for ``np.savez``.
    """
    return f"{case_name.replace('/', '.')}|{method}|{h}"


def save_systems(scratch: Path, systems: dict[str, dict[str, np.ndarray]]) -> None:
    """Write every case's system for the other processes to read.

    Parameters
    ----------
    scratch : Path
        Directory shared with the driver.
    systems : dict[str, dict[str, np.ndarray]]
        Output of :func:`build_systems`.
    """
    np.savez(
        scratch / "systems.npz",
        **{
            f"{name.replace('/', '.')}|{field}": value
            for name, system in systems.items()
            for field, value in system.items()
        },
    )


def load_systems(scratch: Path) -> dict[str, dict[str, np.ndarray]]:
    """Read back what :func:`save_systems` wrote, still in double precision.

    Parameters
    ----------
    scratch : Path
        Directory shared with the driver.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Keyed by case name, then by field.
    """
    archive = np.load(scratch / "systems.npz")
    systems: dict[str, dict[str, np.ndarray]] = {}
    for joined in archive.files:
        name, field = joined.split("|")
        systems.setdefault(name.replace(".", "/"), {})[field] = archive[joined]
    return systems


def _native_summary(
    jacobian: jax.Array,
    sigma: jax.Array,
    threshold: jax.Array | None,
    prior: jax.Array | None,
) -> dict[str, Any]:
    """Return the spectrum and effective count computed in the worker's precision.

    Parameters
    ----------
    jacobian : jax.Array
        ``(m, P)`` Jacobian.
    sigma : jax.Array
        ``(m,)`` noise model.
    threshold : jax.Array | None
        ``mu / w0``, or ``None`` when there is no prior.
    prior : jax.Array | None
        ``(P, P)`` prior curvature, or ``None``.

    Returns
    -------
    dict[str, Any]
        ``spectrum`` and, with a prior, ``effective_spectrum`` and
        ``effective_dense``, all as arrays of the worker's dtype.
    """
    whitened = jacobian / sigma[:, None]
    singular = jnp.linalg.svd(whitened, compute_uv=False)
    spectrum = jnp.zeros(jacobian.shape[1], dtype=jacobian.dtype)
    spectrum = jnp.sort(spectrum.at[: singular.size].set(singular**2))[::-1]
    out: dict[str, Any] = {"spectrum": spectrum}
    if threshold is not None and prior is not None:
        out["effective_spectrum"] = jnp.sum(spectrum / (spectrum + threshold))
        out["effective_dense"] = effective_parameters(whitened.T @ whitened, prior)
    return out


def zero_steps(method: str, x: jax.Array, h_rel: jax.Array, scale: jax.Array) -> int:
    """Count the parameters whose finite-difference step rounds to exactly zero.

    In single precision ``x + h == x`` once ``h`` falls below half a unit in
    the last place of ``x``, so the quotient is ``0 / 0``. That is the
    expected garbage end of the ladder, and it is recorded rather than hidden.

    Parameters
    ----------
    method : str
        ``"forward"``, ``"central"`` or ``"richardson"``.
    x : jax.Array
        ``(P,)`` evaluation point.
    h_rel : jax.Array
        Relative step.
    scale : jax.Array
        ``(P,)`` parameter scale.

    Returns
    -------
    int
        The largest count over the steps the method takes.
    """
    worst = 0
    for fraction in (1.0, 0.5) if method == "richardson" else (1.0,):
        h = fraction * h_rel * scale
        taken = (x + h) - x
        if method != "forward":
            taken = taken + (x - (x - h))
        worst = max(worst, int(jnp.sum(taken == 0.0)))
    return worst


def _plain(value: Any) -> Any:
    """Return an array as a flat list, with non-finite entries as ``None``.

    Parameters
    ----------
    value : Any
        A scalar or array.

    Returns
    -------
    Any
        JSON-safe: strict JSON has no ``NaN``.
    """
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float(array) if np.isfinite(array) else None
    return [float(v) if np.isfinite(v) else None for v in array.ravel()]


def run_worker(scratch: Path) -> None:
    """Measure every Jacobian in this process's precision and archive it.

    The float64 worker builds the systems first and writes them for the float32
    worker; the float32 worker reads them and casts them once, at entry. Every
    input, observable, prior curvature, Jacobian and native spectrum has its
    dtype asserted against this process's precision, and the worker stops on
    the first mismatch.

    Parameters
    ----------
    scratch : Path
        Directory shared with the driver.
    """
    dtype = jnp.float64 if PRECISION == "float64" else jnp.float32
    assert jnp.zeros(()).dtype == dtype, (PRECISION, jnp.zeros(()).dtype)
    if PRECISION == "float64":
        systems = build_systems()
        save_systems(scratch, systems)
    else:
        systems = load_systems(scratch)

    eps = float(jnp.finfo(dtype).eps)
    jacobians: dict[str, np.ndarray] = {}
    native: dict[str, Any] = {}
    meta: dict[str, Any] = {
        "precision": PRECISION,
        "eps": eps,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "dtype_assertions": [],
        "cost": {},
        "warnings": [],
        "phase0": {},
    }

    def check(label: str, array: jax.Array) -> None:
        assert array.dtype == dtype, (label, array.dtype, dtype)
        meta["dtype_assertions"].append([label, str(array.dtype)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for case in CASES:
            name = case["name"]
            print(f"[{PRECISION}] {name}", file=sys.stderr, flush=True)
            system = {k: jnp.asarray(v, dtype=dtype) for k, v in systems[name].items()}
            for field in ("positions", "velocities", "x", "scale", "sigma"):
                check(f"{name} input {field}", system[field])
            f = forward_function(case, system)
            x, scale, sigma = system["x"], system["scale"], system["sigma"]
            threshold = system.get("threshold")
            prior = None
            problem = problem_for(case, system) if case["arm"] != "C" else None
            if problem is not None:
                params = {
                    "positions": system["positions"],
                    "velocities": system["velocities"],
                    "weights": x,
                }
                prior = fisher_information(
                    lambda w, p=problem, q=params: -p.log_prior({**q, "weights": w}), x
                )
                check(f"{name} prior curvature", prior)
            check(f"{name} observable", f(x))
            meta["cost"][name] = {}

            ad = {"ad_forward": jax.jacfwd(f), "ad_reverse": jax.jacrev(f)}
            for method, fn in ad.items():
                compiled, cost = compile_and_time(fn, x)
                meta["cost"][name][method] = cost
                jac = compiled(x)
                check(f"{name} {method} Jacobian", jac)
                jacobians[_key(name, method, "none")] = np.asarray(jac)
                summary = _native_summary(jac, sigma, threshold, prior)
                for field, value in summary.items():
                    check(f"{name} {method} native {field}", value)
                native[_key(name, method, "none")] = {
                    k: _plain(v) for k, v in summary.items()
                }

            for method, builder in FD_BUILDERS.items():
                default = eps ** DEFAULT_EXPONENT[method]
                steps = {f"{h:.0e}": h for h in LADDER}
                steps["default"] = default
                compiled = None
                for label, h in steps.items():
                    h_rel = jnp.asarray(h, dtype=dtype)
                    if compiled is None:
                        compiled, cost = compile_and_time(builder(f), x, h_rel, scale)
                        meta["cost"][name][method] = cost
                    jac = compiled(x, h_rel, scale)
                    check(f"{name} {method} h={label} Jacobian", jac)
                    jacobians[_key(name, method, label)] = np.asarray(jac)
                    summary = _native_summary(jac, sigma, threshold, prior)
                    for field, value in summary.items():
                        check(f"{name} {method} h={label} native {field}", value)
                    native[_key(name, method, label)] = {
                        **{k: _plain(v) for k, v in summary.items()},
                        "zero_steps": zero_steps(method, x, h_rel, scale),
                    }
                meta["default_h"] = meta.get("default_h", {})
                meta["default_h"][method] = default

            if PRECISION == "float64":
                meta["phase0"][name] = phase0_case(case, systems[name], f, prior)
                # The input-representation floor: the double-precision Jacobian
                # at the float32-rounded system, which is where the float32
                # worker actually evaluates.
                rounded = {
                    k: jnp.asarray(np.asarray(v, dtype=np.float32), dtype=dtype)
                    for k, v in systems[name].items()
                }
                jacobians[_key(name, "ref_at_rounded_inputs", "none")] = np.asarray(
                    jax.jacfwd(forward_function(case, rounded))(rounded["x"])
                )
                if prior is not None:
                    jacobians[_key(name, "prior_curvature", "none")] = np.asarray(prior)
        meta["warnings"] = sorted(
            {f"{w.category.__name__}: {w.message}" for w in caught}
        )

    np.savez(scratch / f"jacobians_{PRECISION}.npz", **jacobians)
    (scratch / f"native_{PRECISION}.json").write_text(json.dumps(native))
    (scratch / f"meta_{PRECISION}.json").write_text(json.dumps(meta))


def phase0_case(
    case: dict[str, Any],
    system: dict[str, np.ndarray],
    f: Callable[[jax.Array], jax.Array],
    prior: jax.Array | None,
) -> dict[str, Any]:
    """Run one case's Phase 0 checks, in double precision.

    Parameters
    ----------
    case : dict[str, Any]
        One entry of :data:`CASES`.
    system : dict[str, np.ndarray]
        The case's system.
    f : Callable[[jax.Array], jax.Array]
        The case's forward function.
    prior : jax.Array | None
        Prior curvature at the point, or ``None`` for Arm C.

    Returns
    -------
    dict[str, Any]
        The prior curvature's diagonal and off-diagonal extremes; the shipped
        Hessian-route ``effective_parameters``, which includes the residual
        curvature the Gauss--Newton route leaves out; for Arm A the committed
        sensitivity-figure path, recomputed exactly as it ran; and for Arm B
        whether nornax's default ``reassign_rungs`` changes the rollout.
    """
    out: dict[str, Any] = {}
    x = jnp.asarray(system["x"])
    if prior is not None:
        diagonal = jnp.diag(prior)
        off = prior - jnp.diag(diagonal)
        out["prior_diagonal_min"] = float(jnp.min(diagonal))
        out["prior_diagonal_max"] = float(jnp.max(diagonal))
        out["prior_offdiagonal_max_abs"] = float(jnp.max(jnp.abs(off)))
        problem = problem_for(case, {k: jnp.asarray(v) for k, v in system.items()})
        assert problem is not None
        params = {
            "positions": jnp.asarray(system["positions"]),
            "velocities": jnp.asarray(system["velocities"]),
            "weights": x,
        }
        data_hessian = fisher_information(
            lambda w: -problem.log_likelihood({**params, "weights": w}), x
        )
        out["effective_shipped_hessian"] = float(
            effective_parameters(data_hessian, prior)
        )
    if case["arm"] == "A" and case["set"] != "radial_control":
        # The committed path, line for line: jacfwd, whiten, jnp SVD, pad, sort.
        jacobian = jax.jacfwd(f)(x).reshape(-1, A_NUM)
        whitened = jacobian / jnp.reshape(jnp.asarray(system["sigma"]), (-1, 1))
        singular = jnp.linalg.svd(whitened, compute_uv=False)
        spectrum = np.zeros(A_NUM)
        spectrum[: singular.size] = np.asarray(singular) ** 2
        spectrum = np.sort(spectrum)[::-1]
        threshold = float(system["threshold"])
        committed = json.loads(COMMITTED_SENSITIVITY.read_text())["sets"][case["set"]]
        mine = float(np.sum(spectrum / (spectrum + threshold)))
        out["committed_effective_from_spectrum"] = committed["effective_from_spectrum"]
        out["reproduced_effective_from_spectrum"] = mine
        out["reproduction_relative_difference"] = abs(
            mine - committed["effective_from_spectrum"]
        ) / abs(committed["effective_from_spectrum"])
        out["committed_effective_dense"] = committed["effective"]
        out["committed_threshold"] = json.loads(COMMITTED_SENSITIVITY.read_text())[
            "provenance"
        ]["threshold_mu_over_w0"]
        out["spectrum_max_relative_difference"] = float(
            np.max(
                np.abs(spectrum - np.asarray(committed["spectrum"]))
                / np.asarray(committed["spectrum"])
            )
        )
    if case["arm"] == "B":
        state = {
            "positions": jnp.asarray(system["positions"]),
            "velocities": jnp.asarray(system["velocities"]),
            "weights": x,
        }
        frozen = self_consistent_rollout(reassign_rungs=False)(state)
        default = NornaxRollout.self_consistent(
            self_consistent_rollout().force, dt=B_DT, num_steps=B_STEPS, k_max=0
        )(state)
        out["reassign_rungs_max_abs_difference"] = float(
            max(
                jnp.max(jnp.abs(frozen[leaf] - default[leaf]))
                for leaf in ("positions", "velocities")
            )
        )
    return out


# --- the driver: analysis in double precision ----------------------------------------


def _svd(
    whitened: np.ndarray, num_params: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the padded descending spectrum and both full singular bases.

    Parameters
    ----------
    whitened : np.ndarray
        ``(m, P)`` whitened Jacobian.
    num_params : int
        ``P``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(P,)`` eigenvalues ``d_k = s_k^2``, descending, structural zeros
        last; the ``(P, P)`` right basis whose columns match them; and the
        ``(m, m)`` left basis.
    """
    u, singular, vt = np.linalg.svd(whitened, full_matrices=True)
    spectrum = np.zeros(num_params)
    spectrum[: singular.size] = singular**2
    return spectrum, vt.T, u


def _angles_deg(first: np.ndarray, second: np.ndarray) -> list[float]:
    """Return the principal angles between two column spaces, in degrees.

    Parameters
    ----------
    first : np.ndarray
        ``(P, r)`` basis.
    second : np.ndarray
        ``(P, s)`` basis.

    Returns
    -------
    list[float]
        ``min(r, s)`` angles, largest first; empty if either is empty.
    """
    from scipy.linalg import subspace_angles

    if first.shape[1] == 0 or second.shape[1] == 0:
        return []
    return np.degrees(subspace_angles(first, second)).tolist()


def _mechanism(c: np.ndarray, q: np.ndarray, index: np.ndarray) -> dict[str, Any]:
    """Summarise the cross and quadratic terms over a set of directions.

    Parameters
    ----------
    c : np.ndarray
        ``(P,)`` cross term along each reference direction.
    q : np.ndarray
        ``(P,)`` quadratic term along each reference direction.
    index : np.ndarray
        Which directions to summarise.

    Returns
    -------
    dict[str, Any]
        ``cross_abs`` (sum of ``|c_k|``), ``quadratic`` (sum of ``q_k``),
        ``net`` (sum of ``c_k + q_k``: positive is a lift), and whether the
        quadratic term dominates.
    """
    if index.size == 0:
        return {}
    cross = float(np.sum(np.abs(c[index])))
    quadratic = float(np.sum(q[index]))
    return {
        "directions": index.tolist(),
        "cross_abs": cross,
        "quadratic": quadratic,
        "net": float(np.sum(c[index] + q[index])),
        "quadratic_dominates": bool(quadratic > cross),
    }


def analyse(
    jacobian: np.ndarray,
    reference: np.ndarray,
    sigma: np.ndarray,
    threshold: float | None,
    prior: np.ndarray | None,
    detail: bool = False,
) -> dict[str, Any]:
    """Compare one measured Jacobian with the reference, in double precision.

    Parameters
    ----------
    jacobian : np.ndarray
        ``(m, P)`` measured Jacobian, any precision.
    reference : np.ndarray
        ``(m, P)`` double-precision ``jacfwd`` Jacobian.
    sigma : np.ndarray
        ``(m,)`` noise model.
    threshold : float | None
        ``mu / w0``, or ``None`` when there is no prior.
    prior : np.ndarray | None
        ``(P, P)`` prior curvature, or ``None``.
    detail : bool
        Whether to keep the per-direction arrays (``c_k``, ``q_k``,
        ``q_lift_k``); on for AD and the default steps, off for the ladder,
        whose summaries carry the same story.

    Returns
    -------
    dict[str, Any]
        Every per-method quantity the report and the figure quote.
    """
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if not np.all(np.isfinite(jacobian)):
        return {"finite": False, "num_nonfinite": int(np.sum(~np.isfinite(jacobian)))}
    m, num_params = reference.shape
    rank = min(m, num_params)
    error = jacobian - reference
    column = np.linalg.norm(error, axis=0) / np.linalg.norm(reference, axis=0)
    whitened_ref = reference / sigma[:, None]
    whitened = jacobian / sigma[:, None]
    d_ref, v_ref, u_ref = _svd(whitened_ref, num_params)
    d, v, _ = _svd(whitened, num_params)

    delta = whitened - whitened_ref
    a_v = whitened_ref @ v_ref
    delta_v = delta @ v_ref
    c = 2.0 * np.sum(a_v * delta_v, axis=0)
    q = np.sum(delta_v * delta_v, axis=0)
    cross_matrix = whitened_ref.T @ delta + delta.T @ whitened_ref
    quad_matrix = delta.T @ delta

    lowest = max(1, rank // 4)
    bottom = np.arange(rank - lowest, rank)
    out: dict[str, Any] = {
        "finite": True,
        "jacobian_relative_error": float(
            np.linalg.norm(error) / np.linalg.norm(reference)
        ),
        "worst_column_relative_error": float(np.max(column)),
        "worst_column": int(np.argmax(column)),
        "spectrum": d.tolist(),
        "smallest_nonzero_ratio": float(d[rank - 1] / d_ref[rank - 1]),
        "cross_frobenius": float(np.linalg.norm(cross_matrix)),
        "quadratic_frobenius": float(np.linalg.norm(quad_matrix)),
        "bottom_quarter": _mechanism(c, q, bottom),
    }
    if detail:
        out["c_k"] = c.tolist()
        out["q_k"] = q.tolist()
    if rank < num_params:
        # The structural null space: no Jacobian of this shape can lift its
        # eigenvalues, so the quadratic term shows as a rotation instead.
        null = np.arange(rank, num_params)
        out["null_space"] = _mechanism(c, q, null)
        out["null_space_angles_deg"] = _angles_deg(v_ref[:, null], v[:, null])
    if threshold is not None and prior is not None:
        below = np.flatnonzero(d_ref < threshold)
        out["num_above_threshold"] = int(np.sum(d > threshold))
        out["effective_spectrum"] = float(np.sum(d / (d + threshold)))
        out["effective_dense"] = float(
            effective_parameters(jnp.asarray(whitened.T @ whitened), jnp.asarray(prior))
        )
        out["below_threshold"] = _mechanism(c, q, below)
        if below.size:
            # Which part of the quadratic term can lift. Along a degenerate
            # direction v_k, error that lands in the span U_t of the directions
            # the data already determine is absorbed by rotating the basis, at
            # second order; only error outside that span raises the bottom
            # eigenvalues. To first order the method's bottom eigenvalues are
            # the squared singular values of U_rest^T (A + Delta) V_b, the Schur
            # complement of the determined block -- predicted here and checked
            # against the measured spectrum.
            top = below.min()
            u_top, u_rest = u_ref[:, :top], u_ref[:, top:]
            rotation = np.sum((u_top.T @ delta_v[:, below]) ** 2, axis=0)
            lift = q[below] - rotation
            predicted = np.zeros(below.size)
            if u_rest.shape[1]:
                schur = np.linalg.svd(
                    u_rest.T @ (whitened @ v_ref[:, below]), compute_uv=False
                )
                predicted[: schur.size] = schur**2
            out["below_threshold"].update(
                {
                    "quadratic_rotation": float(np.sum(rotation)),
                    "quadratic_lift": float(np.sum(lift)),
                    "mean_lift_over_threshold": float(np.mean(lift) / threshold),
                    "schur_bottom_spectrum": np.sort(predicted)[::-1].tolist(),
                    "effective_schur": float(
                        np.sum(d[:top] / (d[:top] + threshold))
                        + np.sum(predicted / (predicted + threshold))
                    ),
                }
            )
            if detail:
                out["below_threshold"]["q_lift_k"] = lift.tolist()
        out["below_threshold_angles_deg"] = _angles_deg(
            v_ref[:, below], v[:, num_params - below.size :]
        )
        fisher = jnp.asarray(whitened.T @ whitened)
        eigenvalues = np.asarray(jnp.linalg.eigh(fisher)[0])
        _, shipped = degenerate_directions(
            fisher, rtol=threshold / float(np.max(np.abs(eigenvalues)))
        )
        shipped = np.asarray(shipped)
        out["shipped_degenerate_count"] = int(shipped.shape[1])
        out["shipped_negative_eigenvalues"] = int(np.sum(eigenvalues < 0.0))
        out["shipped_degenerate_angles_deg"] = _angles_deg(v_ref[:, below], shipped)
    else:
        inverse = v @ np.diag(1.0 / d) @ v.T
        out["cramer_rao"] = np.sqrt(np.diag(inverse)).tolist()
        out["condition_number"] = float(d[0] / d[-1])
    return out


def _launch(precision: str, scratch: Path) -> None:
    """Run this script as a worker in a fresh process.

    Parameters
    ----------
    precision : str
        ``"float64"`` or ``"float32"``.
    scratch : Path
        Directory shared with the worker.

    Raises
    ------
    RuntimeError
        If the worker exits non-zero.
    """
    env = dict(os.environ)
    env[_PRECISION_VARIABLE] = precision
    env[_SCRATCH_VARIABLE] = str(scratch)
    env["JAX_ENABLE_X64"] = "1" if precision == "float64" else "0"
    env.setdefault("JAX_PLATFORMS", "cpu")
    status = subprocess.run([sys.executable, str(Path(__file__).resolve())], env=env)
    if status.returncode != 0:
        raise RuntimeError(f"the {precision} worker exited {status.returncode}")


def _git_head(repo: Path) -> str:
    """Return a checkout's HEAD commit, or ``"unknown"``.

    Parameters
    ----------
    repo : Path
        A directory inside the checkout.

    Returns
    -------
    str
        The full commit hash, with ``+dirty`` when the tree has changes.
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("+dirty" if dirty else "")


def measure() -> dict[str, Any]:
    """Run both workers and analyse every Jacobian they measured.

    Returns
    -------
    dict[str, Any]
        JSON-serialisable record of the whole probe.
    """
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        for precision in ("float64", "float32"):
            _launch(precision, scratch)
        return collect(scratch)


def collect(scratch: Path) -> dict[str, Any]:
    """Analyse, in double precision, everything the two workers left in ``scratch``.

    Parameters
    ----------
    scratch : Path
        Directory the workers wrote to.

    Returns
    -------
    dict[str, Any]
        JSON-serialisable record of the whole probe.
    """
    import nornax

    import mimirax

    archive = {
        precision: np.load(scratch / f"jacobians_{precision}.npz")
        for precision in ("float64", "float32")
    }
    jacobians = {p: {k: a[k] for k in a.files} for p, a in archive.items()}
    native = {
        p: json.loads((scratch / f"native_{p}.json").read_text())
        for p in ("float64", "float32")
    }
    meta = {
        p: json.loads((scratch / f"meta_{p}.json").read_text())
        for p in ("float64", "float32")
    }
    systems = load_systems(scratch)

    record: dict[str, Any] = {
        "provenance": {
            "script": "studies/fisher_fd_vs_ad.py",
            "mimirax_commit": _git_head(Path(mimirax.__file__).parent),
            "nornax_commit": _git_head(Path(nornax.__file__).parent),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "machine": f"{platform.system()} {platform.machine()}, CPU only",
            "seeds": {"A": A_SEED, "B": list(B_SEEDS), "C": A_SEED},
            "eps": {p: meta[p]["eps"] for p in meta},
            "ladder_h_rel": list(LADDER),
            "default_exponent": DEFAULT_EXPONENT,
            "default_h_rel": {p: meta[p]["default_h"] for p in meta},
            "timing_repeats": TIMING_REPEATS,
            "arm_c_sources": [list(s) for s in C_SOURCES],
            "arm_c_masses": list(C_MASSES),
            "figure_case": FIGURE_CASE,
            "jax_enable_x64": {p: meta[p]["jax_enable_x64"] for p in meta},
        },
        "dtype_assertions": {
            p: {
                "count": len(meta[p]["dtype_assertions"]),
                "all": meta[p]["dtype_assertions"],
            }
            for p in meta
        },
        "warnings": {p: meta[p]["warnings"] for p in meta},
        "phase0": meta["float64"]["phase0"],
        "cases": {},
    }

    for case in CASES:
        name = case["name"]
        system = systems[name]
        sigma = np.asarray(system["sigma"], dtype=np.float64)
        threshold = float(system["threshold"]) if "threshold" in system else None
        prior_key = _key(name, "prior_curvature", "none")
        prior = jacobians["float64"].get(prior_key)
        reference = jacobians["float64"][_key(name, "ad_forward", "none")]
        rounded = jacobians["float64"][_key(name, "ref_at_rounded_inputs", "none")]
        d_ref, _, _ = _svd(reference / sigma[:, None], reference.shape[1])
        entry: dict[str, Any] = {
            "arm": case["arm"],
            "set": case["set"],
            "seed": case["seed"],
            "m": int(reference.shape[0]),
            "P": int(reference.shape[1]),
            "threshold": threshold,
            "reference_spectrum": d_ref.tolist(),
            "input_rounding_floor": float(
                np.linalg.norm(rounded - reference) / np.linalg.norm(reference)
            ),
            "cost": {p: meta[p]["cost"][name] for p in meta},
            "methods": {},
        }
        if threshold is not None:
            entry["reference_effective_spectrum"] = float(
                np.sum(d_ref / (d_ref + threshold))
            )
            entry["reference_num_above_threshold"] = int(np.sum(d_ref > threshold))
        entry["reference_num_above_1e-3_of_largest"] = int(
            np.sum(d_ref > 1.0e-3 * d_ref[0])
        )
        for precision in ("float64", "float32"):
            rows: dict[str, Any] = {}
            for method in ("ad_forward", "ad_reverse"):
                key = _key(name, method, "none")
                rows[method] = {
                    "none": {
                        **analyse(
                            jacobians[precision][key],
                            reference,
                            sigma,
                            threshold,
                            prior,
                            detail=True,
                        ),
                        "native": native[precision][key],
                    }
                }
            for method in FD_METHODS:
                rows[method] = {}
                labels = [f"{h:.0e}" for h in LADDER] + ["default"]
                for label in labels:
                    key = _key(name, method, label)
                    rows[method][label] = {
                        **analyse(
                            jacobians[precision][key],
                            reference,
                            sigma,
                            threshold,
                            prior,
                            detail=label == "default",
                        ),
                        "native": {
                            k: v
                            for k, v in native[precision][key].items()
                            if label == "default" or k != "spectrum"
                        },
                    }
            entry["methods"][precision] = rows
        record["cases"][name] = entry
    return record


# --- the figure ---------------------------------------------------------------------

# Okabe-Ito, as in sensitivity_figure.py; no claim rests on hue alone -- the arms
# also differ in marker, the precisions in line style.
ARM_STYLE = {
    "A": dict(colour="#009E73", name="A  tracer, linear in w (radial, width 0.6)"),
    "B": dict(colour="#D55E00", name="B  self-consistent (LOSVD 10x10, seeds 1-3)"),
    "C": dict(colour="#0072B2", name="C  three source masses (LOSVD 8x8)"),
}
DEFAULT_MARKERS = {"forward": "v", "central": "o", "richardson": "D"}
PRECISION_COLOUR = {"float64": "#0072B2", "float32": "#D55E00"}


def _error(row: dict[str, Any]) -> float:
    """Return a row's Jacobian error, or ``nan`` where the step rounded to zero.

    Parameters
    ----------
    row : dict[str, Any]
        One analysed Jacobian.

    Returns
    -------
    float
        ``|J - J_ref|_F / |J_ref|_F``.
    """
    return row["jacobian_relative_error"] if row["finite"] else float("nan")


def figure(record: dict[str, Any]) -> Any:
    """Draw the Jacobian error against ``h``, and the figure case's spectrum.

    Parameters
    ----------
    record : dict[str, Any]
        The JSON record, read back from disk.

    Returns
    -------
    Any
        The matplotlib figure.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(15.0, 7.4))
    fig.patch.set_facecolor("#FFFFFF")
    ladder = np.asarray(record["provenance"]["ladder_h_rel"])
    labels = [f"{h:.0e}" for h in ladder]
    defaults = record["provenance"]["default_h_rel"]
    arm_cases = {
        "A": ["A/radial_wide"],
        "B": [n for n in record["cases"] if n.startswith("B/losvd/")],
        "C": ["C/losvd"],
    }
    for arm, style in ARM_STYLE.items():
        names = arm_cases[arm]
        for precision, line in (("float64", "-"), ("float32", "--")):
            curves = np.asarray(
                [
                    [
                        _error(record["cases"][n]["methods"][precision]["central"][lab])
                        for lab in labels
                    ]
                    for n in names
                ]
            )
            left.loglog(
                ladder,
                np.nanmedian(curves, axis=0),
                line,
                color=style["colour"],
                linewidth=2.0,
                label=f"{style['name']}" if precision == "float64" else None,
            )
            if len(names) > 1:
                left.fill_between(
                    ladder,
                    np.nanmin(curves, axis=0),
                    np.nanmax(curves, axis=0),
                    color=style["colour"],
                    alpha=0.18,
                    linewidth=0,
                )
            for method, marker in DEFAULT_MARKERS.items():
                values = [
                    _error(record["cases"][n]["methods"][precision][method]["default"])
                    for n in names
                ]
                left.plot(
                    [defaults[precision][method]] * len(values),
                    values,
                    marker,
                    color=style["colour"],
                    markersize=9,
                    markeredgewidth=1.6,
                    markerfacecolor=(
                        "none" if precision == "float32" else style["colour"]
                    ),
                    linestyle="none",
                )
            if precision == "float32":
                ad = np.median(
                    [
                        _error(
                            record["cases"][n]["methods"][precision]["ad_forward"][
                                "none"
                            ]
                        )
                        for n in names
                    ]
                )
                left.axhline(ad, color=style["colour"], linestyle=":", linewidth=1.4)
    grey = "#505050"
    left.plot([], [], "-", color=grey, label="central difference, float64")
    left.plot([], [], "--", color=grey, label="central difference, float32")
    left.plot(
        [], [], ":", color=grey, label="AD, float32 (float64 AD is the reference)"
    )
    for method, marker in DEFAULT_MARKERS.items():
        left.plot(
            [],
            [],
            marker,
            color=grey,
            linestyle="none",
            label=f"default step, {method}",
        )
    left.set_xlabel("step h, relative to the parameter scale", fontsize=12)
    left.set_ylabel("Jacobian error  |J - J_ref|_F / |J_ref|_F", fontsize=12)
    left.set_ylim(1.0e-14, 30.0)
    left.set_title(
        "Linear arm: no optimum. Nonlinear arms: a U.\n"
        "Single precision: every step is three or more orders worse.",
        fontsize=12.5,
        loc="left",
    )
    left.legend(fontsize=8.6, loc="lower left", framealpha=0.9)
    left.grid(True, which="major", alpha=0.3)

    case = record["cases"][FIGURE_CASE]
    threshold = case["threshold"]
    index = np.arange(1, case["P"] + 1)
    reference = np.asarray(case["reference_spectrum"])
    right.semilogy(
        index,
        reference,
        "-",
        color="#101010",
        linewidth=2.6,
        label=(
            f"reference, float64 AD: effective "
            f"{case['reference_effective_spectrum']:.2f}"
        ),
        zorder=5,
    )
    for precision, colour in PRECISION_COLOUR.items():
        for method, marker in DEFAULT_MARKERS.items():
            row = case["methods"][precision][method]["default"]
            right.semilogy(
                index,
                row["spectrum"],
                "-" if precision == "float64" else "--",
                marker=marker,
                color=colour,
                markersize=4.5,
                linewidth=1.1,
                label=(
                    f"{method}, default h, {precision}: "
                    f"effective {row['effective_spectrum']:.2f}"
                ),
            )
    ad32 = case["methods"]["float32"]["ad_forward"]["none"]
    right.semilogy(
        index,
        ad32["spectrum"],
        ":",
        color="#7F7F7F",
        linewidth=2.0,
        label=f"AD, float32: effective {ad32['effective_spectrum']:.2f}",
    )
    # Below (eps * s_max)^2 even the reference is round-off, so that band is
    # shaded rather than read.
    floor = (record["provenance"]["eps"]["float64"] * np.sqrt(reference[0])) ** 2
    right.axhspan(1.0e-30, floor, color="#C8C8C8", alpha=0.45, linewidth=0)
    right.text(
        case["P"],
        floor / 1000.0,
        "below the reference's own double-precision resolution",
        ha="right",
        fontsize=8.5,
        color=grey,
    )
    right.axhline(threshold, color="#101010", linestyle=":", linewidth=1.6)
    right.text(
        case["P"],
        threshold * 3.0,
        f"prior threshold mu/w0 = {threshold:.1e}",
        ha="right",
        va="bottom",
        fontsize=10,
    )
    right.set_ylim(1.0e-28, 1.0e8)
    right.set_xlim(0.5, case["P"] + 0.5)
    right.set_xlabel("eigenvalue index k  (of 32 orbit weights)", fontsize=12)
    right.set_ylabel("Fisher eigenvalue  (squared singular value)", fontsize=12)
    right.set_title(
        f"{FIGURE_CASE}: a lift shows in the 18 directions below the threshold.\n"
        "Cancellation noise lifts them; central and Richardson in float64 do not.",
        fontsize=12.5,
        loc="left",
    )
    right.legend(fontsize=8.2, loc="lower left", framealpha=0.9)
    right.grid(True, which="major", alpha=0.3)
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 1.0))
    fig.text(
        0.006,
        0.012,
        "Mock systems, not a galaxy and not data: 32 tracer orbits in a softened point "
        "mass (A), 32 self-gravitating bodies (B), three point masses (C). Every Fisher "
        "matrix is J^T C^-1 J at one point; no fit was run.\n"
        "The reference is the float64 jacfwd Jacobian; jacfwd and jacrev agree to "
        "<= 2.6e-14. All spectra are squared singular values of the whitened Jacobian, "
        "computed in float64 for every method so that only the Jacobian differs.\n"
        "Defaults: sqrt(eps), eps^(1/3), eps^(1/5) times the scale, fixed before the "
        "sweep. Float32 steps below 1e-7 round to zero (x + h == x) and are not drawn. "
        "Arm B: median and range over seeds 1-3.",
        fontsize=8.2,
        color=grey,
        ha="left",
        va="bottom",
        linespacing=1.45,
    )
    return fig


def _dumps(value: Any, depth: int = 0) -> str:
    """Serialise to JSON with nested objects indented and flat lists on one line.

    Parameters
    ----------
    value : Any
        A JSON-serialisable value.
    depth : int
        Current nesting depth.

    Returns
    -------
    str
        Strict JSON, readable by eye and a fraction of ``indent=1``'s size.
    """
    pad = " " * (depth + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [
            f"{pad}{json.dumps(k)}: {_dumps(v, depth + 1)}" for k, v in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + " " * depth + "}"
    if isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        items = [f"{pad}{_dumps(v, depth + 1)}" for v in value]
        return "[\n" + ",\n".join(items) + "\n" + " " * depth + "]"
    return json.dumps(value, allow_nan=False)


def main() -> int:
    """Run the probe, write the JSON, read it back and draw from it.

    Returns
    -------
    int
        Process exit status.
    """
    if PRECISION is not None:
        run_worker(Path(os.environ[_SCRATCH_VARIABLE]))
        return 0
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = RESULTS / "fisher_fd_vs_ad.json"
    payload.write_text(_dumps(measure()) + "\n")
    record = json.loads(payload.read_text())
    fig = figure(record)
    for suffix in ("pdf", "png"):
        fig.savefig(RESULTS / f"fisher_fd_vs_ad.{suffix}", dpi=200)
    print(f"wrote {payload} and the figure to {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
