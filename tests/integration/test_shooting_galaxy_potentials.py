"""Multiple shooting recovers a galaxy potential's halo: the S10 recovery gate.

A small self-gravitating disc (N = 64, total mass 1e-2) in nornax's analytic
galaxy potential -- a Miyamoto-Nagai disc field plus an NFW halo
(:mod:`nornax.potentials`, ``f(x, t, args)``, adapted to jaccpot's rollout by
:func:`~mimirax.adapters.jaccpot.rows_field`) -- on jaccpot's direct-sum
rollout. ``theta = {log_halo_mass, log_halo_scale}`` enters the field as the
rollout's ``external_args``; the disc field's parameters are fixed. The final
phase-space state is observed with noise, and the halo is recovered from a
perturbed start with 2 segments and with single shooting. CPU, float64.
``mimirax[jaccpot]`` and a nornax with :mod:`nornax.potentials`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

from mimirax import (
    AugmentedLagrangian,
    GaussianLikelihood,
    InferenceProblem,
    MultipleShootingForwardModel,
)
from mimirax.adapters.jaccpot import DirectSumSegmentEngine

pytest.importorskip("jaccpot", reason="mimirax[jaccpot] is not installed")
pytest.importorskip(
    "jaccpot.runtime.grad_rollout", reason="jaccpot predates differentiable_rollout"
)
potentials = pytest.importorskip(
    "nornax.potentials", reason="nornax predates its analytic potentials"
)

TRUTH = {
    "log_halo_mass": jnp.log(jnp.asarray(5.0)),
    "log_halo_scale": jnp.log(jnp.asarray(3.0)),
}
DISC = potentials.MiyamotoNagaiParams(M=1.0, a=1.0, b=0.1)
GALAXY = potentials.CompositePotential(
    disc=potentials.MiyamotoNagai(), halo=potentials.NFW()
)
DT = 0.01
STEPS = 80
SOFTENING = 0.1


def galaxy_field(x, t, theta):
    """The disc field (fixed) plus the halo whose parameters are ``theta``."""
    halo = potentials.NFWParams(
        M=jnp.exp(theta["log_halo_mass"]), r_s=jnp.exp(theta["log_halo_scale"])
    )
    return GALAXY(x, t, {"disc": DISC, "halo": halo})


def _disc(n, seed=3):
    """Particles on near-circular orbits at R in [0.5, 3], |z| ~ 0.05."""
    k_r, k_phi, k_z = jax.random.split(jax.random.PRNGKey(seed), 3)
    radius = jax.random.uniform(k_r, (n,), minval=0.5, maxval=3.0)
    phi = jax.random.uniform(k_phi, (n,), maxval=2.0 * jnp.pi)
    z = 0.05 * jax.random.normal(k_z, (n,))
    positions = jnp.stack([radius * jnp.cos(phi), radius * jnp.sin(phi), z], -1)
    a = galaxy_field(positions, 0.0, TRUTH)
    radial = -(a[:, 0] * jnp.cos(phi) + a[:, 1] * jnp.sin(phi))
    speed = jnp.sqrt(radius * radial)
    velocities = jnp.stack(
        [-speed * jnp.sin(phi), speed * jnp.cos(phi), jnp.zeros(n)], -1
    )
    return positions, velocities, jnp.full((n,), 1e-2 / n)


def _model(segment_steps):
    x0, v0, masses = _disc(64)
    engine = DirectSumSegmentEngine(
        segment_steps=tuple(segment_steps),
        dt=DT,
        field=galaxy_field,
        softening=SOFTENING,
    )
    return MultipleShootingForwardModel(engine, masses, len(segment_steps), x0, v0)


def _phase_space(state):
    return jnp.concatenate([state["positions"], state["velocities"]], -1).reshape(-1)


def test_recover_the_halo_with_multiple_and_single_shooting() -> None:
    """2 segments and single shooting recover the NFW halo's mass and scale.

    Start at log_halo_mass + 0.3, log_halo_scale - 0.2; the final state observed
    with Gaussian noise sigma = 1e-3. Both converge to the same estimate,
    within the noise of the truth. Measured (L-BFGS inner loop, mu0 = 100; CPU,
    jaccpot s9/session2, nornax s10/shooting-potentials, 2026-11-01): log_halo_mass
    +1.063e-3 (2 segments) / +1.068e-3 (single), log_halo_scale +1.053e-3 /
    +1.057e-3; 12 outer / 357 inner steps against 2 / 21; 42 s for the file.
    """
    model = _model((STEPS // 2,) * 2)
    single_model = _model((STEPS,))
    sigma = 1e-3
    truth = {"theta": TRUTH, "nodes": jax.jit(model.init_nodes)(TRUTH)}
    clean = _phase_space(jax.jit(lambda p: model(p))(truth))
    observed = clean + sigma * jax.random.normal(jax.random.PRNGKey(11), clean.shape)
    likelihood = GaussianLikelihood(sigma=sigma)
    problem = InferenceProblem(model, _phase_space, likelihood, observed)
    single_problem = InferenceProblem(single_model, _phase_space, likelihood, observed)
    start = {
        "log_halo_mass": TRUTH["log_halo_mass"] + 0.3,
        "log_halo_scale": TRUTH["log_halo_scale"] - 0.2,
    }
    driver = AugmentedLagrangian(
        optimizer=optax.lbfgs(), inner_steps=200, max_outer=20, mu0=100.0
    )
    single = driver.minimize(
        single_problem, {"theta": start, "nodes": single_model.init_nodes(start)}
    )
    multiple = driver.minimize(
        problem, {"theta": start, "nodes": jax.jit(model.init_nodes)(start)}
    )
    assert single.converged and multiple.converged
    assert multiple.num_traces == 1
    for name in TRUTH:
        got = float(multiple.params["theta"][name])
        print(
            f"{name}: multiple {got - float(TRUTH[name]):+.3e}, single "
            f"{float(single.params['theta'][name]) - float(TRUTH[name]):+.3e}; outer/inner "
            f"multiple {len(multiple.outer_steps)}/{len(multiple.objective_trace)}, "
            f"single {len(single.outer_steps)}/{len(single.objective_trace)}"
        )
        assert abs(got - float(single.params["theta"][name])) < 1e-5, name
        assert abs(got - float(TRUTH[name])) < 2e-2, name
