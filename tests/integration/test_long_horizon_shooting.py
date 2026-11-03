"""On a chaotic horizon the single-shot gradient explodes; multiple shooting's does not.

The qualitative claim of ``studies/long_horizon_shooting`` (S10 session 2,
track L), pinned cheaply: a small self-gravitating Plummer system (N = 64, mass
0.2, scale 0.3) at the centre of session 1's galaxy (nornax's Miyamoto-Nagai
disc, fixed, plus an NFW halo whose ``theta = (log M, log r_s)`` is the unknown),
on jaccpot's direct-sum rollout (``checkpoint="step"``, the production reverse),
observed every segment (``output="segment_ends"``). Its stars' mutual encounters
make it chaotic, and the halo acts on every orbit, so ``dX/dtheta`` grows like
``exp(lambda t)``.

At a fixed small offset from the truth, the gradient in ``theta`` of

* the single-shooting objective ``f(theta, init_nodes(theta))`` -- one
  trajectory over the whole horizon -- and
* the multiple-shooting augmented Lagrangian at the truth's nodes, zero
  multipliers and ``mu = 1 / sigma^2``

agree on one segment (the two are the same function there) and part
exponentially with the horizon: every segment adds its own bounded share to the
multiple-shooting gradient, while the single shot's grows with the whole
horizon's amplification. CPU, float64, seconds. ``mimirax[jaccpot]`` and a
nornax with :mod:`nornax.potentials`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from mimirax import (
    GaussianLikelihood,
    InferenceProblem,
    MultipleShootingForwardModel,
    augmented_lagrangian_value,
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
N = 64
SOFTENING = 0.015
DT = 0.02
SEGMENT_STEPS = 80
SIGMA = 1.0e-3


def galaxy_field(x, t, theta):
    """The disc field (fixed) plus the halo whose parameters are ``theta``."""
    halo = potentials.NFWParams(
        M=jnp.exp(theta["log_halo_mass"]), r_s=jnp.exp(theta["log_halo_scale"])
    )
    return GALAXY(x, t, {"disc": DISC, "halo": halo})


def _stars(seed=0):
    """Plummer positions (mass 0.2, scale 0.3); velocities in virial balance."""
    k_r, k_d, k_v = jax.random.split(jax.random.PRNGKey(seed), 3)
    u = jax.random.uniform(k_r, (N,), minval=0.02, maxval=0.95)
    r = 0.3 / jnp.sqrt(u ** (-2.0 / 3.0) - 1.0)
    d = jax.random.normal(k_d, (N, 3))
    x = r[:, None] * d / jnp.linalg.norm(d, axis=1, keepdims=True)
    v = jax.random.normal(k_v, (N, 3))
    x, v = x - x.mean(0), v - v.mean(0)
    masses = jnp.full((N,), 0.2 / N)
    diff = x[:, None] - x[None]
    r2 = jnp.sum(diff * diff, -1) + SOFTENING**2
    a_self = -jnp.sum(masses[None, :, None] * diff * r2[..., None] ** -1.5, axis=1)
    a = a_self + galaxy_field(x, 0.0, TRUTH)
    w = -jnp.sum(masses[:, None] * x * a)
    v = v * jnp.sqrt(w / jnp.sum(masses[:, None] * v * v))
    return x, v, masses


def _phase_space(state):
    return jnp.concatenate([state["positions"], state["velocities"]], -1).reshape(-1)


def _gradients(num_segments):
    """|grad_theta| of the single shot and of the multiple-shooting Lagrangian."""
    x0, v0, masses = _stars()
    engine = DirectSumSegmentEngine(
        segment_steps=(SEGMENT_STEPS,) * num_segments,
        dt=DT,
        field=galaxy_field,
        softening=SOFTENING,
    )
    model = MultipleShootingForwardModel(
        engine, masses, num_segments, x0, v0, output="segment_ends"
    )
    nodes = jax.jit(model.init_nodes)(TRUTH)
    clean = _phase_space(model({"theta": TRUTH, "nodes": nodes}))
    observed = clean + SIGMA * jax.random.normal(jax.random.PRNGKey(11), clean.shape)
    problem = InferenceProblem(
        model, _phase_space, GaussianLikelihood(sigma=SIGMA), observed
    )
    multipliers = (jnp.zeros((num_segments - 1, N, 6)),)
    theta = {k: v + 1.0e-3 for k, v in TRUTH.items()}

    def single(th):
        params = {"theta": th, "nodes": model.init_nodes(th)}
        return problem.evaluate(params).negative_log_posterior

    def multiple(th):
        params = {"theta": th, "nodes": nodes}
        return augmented_lagrangian_value(problem, params, multipliers, SIGMA**-2)[0]

    def norm(tree):
        return float(jnp.sqrt(sum(jnp.sum(g * g) for g in tree.values())))

    return (
        norm(jax.jit(jax.grad(single))(theta)),
        norm(jax.jit(jax.grad(multiple))(theta)),
    )


def test_single_shot_gradient_explodes_with_the_horizon_multiple_does_not() -> None:
    """One segment: the same gradient. Six: the single shot's is > 1e4 x larger.

    Measured (CPU, jax 0.11.2, jaccpot s10/fused-segment-engine 5385b164, nornax
    s10/shooting-potentials 48479ae, 2026-11-01): one segment 2.805e4 for both;
    six segments (480 steps) single 1.259e11, multiple 4.306e5 -- 2.9e5 apart,
    the multiple-shooting gradient 15x its one-segment value. 19 s for the file.
    """
    s1, m1 = _gradients(1)
    s6, m6 = _gradients(6)
    print(f"|grad| S=1: single {s1:.3e} multiple {m1:.3e}; S=6: {s6:.3e} {m6:.3e}")
    assert s1 == pytest.approx(m1, rel=1e-12)
    assert s6 / m6 > 1.0e4, (s6, m6)
    assert m6 / m1 < 1.0e2, (m6, m1)
