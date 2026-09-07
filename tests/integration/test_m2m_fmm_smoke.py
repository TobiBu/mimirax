"""One made-to-measure gradient step through jaccpot's FMM. A smoke test.

``mimirax[nornax]`` **and** jaccpot. What this file establishes is that the
pieces connect: jaccpot's ``BlockStepFMM`` satisfies nornax's
``MutualForceModel``, nornax's rollout carries and rebuilds its topology inside
the scan (the EDDA programme's B4), and a made-to-measure objective built on top
of that differentiates to a finite gradient which moves the weights downhill.

**No number from this file goes anywhere near a paper.** It runs at N = 256 on
CPU with a tiny leaf size, takes one optimizer step, and asserts finiteness and
a direction. Whether made-to-measure converges at FMM scale, over how long a
time average, at what rebuild cadence, is Jaccpot-Dynamics I's experiment; a
smoke test that pretended to answer it would be worse than no test.

The system is the two-clump configuration jaccpot's own cross-repo tests use,
and for their reason: a single Gaussian blob at this N has *no far pairs*, so
the FMM degenerates to a direct sum and every claim about a tree is vacuous.
``num_far_pairs > 0`` is asserted before anything else is believed.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from mimirax import (
    EntropyPrior,
    GaussianLikelihood,
    GaussianRadialBins,
    InferenceProblem,
    MadeToMeasure,
    TimeAverage,
    WeightedKernelSum,
)
from mimirax.adapters.nornax import NornaxRollout

pytest.importorskip("nornax", reason="mimirax[nornax] is not installed")
pytest.importorskip("jaccpot", reason="jaccpot is not installed")

from jaccpot.nornax_adapter import BlockStepFMM  # noqa: E402

SOFTENING = 1.0e-2
SIGMA = 0.05
MU = 1.0e-3


def _two_clumps(n=256, seed=25, separation=8.0, sigma=0.6):
    """Two well-separated Gaussian clumps, so the mutual MAC accepts far pairs.

    Copied from jaccpot's ``tests/integration/test_mutual_fmm_nornax.py`` on
    purpose: a system that makes the far field non-empty at ``theta = 0.6`` is a
    property of the configuration, and reusing the one jaccpot validated is
    cheaper and more honest than inventing another. Masses are unequal because
    equal masses make a target-centric gather accidentally antisymmetric.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    centres = ([-separation / 2.0, 0.0, 0.0], [separation / 2.0, 0.0, 0.0])
    positions = np.concatenate([rng.normal(c, sigma, (half, 3)) for c in centres])
    velocities = rng.normal(0.0, 0.05, (n, 3))
    masses = rng.uniform(0.5, 1.5, n)
    return (
        jnp.asarray(positions, dtype=jnp.float64),
        jnp.asarray(velocities, dtype=jnp.float64),
        jnp.asarray(masses, dtype=jnp.float64),
    )


def test_one_made_to_measure_step_through_the_fmm_is_finite() -> None:
    """Item 6: case (a) on ``BlockStepFMM``, one gradient step, finite. A smoke.

    ``topology_backend="device"`` is what makes ``rebuild_state`` traceable, so
    the rollout rebuilds the tree inside its own scan at every base step
    (``rebuild_every=1``) rather than on the host. The topology is severed from
    the gradient there, so what autodiff returns is the exact fixed-topology
    gradient on each one-step segment -- which is the substantive content of B4
    and the reason this composition is legitimate at all.

    Asserted: a non-empty far field, a finite gradient with a non-zero norm, a
    finite objective, positive weights after one step, and that the step lowers
    the objective. Nothing about magnitude, accuracy or convergence.

    The one step is the **classic** multiplicative update at ``epsilon = 1e-9``,
    not an Adam step, and the reason is worth recording. Measured here:
    ``F = 7.623e+03`` with ``|dF/dw| = 6.47e+04`` at the true weights against
    data offset by 2 %, and the classic step then gives ``dF = -3.98`` (and
    -39.6, -376, -1517 at ``epsilon`` of 1e-8, 1e-7, 1e-6). A single *Adam* step
    goes **uphill** at any learning rate tried -- Adam's first update is
    scale-free, so it moves every log-weight by the full learning rate whatever
    the local curvature, and this objective's curvature will not take that. So
    will plain SGD in log-weights, at ``dF/du = w dF/dw`` of order 6e4. None of
    that is a statement about the FMM gradient, which is what this test checks;
    it is a statement about step sizes on a stiff objective, and it is why the
    descent here is taken along the gradient with a step small enough to mean it.
    """
    k_max, dt = 1, 2.0e-3
    positions, velocities, masses = _two_clumps()
    fmm = BlockStepFMM(
        softening=SOFTENING,
        k_max=k_max,
        theta=0.6,
        max_order=4,
        leaf_size=16,
        topology_backend="device",
    )
    fmm.prepare(positions, masses)
    assert (
        int(fmm.state.num_far_pairs) > 0
    ), "no far pairs: the FMM is a direct sum here"

    rollout = NornaxRollout.self_consistent(
        fmm,
        dt=dt,
        num_steps=2,
        k_max=k_max,
        rebuild_fn=fmm.rebuild_state,
        rebuild_every=1,
    )
    observable = TimeAverage(
        WeightedKernelSum(
            GaussianRadialBins(centres=jnp.asarray([3.0, 4.0, 5.0]), width=1.0)
        )
    )
    params = {"positions": positions, "velocities": velocities, "weights": masses}
    observed = observable(rollout(params)) * 1.02
    problem = InferenceProblem(
        forward=rollout,
        observable=observable,
        likelihood=GaussianLikelihood(sigma=SIGMA),
        observed=observed,
        priors=(EntropyPrior(mu=MU, reference=masses),),
    )
    objective = problem.negative_log_posterior

    stepped = MadeToMeasure(num_steps=1, epsilon=1.0e-9).force_of_change(
        objective, params
    )
    before = float(stepped.objective_trace[0])
    after = float(objective(stepped.params))
    weights = stepped.params["weights"]
    assert np.isfinite(before), before
    assert np.isfinite(after), after
    assert jnp.all(jnp.isfinite(weights))
    assert jnp.all(weights > 0.0)
    assert not jnp.array_equal(weights, masses), "the step did not move the weights"
    assert after < before, f"F went {before:.6e} -> {after:.6e}"
