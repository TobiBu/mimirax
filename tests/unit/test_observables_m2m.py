"""The made-to-measure observable, its reference kernel, and time averaging."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from mimirax import (
    OBSERVABLES,
    GaussianRadialBins,
    Observable,
    TimeAverage,
    WeightedKernelSum,
)


@pytest.fixture()
def trajectory(key) -> dict[str, jnp.ndarray]:
    """A five-step recorded trajectory of seven particles, weights included."""
    k_pos, k_vel, k_w = jax.random.split(key, 3)
    steps, n = 5, 7
    weights = 0.5 + jax.random.uniform(k_w, (n,))
    return {
        "positions": jax.random.normal(k_pos, (steps, n, 3)),
        "velocities": 0.4 * jax.random.normal(k_vel, (steps, n, 3)),
        "weights": jnp.broadcast_to(weights, (steps, n)),
    }


def test_the_new_observables_are_registered_and_satisfy_the_protocol() -> None:
    """All three are in the registry and structurally Observables."""
    for name in ("weighted_kernel_sum", "time_average", "gaussian_radial_bins"):
        assert name in OBSERVABLES
    kernel = GaussianRadialBins(centres=jnp.asarray([1.0]), width=0.5)
    inner = WeightedKernelSum(kernel)
    assert isinstance(kernel, Observable)
    assert isinstance(inner, Observable)
    assert isinstance(TimeAverage(inner), Observable)


def test_weighted_kernel_sum_is_the_weights_contracted_with_the_kernel() -> None:
    """y_j = sum_i w_i K_ji, and kernel_values exposes K unweighted."""
    kernel_matrix = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    observable = WeightedKernelSum(lambda state: kernel_matrix)
    state = {"weights": jnp.asarray([1.0, 0.5, 2.0])}
    assert jnp.allclose(observable.kernel_values(state), kernel_matrix)
    assert jnp.allclose(observable(state), jnp.asarray([12.5, 16.0]))


def test_weighted_kernel_sum_is_linear_in_the_weights() -> None:
    """Its Jacobian in the weights is exactly the kernel, which is what M2M assumes."""
    kernel_matrix = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    observable = WeightedKernelSum(lambda state: kernel_matrix)
    jacobian = jax.jacobian(lambda w: observable({"weights": w}))(jnp.ones(3))
    assert jnp.allclose(jacobian, kernel_matrix.T, atol=1e-14)


def test_gaussian_radial_bins_stacks_its_moments_in_order() -> None:
    """Columns are bins-per-moment, in the order the moments are named."""
    centres = jnp.asarray([0.5, 1.0, 2.0])
    state = {
        "positions": jnp.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        "velocities": jnp.asarray([[0.0, 3.0, 0.0], [1.0, 0.0, 0.0]]),
    }
    kernel = GaussianRadialBins(centres=centres, width=0.4)
    values = kernel(state)
    assert values.shape == (2, 6)
    mass, v2 = values[:, :3], values[:, 3:]
    # The v2 block is the mass block scaled by each particle's own |v|^2.
    assert jnp.allclose(v2, mass * jnp.asarray([9.0, 1.0])[:, None], atol=1e-14)
    # The mass block is the Gaussian of radius minus centre; particle 0 sits on
    # the middle bin's centre, so that entry is exactly 1.
    assert float(mass[0, 1]) == pytest.approx(1.0)
    assert float(mass[1, 2]) == pytest.approx(1.0)


def test_gaussian_radial_bins_rejects_unknown_moments() -> None:
    """A misspelled moment fails at construction, listing what is supported."""
    with pytest.raises(ValueError, match="unsupported moments"):
        GaussianRadialBins(centres=jnp.ones(1), width=0.5, moments=("sigma",))
    with pytest.raises(ValueError, match="at least one"):
        GaussianRadialBins(centres=jnp.ones(1), width=0.5, moments=())


def test_time_average_is_the_mean_of_the_per_step_values(trajectory) -> None:
    """The default average is the plain mean over the recorded axis."""
    kernel = GaussianRadialBins(centres=jnp.asarray([0.5, 1.0, 1.5]), width=0.4)
    observable = TimeAverage(WeightedKernelSum(kernel))
    per_step = observable.step_values(trajectory)
    assert per_step.shape == (5, 6)
    assert jnp.allclose(observable(trajectory), jnp.mean(per_step, axis=0), atol=1e-14)


def test_time_average_weights_are_normalized_and_end_weighted(trajectory) -> None:
    """Exponential smoothing keeps sum(w) = 1 and puts the most weight last."""
    kernel = GaussianRadialBins(centres=jnp.asarray([1.0]), width=0.4)
    smoothed = TimeAverage(WeightedKernelSum(kernel), decay_steps=2.0)
    weights = smoothed.weights(5)
    assert float(jnp.sum(weights)) == pytest.approx(1.0)
    assert jnp.all(jnp.diff(weights) > 0.0)
    # exp(-age / decay) up to the shared normalization.
    ratio = weights[1:] / weights[:-1]
    assert jnp.allclose(ratio, jnp.exp(1.0 / 2.0), atol=1e-12)
    assert jnp.allclose(
        smoothed(trajectory),
        jnp.tensordot(weights, smoothed.step_values(trajectory), axes=(0, 0)),
        atol=1e-14,
    )


def test_time_average_recovers_the_last_step_as_the_decay_vanishes(trajectory) -> None:
    """A very short smoothing time is the instantaneous value, as it must be."""
    kernel = GaussianRadialBins(centres=jnp.asarray([0.8, 1.6]), width=0.5)
    inner = WeightedKernelSum(kernel)
    smoothed = TimeAverage(inner, decay_steps=1.0e-3)
    last = {leaf: value[-1] for leaf, value in trajectory.items()}
    assert jnp.allclose(smoothed(trajectory), inner(last), atol=1e-12)


def test_time_average_rejects_a_non_positive_decay() -> None:
    """A zero or negative smoothing time is a construction error, not a silent mean."""
    inner = WeightedKernelSum(lambda state: jnp.ones((1, 1)))
    with pytest.raises(ValueError, match="decay_steps must be > 0"):
        TimeAverage(inner, decay_steps=0.0)


def test_time_average_gradient_in_the_weights_is_the_averaged_kernel(
    trajectory,
) -> None:
    """d y_bar / d w is the time-averaged kernel, exactly, in closed form.

    The whole point of stacking the weights along the time axis: the gradient
    that reaches a single ``(n,)`` weight vector is the sum over the recorded
    steps of each step's kernel, times the averaging weight. Checked against
    the hand-written expression rather than a finite difference, since the map
    is linear and the exact answer is available.
    """
    kernel = GaussianRadialBins(centres=jnp.asarray([0.5, 1.2]), width=0.45)
    observable = TimeAverage(WeightedKernelSum(kernel))
    steps, n = trajectory["positions"].shape[:2]
    weights = trajectory["weights"][0]

    def prediction(w):
        state = dict(trajectory)
        state["weights"] = jnp.broadcast_to(w, (steps, n))
        return observable(state)

    got = jax.jacobian(prediction)(weights)
    per_step = jnp.stack(
        [
            kernel({leaf: value[t] for leaf, value in trajectory.items()})
            for t in range(steps)
        ]
    )
    want = jnp.mean(per_step, axis=0).T
    assert jnp.allclose(got, want, atol=1e-14)
