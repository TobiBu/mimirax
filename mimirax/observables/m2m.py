"""The made-to-measure observable, a reference kernel for it, and time averaging.

Made-to-measure is a statement about one shape of observable. Every quantity
the method fits is *linear in the particle weights*,

.. math:: y_j = \\sum_i w_i K_j(z_i),

with ``z_i`` particle ``i``'s phase-space coordinates and ``K_j`` the kernel of
observable ``j`` -- a binned density, a mass-weighted velocity moment, a
surface-brightness pixel. :class:`WeightedKernelSum` is that form, with the
kernel supplied by the caller, and it is the seam the *classic* algorithm needs:
Syer & Tremaine's force of change is driven by ``K_j(z_i)`` per particle, not by
``y_j`` alone, so the kernel has to be reachable and not buried inside a
closure. :meth:`WeightedKernelSum.kernel_values` is that access.

:class:`TimeAverage` is the second half of the method. A made-to-measure fit
compares the data not with an instantaneous model value but with one averaged
along the orbits, which is what turns a snapshot of ``N`` particles into a
statement about a *stationary* distribution function. It wraps any inner
observable and averages it over the leading axis of a recorded trajectory --
:class:`mimirax.adapters.nornax.NornaxRollout`'s output, or any pytree whose
leaves all carry a time axis.

What is **not** claimed here: that a given number of recorded steps is a long
enough average, or that a given kernel makes the weights identifiable. Both are
measurements, and :mod:`mimirax.diagnostics` is where they are made.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from mimirax._typing import PyTree

__all__ = ["GaussianRadialBins", "TimeAverage", "WeightedKernelSum"]


@dataclass(frozen=True)
class WeightedKernelSum:
    """The made-to-measure observable ``y_j = sum_i w_i K_j(z_i)``.

    Differentiable in the weights **and** in whatever the kernel reads, so a fit
    over the weights alone and a fit that also moves the initial conditions use
    the same observable.

    The state is a mapping: :attr:`weights_key` selects the weight leaf and the
    kernel is handed the whole state, so a kernel is free to use positions,
    velocities or anything else the forward model recorded.

    Attributes
    ----------
    kernel : Callable[[PyTree], Array]
        ``state -> (n, m)``: particle ``i``'s contribution to observable ``j``.
        Must be differentiable in the state's array leaves.
    weights_key : str
        Which leaf of the state holds the ``(n,)`` weights.
    """

    kernel: Callable[[PyTree], Array]
    weights_key: str = "weights"

    def kernel_values(self, state: PyTree) -> Array:
        """Return the raw ``(n, m)`` kernel, unweighted.

        The classic force-of-change update needs this, and so does any
        hand-written check of the residual gradient
        ``sum_j Delta_j K_ji / sigma_j^2`` against the one autodiff produces.

        Parameters
        ----------
        state : PyTree
            A state the kernel accepts.

        Returns
        -------
        Array
            ``(n, m)`` kernel values.
        """
        return jnp.asarray(self.kernel(state))

    def __call__(self, state: PyTree) -> Array:
        """Contract the weights with the kernel.

        Parameters
        ----------
        state : PyTree
            A mapping carrying :attr:`weights_key` and whatever the kernel reads.

        Returns
        -------
        Array
            ``(m,)`` predicted observables.
        """
        weights = jnp.asarray(state[self.weights_key])
        return weights @ self.kernel_values(state)


@dataclass(frozen=True)
class GaussianRadialBins:
    """A reference kernel: soft radial bins, and their mass-weighted ``v^2`` moment.

    The doubles' counterpart on the observable side. Real made-to-measure
    kernels -- line-of-sight velocity distributions, IFU pixels, stream tracks --
    belong to the science that uses them (D-027 keeps domain observables in the
    domain packages); this exists so the method has something to run on today,
    the way :class:`~mimirax.observables.ProjectedPositions` does.

    The bins are Gaussian rather than top-hat because a top-hat's derivative
    with respect to a position is zero almost everywhere and a delta on a
    measure-zero set: a fit that also moves the particles would see no gradient
    from it. The Gaussians overlap, which makes the recovered weights genuinely
    degenerate in a controlled, reportable way rather than accidentally
    well-conditioned -- read the degeneracy off
    :func:`mimirax.diagnostics.fisher_information`, do not tune it away.

    Each requested moment contributes ``len(centres)`` columns, in the order
    :attr:`moments` lists them, so the kernel is ``(n, len(centres) *
    len(moments))``.

    Attributes
    ----------
    centres : Array
        ``(b,)`` bin centres in radius.
    width : float
        Gaussian bin width, in the same units as :attr:`centres`.
    moments : tuple[str, ...]
        Which moments to stack: ``"mass"`` for the binned weight itself and
        ``"v2"`` for the binned weight times ``|v|^2`` (a mass-weighted kinetic
        moment, linear in the weights as the method requires -- a *ratio* such
        as a velocity dispersion is not).

    Raises
    ------
    ValueError
        If :attr:`moments` is empty or names a moment that is not supported.
    """

    centres: Array
    width: float
    moments: tuple[str, ...] = ("mass", "v2")

    def __post_init__(self) -> None:
        """Reject an empty or unknown ``moments`` at construction.

        Raises
        ------
        ValueError
            If :attr:`moments` is empty or contains an unsupported name.
        """
        supported = ("mass", "v2")
        if not self.moments:
            raise ValueError(f"moments must name at least one of {supported}")
        unknown = tuple(m for m in self.moments if m not in supported)
        if unknown:
            raise ValueError(f"unsupported moments {unknown}; supported: {supported}")

    def __call__(self, state: PyTree) -> Array:
        """Evaluate the kernel at one state.

        Parameters
        ----------
        state : PyTree
            A mapping with ``"positions"`` ``(n, 3)`` and, when ``"v2"`` is
            among :attr:`moments`, ``"velocities"`` ``(n, 3)``.

        Returns
        -------
        Array
            ``(n, b * len(moments))`` kernel values.
        """
        positions = jnp.asarray(state["positions"])
        radius = jnp.linalg.norm(positions, axis=-1)
        centres = jnp.asarray(self.centres)
        z = (radius[:, None] - centres[None, :]) / self.width
        bins = jnp.exp(-0.5 * z * z)
        columns = []
        for moment in self.moments:
            if moment == "mass":
                columns.append(bins)
            else:
                velocities = jnp.asarray(state["velocities"])
                v2 = jnp.sum(velocities * velocities, axis=-1)
                columns.append(bins * v2[:, None])
        return jnp.concatenate(columns, axis=1)


@dataclass(frozen=True)
class TimeAverage:
    """Average an inner observable over a recorded trajectory's leading axis.

    ``state`` is a pytree whose every array leaf has a leading ``t`` axis -- the
    stacked records a rollout returns. The inner observable is ``vmap``-ed over
    that axis and the results are averaged, so the inner observable is written
    for a single snapshot and never has to know it is being time-averaged.

    Two averages are available. The default is the plain mean, which is the
    right thing when the recorded window is what the fit is meant to average
    over. With :attr:`decay_steps` set, the weights are exponential in the step
    index and normalized to sum to one, ``w_t propto exp(-(t_last - t) /
    decay_steps)``: the discrete counterpart of Syer & Tremaine's temporal
    smoothing ``d y~ / dt = alpha (y - y~)``, with ``alpha dt = 1 /
    decay_steps``. It is in units of *recorded steps*, not of time, so the
    observable stays independent of the integrator's step size; a driver that
    wants a physical timescale converts once, at the call site.

    Neither choice makes a claim about whether the window is long enough for the
    average to represent a stationary state. That is measured -- vary the number
    of recorded steps and watch the fitted weights -- not assumed.

    Attributes
    ----------
    inner : Callable[[PyTree], Array]
        The per-snapshot observable, an :class:`~mimirax.protocols.Observable`.
    decay_steps : float | None
        Exponential-smoothing timescale in recorded steps; ``None`` (the
        default) gives the plain mean.

    Raises
    ------
    ValueError
        If :attr:`decay_steps` is given and is not strictly positive.
    """

    inner: Callable[[PyTree], Array]
    decay_steps: float | None = None

    def __post_init__(self) -> None:
        """Reject a non-positive ``decay_steps`` at construction.

        Raises
        ------
        ValueError
            If :attr:`decay_steps` is not ``None`` and not strictly positive.
        """
        if self.decay_steps is not None and not self.decay_steps > 0.0:
            raise ValueError(f"decay_steps must be > 0; got {self.decay_steps}")

    def step_values(self, state: PyTree) -> Array:
        """Return the inner observable at every recorded step, unaveraged.

        Parameters
        ----------
        state : PyTree
            A pytree whose array leaves all carry a leading ``t`` axis.

        Returns
        -------
        Array
            ``(t, m)`` per-step predictions.
        """
        return jnp.asarray(jax.vmap(self.inner)(state))

    def weights(self, num_steps: int) -> Array:
        """Return the ``(t,)`` averaging weights, which sum to one.

        Parameters
        ----------
        num_steps : int
            How many recorded steps to weight.

        Returns
        -------
        Array
            Uniform ``1 / t`` weights by default, exponential weights ending at
            the last step when :attr:`decay_steps` is set.
        """
        if self.decay_steps is None:
            return jnp.full((num_steps,), 1.0 / num_steps)
        age = jnp.arange(num_steps - 1, -1, -1)
        return jax.nn.softmax(-age / self.decay_steps)

    def __call__(self, state: PyTree) -> Array:
        """Average the inner observable over the recorded steps.

        Parameters
        ----------
        state : PyTree
            A pytree whose array leaves all carry a leading ``t`` axis.

        Returns
        -------
        Array
            ``(m,)`` time-averaged prediction.
        """
        per_step = self.step_values(state)
        weights = self.weights(per_step.shape[0]).astype(per_step.dtype)
        return jnp.tensordot(weights, per_step, axes=(0, 0))
