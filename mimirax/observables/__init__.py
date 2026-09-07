"""Observation operators and their registry.

An observable maps a forward model's state to data space; see
:class:`mimirax.protocols.Observable`. New observables are registered here with
``OBSERVABLES.register(name, cls)`` -- no core module changes.
"""

from mimirax._registry import Registry
from mimirax.observables.m2m import (
    GaussianRadialBins,
    TimeAverage,
    WeightedKernelSum,
)
from mimirax.observables.reference import IdentityObservable, ProjectedPositions
from mimirax.protocols import Observable

__all__ = [
    "OBSERVABLES",
    "GaussianRadialBins",
    "IdentityObservable",
    "ProjectedPositions",
    "TimeAverage",
    "WeightedKernelSum",
    "make_observable",
]

OBSERVABLES: Registry[type] = Registry("observable")
OBSERVABLES.register("identity", IdentityObservable)
OBSERVABLES.register("projected_positions", ProjectedPositions)
OBSERVABLES.register("weighted_kernel_sum", WeightedKernelSum)
OBSERVABLES.register("time_average", TimeAverage)
OBSERVABLES.register("gaussian_radial_bins", GaussianRadialBins)


def make_observable(name: str, **kwargs: object) -> Observable:
    """Instantiate a registered observable by name.

    Parameters
    ----------
    name : str
        A key of :data:`OBSERVABLES`.
    **kwargs : object
        Constructor arguments for the registered class.

    Returns
    -------
    Observable
        The instance.
    """
    return OBSERVABLES.get(name)(**kwargs)
