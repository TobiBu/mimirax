"""Likelihoods and misfits, and their registry.

A likelihood compares a prediction with an observation and returns a
log-probability; see :class:`mimirax.protocols.Likelihood`. Register new ones
with ``LIKELIHOODS.register(name, cls)``.
"""

from mimirax._registry import Registry
from mimirax.likelihoods.gaussian import GaussianLikelihood, chi_squared

__all__ = ["LIKELIHOODS", "GaussianLikelihood", "chi_squared"]

LIKELIHOODS: Registry[type] = Registry("likelihood")
LIKELIHOODS.register("gaussian", GaussianLikelihood)
