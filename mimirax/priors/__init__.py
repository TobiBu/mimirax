"""Priors and regularizers, and their registry.

A prior is a log-density over parameters; a regularizer is a prior that says
so. See :class:`mimirax.protocols.Prior`. Register new ones with
``PRIORS.register(name, cls)``.
"""

from mimirax._registry import Registry
from mimirax.priors.reference import EntropyPrior, GaussianPrior, L2Regularizer

__all__ = ["PRIORS", "EntropyPrior", "GaussianPrior", "L2Regularizer"]

PRIORS: Registry[type] = Registry("prior")
PRIORS.register("gaussian", GaussianPrior)
PRIORS.register("l2", L2Regularizer)
PRIORS.register("entropy", EntropyPrior)
