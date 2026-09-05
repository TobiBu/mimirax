"""Diagnostics of an inference: residuals, convergence traces, degeneracies.

These are diagnostics of the *fit*, not of the simulation -- energy and
momentum conservation belong to the integrator package (``nornax.diagnostics``).
"""

from mimirax.diagnostics.convergence import has_converged, relative_change
from mimirax.diagnostics.degeneracy import (
    degenerate_directions,
    fisher_information,
    profile_likelihood,
)
from mimirax.diagnostics.residuals import normalized_residuals, residuals, rms_residual

__all__ = [
    "degenerate_directions",
    "fisher_information",
    "has_converged",
    "normalized_residuals",
    "profile_likelihood",
    "relative_change",
    "residuals",
    "rms_residual",
]
