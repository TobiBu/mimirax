"""Analytic test doubles: forward models and force models with exact gradients.

Imported explicitly (``import mimirax.testing``), not from the package root, so
the doubles stay out of the user-facing namespace.
"""

from mimirax.testing.doubles import (
    DirectSumGravity,
    LinearForwardModel,
    SoftenedPointMassField,
    make_linear_problem,
)

__all__ = [
    "DirectSumGravity",
    "LinearForwardModel",
    "SoftenedPointMassField",
    "make_linear_problem",
]
