"""Shared jaxtyping shape aliases for mimirax array annotations.

These aliases document (and, when ``MIMIRAX_RUNTIME_TYPECHECK=1``, enforce) the
array shapes flowing through the inference layer. Axis ``n`` is the particle or
parameter count, ``m`` the number of data points, ``t`` the number of targets
an observable is evaluated at; jaxtyping binds each consistently within a single
call, so a mismatch between, say, ``positions`` and ``masses`` is caught.

The vocabulary is shared with the flake8 ``--builtins`` list in
``.pre-commit-config.yaml``; a new single-identifier axis has to be added there
too, or pyflakes reports it as an undefined name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

from jax import Array
from jaxtyping import Float

__all__ = [
    "DataVector",
    "Matrix",
    "PerParticle",
    "PyTree",
    "Scalar",
    "ScalarLike",
    "SigmaLike",
    "Vec3",
]

# Any JAX pytree of arrays -- parameters, states, predictions. Deliberately
# ``Any``: the whole point of the layer is that the parameter structure is the
# caller's, and every core path works on it through ``jax.tree_util``.
PyTree: TypeAlias = Any

if TYPE_CHECKING:
    # A static checker sees a plain ``Array``: pyright parses jaxtyping's
    # dimension strings as forward references and rejects ``"n 3"`` as a type
    # expression. The shapes are enforced at runtime, not statically, so this
    # loses nothing the checker could have used.
    Vec3: TypeAlias = Array
    PerParticle: TypeAlias = Array
    DataVector: TypeAlias = Array
    Matrix: TypeAlias = Array
    Scalar: TypeAlias = Array
else:
    # Per-particle 3-vector field: positions, velocities, accelerations.
    Vec3 = Float[Array, "n 3"]
    # Per-particle scalar field (e.g. masses).
    PerParticle = Float[Array, "n"]
    # A flat data vector (an observable's output, or an observation).
    DataVector = Float[Array, "m"]
    # A dense linear map from parameters to data.
    Matrix = Float[Array, "m n"]
    # A 0-d array scalar (a log-density, a loss).
    Scalar = Float[Array, ""]

# A scalar accepted at an API boundary: either a 0-d array or a Python float.
ScalarLike: TypeAlias = Scalar | float

# A noise width: one number for every data point, or one per point. The array
# form is what a data vector stacking incommensurate moments needs -- see
# ``mimirax.likelihoods.GaussianLikelihood``. Deliberately not shape-annotated:
# it broadcasts against the data, so a scalar, an ``(m,)`` vector and an
# ``(1,)`` array are all valid and jaxtyping would bind ``m`` from it.
SigmaLike: TypeAlias = Array | float
