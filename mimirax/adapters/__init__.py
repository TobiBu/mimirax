"""Solver glue, behind optional extras. Nothing here is imported eagerly.

``import mimirax`` must succeed with no solver installed, so this package
exposes no names at import time. Import an adapter module explicitly --
``from mimirax.adapters.jaccpot import JaccpotForceModel`` -- and it imports its
solver lazily, raising an ``ImportError`` that names the extra to install.

The dependency direction is the ecosystem's rule: an adapter imports the
solver; the solver never imports mimirax; the mimirax core imports neither.
"""

__all__: list[str] = []
