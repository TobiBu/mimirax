"""Sphinx configuration for the mimirax documentation."""

from __future__ import annotations

import os
import sys

# Make the package importable for autodoc (repo root is the parent of docs/).
sys.path.insert(0, os.path.abspath(".."))

project = "mimirax"
author = "AstroAI Lab"
copyright = "AstroAI Lab"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# numpy-style docstrings; types come from the (type-checked) signatures.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
# The solver adapters import packages that are not installed where the docs build.
autodoc_mock_imports = ["jaccpot", "nornax", "odisseo"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}
# The docs build offline in CI; missing intersphinx inventories must not fail it.
intersphinx_disabled_reftypes = ["*"]

myst_enable_extensions = ["deflist", "colon_fence"]

html_theme = "furo"
html_title = "mimirax"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

nitpicky = False
