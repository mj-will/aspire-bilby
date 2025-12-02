"""Sphinx configuration for the aspire-bilby documentation."""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath("..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import aspire_bilby  # noqa: E402

project = "aspire-bilby"
copyright = "2025, Michael J. Williams"
author = "Michael J. Williams"
version = aspire_bilby.__version__
release = aspire_bilby.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "autoapi.extension",
]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_preprocess_types = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autoapi_type = "python"
autoapi_dirs = ["../src/aspire_bilby/"]
autoapi_add_toctree_entry = True
autoapi_options = [
    "members",
    "imported-members",
    "show-inheritance",
    "show-module-summary",
    "undoc-members",
]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = "aspire-bilby"
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/mj-will/aspire-bilby",
    "repository_branch": "main",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
