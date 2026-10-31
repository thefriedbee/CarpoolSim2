"""Sphinx configuration for the carpoolsim documentation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "CarpoolSim2"
author = "carpoolsim"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]
autosummary_generate = True
autosummary_imported_members = True
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

html_theme = "pydata_sphinx_theme"
html_title = "CarpoolSim2"
html_theme_options = {
    "show_nav_level": 2,
    "navigation_depth": 3,
    "secondary_sidebar_items": ["page-toc"],
}
# html_sidebars = {"**": []}
myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence"]
