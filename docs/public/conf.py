# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sphinx configuration for ovannotators documentation."""

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

sys.path.insert(0, os.path.abspath("../../source"))

project = "ovannotators"
copyright = "2026, NVIDIA CORPORATION & AFFILIATES"
author = "NVIDIA CORPORATION & AFFILIATES"

# Single source of truth: derive the version from the installed package
# metadata (declared in pyproject.toml) rather than hard-coding it here.
try:
    release = _pkg_version("ovannotators")
except PackageNotFoundError:
    release = "0.0.0+unknown"
# Short X.Y version for the Sphinx ``version`` field.
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "nvidia_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "pygments_light_style": "sas",
    "pygments_dark_style": "github-dark",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA-Omniverse/ovannotators",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}

# -- Options for autodoc -----------------------------------------------------
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autosummary_generate = True

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = True
