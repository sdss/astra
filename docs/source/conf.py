# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))


# -- Project information -----------------------------------------------------

project = "astra"
copyright = "2022, Andy Casey"
author = "Andy Casey"

# The full version, including alpha/beta/rc tags
release = "0.3"


extensions = [
    # Sphinx's own extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    # Our custom extension, only meant for Furo's own documentation.
    # External stuff
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_inline_tabs",
]
autosummary_generate = True
autosummary_mock_imports = [
    "astra.database.catalogdb",
    "astra.database.targetdb",
    "astra.database.apogee_drpdb",
    "astra.database.sdss5db",
    "astra.contrib.thecannon",
    "astra.contrib.thecannon_new"
]

add_module_names = False
autodoc_default_flags = ["members"]
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__,__call__",
    'show-inheritance': True,
}
templates_path = ["_templates"]


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


html_theme = "furo"
html_title = "Astra"
language = "en"

html_static_path = ["_static"]
html_favicon = "_static/sdss-v-icon.png"
html_logo = "_static/sdss-v-logo-square.png"
# html_theme_options = {
#    "light_logo": "ads-logo-light-square.png",
#    "dark_logo": "ads-logo-square.png"
# }
html_css_files = ["pied-piper-admonition.css"]
"""
html_theme_options = {
    "announcement": (
        "If things look broken, try disabling your ad-blocker. "
        "It's because 'ads' seems a lot like <b>ad</b>(<i>vertisement</i>)<b>s</b>, "
        "and there's not much I can do about that!"
    )
}
"""
