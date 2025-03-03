"""Sphinx configuration file for the Neural Interactive Proofs documentation."""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Neural Interactive Proofs"
copyright = "2025, Sam Adam-Day and Lewis Hammond"
author = "Sam Adam-Day and Lewis Hammond"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_tabs.tabs",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.autoprogram",
    "numpydoc",
    # "autoapi.extension",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["tabs.css"]


# -- Options for NumpyDoc ----------------------------------------------

numpydoc_show_class_members = False
napoleon_custom_sections = ["Shapes", "Class Attributes"]

# -- Options for autosummary and autodoc ----------------------------------------------

sys.path.insert(0, str(root_dir.joinpath("scripts")))
sys.path.insert(0, str(root_dir))

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "tensordict": ("https://pytorch.org/tensordict/stable", None),
    "torchrl": ("https://pytorch.org/rl/stable", None),
}

# -- Options for sphinxcontrib-bibtex ----------------------------------------------
bibtex_bibfiles = ["references.bib"]
