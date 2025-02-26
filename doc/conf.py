"""Sphinx configuration file for the Neural Interactive Proofs documentation."""

import sys
from pathlib import Path
from string import Template

from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment

import markdown

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

sys.path.insert(0, str(Path("..", "scripts").resolve()))

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


def generate_splash_page(app: Sphinx, env: BuildEnvironment) -> list[str]:
    """Generate the splash page from the markdown source."""

    with open(app.config.splash_source_filename, "r", encoding="utf-8") as input_file:
        text = input_file.read()
    content = markdown.markdown(text, extensions=["extra"])

    # Try to extract the title from the markdown; default to a generic title
    for line in text.split("\n"):
        stripped_line = line.strip()
        if stripped_line.startswith("# "):
            title = stripped_line[2:]
            break
    else:
        title = "Neural Interactive Proofs"

    # Load the template and substitute the title and content
    with open(app.config.splash_template_path, "r", encoding="utf-8") as template_file:
        html_template = Template(template_file.read())
    html = html_template.substitute(title=title, content=content)

    # Write the output
    with open(
        Path(app.outdir, app.config.splash_output_filename), "w", encoding="utf-8"
    ) as output_file:
        output_file.write(html)

    return []


def setup(app: Sphinx) -> None:
    """Set up the Sphinx application, registering the splash page generator."""
    app.connect("env-updated", generate_splash_page)

    app.add_config_value(
        "splash_template_path", Path("_templates", "splash", "template.html"), ""
    )
    app.add_config_value("splash_source_filename", "splash.md", "")
    app.add_config_value("splash_output_filename", "splash.html", "")
