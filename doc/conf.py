"""Sphinx configuration file for the Neural Interactive Proofs documentation."""

import sys
from pathlib import Path
from string import Template
from typing import Literal, Any
import importlib
import inspect

from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment

import markdown

import pypandoc

_root_path = Path(__file__).parent.parent
_docs_path = Path(__file__).parent

_templates_dir_name = "_templates"
_templates_path = _docs_path / _templates_dir_name

_footer_links_path = _templates_path / "theme" / "footer-links.html"
_mermaid_init_js_path = _templates_path / "mermaid" / "init.js"
_mermaid_theme_css_path = _templates_path / "mermaid" / "theme.css"

_github_tree_url = "https://github.com/SamAdamDay/neural-interactive-proofs/tree/main"

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Neural Interactive Proofs"
copyright = "2025, Sam Adam-Day and Lewis Hammond"
author = "Sam Adam-Day and Lewis Hammond"

# -- Path setup ----------------------------------------------------------------

sys.path.insert(0, str(_root_path / "scripts"))
sys.path.insert(0, str(_root_path))
sys.path.insert(0, str(_root_path / "doc" / "extensions"))

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
    "sphinxcontrib.mermaid",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.linkcode",
]

templates_path = [_templates_dir_name]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

root_doc = "docs/index"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

with open(_footer_links_path, "r") as footer_links_file:
    _footer_links = footer_links_file.read()

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["tabs.css", "extra.css"]
html_theme_options = {"extra_footer": _footer_links}
html_favicon = "_static/favicon.ico"


# -- Options for Napoleon ----------------------------------------------

napoleon_custom_sections = ["Shapes"]
napoleon_use_rtype = False


# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "torchvision": ("https://pytorch.org/vision/stable", None),
    "tensordict": ("https://pytorch.org/tensordict/stable", None),
    "torchrl": ("https://pytorch.org/rl/stable", None),
}

# -- Options for sphinxcontrib-bibtex ----------------------------------------------
bibtex_bibfiles = ["references.bib"]


# -- Options for sphinxcontrib-mermaid ----------------------------------------------

mermaid_version = "11.2.0"
mermaid_include_elk = "0.1.4"

with open(_mermaid_init_js_path, "r") as mermaid_init_js_file:
    _mermaid_init_js_template = Template(mermaid_init_js_file.read())
    mermaid_init_js = _mermaid_init_js_template.substitute(
        mermaid_version=mermaid_version,
        elk_version=mermaid_include_elk,
    )


def generate_splash_page(app: Sphinx, env: BuildEnvironment) -> list[str]:
    """Generate the splash page from the markdown source."""

    source_path = _docs_path.joinpath(app.config.splash_source_path).resolve()
    with open(source_path, "r", encoding="utf-8") as input_file:
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
    template_path = _docs_path.joinpath(app.config.splash_template_path).resolve()
    with open(template_path, "r", encoding="utf-8") as template_file:
        html_template = Template(template_file.read())
    html = html_template.substitute(title=title, content=content)

    # Write the output
    output_path = Path(app.outdir, app.config.splash_output_path).resolve()
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(html)

    return []


def generate_changelog(app: Sphinx, env: BuildEnvironment) -> list[str]:
    """Convert the CHANGELOG.md file to a RST file."""

    # Convert the markdown to RST
    source_path = _root_path.joinpath(app.config.changelog_path).resolve()
    changelog_rst = pypandoc.convert_file(str(source_path), "rst", format="md")

    # Write the output if it has changed
    output_path = _docs_path.joinpath(app.config.changelog_output_path).resolve()
    try:
        with open(output_path, "r", encoding="utf-8") as output_file:
            current_changelog_rst = output_file.read()
    except FileNotFoundError:
        current_changelog_rst = ""
    if changelog_rst != current_changelog_rst:
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(changelog_rst)

    return []


def skip(
    app: Sphinx,
    what: Literal["module", "class", "exception", "function", "method", "attribute"],
    name: str,
    obj: Any,
    would_skip: bool,
    options: dict[str, bool],
) -> bool:
    """Skip certain methods and attributes from the documentation."""

    # We only want to include methods that are part of the nip library
    if what == "method":
        return (
            would_skip
            or not hasattr(obj, "__module__")
            or obj.__module__ is None
            or not obj.__module__.startswith("nip")
        )

    # We exclude all private attributes. Ideally we'd like to exclude attributes not
    # defined in the the nip library, but this is not possible with the current
    # implementation of autodoc-skip-member
    elif what == "attribute":
        return would_skip or name.startswith("_")

    return would_skip


def linkcode_resolve(
    domain: str, info: dict[Literal["module", "fullname"], str]
) -> str | None:
    """Resolve a link to the source code for a Python object.

    Parameters
    ----------
    domain : str
        The domain of the object. This should be "py", because we only document Python
        objects.
    info : dict[Literal["module", "fullname"], str]
        Information about the object to link to. This should contain the module name and
        the full name of the object.

    Returns
    -------
    link : str or None
        The URL to the source code for the object, or None if the source code cannot be
        found.

    Note
    ----
    This function is based on the responses to `this GitHub issue
    <https://github.com/readthedocs/sphinx-autoapi/issues/202>`_.
    """

    if domain != "py":
        raise NotImplementedError(
            f"Source code linking for domain {domain!r} not implemented"
        )

    module_name = info["module"]
    object_name = info["fullname"]

    if module_name == "":
        return None

    # Keep transferring the last part of the module name to the object name and trying
    # to import the module until we either successfully import the module or run out of
    # parts of the module name. This allows us to handle classes
    while True:
        try:
            obj = importlib.import_module(module_name)
        except ImportError:
            if "." in module_name:
                module_name, _, object_name_prefix = module_name.rpartition(".")
                object_name = f"{object_name_prefix}.{object_name}"
            else:
                return None
        else:
            break

    # Walk through the object name, resolving each part in turn by getting the attribute
    # from the previous object
    while object_name != "":
        object_name_prefix, _, object_name = object_name.partition(".")
        try:
            obj = getattr(obj, object_name_prefix)
        except AttributeError:
            return None

    # Unwrap any decorators before getting the source code
    obj = inspect.unwrap(obj)

    # Get the relative path to the source file
    try:
        absolute_file_path = inspect.getsourcefile(obj)
    except TypeError:
        return None
    if absolute_file_path is None or absolute_file_path == "<string>":
        return None
    relative_file_path = Path(absolute_file_path).relative_to(_root_path)

    lines, starting_line_number = inspect.getsourcelines(obj)
    end_line_number = starting_line_number + len(lines) - 1

    return (
        f"{_github_tree_url}/{relative_file_path}"
        f"#L{starting_line_number}-L{end_line_number}"
    )


def setup(app: Sphinx) -> None:
    """Set up the Sphinx application."""

    # Add the splash page generation to the build process
    app.add_config_value("splash_template_path", "_templates/splash/template.html", "")
    app.add_config_value("splash_source_path", "index.md", "")
    app.add_config_value("splash_output_path", "index.html", "")
    app.connect("env-updated", generate_splash_page)

    # Add the CHANGELOG.rst generation to the build process
    app.add_config_value("changelog_path", "CHANGELOG.md", "html")
    app.add_config_value("changelog_output_path", "docs/changelog.rst", "html")
    app.connect("config-inited", generate_changelog)

    # Add the skip function to the autodoc-skip-member event
    app.connect("autodoc-skip-member", skip)
