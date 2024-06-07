"""Configuration file for the Sphinx documentation builder."""

#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

PROJECT = "piQture"
COPYRIGHT_NOTICE = "2024, Saasha Joshi"
AUTHOR = "Saasha Joshi"
RELEASE = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# pylint: disable=invalid-name
# html_theme = "alabaster"
html_theme = "furo"
html_static_path = ["_static"]
