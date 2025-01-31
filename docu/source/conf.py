# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Asegura acceso al paquete


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pytopicgram'
copyright = '2025, SAIL'
author = 'SAIL'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',      # Documentación automática de docstrings
    'sphinx.ext.napoleon',     # Soporte para Google y NumPy docstrings
    'sphinx.ext.viewcode'     # Agregar enlaces al código fuente
]

autosummary_generate = True  # Permite generar los archivos automáticamente

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Tema de ReadTheDocs
html_theme = 'sphinx_rtd_theme'
#html_theme = 'alabaster'

#html_static_path = ['_static']
