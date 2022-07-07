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
# import mock
 
# MOCK_MODULES = ['matplotlib', 'matplotlib.pyplot', 'numpy', 'scipy', 'networkx']
# for mod_name in MOCK_MODULES:
# 	sys.modules[mod_name] = mock.Mock()

import os
import sys
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath("../../"))
print(sys.path)

from openqaoa.qaoa_parameters.baseparams import shapedArray



# -- Project information -----------------------------------------------------

project = 'OpenQAOA'
copyright = '2022, Entropica Labs'
author = 'Entropica Labs'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.todo', 
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# html_logo = 'Entropica_logo.png'
# html_favicon ='favicon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- More customizations ----------------------------------------------------
# Document __init__ and __call__ functions
# def skip(app, what, name, obj, would_skip, options):
#     if name == "__call__":
#         print("Documenting Call")
#         return False

#     if type(obj) == shapedArray:
#         return True
#     return would_skip


# def setup(app):
#     app.connect(skip)
