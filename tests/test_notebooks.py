"""
Tests to run the notebooks. They work by converting the notebooks
to a python script via nbconvert and then running the resulting .py file.
"""

import subprocess
import pytest
import importlib
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError

import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')


def notebook_test_function(name):

    with open(name) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='env')

    ep.preprocess(nb)


# @pytest.mark.notebook
def test_1_Workflows_example():
    notebook_test_function("./examples/Workflows_example.ipynb")


# @pytest.mark.notebook
def test_2_hamiltonian_tutorial():
    notebook_test_function("./examples/hamiltonian_tutorial.ipynb")

# @pytest.mark.notebook
def test_3_openqaoa_example_vectorised():
    notebook_test_function("./examples/openqaoa_example_vectorised.ipynb")

# # @pytest.mark.notebook
# def test_4_openqaoa_example():
#     notebook_test_function("./examples/openqaoa_example.ipynb")

# @pytest.mark.notebook
def test_5_QCS_pyquil_test():
    notebook_test_function("./examples/QCS_pyquil_test.ipynb")


# @pytest.mark.notebook
def test_6_RQAOA_example():
    notebook_test_function("./examples/RQAOA_example.ipynb")


# @pytest.mark.notebook
def test_7_test_backends_correctness():
    notebook_test_function("./examples/test_backends_correctness.ipynb")

# @pytest.mark.notebook
def test_8_testing_decompositions():
    notebook_test_function("./examples/testing_decompositions.ipynb")

# @pytest.mark.notebook
def test_9_testing_variate_params():
    notebook_test_function("./examples/testing_variate_params.ipynb")
