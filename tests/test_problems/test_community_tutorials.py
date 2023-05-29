import sys
import os
import pytest

from test_notebooks import notebook_test_function

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../../")

# Community Tutorials
# @pytest.mark.notebook
def test_tutorial_quantum_approximate_optimization_algorithm():
    notebook_test_function(
        "./examples/community_tutorials/01_tutorial_quantum_approximate_optimization_algorithm.ipynb"
    )


# @pytest.mark.notebook
def test_docplex_example():
    notebook_test_function("./examples/community_tutorials/02_docplex_example.ipynb")


# @pytest.mark.notebook
def test_portfolio_optimization():
    notebook_test_function(
        "./examples/community_tutorials/03_portfolio_optimization.ipynb"
    )

# @pytest.mark.notebook
def test_binpacking():
    notebook_test_function("./examples/community_tutorials/04_binpacking.ipynb")
