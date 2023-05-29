import sys
import os
import pytest

from test_notebooks import notebook_test_function

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../../")

# @pytest.mark.notebook
def test_01_workflows_example():
    notebook_test_function("./examples/01_workflows_example.ipynb")

# @pytest.mark.notebook
def test_02_simulators_comparison():
    notebook_test_function("./examples/02_simulators_comparison.ipynb")

# @pytest.mark.notebook
def test_06_fast_qaoa_simulator():
    notebook_test_function("./examples/06_fast_qaoa_simulator.ipynb")

@pytest.mark.qpu
def test_10_workflows_on_Amazon_braket():
    notebook_test_function("./examples/10_workflows_on_Amazon_braket.ipynb")


@pytest.mark.qpu
def test_14_benchmark():
    notebook_test_function("./examples/14_qaoa_benchmark.ipynb")