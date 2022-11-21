"""
Tests to run the notebooks. They work by converting the notebooks
to a python script via nbconvert and then running the resulting .py file.
"""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pytest

import sys, os
import subprocess
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')


def notebook_test_function(name):

    
    with open(name, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='env')

    ep.preprocess(nb)


# @pytest.mark.notebook
def test_01_workflows_example():
    notebook_test_function("./examples/01_workflows_example.ipynb")

# @pytest.mark.notebook
def test_02_simulators_comparison():
    notebook_test_function("./examples/02_simulators_comparison.ipynb")

@pytest.mark.qpu
def test_03_qaoa_on_qpus():
    notebook_test_function("./examples/03_qaoa_on_qpus.ipynb")

# @pytest.mark.notebook
def test_04_qaoa_variational_parameters():
    notebook_test_function("./examples/04_qaoa_variational_parameters.ipynb")

# @pytest.mark.notebook
def test_05_advanced_parameterization():
    notebook_test_function("./examples/05_advanced_parameterization.ipynb")

# @pytest.mark.notebook
def test_06_fast_qaoa_simulator():
    notebook_test_function("./examples/06_fast_qaoa_simulator.ipynb")

# @pytest.mark.notebook
def test_07_cost_landscapes_w_manual_mode():
    notebook_test_function("./examples/07_cost_landscapes_w_manual_mode.ipynb")

# @pytest.mark.notebook
def test_08_results_example():
    notebook_test_function("./examples/08_results_example.ipynb")

def test_09_RQAOA_example():
    notebook_test_function("./examples/09_RQAOA_example.ipynb")

@pytest.mark.qpu
def test_10_workflows_on_Amazon_braket():
    notebook_test_function("./examples/10_workflows_on_Amazon_braket.ipynb")
    
def test_11_Mixer_example():
    notebook_test_function("./examples/11_Mixer_example.ipynb")
       
### Community Tutorials

# @pytest.mark.notebook
def test_tutorial_quantum_approximate_optimization_algorithm():
    notebook_test_function("./examples/community_tutorials/01_tutorial_quantum_approximate_optimization_algorithm.ipynb")

# @pytest.mark.notebook
def test_docplex_example():
    notebook_test_function("./examples/community_tutorials/02_docplex_example.ipynb")
    
# @pytest.mark.notebook
def test_portfolio_optimization():
    notebook_test_function("./examples/community_tutorials/03_portfolio_optimization.ipynb")
    
# @pytest.mark.notebook
def test_binpacking():
    notebook_test_function("./examples/community_tutorials/04_binpacking.ipynb")


