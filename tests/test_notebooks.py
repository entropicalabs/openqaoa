"""
Tests to run the notebooks. They work by converting the notebooks
to a python script via nbconvert and then running the resulting .py file.
"""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

import sys, os
import subprocess
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')


def notebook_test_function(name):

    env_name = str(subprocess.run("which python", shell=True,capture_output=True).stdout)
    env_name = env_name.split('env')[1]
    env_name = env_name.split(os.sep)[1][:-3]
    
    with open(name, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name=env_name)

    ep.preprocess(nb)


# @pytest.mark.notebook
def test_1_workflows_example():
    notebook_test_function("./examples/1_workflows_example.ipynb")

# @pytest.mark.notebook
def test_2_simulators_comparison():
    notebook_test_function("./examples/2_simulators_comparison.ipynb")

# @pytest.mark.notebook
# def test_3_qaoa_on_qpus():
#     notebook_test_function("./examples/3_qaoa_on_qpus.ipynb")

# @pytest.mark.notebook
def test_4_qaoa_variational_parameters():
    notebook_test_function("./examples/4_qaoa_variational_parameters.ipynb")

# @pytest.mark.notebook
def test_5_advanced_parameterization():
    notebook_test_function("./examples/5_advanced_parameterization.ipynb")


# @pytest.mark.notebook
def test_6_fast_qaoa_simulator():
    notebook_test_function("./examples/6_fast_qaoa_simulator.ipynb")

# @pytest.mark.notebook
def test_7_cost_landscapes_w_manual_mode():
    notebook_test_function("./examples/7_cost_landscapes_w_manual_mode.ipynb")

# @pytest.mark.notebook
def test_8_results_example():
    notebook_test_function("./examples/8_results_example.ipynb")

# @pytest.mark.notebook
def test_9_RQAOA_example():
    notebook_test_function("./examples/9_RQAOA_example.ipynb")

