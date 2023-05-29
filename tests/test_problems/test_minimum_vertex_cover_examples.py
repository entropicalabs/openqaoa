import sys
import os
import pytest

from test_notebooks import notebook_test_function

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../../")

# @pytest.mark.notebook
def test_05_advanced_parameterization():
    notebook_test_function("./examples/05_advanced_parameterization.ipynb")

def test_09_RQAOA_example():
    notebook_test_function("./examples/09_RQAOA_example.ipynb")

def test_11_Mixer_example():
    notebook_test_function("./examples/11_Mixer_example.ipynb")

def test_13_optimizers():
    notebook_test_function("./examples/13_optimizers.ipynb")