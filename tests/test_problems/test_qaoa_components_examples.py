import sys
import os
import pytest

from test_notebooks import notebook_test_function

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../../")


# @pytest.mark.notebook
def test_04_qaoa_variational_parameters():
    notebook_test_function("./examples/04_qaoa_variational_parameters.ipynb")

# @pytest.mark.notebook
def test_07_cost_landscapes_w_manual_mode():
    notebook_test_function("./examples/07_cost_landscapes_w_manual_mode.ipynb")