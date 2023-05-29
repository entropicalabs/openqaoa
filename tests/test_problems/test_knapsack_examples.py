import sys
import os
import pytest

from test_notebooks import notebook_test_function

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../../")

# @pytest.mark.notebook
def test_08_results_example():
    notebook_test_function("./examples/08_results_example.ipynb")

def test_X_dumping_data():
    notebook_test_function("./examples/X_dumping_data.ipynb")