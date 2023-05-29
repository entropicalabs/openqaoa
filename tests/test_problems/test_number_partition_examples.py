import sys
import os
import pytest

from test_notebooks import notebook_test_function

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../../")

@pytest.mark.qpu
def test_03_qaoa_on_qpus():
    notebook_test_function("./examples/03_qaoa_on_qpus.ipynb")