import sys
import os
from test_notebooks import notebook_test_function

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../../")

# @pytest.mark.notebook
def test_01_workflows_example():
    notebook_test_function("./examples/01_workflows_example.ipynb")