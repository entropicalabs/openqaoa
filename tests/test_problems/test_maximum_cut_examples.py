import sys
import os
import test_notebooks

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../../")

# @pytest.mark.notebook
def test_01_workflows_example():
    notebook_test_function("./examples/01_workflows_example.ipynb")