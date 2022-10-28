import warnings
import unittest

import networkx as nx
from openqaoa.workflows.optimizer import QAOA
from openqaoa.devices import create_device
from openqaoa.problems.problem import MinimumVertexCover
from openqaoa.optimizers.training_vqa import CustomScipyGradientOptimizer


#create a problem
g = nx.circulant_graph(4, [1])
problem = MinimumVertexCover(g, field =1.0, penalty=10)
qubo_problem = problem.get_qubo_problem()


class TestPennylaneOptimizers(unittest.TestCase):

    def _run_method(self, method):
        " function tu run the test for any method "
        q = QAOA()
        q.set_classical_optimizer(method=method, maxiter=3, jac='finite_difference')
        q.compile(qubo_problem) 
        q.optimize()

        assert len(q.results.most_probable_states['solutions_bitstrings'][0]) > 0

    def test_pennylane_optimizers(self):
        " function to run the tests for pennylane optimizers "
        list_optimizers = CustomScipyGradientOptimizer.CUSTOM_GRADIENT_OPTIMIZERS

        i = 0
        for opt in list_optimizers:
            if opt.split('_')[0] == "pennylane":
                self._run_method(opt)
                i += 1

        assert i == sum([1 for opt in list_optimizers if  "pennylane" in opt])




if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=PendingDeprecationWarning)
        unittest.main()
