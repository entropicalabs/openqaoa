import warnings
import unittest

import networkx as nx
from openqaoa.workflows.optimizer import QAOA
from openqaoa.devices import create_device
from openqaoa.problems.problem import MaximumCut
from openqaoa.optimizers.training_vqa import CustomScipyGradientOptimizer


#create a problem
nodes = 4
edge_probability = 0.6
g = nx.generators.fast_gnp_random_graph(n=nodes,p=edge_probability)
maxcut_prob = MaximumCut(g)
maxcut_qubo = maxcut_prob.get_qubo_problem()


class TestPennylaneOptimizers(unittest.TestCase):

    def _run_method(self, method):
        " function tu run the test for any method "
        q = QAOA()
        device = create_device(location='local', name='qiskit.statevector_simulator')
        q.set_device(device)


        q.set_circuit_properties(p=2, param_type='standard', init_type='rand', mixer_hamiltonian='x')
        q.set_backend_properties(prepend_state=None, append_state=None)
        q.set_classical_optimizer(method=method, maxiter=4, optimizer_options = {'blocking':False, 'resamplings': 0},
                                optimization_progress=True, cost_progress=True, parameter_log=True, jac='finite_difference')

        q.compile(maxcut_qubo) 
        q.optimize()

    def test_pennylane_optimizers(self):
        " function to run the tests for pennylane optimizers "
        list_optimizers = CustomScipyGradientOptimizer.CUSTOM_GRADIENT_OPTIMIZERS

        for opt in list_optimizers:
            if opt.split()[0] == "pennylane":
                self._run_method(opt)




if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=PendingDeprecationWarning)
        unittest.main()
