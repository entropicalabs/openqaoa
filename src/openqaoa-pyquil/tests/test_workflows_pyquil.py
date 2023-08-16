import networkx as nw
import unittest

from openqaoa import QAOA
from openqaoa.backends import create_device
from openqaoa.problems import MinimumVertexCover
from openqaoa_pyquil.backends import DevicePyquil
from openqaoa_pyquil.backends import QAOAPyQuilWavefunctionSimulatorBackend

class TestingVanillaQAOA(unittest.TestCase):

    def test_set_backend_properties_check_backend_pyquil_statevector(self):

        """
        Check if the backend returned by set_backend_properties is correct
        Based on the input device. For pyquil statevector simulator.
        Also Checks if defaults from workflows are used in the backend.
        """

        nodes = 6
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes, p=edge_probability)
        problem = MinimumVertexCover(g, field=1.0, penalty=10)
        qubo_problem = problem.qubo

        q = QAOA()
        q.set_device(
            create_device(location="local", name="pyquil.statevector_simulator")
        )
        q.compile(problem=qubo_problem)

        self.assertEqual(type(q.backend), QAOAPyQuilWavefunctionSimulatorBackend)

        self.assertEqual(q.backend.init_hadamard, True)
        self.assertEqual(q.backend.prepend_state, None)
        self.assertEqual(q.backend.append_state, None)
        self.assertEqual(q.backend.cvar_alpha, 1)

        self.assertRaises(AttributeError, lambda: q.backend.n_shots)
        

