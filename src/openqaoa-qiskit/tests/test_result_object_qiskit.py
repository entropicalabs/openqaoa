import unittest
import networkx as nx

from openqaoa import QAOA
from openqaoa.problems import MinimumVertexCover
from openqaoa.backends import create_device


class TestingLoggerClass(unittest.TestCase):
    def test_plot_probabilities(self):
        # Create the problem
        g = nx.circulant_graph(6, [1])
        vc = MinimumVertexCover(g, field=1.0, penalty=10).qubo

        # shot based simulator:
        q_shot = QAOA()
        q_shot_dev = create_device(location="local", name="qiskit.shot_simulator")
        q_shot.set_device(q_shot_dev)

        q_shot.compile(vc)
        q_shot.optimize()

        q_shot.result.plot_probabilities()
