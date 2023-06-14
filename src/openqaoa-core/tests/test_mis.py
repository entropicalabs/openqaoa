import unittest
import networkx as nx
import numpy as np
from openqaoa.problems import MIS


def terms_list_equality(terms_list1, terms_list2):
    """
    Check the terms equality between two terms list
    where the order of edges do not matter.
    """
    if len(terms_list1) != len(terms_list2):
        bool = False
    else:
        for term1, term2 in zip(terms_list1, terms_list2):
            bool = True if (term1 == term2 or term1 == term2[::-1]) else False

    return bool


def terms_list_isclose(terms_list1, terms_list2):
    """
    Check if the distance between two terms list
    where the order of edges do not matter.
    """
    if len(terms_list1) != len(terms_list2):
        bool = False
    else:
        for term1, term2 in zip(terms_list1, terms_list2):
            bool = (
                True
                if np.isclose(term1, term2) or np.isclose(term1, term2[::-1])
                else False
            )

    return bool

class TestMIS(unittest.TestCase):
    """Tests for the MIS class"""
    def test_mis_terms_weights_constant(self):
        """Test that MIS creates a correct QUBO from the provided graph"""

        gr = nx.generators.random_graphs.fast_gnp_random_graph(n=5, p=0.8, seed=1234)
        gr_edges = [
            [0, 3],
            [0, 4],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 4],
            [0],
            [1],
            [2],
            [3],
            [4],
        ]
        gr_weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -1.0, -1.0]

        mis_prob_qubo = MIS(gr).qubo

        self.assertTrue(terms_list_equality(gr_edges, mis_prob_qubo.terms))
        self.assertEqual(gr_weights, mis_prob_qubo.weights)
        self.assertEqual(0.5, mis_prob_qubo.constant)

    def test_mis_random_problem(self):
        """Test MIS random instance method"""

        seed = 1234
        gr = nx.generators.random_graphs.fast_gnp_random_graph(n=10, p=0.8, seed=seed)
        mis_manual_prob = MIS(gr).qubo

        np.random.seed(1234)
        mis_random_prob = MIS.random_instance(
            n_nodes=10, edge_probability=0.8, seed=seed
        ).qubo

        self.assertTrue(
            terms_list_equality(mis_manual_prob.terms, mis_random_prob.terms)
        )
        self.assertEqual(mis_manual_prob.weights, mis_random_prob.weights)
        self.assertEqual(mis_manual_prob.constant, mis_random_prob.constant)

    def test_mis_type_checking(self):
        """
        Checks if the type-checking returns the right error.
        """

        # graph type-check
        graph_list = [(1, 2), {"node1": 1, "node2": 2}, np.array([1, 2])]

        for each_graph in graph_list:
            with self.assertRaises(TypeError) as e:
                MIS(G=each_graph)
            self.assertEqual(
                "Input problem graph must be a networkx Graph.", str(e.exception)
            )

    def test_mis_classical_sol(self):
        """Test the maximal independent set random instance method classical solution"""

        seed = 1234
        np.random.seed(seed)
        mis_sol = MIS.random_instance(
            n_nodes=10, edge_probability=0.7, seed=seed
        ).classical_solution()

        sol = {
            "x_0": 1,
            "x_1": 1,
            "x_2": 0,
            "x_3": 0,
            "x_4": 0,
            "x_5": 0,
            "x_6": 0,
            "x_7": 0,
            "x_8": 1,
            "x_9": 0,
        }

        self.assertEqual(mis_sol, sol)

    def test_mis_plot(self):
        """Test maximal independent set random instance method"""
        from matplotlib.pyplot import Figure

        seed = 1234
        mis_random_prob = MIS.random_instance(
            n_nodes=10, edge_probability=0.7, seed=seed
        )
        sol = {
            "x_0": 1,
            "x_1": 1,
            "x_2": 0,
            "x_3": 0,
            "x_4": 0,
            "x_5": 0,
            "x_6": 0,
            "x_7": 0,
            "x_8": 1,
            "x_9": 0,
        }
        fig = mis_random_prob.plot_solution(sol)
        self.assertTrue(isinstance(fig, Figure))


if __name__ == "__main__":
    unittest.main()