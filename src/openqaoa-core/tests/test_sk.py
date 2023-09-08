import unittest
import networkx as nx
import numpy as np
from openqaoa.problems import SK


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


class TestSK(unittest.TestCase):
    """Tests for the SK class"""

    def test_sk_terms_weights_constant(self):
        """Test that SK creates a correct QUBO from the provided graph"""

        n_nodes = 5
        rng = np.random.default_rng(1234)
        G = nx.Graph()
        G.add_weighted_edges_from(
            [
                [i, j, round(2 * rng.random() - 1)]
                for i in range(n_nodes)
                for j in range(i + 1, n_nodes)
            ]
        )
        gr_edges = [[0, 1], [0, 3], [1, 3], [1, 4], [2, 4], [0], [1], [2], [3], [4]]
        gr_weights = [-1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        sk_prob_qubo = SK(G).qubo
        self.assertTrue(terms_list_equality(gr_edges, sk_prob_qubo.terms))
        self.assertEqual(gr_weights, sk_prob_qubo.weights)
        self.assertEqual(0.0, sk_prob_qubo.constant)

    def test_sk_random_problem(self):
        """Test SK random instance method"""

        seed = 123
        n_nodes = 5
        mu = 0
        sigma = 1
        rng = np.random.default_rng(seed)
        G = nx.Graph()
        G.add_weighted_edges_from(
            [
                [i, j, round(rng.normal(loc=mu, scale=sigma), 3)]
                for i in range(n_nodes)
                for j in range(i + 1, n_nodes)
            ]
        )
        sk_manual_prob = SK(G).qubo

        sk_random_prob = SK.random_instance(n_nodes=n_nodes, seed=seed).qubo

        self.assertTrue(terms_list_equality(sk_manual_prob.terms, sk_random_prob.terms))
        self.assertEqual(sk_manual_prob.weights, sk_random_prob.weights)
        self.assertEqual(sk_manual_prob.constant, sk_random_prob.constant)

    def test_sk_type_checking(self):
        """
        Checks if the type-checking returns the right error.
        """

        # graph type-check
        graph_list = [(1, 2), {"node1": 1, "node2": 2}, np.array([1, 2])]

        for each_graph in graph_list:
            with self.assertRaises(TypeError) as e:
                SK(G=each_graph)
            self.assertEqual(
                "Input problem graph must be a networkx Graph.", str(e.exception)
            )

    def test_sk_classical_sol(self):
        """Test the SK random instance method classical solution"""

        seed = 1234
        sk_sol = SK.random_instance(n_nodes=2, seed=seed).classical_solution()

        sol = {"x_0": 1, "x_1": 0}

        self.assertEqual(sk_sol, sol)

    def test_sk_plot(self):
        """Test SK random instance method"""
        import matplotlib.pyplot as plt

        seed = 1234
        sk_random_prob = SK.random_instance(n_nodes=10, seed=seed)
        sol = {
            "x_0": 1.0,
            "x_1": 1.0,
            "x_2": 0,
            "x_3": 1.0,
            "x_4": 0,
            "x_5": 1.0,
            "x_6": 0,
            "x_7": 0,
            "x_8": 0,
            "x_9": 1.0,
        }

        fig, ax = plt.subplots()
        sk_random_prob.plot_solution(sol, ax=ax)
        self.assertTrue(isinstance(ax, plt.Axes))


if __name__ == "__main__":
    unittest.main()
