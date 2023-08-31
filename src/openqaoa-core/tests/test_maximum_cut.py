import unittest
import networkx as nx
import numpy as np
from openqaoa.problems import MaximumCut


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


class TestMaximumCut(unittest.TestCase):
    """Tests for MaximumCut class"""

    def test_maximumcut_terms_weights_constant(self):
        """Test that MaximumCut creates a correct QUBO from the provided graph"""

        gr = nx.generators.random_graphs.fast_gnp_random_graph(n=10, p=0.8)
        gr_edges = [list(edge) for edge in gr.edges()]
        gr_weights = [1] * len(gr_edges)

        maxcut_prob_qubo = MaximumCut(gr).qubo

        self.assertTrue(terms_list_equality(gr_edges, maxcut_prob_qubo.terms))
        self.assertEqual(gr_weights, maxcut_prob_qubo.weights)
        self.assertEqual(0, maxcut_prob_qubo.constant)

    def test_maximumcut_random_problem(self):
        """Test MaximumCut random instance method"""

        seed = 1234
        gr = nx.generators.random_graphs.fast_gnp_random_graph(n=10, p=0.8, seed=seed)
        maxcut_manual_prob = MaximumCut(gr).qubo

        np.random.seed(1234)
        maxcut_random_prob = MaximumCut.random_instance(
            n_nodes=10, edge_probability=0.8, seed=seed
        ).qubo

        self.assertTrue(
            terms_list_equality(maxcut_manual_prob.terms, maxcut_random_prob.terms)
        )
        self.assertEqual(maxcut_manual_prob.weights, maxcut_random_prob.weights)
        self.assertEqual(maxcut_manual_prob.constant, maxcut_random_prob.constant)

    def test_maximumcut_type_checking(self):
        """
        Checks if the type-checking returns the right error.
        """

        # graph type-check
        graph_list = [(1, 2), {"node1": 1, "node2": 2}, np.array([1, 2])]

        for each_graph in graph_list:
            with self.assertRaises(TypeError) as e:
                MaximumCut(G=each_graph)
            self.assertEqual(
                "Input problem graph must be a networkx Graph.", str(e.exception)
            )


if __name__ == "__main__":
    unittest.main()
