import unittest
import networkx as nx
import numpy as np
from openqaoa.problems import KColor


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


class TestKColor(unittest.TestCase):
    """Tests for the KColor class"""

    def test_kcolor_terms_weights_constant(self):
        """Test that KColor creates a correct Docplex model from the provided graph"""

        gr = nx.generators.random_graphs.fast_gnp_random_graph(n=3, p=0.8, seed=1234)
        k = 3

        kcolor_prob = KColor(gr, k)
        mdl = kcolor_prob.docplex_model
        qubo = kcolor_prob.qubo
        terms = [
            [3, 6],
            [4, 7],
            [8, 5],
            [0, 1],
            [0, 2],
            [1, 2],
            [3, 4],
            [3, 5],
            [4, 5],
            [6, 7],
            [8, 6],
            [8, 7],
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
        ]
        weights = [
            1.0,
            1.0,
            1.0,
            6.5,
            6.5,
            6.5,
            6.5,
            6.5,
            6.5,
            6.5,
            6.5,
            6.5,
            -6.5,
            -6.5,
            -6.5,
            -7.5,
            -7.5,
            -7.5,
            -7.5,
            -7.5,
            -7.5,
        ]
        constant = 42.0
        # Verify the number of variables
        num_vertices = gr.number_of_nodes()
        num_colors = k
        expected_num_vars = num_vertices * num_colors
        self.assertEqual(mdl.number_of_variables, expected_num_vars)

        # Verify the constraints
        expected_num_constraints = num_vertices
        self.assertEqual(mdl.number_of_constraints, expected_num_constraints)

        # Verify the qubo
        self.assertEqual(qubo.terms, terms)
        self.assertEqual(qubo.weights, weights)
        self.assertEqual(qubo.constant, constant)

    def test_kcolor_random_problem(self):
        """Test KColor random instance method"""
        seed = 1234
        n_nodes = 10
        edge_probability = 0.8
        k = 4
        np.random.seed(seed)
        kcolor_manual_prob = KColor.random_instance(
            n_nodes=n_nodes, edge_probability=edge_probability, k=k, seed=seed
        )
        kcolor_random_prob = KColor(
            nx.generators.random_graphs.fast_gnp_random_graph(
                n=n_nodes, p=edge_probability, seed=seed
            ),
            k,
        )

        # Verify the graph is the same
        self.assertEqual(kcolor_manual_prob.G.nodes(), kcolor_random_prob.G.nodes())
        self.assertEqual(kcolor_manual_prob.G.edges(), kcolor_random_prob.G.edges())

        # Verify the number of colors
        self.assertEqual(k, kcolor_manual_prob.k)

        # Verify the Docplex models are the same
        self.assertEqual(
            kcolor_manual_prob.docplex_model.export_to_string(),
            kcolor_random_prob.docplex_model.export_to_string(),
        )

    def test_kcolor_type_checking(self):
        """
        Checks if the type-checking returns the right error.
        """

        # graph type-check
        graph_list = [(1, 2), {"node1": 1, "node2": 2}, np.array([1, 2])]

        for each_graph in graph_list:
            with self.assertRaises(TypeError) as e:
                KColor(G=each_graph, k=3)
            self.assertEqual(
                "Input problem graph must be a networkx Graph.", str(e.exception)
            )

    def test_kcolor_classical_sol(self):
        """Test the k-color random instance method classical solution"""

        seed = 1234
        n_nodes = 10
        edge_probability = 0.1
        k = 3

        np.random.seed(seed)
        kcolor_sol = KColor.random_instance(
            n_nodes=n_nodes, edge_probability=edge_probability, k=k, seed=seed
        ).classical_solution(string=True)
        # Verify the solution size
        self.assertTrue(len(kcolor_sol) == n_nodes * k)

        # Verify each vertex has just a color
        for i in range(n_nodes):
            self.assertTrue(kcolor_sol[i * k : (i + 1) * k].count("1") == 1)

    def test_kcolor_plot(self):
        """Test k-color random instance method"""
        import matplotlib.pyplot as plt

        seed = 1234
        n_nodes = 10
        edge_probability = 0.7
        k = 3

        kcolor_random_prob = KColor.random_instance(
            n_nodes=n_nodes, edge_probability=edge_probability, k=k, seed=seed
        )
        sol = kcolor_random_prob.classical_solution()
        fig, ax = plt.subplots()
        kcolor_random_prob.plot_solution(sol, ax=ax)

        self.assertTrue(isinstance(ax, plt.Axes))


if __name__ == "__main__":
    unittest.main()
