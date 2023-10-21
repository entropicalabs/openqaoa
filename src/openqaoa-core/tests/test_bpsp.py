import unittest
import networkx as nx
import numpy as np
from openqaoa.problems import BPSP


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


class TestBPSP(unittest.TestCase):

    def test_bpsp_terms_weights_constant(self):
        """Test that BPSP creates a correct QUBO from the provided graph"""

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
        gr_weights = [-1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # You might want to adjust these expected values

        bpsp_prob_qubo = BPSP(G).qubo
        self.assertTrue(terms_list_equality(gr_edges, bpsp_prob_qubo.terms))
        self.assertEqual(gr_weights, bpsp_prob_qubo.weights)
        self.assertEqual(0.0, bpsp_prob_qubo.constant)

    def setUp(self):
        # This method is run before each test. It's used to set up testing data.
        self.bp = BPSP(np.array([0, 1, 0, 1]))
        
    def test_car_sequence_raises(self):
        # Test if ValueErrors are raised as expected
        with self.assertRaises(ValueError):
            self.bp.car_sequence = np.array([0, 1, 0])
        with self.assertRaises(ValueError):
            self.bp.car_sequence = np.array([0, 0, 1, 2, 2, 3, 3, 3])

    def test_random_instance(self):
        # Test random instance generation
        instance = BPSP.random_instance(num_cars=2)
        self.assertTrue(all(np.array([0, 1, 0, 1]) == instance.car_sequence) or 
                        all(np.array([1, 0, 1, 0]) == instance.car_sequence))

    def test_car_pos(self):
        self.assertEqual(self.bp.car_positions, {0: (0, 2), 1: (1, 3)})

    def test_graph(self):
        # Add a basic test for graph generation (more rigorous tests would require inspecting the graph)
        self.assertEqual(len(self.bp.bpsp_graph.nodes), 2)

    def test_docplex_bpsp_model(self):
        # Test for the docplex model (basic check)
        self.assertIsNotNone(self.bp.docplex_bpsp_model)

    def test_cplex_solution(self):
        # Assuming cplex is installed and functional
        solution, objective_value = self.bp.cplex_solution()
        self.assertEqual(len(solution), 4)
        self.assertIn(objective_value, [1, 3])  # Depending on the BPSP instance solution

    def test_qaoa_solution(self):
        sequence, color_swaps = self.bp.qaoa_solution("1010")
        self.assertEqual(sequence, [1, 0, 0, 1])
        self.assertEqual(color_swaps, 3)

    def test_redfirst_solution(self):
        sequence, color_swaps = self.bp.redfirst_solution()
        self.assertEqual(sequence, [1, 1, 0, 0])
        self.assertEqual(color_swaps, 2)

    def test_greedy_solution(self):
        sequence, color_swaps = self.bp.greedy_solution()
        self.assertEqual(sequence, [0, 0, 1, 1])
        self.assertEqual(color_swaps, 2)

# ... other tests ...

if __name__ == "__main__":
    unittest.main()
