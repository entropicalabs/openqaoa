import unittest
import numpy as np
import networkx as nx
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


class TestBPSP(unittest.TestCase):
    """
    Test suite for the BPSP problem.

    This test suite checks the functionality of methods related to the BPSP problem,
    including QUBO creation, car sequencing, solution methods, and more.
    """

    def setUp(self):
        """
        Set up testing data before each test.

        This method initializes a BPSP instance with a specific car sequence.
        """
        self.bp = BPSP(np.array([0, 1, 0, 1]))

    def test_bpsp_terms_weights_constant(self):
        """
        Test the correct QUBO formation from a provided graph.

        This method validates that BPSP creates the expected QUBO by comparing 
        its terms and weights with predefined expected values.
        """
        car_sequence = [3, 1, 0, 0, 2, 2, 4, 4, 3, 1]
        G = nx.Graph()
        G.add_weighted_edges_from([[3, 1, -2], [3, 4, -1], [1, 0, -1], [0, 2, 1], [2, 4, 1]])

        gr_edges = [[3, 1], [3, 4], [1, 0], [0, 2], [2, 4]]
        gr_weights = [-2, -1, -1, 1, 1]

        bpsp_prob_qubo = BPSP(car_sequence).qubo
        self.assertTrue(terms_list_equality(gr_edges, bpsp_prob_qubo.terms))
        self.assertEqual(gr_weights, bpsp_prob_qubo.weights)
        self.assertEqual(0.0, bpsp_prob_qubo.constant)

    def test_car_sequence_raises(self):
        """
        Test if ValueErrors are raised for invalid car sequences.

        Certain car sequences are expected to be invalid based on their content.
        This method checks if setting such sequences raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.bp.car_sequence = np.array([0, 1, 0])
        with self.assertRaises(ValueError):
            self.bp.car_sequence = np.array([0, 0, 1, 2, 2, 3, 3, 3])

    def test_random_instance(self):
        """
        Test the generation of a random BPSP instance.

        Validates that the generated random BPSP instance has one of the
        expected car sequences.
        """
        instance = BPSP.random_instance(num_cars=2)
        self.assertTrue(all(np.array([0, 1, 0, 1]) == instance.car_sequence) or 
                        all(np.array([1, 1, 0, 0]) == instance.car_sequence) or 
                        all(np.array([0, 0, 1, 1]) == instance.car_sequence) or 
                        all(np.array([1, 0, 1, 0]) == instance.car_sequence))

    def test_car_pos(self):
        """Test the retrieval of car positions."""
        self.assertEqual(self.bp.car_positions, {0: (0, 2), 1: (1, 3)})

    def test_graph(self):
        """Test the generation of a graph representation of the BPSP instance."""
        self.assertEqual(len(self.bp.bpsp_graph.nodes), 2)

    def test_docplex_bpsp_model(self):
        """Test if the docplex model representation of BPSP is generated."""
        self.assertIsNotNone(self.bp.docplex_bpsp_model)

    def test_cplex_solution(self):
        """
        Test the solution of the BPSP problem using CPLEX.

        This test assumes that CPLEX is installed and functional. It checks
        the length of the solution and the objective value.
        """
        solution, objective_value = self.bp.cplex_solution()
        self.assertEqual(len(solution), 4)
        self.assertIn(objective_value, [1, 3])

    def test_qaoa_solution(self):
        """Test the solution of the BPSP problem using QAOA."""
        sequence, color_swaps = self.bp.qaoa_solution('10')
        self.assertEqual(sequence, [1, 0, 0, 1])
        self.assertEqual(color_swaps, 2)

    def test_redfirst_solution(self):
        """Test the solution of the BPSP problem using the Red-First method."""
        sequence, color_swaps = self.bp.redfirst_solution()
        self.assertEqual(sequence, [1, 1, 0, 0])
        self.assertEqual(color_swaps, 1)

    def test_greedy_solution(self):
        """Test the solution of the BPSP problem using the Greedy method."""
        sequence, color_swaps = self.bp.greedy_solution()
        self.assertEqual(sequence, [0, 0, 1, 1])
        self.assertEqual(color_swaps, 1)




if __name__ == "__main__":
    unittest.main()
