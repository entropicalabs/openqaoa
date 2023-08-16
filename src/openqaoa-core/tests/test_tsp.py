import unittest
import networkx as nx
import numpy as np
from openqaoa.problems import TSP, TSP_LP


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

class TestTSP(unittest.TestCase):
    """Tests for TSP class"""
    def test_tsp_terms_weights_constant(self):
        """Testing TSP problem creation"""
        city_coordinates = [(4, 1), (4, 4), (3, 3), (1, 3.5)]
        expected_terms = [
            [0, 3],
            [1, 4],
            [2, 5],
            [0, 6],
            [1, 7],
            [8, 2],
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
            [0, 4],
            [1, 3],
            [3, 7],
            [4, 6],
            [0, 5],
            [2, 3],
            [8, 3],
            [5, 6],
            [1, 5],
            [2, 4],
            [8, 4],
            [5, 7],
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
        expected_weights = [
            3.905124837953327,
            3.905124837953327,
            3.905124837953327,
            3.905124837953327,
            3.905124837953327,
            3.905124837953327,
            3.905124837953327,
            3.905124837953327,
            3.905124837953327,
            3.905124837953327,
            3.905124837953327,
            3.905124837953327,
            3.905124837953327,
            3.905124837953327,
            3.905124837953327,
            3.905124837953327,
            3.905124837953327,
            3.905124837953327,
            0.3535533905932738,
            0.3535533905932738,
            0.7603453162872774,
            0.7603453162872774,
            0.3535533905932738,
            0.3535533905932738,
            0.5153882032022076,
            0.5153882032022076,
            0.7603453162872774,
            0.7603453162872774,
            0.5153882032022076,
            0.5153882032022076,
            -10.424148382787205,
            -9.797225258452029,
            -11.038545614372802,
            -10.038047089667756,
            -9.548132863497614,
            -10.361716714885624,
            -10.424148382787205,
            -9.797225258452029,
            -11.038545614372802,
        ]
        expected_constant = 62.51983851122417
        tsp_qubo = TSP(city_coordinates).qubo
        self.assertTrue(terms_list_equality(expected_terms, tsp_qubo.terms))
        self.assertEqual(expected_weights, tsp_qubo.weights)
        self.assertEqual(expected_constant, tsp_qubo.constant)

    def test_tsp_random_instance(self):
        """Testing the random_instance method of the TSP problem class"""
        rng = np.random.default_rng(1234)
        n_cities = 4

        box_size = np.sqrt(n_cities)
        city_coordinates = list(map(tuple, box_size * rng.random(size=(n_cities, 2))))

        tsp_prob = TSP(city_coordinates).qubo

        tsp_prob_random = TSP.random_instance(n_cities=n_cities, seed=1234).qubo

        self.assertTrue(terms_list_equality(tsp_prob_random.terms, tsp_prob.terms))
        self.assertEqual(tsp_prob_random.weights, tsp_prob.weights)
        self.assertEqual(tsp_prob_random.constant, tsp_prob.constant)

    def test_tsp_type_checking(self):
        """
        Checks if the type-checking returns the right error.
        """
        # If nothing is given, must return a ValueError
        with self.assertRaises(ValueError) as e:
            TSP()
        self.assertEqual(
            "Input missing: city coordinates, distance matrix or (weighted graph) required",
            str(e.exception),
        )

        # coordinates type-check
        coordinates_list = [(1, 2), {"test": "oh", "test1": "oh"}, np.array([1, 2])]

        for each_coordinates in coordinates_list:
            with self.assertRaises(TypeError) as e:
                TSP(each_coordinates)
            self.assertEqual("The coordinates should be a list", str(e.exception))

        coordinates_list = [[[1, 2], [2, 1]], [np.array([1, 2]), np.array([2, 1])]]
        for each_coordinates in coordinates_list:
            with self.assertRaises(TypeError) as e:
                TSP(each_coordinates)
            self.assertEqual(
                "The coordinates should be contained in a tuple", str(e.exception)
            )

        coordinates_list = [
            [("oh", "num"), ("num", "oh")],
            [(np.array(1), np.array(2)), (np.array(2), np.array(1))],
        ]
        for each_coordinates in coordinates_list:
            with self.assertRaises(TypeError) as e:
                TSP(each_coordinates)
            self.assertEqual(
                "The coordinates must be of type float or int", str(e.exception)
            )

        # coordinates type-check
        distance_matrices = [
            (1, 2),
            np.array([[1, 2], [3, 4]]),
            {"test": "oh", "test1": "oh"},
        ]

        for distance_matrix in distance_matrices:
            with self.assertRaises(TypeError) as e:
                TSP(distance_matrix=distance_matrix)
            self.assertEqual("The distance matrix should be a list", str(e.exception))

        # Distance matrix type-check
        distance_matrices = [[(1, 2), (2, 1)], [np.array([1, 2]), np.array([2, 1])]]
        for distance_matrix in distance_matrices:
            with self.assertRaises(TypeError) as e:
                TSP(distance_matrix=distance_matrix)
            self.assertEqual(
                "Each row in the distance matrix should be a list", str(e.exception)
            )

        distance_matrices = [
            [["oh", "num"], ["num", "oh"]],
            [[np.array(1), np.array(2)], [np.array(2), np.array(1)]],
        ]
        for distance_matrix in distance_matrices:
            with self.assertRaises(TypeError) as e:
                TSP(distance_matrix=distance_matrix)
            self.assertEqual(
                "The distance matrix entries must be of type float or int",
                str(e.exception),
            )

        distance_matrix = [[1, 2.3], [-2, 3]]
        with self.assertRaises(ValueError) as e:
            TSP(distance_matrix=distance_matrix)
        self.assertEqual("Distances should be positive", str(e.exception))

        # Graph type-check
        G = nx.complete_graph(5)
        for u, v in G.edges():
            G[u][v]["weight"] = "a"

        with self.assertRaises(TypeError) as e:
            TSP(G=G)
        self.assertEqual(
            "The edge weights must be of type float or int", str(e.exception)
        )

        for u, v in G.edges():
            G[u][v]["weight"] = -2.0

        with self.assertRaises(ValueError) as e:
            TSP(G=G)
        self.assertEqual("Edge weights should be positive", str(e.exception))

    # TESTING TSP PROBLEM CLASS
class TestTSP_LP(unittest.TestCase):
    """Tests for TSP LP class"""
    def test_tsp_lp_terms_weights_constant(self):
        """Testing TSP LP problem creation"""
        tsp_qubo = TSP_LP.random_instance(n_nodes=3, seed=1234).qubo
        expected_terms = [[0, 1], [0, 2], [1, 2], [0], [1], [2]]
        expected_weights = [2.0, 2.0, 2.0, 7.66823080091817, 7.707925770071554, 7.704586691688892]
        expected_constant = 18.919256737321383
        self.assertTrue(terms_list_equality(expected_terms, tsp_qubo.terms))
        self.assertEqual(expected_weights, tsp_qubo.weights)
        self.assertEqual(expected_constant, tsp_qubo.constant)

    def test_tsp_lp_length(self):
        """Testing TSP LP problem creation"""
        cities = 3
        tsp = TSP_LP.random_instance(n_nodes=cities, seed=111)
        solution = tsp.classical_solution()
        distance_expected = 2.503342058155561
        self.assertEqual(distance_expected, tsp.get_distance(solution))
    
    def test_tsp_lp_plot(self):
        """Testing TSP LP problem creation"""
        from matplotlib.pyplot import Figure
        cities = 6
        tsp = TSP_LP.random_instance(n_nodes=cities, seed=123, subtours=[[1,3,4]])
        solution = tsp.classical_solution()
        fig = tsp.plot_solution(solution)  
        self.assertTrue(isinstance(fig, Figure))

if __name__ == "__main__":
    unittest.main()