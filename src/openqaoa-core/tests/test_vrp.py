import unittest
import networkx as nx
import numpy as np
from openqaoa.problems import VRP


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


class TestVRP(unittest.TestCase):
    """Tests for VRP class"""

    def test_vrp_terms_weights_constant(self):
        """Testing VRP problem creation"""
        pos = [[4, 1], [4, 4], [3, 3]]  # nodes position x, y
        n_vehicles = 1
        n_nodes = len(pos)
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        for i in range(n_nodes - 1):
            for j in range(i + 1, n_nodes):
                r = np.sqrt((pos[i][0] - pos[j][0]) ** 2 + (pos[i][1] - pos[j][1]) ** 2)
                G.add_weighted_edges_from([(i, j, r)])
        vrp_qubo = VRP(G, n_vehicles, pos).qubo
        expected_terms = [[0, 1], [0, 2], [1, 2], [0], [1], [2]]
        expected_weights = [2.0, 2.0, 2.0, 6.5, 6.881966011250105, 7.292893218813452]
        expected_constant = 21.32514076993644

        self.assertTrue(terms_list_equality(expected_terms, vrp_qubo.terms))
        self.assertEqual(expected_weights, vrp_qubo.weights)
        self.assertEqual(expected_constant, vrp_qubo.constant)

    def test_vrp_random_instance(self):
        """Testing the random_instance method of the VRP problem class"""
        seed = 1234
        rng = np.random.default_rng(seed)
        n_nodes = 3
        n_vehicles = 1
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        pos = [[0, 0]]
        pos += [list(2 * rng.random(2) - 1) for _ in range(n_nodes - 1)]
        for i in range(n_nodes - 1):
            for j in range(i + 1, n_nodes):
                r = np.sqrt((pos[i][0] - pos[j][0]) ** 2 + (pos[i][1] - pos[j][1]) ** 2)
                G.add_weighted_edges_from([(i, j, r)])

        vrp_prob = VRP(G, n_vehicles, pos).qubo
        vrp_prob_random = VRP.random_instance(
            n_nodes=n_nodes, n_vehicles=n_vehicles, seed=seed
        ).qubo

        self.assertTrue(terms_list_equality(vrp_prob_random.terms, vrp_prob.terms))
        self.assertEqual(vrp_prob_random.weights, vrp_prob.weights)
        self.assertEqual(vrp_prob_random.constant, vrp_prob.constant)

    def test_vrp_matrix_input(self):
        """Testing the matrix input method of the VRP problem class"""
        matrix = [[0, 1.5, 3, 4.5], [0, 0, 5.5, 6.5], [0, 0, 0, 7.5], [0, 0, 0, 0]]
        n_vehicles = 1
        vrp = VRP.from_distance_matrix(matrix=matrix, n_vehicles=n_vehicles)
        vrp_qubo = vrp.qubo
        terms = [
            [0, 1],
            [0, 2],
            [1, 2],
            [0, 3],
            [0, 4],
            [3, 4],
            [1, 3],
            [1, 5],
            [3, 5],
            [2, 4],
            [2, 5],
            [4, 5],
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
        ]

        weights = [
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            3.25,
            2.5,
            1.75,
            1.25,
            0.75,
            0.25,
        ]
        constant = 30.25
        self.assertTrue(terms_list_equality(vrp_qubo.terms, terms))
        self.assertEqual(vrp_qubo.weights, weights)
        self.assertEqual(vrp_qubo.constant, constant)

    def test_vrp_coordinates_input(self):
        """Testing the coordinates input of the VRP problem class"""
        coordinates = [[0, 1], [0, 2], [0, 4], [3, 1]]
        n_vehicles = 1
        vrp = VRP.from_coordinates(pos=coordinates, n_vehicles=n_vehicles)
        vrp_qubo = vrp.qubo
        terms = [
            [0, 1],
            [0, 2],
            [1, 2],
            [0, 3],
            [0, 4],
            [3, 4],
            [1, 3],
            [1, 5],
            [3, 5],
            [2, 4],
            [2, 5],
            [4, 5],
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
        ]

        weights = [
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            3.5,
            2.5,
            2.5,
            3.0,
            2.418861169915811,
            1.878679656440358,
        ]
        constant = 24.20245917364383
        self.assertTrue(terms_list_equality(vrp_qubo.terms, terms))
        self.assertEqual(vrp_qubo.weights, weights)
        self.assertEqual(vrp_qubo.constant, constant)

    def test_vrp_random_instance_unbalanced(self):
        """
        Testing the random_instance method of the VRP problem class using the
        unbalanced penalization method
        """
        seed = 1234
        n_nodes = 8
        n_vehicles = 2
        n_vars = n_nodes * (n_nodes - 1) // 2

        vrp_prob_random = VRP.random_instance(
            n_nodes=n_nodes,
            n_vehicles=n_vehicles,
            seed=seed,
            method="unbalanced",
            penalty=3 * [0.1],
        ).qubo

        self.assertTrue(vrp_prob_random.n == n_vars)
        self.assertTrue(vrp_prob_random.weights[0] == 0.05)
        self.assertTrue(vrp_prob_random.weights[-1] == -0.793013310265329)
        self.assertTrue(vrp_prob_random.terms[0] == [0, 1])
        self.assertTrue(vrp_prob_random.terms[-1] == [27])

    def test_vrp_plot(self):
        """
        Testing the random_instance method of the VRP problem class using the
        unbalanced penalization method
        """
        import matplotlib.pyplot as plt

        seed = 1234
        np.random.seed(seed)
        n_nodes = 8
        n_vehicles = 2

        vrp_prob_random = VRP.random_instance(
            n_nodes=n_nodes, n_vehicles=n_vehicles, seed=seed
        )
        vrp_sol = vrp_prob_random.classical_solution()
        vrp_sol_str = vrp_prob_random.classical_solution(string=True)

        self.assertTrue(isinstance(vrp_prob_random.plot_solution(vrp_sol), plt.Figure))
        self.assertTrue(
            isinstance(vrp_prob_random.plot_solution(vrp_sol_str), plt.Figure)
        )

    def test_vrp_type_checking(self):
        """
        Checks if the type-checking returns the right error.
        """
        # Wrong method is called
        with self.assertRaises(ValueError) as e:
            VRP.random_instance(n_nodes=6, n_vehicles=2, method="method")
        self.assertEqual(
            "The method 'method' is not valid.",
            str(e.exception),
        )
        # Penalization terms in unblanced method is not equal to three
        with self.assertRaises(ValueError) as e:
            VRP.random_instance(
                n_nodes=6, n_vehicles=2, method="unbalanced", penalty=[0.1]
            )
        self.assertEqual(
            "The penalty must have 3 parameters [lambda_0, lambda_1, lambda_2]",
            str(e.exception),
        )

        # paths with an unfeasible solution
        with self.assertRaises(ValueError) as e:
            vrp = VRP.random_instance(n_nodes=6, n_vehicles=2, seed=1234)
            sol = vrp.classical_solution()
            sol["x_0_1"] = (
                sol["x_0_1"] + 1
            ) % 2  # Changing one value in the solution to make it an unfeasible solution
            vrp.paths_subtours(sol)
        self.assertEqual(
            "Solution provided does not fulfill all the path conditions.",
            str(e.exception),
        )
        # subtours with an unfeasible direction (broke subtour)
        with self.assertRaises(ValueError) as e:
            vrp = VRP.random_instance(
                n_nodes=10, n_vehicles=2, subtours=[[]], seed=1234
            )
            sol = vrp.classical_solution()
            sol["x_1_2"] = (
                sol["x_1_2"] + 1
            ) % 2  # Changing one value in the solution to make it an unfeasible solution
            vrp.paths_subtours(sol)
        self.assertEqual(
            "The subtours in the solution are broken.",
            str(e.exception),
        )
        # Test unfeasible problem does not return classical solution
        with self.assertRaises(ValueError) as e:
            vrp = VRP.random_instance(n_nodes=3, n_vehicles=3, seed=1234)
            sol = vrp.classical_solution()
            vrp.paths_subtours(sol)
        self.assertEqual(
            "Solution not found: integer infeasible.",
            str(e.exception),
        )
        # Test wrong graph input
        with self.assertRaises(TypeError) as e:
            G = [[0, 1, 2], [1, 2, 3]]
            vrp = VRP(G, n_vehicles=2, pos=[[0, 1], [1, 2]])
        self.assertEqual(
            "Input problem graph must be a networkx Graph.",
            str(e.exception),
        )
        # Test different size between graph nodes and x, y positions in pos
        with self.assertRaises(ValueError) as e:
            G = nx.Graph()
            G.add_nodes_from(range(3))
            vrp = VRP(G, n_vehicles=2, pos=[[0, 1], [1, 2]])
        self.assertEqual(
            "The number of nodes in G is 3 while the x, y coordinates in pos is 2",
            str(e.exception),
        )
        # Test different size between colors and number of vehicles
        with self.assertRaises(ValueError) as e:
            seed = 1234
            vrp = VRP.random_instance(n_nodes=6, n_vehicles=2, seed=seed)
            vrp.plot_solution(vrp.classical_solution(), colors=["tab:blue"])
        self.assertEqual(
            "The length of colors 1 and the number of vehicles 2 do not match",
            str(e.exception),
        )


if __name__ == "__main__":
    unittest.main()
