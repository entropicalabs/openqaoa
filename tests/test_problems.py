import unittest
import networkx as nx
import numpy as np
from random import randint, random
from openqaoa.problems import (
    NumberPartition,
    QUBO,
    TSP,
    Knapsack,
    ShortestPath,
    SlackFreeKnapsack,
    MaximumCut,
    MinimumVertexCover,
    BinPacking,
)
from openqaoa.utilities import convert2serialize
from openqaoa.problems.helper_functions import create_problem_from_dict


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
            bool = True if np.isclose(term1, term2) or np.isclose(term1, term2[::-1]) else False

    return bool


class TestProblem(unittest.TestCase):

    # TESTING QUBO CLASS METHODS
    def test_qubo_terms_and_weight_same_size(self):
        """
        Test that creating a QUBO problem with invalid terms and weights
        sizes raises an exception and check the constant is detected correctly
        """
        n = 2
        terms_wrong = [[0], [1]]
        terms_correct = [[0], [1], []]
        weights = [1, 2, 3]

        with self.assertRaises(ValueError):
            qubo_problem = QUBO(n, terms_wrong, weights)

        qubo_problem = QUBO(n, terms_correct, weights)
        self.assertEqual(qubo_problem.constant, 3)

    def test_qubo_cleaning_terms(self):
        """Test that cleaning terms works for a QUBO problem"""
        terms = [[1, 2], [0], [2, 3], [2, 1], [0]]
        weights = [3, 4, -3, -2, -1]

        cleaned_terms = [[1, 2], [0], [2, 3]]
        self.assertEqual(QUBO.clean_terms_and_weights(terms, weights)[0], cleaned_terms)

    def test_qubo_cleaning_weights(self):
        """Test that cleaning weights works for a QUBO problem"""
        terms = [[1, 2], [0], [2, 3], [2, 1], [0]]
        weights = [3, 4, -3, -2, -1]

        cleaned_weights = [1, 3, -3]
        self.assertEqual(
            QUBO.clean_terms_and_weights(terms, weights)[1], cleaned_weights
        )

    def test_qubo_ising_conversion(self):
        """Test that conversion to Ising formulation works for a QUBO problem"""
        # Small instance
        n = 2
        terms = [[1, 1], [0, 0]]
        weights = [3, 4]

        expected_ising_terms = [[0], [1], []]
        expected_ising_weights = [-2, -1.5, 3.5]

        ising_terms, ising_weights = QUBO.convert_qubo_to_ising(n, terms, weights)
        self.assertEqual(expected_ising_terms, ising_terms)
        self.assertEqual(expected_ising_weights, ising_weights)

        # Larger instance
        n = 4
        terms = [[1, 2], [0], [2, 3], [2, 1], [0]]
        weights = [3, 4, -3, -2, -1]

        expected_ising_terms = [[1, 2], [2, 3], [2, 1], [0], [1], [2], [3], []]
        expected_ising_weights = [0.75, -0.75, -0.5, -1.5, -0.25, 0.5, 0.75, 1.0]

        ising_terms, ising_weights = QUBO.convert_qubo_to_ising(n, terms, weights)
        self.assertEqual(expected_ising_terms, ising_terms)
        self.assertEqual(expected_ising_weights, ising_weights)

    def test_qubo_type_checking(self):
        """
        Checks if the type-checking returns the right error.
        """

        # n type-check
        n_list = [1.5, "test", [], (), {}, np.array(1)]
        terms = [[0], []]
        weights = [1, 2]

        with self.assertRaises(TypeError) as e:
            for each_n in n_list:
                QUBO(each_n, terms, weights)
            self.assertEqual(
                "The input parameter, n, has to be of type int", str(e.exception)
            )

        n_list = [-1, 0]
        with self.assertRaises(TypeError) as e:
            for each_n in n_list:
                QUBO(each_n, terms, weights)
            self.assertEqual(
                "The input parameter, n, must be a positive integer greater than 0",
                str(e.exception),
            )

        # weights type-check
        n = 1
        terms = [[0], []]
        weights_list = [{"test": "oh", "test1": "oh"}, np.array([1, 2])]

        for each_weights in weights_list:
            with self.assertRaises(TypeError) as e:
                QUBO(n, terms, each_weights)
            self.assertEqual(
                "The input parameter weights must be of type of list or tuple",
                str(e.exception),
            )

        weights_list = [["test", "oh"], [np.array(1), np.array(2)]]
        for each_weights in weights_list:
            with self.assertRaises(TypeError) as e:
                QUBO(n, terms, each_weights)
            self.assertEqual(
                "The elements in weights list must be of type float or int.",
                str(e.exception),
            )

        # terms type-check
        n = 1
        terms_list = [{"test": [0], "test1": []}]
        weights = [1, 2]
        for each_terms in terms_list:
            with self.assertRaises(TypeError) as e:
                QUBO(n, each_terms, weights)
            self.assertEqual(
                "The input parameter terms must be of type of list or tuple",
                str(e.exception),
            )

    def test_qubo_metadata(self):
        """Test that metadata is correctly stored"""
        qubo_problem = QUBO.random_instance(3)
        qubo_problem.set_metadata({"tag1": "value1", "tag2": "value2"})
        qubo_problem.set_metadata({"tag2": "value2.0"})

        assert (
            qubo_problem.metadata["tag1"] == "value1"
        ), "qubo metadata is not well set"
        assert (
            qubo_problem.metadata["tag2"] == "value2.0"
        ), "qubo metadata is not well set, should have overwritten previous value"

        error = False
        try:
            qubo_problem.set_metadata({"tag10": complex(1, 2)})
        except:
            error = True
        assert (
            error
        ), "Should have thrown an error when setting metadata that is not json serializable"

        error = False
        try:
            qubo_problem.set_metadata({(1, 2): "value"})
        except:
            error = True
        assert (
            error
        ), "Should have thrown an error when setting key metadata that is not json serializable"

    def test_qubo_problem_instance_serializable(self):
        """test that when problem instance is not serializable, it throws an error"""

        qubo = QUBO.random_instance(3)

        error = False
        try:
            qubo.problem_instance = {"tag10": complex(1, 2)}
        except:
            error = True
        assert (
            error
        ), "Should have thrown an error when setting qubo problem instance that is not json serializable"

        error = False
        try:
            qubo.problem_instance = {(1, 2): "value"}
        except:
            error = True
        assert (
            error
        ), "Should have thrown an error when setting key qubo problem instance that is not json serializable"

    # TESTING NUMBER PARITION CLASS

    def test_number_partitioning_terms_weights_constant(self):
        """Test that Number Partitioning creates the correct terms, weights, constant"""
        list_numbers = [1, 2, 3]
        expected_terms = [[0, 1], [0, 2], [1, 2]]
        expected_weights = [4, 6, 12]
        expected_constant = 14

        np_problem = NumberPartition(list_numbers)
        qubo_problem = np_problem.qubo

        self.assertTrue(terms_list_equality(qubo_problem.terms, expected_terms))
        self.assertEqual(qubo_problem.weights, expected_weights)
        self.assertEqual(qubo_problem.constant, expected_constant)

    def test_number_partitioning_random_problem(self):
        """Test randomly generated NumberPartition problem"""
        # regenerate the same numbers randomly
        rng = np.random.default_rng(1234)
        random_numbers_list = list(map(int, rng.integers(1, 10, size=5)))
        manual_np_prob = NumberPartition(random_numbers_list).qubo

        np_prob_random = NumberPartition.random_instance(n_numbers=5, seed=1234).qubo

        self.assertTrue(terms_list_equality(np_prob_random.terms, manual_np_prob.terms))
        self.assertEqual(np_prob_random.weights, manual_np_prob.weights)
        self.assertEqual(np_prob_random.constant, manual_np_prob.constant)

    def test_num_part_type_checking(self):
        """
        Checks if the type-checking returns the right error.
        """

        # numbers type-check
        numbers_list = [(1, 2), {"test": 1, "test1": 2}, np.array([1, 2])]

        for each_number in numbers_list:
            with self.assertRaises(TypeError) as e:
                NumberPartition(numbers=each_number)
            self.assertEqual(
                "The input parameter, numbers, has to be a list", str(e.exception)
            )

        numbers_list = [[0.1, 1], [np.array(1), np.array(2)]]

        for each_number in numbers_list:
            with self.assertRaises(TypeError) as e:
                NumberPartition(numbers=each_number)
            self.assertEqual(
                "The elements in numbers list must be of type int.", str(e.exception)
            )

    # TESTING MAXIMUMCUT CLASS

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

    # TESTING KNAPSACK CLASS

    def test_knapsack_terms_weights_constant(self):
        """Test that Knapsack creates the correct QUBO problem"""

        values = [2, 4, 3, 5]
        weights = [3, 6, 9, 1]
        weight_capacity = 15
        n_qubits = len(values) + int(np.ceil(np.log2(weight_capacity)))
        penalty = 2 * max(values)
        knap_terms = [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
            [4, 5],
            [4, 6],
            [4, 7],
            [5, 6],
            [5, 7],
            [6, 7],
            [0, 4],
            [0, 5],
            [0, 6],
            [0, 7],
            [1, 4],
            [1, 5],
            [1, 6],
            [1, 7],
            [2, 4],
            [2, 5],
            [2, 6],
            [2, 7],
            [3, 4],
            [3, 5],
            [3, 6],
            [3, 7],
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
        ]
        knap_weights = [
            10.0,
            20.0,
            40.0,
            40.0,
            80.0,
            160.0,
            90.0,
            135.0,
            15.0,
            270.0,
            30.0,
            45.0,
            15.0,
            30.0,
            45.0,
            5.0,
            30.0,
            60.0,
            90.0,
            10.0,
            60.0,
            120.0,
            180.0,
            20.0,
            120.0,
            240.0,
            360.0,
            40.0,
            -20.0,
            -40.0,
            -80.0,
            -160.0,
            -59.0,
            -118.0,
            -178.5,
            -17.5,
        ]
        knap_constant = 563.0

        knapsack_prob_qubo = Knapsack(values, weights, weight_capacity, penalty).qubo

        self.assertTrue(terms_list_equality(knap_terms, knapsack_prob_qubo.terms))
        self.assertEqual(knap_weights, knapsack_prob_qubo.weights)
        self.assertEqual(knap_constant, knapsack_prob_qubo.constant)
        self.assertEqual(n_qubits, knapsack_prob_qubo.n)

    def test_knapsack_random_problem(self):
        """Test random instance method of Knapsack problem class"""

        rng = np.random.default_rng(1234)
        n_items = 5
        values = list(map(int, rng.integers(1, n_items, size=n_items)))
        weights = list(map(int, rng.integers(1, n_items, size=n_items)))
        weight_capacity = int(
            rng.integers(np.min(weights) * n_items, np.max(weights) * n_items)
        )
        penalty = 2 * np.max(values)

        knap_manual = Knapsack(values, weights, weight_capacity, int(penalty)).qubo

        knap_random_instance = Knapsack.random_instance(n_items=n_items, seed=1234).qubo

        self.assertTrue(
            terms_list_equality(knap_manual.terms, knap_random_instance.terms)
        )
        self.assertEqual(knap_manual.weights, knap_random_instance.weights)
        self.assertEqual(knap_manual.constant, knap_random_instance.constant)
        self.assertEqual(knap_manual.n, knap_random_instance.n)

    def test_knapsack_random_problem_smallsize(self):
        """Test random instance method of Knapsack problem class"""

        rng = np.random.default_rng(1234)
        n_items = 3
        values = list(map(int, rng.integers(1, n_items, size=n_items)))
        weights = list(map(int, rng.integers(1, n_items, size=n_items)))
        weight_capacity = int(
            rng.integers(np.min(weights) * n_items, np.max(weights) * n_items)
        )
        penalty = 2 * np.max(values)

        knap_manual = Knapsack(values, weights, weight_capacity, int(penalty)).qubo

        knap_random_instance = Knapsack.random_instance(n_items=n_items, seed=1234).qubo

        self.assertTrue(
            terms_list_equality(knap_manual.terms, knap_random_instance.terms)
        )
        self.assertEqual(knap_manual.weights, knap_random_instance.weights)
        self.assertEqual(knap_manual.constant, knap_random_instance.constant)
        self.assertEqual(knap_manual.n, knap_random_instance.n)

    def test_knapsack_type_checking(self):
        """
        Checks if the type-checking returns the right error.
        """

        # values type-check
        weights = [1, 2]
        weight_capacity = 5
        penalty = 0.1
        values_list = [(1, 2), {"test": "oh", "test1": "oh"}, np.array([1, 2])]

        for each_values in values_list:
            with self.assertRaises(TypeError) as e:
                Knapsack(each_values, weights, weight_capacity, penalty)
            self.assertEqual(
                "The input parameter, values, has to be a list", str(e.exception)
            )

        values_list = [["test", "oh"], [np.array(1), np.array(2)], [0.1, 0.5]]
        for each_values in values_list:
            with self.assertRaises(TypeError) as e:
                Knapsack(each_values, weights, weight_capacity, penalty)
            self.assertEqual(
                "The elements in values list must be of type int.", str(e.exception)
            )

        # weights type-check
        values = [1, 2]
        weight_capacity = 5
        penalty = 0.1
        weights_list = [(1, 2), {"test": "oh", "test1": "oh"}, np.array([1, 2])]

        for each_weights in weights_list:
            with self.assertRaises(TypeError) as e:
                Knapsack(values, each_weights, weight_capacity, penalty)
            self.assertEqual(
                "The input parameter, weights, has to be a list", str(e.exception)
            )

        weights_list = [["test", "oh"], [np.array(1), np.array(2)], [0.1, 0.5]]
        for each_weights in weights_list:
            with self.assertRaises(TypeError) as e:
                Knapsack(values, each_weights, weight_capacity, penalty)
            self.assertEqual(
                "The elements in weights list must be of type int.", str(e.exception)
            )

        # weight capacity type-check
        values = [1, 2]
        weights = [1, 2]
        weight_capacity_list = [0.5, np.array(1), np.array(0.5), "oh"]
        penalty = 0.1

        for each_weight_capacity in weight_capacity_list:
            with self.assertRaises(TypeError) as e:
                Knapsack(values, weights, each_weight_capacity, penalty)
            self.assertEqual(
                "The input parameter, weight_capacity, has to be of type int",
                str(e.exception),
            )

        with self.assertRaises(TypeError) as e:
            Knapsack(values, weights, -1, penalty)
        self.assertEqual(
            "The input parameter, weight_capacity, must be a positive integer greater than 0",
            str(e.exception),
        )

        # penalty capacity type-check
        values = [1, 2]
        weights = [1, 2]
        penalty_list = [np.array(1), np.array(0.5), "oh"]
        weight_capacity = 5

        for each_penalty in penalty_list:
            with self.assertRaises(TypeError) as e:
                Knapsack(values, weights, weight_capacity, each_penalty)
            self.assertEqual(
                "The input parameter, penalty, has to be of type float or int",
                str(e.exception),
            )

    # TESTING SLACKFREEKNAPSACK CLASS

    def test_slackfreeknapsack_terms_weights_constant(self):
        """Test that SlackFree Knapsack creates the correct QUBO problem"""

        values = [2, 4, 3, 5]
        weights = [3, 6, 9, 1]
        weight_capacity = 15
        n_qubits = len(values)
        penalty = 2 * max(values)
        slknap_terms = [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
            [0],
            [1],
            [2],
            [3],
        ]
        slknap_weights = [
            90.0,
            135.0,
            15.0,
            270.0,
            30.0,
            45.0,
            166.0,
            332.0,
            496.5,
            57.5,
        ]
        slknap_constant = 613.0

        slknapsack_prob_qubo = SlackFreeKnapsack(
            values, weights, weight_capacity, penalty
        ).qubo

        self.assertTrue(terms_list_equality(slknap_terms, slknapsack_prob_qubo.terms))
        self.assertEqual(slknap_weights, slknapsack_prob_qubo.weights)
        self.assertEqual(slknap_constant, slknapsack_prob_qubo.constant)
        self.assertEqual(n_qubits, slknapsack_prob_qubo.n)

    def test_slackfreeknapsack_random_problem(self):
        """Test random instance method of SlackFree Knapsack problem class"""

        rng = np.random.default_rng(1234)
        n_items = 5
        values = list(map(int, rng.integers(1, n_items, size=n_items)))
        weights = list(map(int, rng.integers(1, n_items, size=n_items)))
        weight_capacity = int(
            rng.integers(np.min(weights) * n_items, np.max(weights) * n_items)
        )
        penalty = 2 * np.max(values)

        slknap_manual = SlackFreeKnapsack(
            values, weights, weight_capacity, int(penalty)
        ).qubo

        slknap_random_instance = SlackFreeKnapsack.random_instance(
            n_items=n_items, seed=1234
        ).qubo

        self.assertTrue(
            terms_list_equality(slknap_manual.terms, slknap_random_instance.terms)
        )
        self.assertEqual(slknap_manual.weights, slknap_random_instance.weights)
        self.assertEqual(slknap_manual.constant, slknap_random_instance.constant)
        self.assertEqual(slknap_manual.n, slknap_random_instance.n)

    # TESTING MINIMUMVERTEXCOVER CLASS

    def test_mvc_terms_weights_constant(self):
        """Test terms,weights,constant of QUBO generated by MVC class"""

        mvc_terms = [
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
        mvc_weights = [1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 2.0, 2.0, 2.0, 3.25, 3.25]
        mvc_constant = 10.0

        gr = nx.generators.fast_gnp_random_graph(5, 0.8, seed=1234)
        mvc_prob = MinimumVertexCover(gr, field=1.0, penalty=5).qubo

        self.assertTrue(terms_list_equality(mvc_terms, mvc_prob.terms))
        self.assertEqual(mvc_weights, mvc_prob.weights)
        self.assertEqual(mvc_constant, mvc_prob.constant)

    def test_mvc_random_problem(self):
        """Test the random_instance method of MVC class"""
        mvc_terms = [
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
        mvc_weights = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 4.5, 4.5, 4.5, 7.0, 7.0]
        mvc_constant = 17.5

        mvc_prob_random = MinimumVertexCover.random_instance(
            n_nodes=5, edge_probability=0.8, seed=1234
        ).qubo

        self.assertTrue(terms_list_equality(mvc_terms, mvc_prob_random.terms))
        self.assertEqual(mvc_weights, mvc_prob_random.weights)
        self.assertEqual(mvc_constant, mvc_prob_random.constant)

    def test_mvc_type_checking(self):
        """
        Checks if the type-checking returns the right error.
        """

        # graph type-check
        graph_list = [(1, 2), {"node1": 1, "node2": 2}, np.array([1, 2])]
        field = 0.1
        penalty = 0.1

        for each_graph in graph_list:
            with self.assertRaises(TypeError) as e:
                MinimumVertexCover(each_graph, field, penalty)
            self.assertEqual(
                "Input problem graph must be a networkx Graph.", str(e.exception)
            )

        # field capacity type-check
        graph = nx.circulant_graph(6, [1])
        field_list = [np.array(1), np.array(0.5), "oh"]
        penalty = 0.1

        for each_field in field_list:
            with self.assertRaises(TypeError) as e:
                MinimumVertexCover(graph, each_field, penalty)
            self.assertEqual(
                "The input parameter, field, has to be of type float or int",
                str(e.exception),
            )

        # penalty capacity type-check
        graph = nx.circulant_graph(6, [1])
        field = 0.1
        penalty_list = [np.array(1), np.array(0.5), "oh"]

        for each_penalty in penalty_list:
            with self.assertRaises(TypeError) as e:
                MinimumVertexCover(graph, field, each_penalty)
            self.assertEqual(
                "The input parameter, penalty, has to be of type float or int",
                str(e.exception),
            )

    # TESTING TSP PROBLEM CLASS

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
        for (u, v) in G.edges():
            G[u][v]["weight"] = "a"

        with self.assertRaises(TypeError) as e:
            TSP(G=G)
        self.assertEqual(
            "The edge weights must be of type float or int", str(e.exception)
        )

        for (u, v) in G.edges():
            G[u][v]["weight"] = -2.0

        with self.assertRaises(ValueError) as e:
            TSP(G=G)
        self.assertEqual("Edge weights should be positive", str(e.exception))

    # TESTING SHORTESTPATH PROBLEM CLASS

    def test_shortestpath_terms_weights_constant(self):
        """Test terms,weights,constant of QUBO generated by Shortest Path class"""

        sp_terms = [
            [0],
            [1],
            [2],
            [3],
            [1],
            [1, 2],
            [2, 1],
            [2],
            [2],
            [2, 3],
            [3, 2],
            [3],
            [0],
            [0, 1],
            [1],
            [1, 3],
            [0, 3],
            [3, 1],
            [3],
        ]
        sp_weights = [1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 4, -4, 1, 1, -4, 1, 1]
        conv_sp_terms = [
            [1, 2],
            [2, 1],
            [2, 3],
            [3, 2],
            [0, 1],
            [1, 3],
            [0, 3],
            [3, 1],
            [0],
            [1],
            [2],
            [3],
            [],
        ]
        conv_sp_weights = [
            0.25,
            0.25,
            0.25,
            0.25,
            -1.0,
            0.25,
            -1.0,
            0.25,
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            2.5,
        ]
        sp_qubo_terms = [[1, 2], [2, 3], [0, 1], [1, 3], [0, 3], [0], [1], [2], [3]]
        sp_qubo_weights = [0.5, 0.5, -1.0, 0.5, -1.0, -0.5, -0.5, -0.5, -0.5]
        sp_qubo_constant = 2.5

        gr = nx.generators.fast_gnp_random_graph(3, 1, seed=1234)
        for (u, v) in gr.edges():
            gr.edges[u, v]["weight"] = 1
        for w in gr.nodes():
            gr.nodes[w]["weight"] = 1
        source, dest = 0, 2
        sp = ShortestPath(gr, source, dest)
        n_variables = sp.G.number_of_nodes() + sp.G.number_of_edges() - 2
        bin_terms, bin_weights = sp.terms_and_weights()
        terms, weights = QUBO.convert_qubo_to_ising(n_variables, bin_terms, bin_weights)
        qubo = sp.qubo

        self.assertTrue(terms_list_equality(bin_terms, sp_terms))
        self.assertEqual(list(bin_weights), sp_weights)
        self.assertTrue(terms_list_equality(terms, conv_sp_terms))
        self.assertEqual(list(weights), conv_sp_weights)
        self.assertTrue(terms_list_equality(sp_qubo_terms, qubo.terms))
        self.assertEqual(sp_qubo_weights, qubo.weights)
        self.assertEqual(sp_qubo_constant, qubo.constant)

    def test_shortestpath_random_instance(self):
        """Test random instance method of Shortest Path problem class"""
        sp_rand_terms = [[1, 2], [2, 3], [0, 1], [1, 3], [0, 3], [0], [1], [2], [3]]
        sp_rand_weights = [0.5, 0.5, -1.0, 0.5, -1.0, -0.5, -0.5, -0.5, -0.5]
        sp_rand_constant = 2.5

        gr = nx.generators.fast_gnp_random_graph(3, 1, seed=1234)
        for (u, v) in gr.edges():
            gr.edges[u, v]["weight"] = 1.0
        for w in gr.nodes():
            gr.nodes[w]["weight"] = 1.0
        sp_prob = ShortestPath.random_instance(
            n_nodes=3, edge_probability=1, seed=1234, source=0, dest=2
        ).qubo

        self.assertTrue(terms_list_equality(sp_rand_terms, sp_prob.terms))
        self.assertEqual(sp_rand_weights, sp_prob.weights)
        self.assertEqual(sp_rand_constant, sp_prob.constant)

        self.assertEqual(sp_prob.terms, ShortestPath(gr, 0, 2).qubo.terms)
        self.assertEqual(sp_prob.weights, ShortestPath(gr, 0, 2).qubo.weights)
        self.assertEqual(sp_prob.constant, ShortestPath(gr, 0, 2).qubo.constant)

    def test_assertion_error(self):
        def test_assertion_fn():
            n_row = 1
            n_col = 1

            G = nx.triangular_lattice_graph(n_row, n_col)
            G = nx.convert_node_labels_to_integers(G)
            G.remove_edges_from(nx.selfloop_edges(G))

            node_weights = np.round(np.random.rand(len(G.nodes())), 3)
            edge_weights = np.round(np.random.rand(len(G.edges())), 3)

            node_dict = dict(zip(list(G.nodes()), node_weights))
            edge_dict = dict(zip(list(G.edges()), edge_weights))

            nx.set_edge_attributes(G, values=edge_dict, name="weight")
            nx.set_node_attributes(G, values=node_dict, name="weight")

            shortest_path_problem = ShortestPath(G, 0, -1)
            shortest_path_qubo = shortest_path_problem.qubo

        self.assertRaises(Exception, test_assertion_fn)

    def __generate_random_problems(self):
        problems_random_instances = {
            "tsp": TSP.random_instance(n_cities=randint(2, 15)),
            "number_partition": NumberPartition.random_instance(
                n_numbers=randint(2, 15)
            ),
            "maximum_cut": MaximumCut.random_instance(
                n_nodes=randint(2, 15), edge_probability=random()
            ),
            "knapsack": Knapsack.random_instance(n_items=randint(2, 15)),
            "slack_free_knapsack": SlackFreeKnapsack.random_instance(
                n_items=randint(2, 15)
            ),
            "minimum_vertex_cover": MinimumVertexCover.random_instance(
                n_nodes=randint(2, 15), edge_probability=random()
            ),
            "shortest_path": ShortestPath.random_instance(
                n_nodes=randint(3, 15), edge_probability=random()
            ),
            "bin_packing": BinPacking.random_instance(
            ),
        }
        qubo_random_instances = {
            k: v.qubo for k, v in problems_random_instances.items()
        }
        qubo_random_instances["generic_qubo"] = QUBO.random_instance(randint(2, 15))

        return problems_random_instances, qubo_random_instances

    def test_problem_instance(self):
        """
        Test problem instance method of the QUBO class.
        From the random instance of all the different problems, we generate the QUBO problem out of it and then we check if the problem instance attribute is correct, by comparing the keys of the problem instance with the expected keys.
        """

        _, qubos = self.__generate_random_problems()

        expected_keys = {
            "tsp": ["problem_type", "n_cities", "G", "A", "B"],
            "number_partition": ["problem_type", "numbers", "n_numbers"],
            "maximum_cut": ["problem_type", "G"],
            "knapsack": [
                "problem_type",
                "values",
                "weights",
                "weight_capacity",
                "penalty",
                "n_items",
            ],
            "slack_free_knapsack": [
                "problem_type",
                "values",
                "weights",
                "weight_capacity",
                "penalty",
                "n_items",
            ],
            "minimum_vertex_cover": ["problem_type", "G", "field", "penalty"],
            "shortest_path": ["problem_type", "G", "source", "dest"],
            "bin_packing":["problem_type", "weights", "weight_capacity","penalty",
                           "n_items", "method", "simplifications", "n_bins",
                           'min_bins', 'solution'],
            "generic_qubo": ["problem_type"],
        }

        for k, v in qubos.items():
            assert (
                list(v.problem_instance.keys()) == expected_keys[k]
            ), "Problem instance keys are not correct for problem type {}".format(k)
            assert (
                k == v.problem_instance["problem_type"]
            ), "Problem type is not correct for problem type {}".format(k)

    def test_problem_from_instance_dict(self):
        """
        Test problem from instance method of the problem class.
        """
        problem_mapper = {
            "generic_qubo": QUBO,
            "tsp": TSP,
            "number_partition": NumberPartition,
            "maximum_cut": MaximumCut,
            "knapsack": Knapsack,
            "slack_free_knapsack": SlackFreeKnapsack,
            "minimum_vertex_cover": MinimumVertexCover,
            "shortest_path": ShortestPath,
            "bin_packing": BinPacking,
        }

        problems, qubos = self.__generate_random_problems()

        for type in qubos:
            if type == "generic_qubo":
                continue

            problem_instance = qubos[type].problem_instance.copy()

            problem = create_problem_from_dict(problem_instance)
            if problem_instance["problem_type"] == "bin_packing":
                print(problem.problem_instance)
                print()
                print(problems[type].problem_instance)
            assert (
                problem.problem_instance == problems[type].problem_instance
            ), "Problem from instance method is not correct for problem type {}".format(
                type
            )
            assert convert2serialize(problem) == convert2serialize(
                problems[type]
            ), "Problem from instance method is not correct for problem type {}".format(
                type
            )

    def test_qubo_from_dict(self):
        """
        Test qubo from dict method of the QUBO class.
        """

        _, qubos = self.__generate_random_problems()
        for _, qubo in qubos.items():

            qubo_dict = qubo.asdict()

            new_qubo = QUBO.from_dict(qubo_dict)

            for term, new_term in zip(qubo.terms, new_qubo.terms):
                assert set(term) == set(
                    new_term
                ), "QUBO from dict method is not correct for problem type {}, terms compared: {}, {}".format(
                    qubo.problem_instance["problem_type"], term, new_term
                )

                assert set(qubo.weights) == set(
                    new_qubo.weights
                ), "QUBO from dict method is not correct for problem type {}".format(
                    qubo.problem_instance["problem_type"]
                )

                for key in qubo.__dict__:
                    if key != "terms" and key != "weights":
                        assert (
                            qubo.__dict__[key] == new_qubo.__dict__[key]
                        ), "QUBO from dict method is not correct for problem type {}".format(
                            qubo.problem_instance["problem_type"]
                        )

    # TESTING BINPACKING CLASS
    def test_binpacking_terms_weights_constant(self):
        """Test that BinPacking creates a correct QUBO from the provided weights and terms"""
        
        terms = [[2, 3],
                 [4, 5],
                 [0, 2],
                 [0, 4],
                 [0, 6],
                 [2, 4],
                 [2, 6],
                 [4, 6],
                 [1, 3],
                 [1, 5],
                 [1, 7],
                 [3, 5],
                 [3, 7],
                 [5, 7],
                 [0],
                 [1],
                 [2],
                 [3],
                 [4],
                 [5],
                 [6],
                 [7]]

        weights = [1.5,
         1.5,
         -0.75,
         -1.0,
         -1.5,
         0.5,
         0.75,
         1.0,
         -0.75,
         -1.0,
         -1.5,
         0.5,
         0.75,
         1.0,
         1.25,
         1.25,
         -0.875,
         -0.875,
         -1.1666666666666665,
         -1.1666666666666665,
         -1.75,
         -1.75]
        constant = 10.083333333333332
        weights_list = [3, 4]
        weight_capacity = 6
        binpacking_prob_qubo = BinPacking(weights_list, weight_capacity, simplifications=False).qubo
        self.assertTrue(terms_list_equality(terms, binpacking_prob_qubo.terms))
        self.assertTrue(terms_list_isclose(weights, binpacking_prob_qubo.weights))
        self.assertTrue(np.isclose(constant, binpacking_prob_qubo.constant))

    def test_binpacking_terms_weights_constant_simplified(self):
        """Test that BinPacking creates a correct QUBO from the provided weights and terms"""
        
        terms = [[1, 2],
                 [1, 3],
                 [2, 3],
                 [4, 5],
                 [4, 6],
                 [5, 6],
                 [1, 4],
                 [1, 7],
                 [4, 7],
                 [2, 5],
                 [8, 2],
                 [8, 5],
                 [0, 3],
                 [0, 6],
                 [0, 9],
                 [3, 6],
                 [9, 3],
                 [9, 6],
                 [0],
                 [1],
                 [2],
                 [3],
                 [4],
                 [5],
                 [6],
                 [7],
                 [8],
                 [9]]

        weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.28, 0.35, 0.2, 0.4, 0.5, -0.4,
                    -0.5, -1.0, 0.2, 0.4, 0.5, 0.4, -1.08, -0.96, -1.36, -1.1, -0.95,
                    -1.45, -0.14, 0.1, -0.9]
        constant = 9.29
        weights_list = [3, 4, 5]
        weight_capacity = 10
        binpacking_prob_qubo = BinPacking(weights_list, weight_capacity).qubo

        self.assertTrue(terms_list_equality(terms, binpacking_prob_qubo.terms))
        self.assertTrue(terms_list_isclose(weights, binpacking_prob_qubo.weights))
        self.assertTrue(np.isclose(constant, binpacking_prob_qubo.constant))
        
    def test_binpacking_terms_weights_constant_unbalanced(self):
            """Test that BinPacking creates a correct QUBO from the provided weights and terms
            using the unbalanced penalization encoding"""
            terms = [[1, 2],
                     [1, 3],
                     [2, 3],
                     [4, 5],
                     [4, 6],
                     [5, 6],
                     [1, 4],
                     [2, 5],
                     [0, 3],
                     [0, 6],
                     [3, 6],
                     [0],
                     [1],
                     [2],
                     [3],
                     [4],
                     [5],
                     [6]]

            weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05, -0.1, -0.125,
                       0.05, -0.475, -0.97, -0.91, -1.01, -0.9625, -0.8875, -1.0125]
            constant = 6.8775
            
            weights_list = [3, 4, 5]
            weight_capacity = 10
            binpacking_prob_qubo = BinPacking(weights_list, weight_capacity, method="unbalanced").qubo

            self.assertTrue(terms_list_equality(terms, binpacking_prob_qubo.terms))
            self.assertTrue(terms_list_isclose(weights, binpacking_prob_qubo.weights))
            self.assertTrue(np.isclose(constant, binpacking_prob_qubo.constant))

    def test_binpacking_terms_penalizations_terms_unbalanced(self):
            """Test that BinPacking creates a correct QUBO from the provided weights and terms
            using the unbalanced penalization encoding given the penalization terms"""
            terms = [[1, 2], [0, 2], [0], [1], [2]]

            weights = [0.5, -0.25, -0.25, -0.0625, -0.125]
            constant = 1.953125
            
            weights_list = [3, 4]
            weight_capacity = 8
            penalty = [1, 1, 1]
            binpacking_prob_qubo = BinPacking(weights_list, weight_capacity, penalty=penalty, method="unbalanced", simplifications=True).qubo

            self.assertTrue(terms_list_equality(terms, binpacking_prob_qubo.terms))
            self.assertTrue(terms_list_isclose(weights, binpacking_prob_qubo.weights))
            self.assertTrue(np.isclose(constant, binpacking_prob_qubo.constant))

    def test_binpacking_terms_penalizations_terms_slack(self):
            """Test that BinPacking creates a correct QUBO from the provided weights and terms
            using the unbalanced penalization encoding"""
            terms = [[1, 2], [1, 3], [0, 2], [0, 4], [2, 4], [0], [1], [2], [3], [4]]

            weights = [0.5, 0.15625, -0.25, -0.5, 0.25, -0.25, 0.03125, -0.125, 0.0390625, -0.25]
            constant = 2.7890625
            
            weights_list = [3, 4]
            weight_capacity = 8
            penalty = [1]
            binpacking_prob_qubo = BinPacking(weights_list, weight_capacity, penalty=penalty, method="slack", simplifications=True).qubo

            self.assertTrue(terms_list_equality(terms, binpacking_prob_qubo.terms))
            self.assertTrue(terms_list_isclose(weights, binpacking_prob_qubo.weights))
            self.assertTrue(np.isclose(constant, binpacking_prob_qubo.constant))

    def test_binpacking_random_problem(self):
        """Test Bin Packing random instance method"""

        seed = 1234
        np.random.seed(seed)
        min_weight = 1
        max_weight = 7
        n_items = 3
        weight_capacity = 15
        weights = list(np.random.randint(min_weight, max_weight, n_items))
        binpacking_manual_prob = BinPacking(weights, weight_capacity).qubo

        binpacking_random_prob = BinPacking.random_instance(
            n_items=3, seed=seed, weight_capacity=weight_capacity
        ).qubo

        self.assertTrue(
            terms_list_equality(binpacking_manual_prob.terms, binpacking_random_prob.terms)
        )
        self.assertEqual(binpacking_manual_prob.weights, binpacking_random_prob.weights)
        self.assertEqual(binpacking_manual_prob.constant, binpacking_random_prob.constant)

    def test_binpacking_classical_sol(self):
        """Test the Bin Packing random instance method classical solution"""

        seed = 1234
        np.random.seed(seed)
        binpacking_sol = BinPacking.random_instance(
            n_items=3, seed=seed).classical_solution()
        
        sol = {'y_0': 1,
             'y_1': 0,
             'y_2': 0,
             'x_0_0': 1,
             'x_0_1': 0,
             'x_0_2': 0,
             'x_1_0': 1,
             'x_1_1': 0,
             'x_1_2': 0,
             'x_2_0': 1,
             'x_2_1': 0,
             'x_2_2': 0}

        self.assertEqual(binpacking_sol, sol)

    def test_binpacking_plot(self):
        """Test Bin Packing random instance method"""
        from matplotlib.pyplot import Figure
        seed = 1234
        binpacking_random_prob = BinPacking.random_instance(
            n_items=3, seed=seed
        )
        sol = {'y_0': 1,
         'y_1': 0,
         'y_2': 0,
         'x_0_0': 1,
         'x_0_1': 0,
         'x_0_2': 0,
         'x_1_0': 1,
         'x_1_1': 0,
         'x_1_2': 0,
         'x_2_0': 1,
         'x_2_1': 0,
         'x_2_2': 0}
        fig = binpacking_random_prob.plot_solution(sol)
        self.assertTrue(
            isinstance(fig, Figure)
        )

    def test_binpacking_method_checking(self):
        """
        Checks if the method-checking returns the right error.
        """
        weights = [3, 5, 7]
        weight_capacity = 15
        method = "random"
        with self.assertRaises(ValueError) as e:
            BinPacking(weights, weight_capacity, method=method)
        self.assertEqual(
            f"The method '{method}' is not a valid method. Choose between 'slack' and 'unbalanced'", str(e.exception)
        )

    def test_binpacking_random_problem_checking(self):
        """
        Checks if the random min_weight equal to max_weight returns the right error.
        """
        min_weight = 5
        max_weight = 5
        with self.assertRaises(ValueError) as e:
            BinPacking.random_instance(min_weight=min_weight, max_weight=max_weight)
        self.assertEqual(
            f"min_weight: {min_weight} must be < max_weight:{max_weight}", str(e.exception)
        )  

    def test_binpacking_classical_sol_checking(self):
        """
        Checks if the unfeasible classical solution returns the right error.
        """
        weights = [10, 10]
        weight_capacity = 8
        with self.assertRaises(ValueError) as e:
            BinPacking(weights=weights, weight_capacity=weight_capacity).classical_solution()
        self.assertEqual(
            'solution not found: integer infeasible', str(e.exception)
        )

    def test_binpacking_input_weights(self):
        """
        Checks if the unfeasible classical solution returns the right error.
        """
        weights = [10.1, 10]
        weight_capacity = 8
        with self.assertRaises(TypeError) as e:
            BinPacking(weights=weights, weight_capacity=weight_capacity).classical_solution()
        self.assertEqual(
            f"The weights must be integer numbers. Format {type(weights[0])} found.", str(e.exception)
        )

    def test_binpacking_input_weight_capacity(self):
        """
        Checks if the unfeasible classical solution returns the right error.
        """
        weights = [10, 10]
        weight_capacity = 8.1
        with self.assertRaises(TypeError) as e:
            BinPacking(weights=weights, weight_capacity=weight_capacity).classical_solution()
        self.assertEqual(
            f"The weight_capacity must be integer. Format {type(weight_capacity)} found.", str(e.exception)
        )

if __name__ == "__main__":
    unittest.main()
