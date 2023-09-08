import unittest
import numpy as np
from openqaoa.problems import Knapsack, SlackFreeKnapsack


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


class TestKnapsack(unittest.TestCase):
    """Tests for Knapsack class"""

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


class TestSlackFreeKnapsack(unittest.TestCase):
    """Tests for SlackFreeKnapsack class"""

    def test_slackfreeknapsack_terms_weights_constant(self):
        """Test that SlackFreeKnapsack creates the correct QUBO problem"""

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
        """Test random instance method of SlackFreeKnapsack problem class"""

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


if __name__ == "__main__":
    unittest.main()
