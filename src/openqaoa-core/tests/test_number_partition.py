import unittest
import numpy as np
from openqaoa.problems import NumberPartition


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

class TestNumberPartition(unittest.TestCase):
    """Tests for NumberPartition class"""
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


if __name__ == "__main__":
    unittest.main()
