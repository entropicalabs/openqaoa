import unittest
import numpy as np
from openqaoa.problems import BinPacking


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


class TestBinPacking(unittest.TestCase):
    """Tests for the BinPacking class"""
    def test_binpacking_terms_weights_constant(self):
        """Test that BinPacking creates a correct QUBO from the provided weights and terms"""

        terms = [
            [2, 3],
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
            [7],
        ]

        weights = [
            1.5,
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
            -1.75,
        ]
        constant = 10.083333333333332
        weights_list = [3, 4]
        weight_capacity = 6
        binpacking_prob_qubo = BinPacking(
            weights_list, weight_capacity, simplifications=False
        ).qubo
        self.assertTrue(terms_list_equality(terms, binpacking_prob_qubo.terms))
        self.assertTrue(terms_list_isclose(weights, binpacking_prob_qubo.weights))
        self.assertTrue(np.isclose(constant, binpacking_prob_qubo.constant))

    def test_binpacking_terms_weights_constant_simplified(self):
        """Test that BinPacking creates a correct QUBO from the provided weights and terms"""

        terms = [
            [1, 2],
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
            [9],
        ]

        weights = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.2,
            0.28,
            0.35,
            0.2,
            0.4,
            0.5,
            -0.4,
            -0.5,
            -1.0,
            0.2,
            0.4,
            0.5,
            0.4,
            -1.08,
            -0.96,
            -1.36,
            -1.1,
            -0.95,
            -1.45,
            -0.14,
            0.1,
            -0.9,
        ]
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
        terms = [
            [1, 2],
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
            [6],
        ]

        weights = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.05,
            0.05,
            -0.1,
            -0.125,
            0.05,
            -0.475,
            -0.97,
            -0.91,
            -1.01,
            -0.9625,
            -0.8875,
            -1.0125,
        ]
        constant = 6.8775

        weights_list = [3, 4, 5]
        weight_capacity = 10
        binpacking_prob_qubo = BinPacking(
            weights_list, weight_capacity, method="unbalanced"
        ).qubo

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
        binpacking_prob_qubo = BinPacking(
            weights_list,
            weight_capacity,
            penalty=penalty,
            method="unbalanced",
            simplifications=True,
        ).qubo

        self.assertTrue(terms_list_equality(terms, binpacking_prob_qubo.terms))
        self.assertTrue(terms_list_isclose(weights, binpacking_prob_qubo.weights))
        self.assertTrue(np.isclose(constant, binpacking_prob_qubo.constant))

    def test_binpacking_terms_penalizations_terms_slack(self):
        """Test that BinPacking creates a correct QUBO from the provided weights and terms
        using the unbalanced penalization encoding"""
        terms = [[1, 2], [1, 3], [0, 2], [0, 4], [2, 4], [0], [1], [2], [3], [4]]

        weights = [
            0.5,
            0.15625,
            -0.25,
            -0.5,
            0.25,
            -0.25,
            0.03125,
            -0.125,
            0.0390625,
            -0.25,
        ]
        constant = 2.7890625

        weights_list = [3, 4]
        weight_capacity = 8
        penalty = [1]
        binpacking_prob_qubo = BinPacking(
            weights_list,
            weight_capacity,
            penalty=penalty,
            method="slack",
            simplifications=True,
        ).qubo

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
            terms_list_equality(
                binpacking_manual_prob.terms, binpacking_random_prob.terms
            )
        )
        self.assertEqual(binpacking_manual_prob.weights, binpacking_random_prob.weights)
        self.assertEqual(
            binpacking_manual_prob.constant, binpacking_random_prob.constant
        )

    def test_binpacking_classical_sol(self):
        """Test the Bin Packing random instance method classical solution"""

        seed = 1234
        np.random.seed(seed)
        binpacking_sol = BinPacking.random_instance(
            n_items=3, seed=seed
        ).classical_solution()

        sol = {
            "y_0": 1,
            "y_1": 0,
            "y_2": 0,
            "x_0_0": 1,
            "x_0_1": 0,
            "x_0_2": 0,
            "x_1_0": 1,
            "x_1_1": 0,
            "x_1_2": 0,
            "x_2_0": 1,
            "x_2_1": 0,
            "x_2_2": 0,
        }

        self.assertEqual(binpacking_sol, sol)

    def test_binpacking_plot(self):
        """Test Bin Packing random instance method"""
        from matplotlib.pyplot import Figure

        seed = 1234
        binpacking_random_prob = BinPacking.random_instance(n_items=3, seed=seed)
        sol = {
            "y_0": 1,
            "y_1": 0,
            "y_2": 0,
            "x_0_0": 1,
            "x_0_1": 0,
            "x_0_2": 0,
            "x_1_0": 1,
            "x_1_1": 0,
            "x_1_2": 0,
            "x_2_0": 1,
            "x_2_1": 0,
            "x_2_2": 0,
        }
        fig = binpacking_random_prob.plot_solution(sol)
        self.assertTrue(isinstance(fig, Figure))

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
            f"The method '{method}' is not a valid method. Choose between 'slack' and 'unbalanced'",
            str(e.exception),
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
            f"min_weight: {min_weight} must be < max_weight:{max_weight}",
            str(e.exception),
        )

    def test_binpacking_classical_sol_checking(self):
        """
        Checks if the unfeasible classical solution returns the right error.
        """
        weights = [10, 10]
        weight_capacity = 8
        with self.assertRaises(ValueError) as e:
            BinPacking(
                weights=weights, weight_capacity=weight_capacity
            ).classical_solution()
        self.assertEqual("solution not found: integer infeasible", str(e.exception))

    def test_binpacking_input_weights(self):
        """
        Checks if the unfeasible classical solution returns the right error.
        """
        weights = [10.1, 10]
        weight_capacity = 8
        with self.assertRaises(TypeError) as e:
            BinPacking(
                weights=weights, weight_capacity=weight_capacity
            ).classical_solution()
        self.assertEqual(
            f"The weights must be integer numbers. Format {type(weights[0])} found.",
            str(e.exception),
        )

    def test_binpacking_input_weight_capacity(self):
        """
        Checks if the unfeasible classical solution returns the right error.
        """
        weights = [10, 10]
        weight_capacity = 8.1
        with self.assertRaises(TypeError) as e:
            BinPacking(
                weights=weights, weight_capacity=weight_capacity
            ).classical_solution()
        self.assertEqual(
            f"The weight_capacity must be integer. Format {type(weight_capacity)} found.",
            str(e.exception),
        )


if __name__ == "__main__":
    unittest.main()
