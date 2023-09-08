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
            [0, 7],
            [0, 8],
            [2, 4],
            [2, 6],
            [2, 7],
            [8, 2],
            [4, 6],
            [4, 7],
            [8, 4],
            [6, 7],
            [8, 6],
            [8, 7],
            [1, 3],
            [1, 5],
            [1, 9],
            [1, 10],
            [1, 11],
            [3, 5],
            [9, 3],
            [10, 3],
            [3, 11],
            [9, 5],
            [10, 5],
            [11, 5],
            [9, 10],
            [9, 11],
            [10, 11],
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
            [10],
            [11],
        ]

        weights = [
            1.5,
            1.5,
            -27.0,
            -36.0,
            -9.0,
            -18.0,
            -27.0,
            18.0,
            4.5,
            9.0,
            13.5,
            6.0,
            12.0,
            18.0,
            3.0,
            4.5,
            9.0,
            -27.0,
            -36.0,
            -9.0,
            -18.0,
            -27.0,
            18.0,
            4.5,
            9.0,
            13.5,
            6.0,
            12.0,
            18.0,
            3.0,
            4.5,
            9.0,
            62.5,
            62.5,
            -31.5,
            -31.5,
            -42.0,
            -42.0,
            -10.5,
            -21.0,
            -31.5,
            -10.5,
            -21.0,
            -31.5,
        ]
        constant = 190.0
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
            [8, 1],
            [1, 9],
            [4, 7],
            [8, 4],
            [9, 4],
            [8, 7],
            [9, 7],
            [8, 9],
            [2, 5],
            [2, 10],
            [2, 11],
            [2, 12],
            [2, 13],
            [10, 5],
            [11, 5],
            [12, 5],
            [5, 13],
            [10, 11],
            [10, 12],
            [10, 13],
            [11, 12],
            [11, 13],
            [12, 13],
            [0, 3],
            [0, 6],
            [0, 14],
            [0, 15],
            [0, 16],
            [0, 17],
            [3, 6],
            [3, 14],
            [3, 15],
            [16, 3],
            [17, 3],
            [6, 14],
            [6, 15],
            [16, 6],
            [17, 6],
            [14, 15],
            [16, 14],
            [17, 14],
            [16, 15],
            [17, 15],
            [16, 17],
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
            [10],
            [11],
            [12],
            [13],
            [14],
            [15],
            [16],
            [17],
        ]

        weights = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            20.0,
            4.0,
            8.0,
            16.0,
            5.0,
            10.0,
            20.0,
            2.0,
            4.0,
            8.0,
            20.0,
            4.0,
            8.0,
            16.0,
            12.0,
            5.0,
            10.0,
            20.0,
            15.0,
            2.0,
            4.0,
            3.0,
            8.0,
            6.0,
            12.0,
            -40.0,
            -50.0,
            -10.0,
            -20.0,
            -40.0,
            -30.0,
            20.0,
            4.0,
            8.0,
            16.0,
            12.0,
            5.0,
            10.0,
            20.0,
            15.0,
            2.0,
            4.0,
            3.0,
            8.0,
            6.0,
            12.0,
            89.5,
            -9.0,
            3.0,
            -37.0,
            -11.0,
            4.0,
            -46.0,
            -2.0,
            -4.0,
            -8.0,
            1.0,
            2.0,
            4.0,
            3.0,
            -9.0,
            -18.0,
            -36.0,
            -27.0,
        ]

        constant = 201.5
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
            5.0,
            5.0,
            -10.0,
            -12.5,
            5.0,
            -2.5,
            3.8000000000000007,
            9.8,
            -0.20000000000000018,
            5.0,
            12.5,
            0.0,
        ]
        constant = 51.89999999999999

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

        weights = [0.5, -16.0, -12.5, 10.0, 6.0]
        constant = 34.0

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
        terms = [
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [3, 4],
            [3, 5],
            [4, 5],
            [0, 2],
            [0, 6],
            [0, 7],
            [0, 8],
            [0, 9],
            [2, 6],
            [2, 7],
            [8, 2],
            [9, 2],
            [6, 7],
            [8, 6],
            [9, 6],
            [8, 7],
            [9, 7],
            [8, 9],
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
            0.5,
            2.0,
            4.0,
            4.0,
            1.0,
            1.0,
            2.0,
            -16.0,
            -4.0,
            -8.0,
            -16.0,
            -4.0,
            2.0,
            4.0,
            8.0,
            2.0,
            1.0,
            2.0,
            0.5,
            4.0,
            1.0,
            2.0,
            15.5,
            2.0,
            -8.0,
            0.5,
            1.0,
            1.0,
            -2.0,
            -4.0,
            -8.0,
            -2.0,
        ]
        constant = 38.0

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
        rng = np.random.default_rng(seed)
        min_weight = 1
        max_weight = 7
        n_items = 3
        weight_capacity = 15
        weights = list(rng.integers(min_weight, max_weight, n_items, dtype=int))
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
        binpacking_sol = BinPacking.random_instance(
            n_items=3, seed=seed
        ).classical_solution()

        sol = {
            "y_0": 1,
            "y_1": 1,
            "y_2": 0,
            "x_0_0": 1,
            "x_0_1": 0,
            "x_0_2": 0,
            "x_1_0": 0,
            "x_1_1": 1,
            "x_1_2": 0,
            "x_2_0": 0,
            "x_2_1": 1,
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
