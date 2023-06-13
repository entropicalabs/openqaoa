import unittest
from random import randint, random
import numpy as np
from openqaoa.problems import (
    NumberPartition,
    QUBO,
    TSP,
    Knapsack,
    ShortestPath,
    SlackFreeKnapsack,
    MaximumCut,
    MinimumVertexCover,
    VRP,
    PortfolioOptimization,
    MIS,
    BinPacking,
)
from openqaoa.utilities import convert2serialize
from openqaoa.problems.helper_functions import create_problem_from_dict


class TestQUBO(unittest.TestCase):
    """Tests for the QUBO class"""
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
            "vehicle_routing": VRP.random_instance(),
            "maximal_independent_set": MIS.random_instance(
                n_nodes=randint(3, 15), edge_probability=random()
            ),
            "bin_packing": BinPacking.random_instance(),
            "portfolio_optimization": PortfolioOptimization.random_instance(),
        }
        qubo_random_instances = {
            k: v.qubo for k, v in problems_random_instances.items()
        }
        qubo_random_instances["generic_qubo"] = QUBO.random_instance(randint(2, 15))
        return problems_random_instances, qubo_random_instances

    def test_problem_instance(self):
        """
        Test problem instance method of the QUBO class.
        From the random instance of all the different problems, we generate the
        QUBO problem out of it and then we check if the problem instance
        attribute is correct, by comparing the keys of the problem instance
        with the expected keys.
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
            "vehicle_routing": [
                "problem_type",
                "G",
                "pos",
                "n_vehicles",
                "depot",
                "subtours",
                "method",
                "penalty",
            ],
            "maximal_independent_set": ["problem_type", "G", "penalty"],
            "bin_packing": [
                "problem_type",
                "weights",
                "weight_capacity",
                "penalty",
                "n_items",
                "method",
                "simplifications",
                "n_bins",
                "min_bins",
                "solution",
            ],
            "portfolio_optimization": [
                "problem_type",
                "mu",
                "sigma",
                "risk_factor",
                "penalty",
                "num_assets",
                "budget",
            ],
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
        problems, qubos = self.__generate_random_problems()

        for type in qubos:
            if type == "generic_qubo":
                continue

            problem_instance = qubos[type].problem_instance.copy()

            problem = create_problem_from_dict(problem_instance)

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


if __name__ == "__main__":
    unittest.main()
