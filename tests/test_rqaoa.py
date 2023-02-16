import numpy as np
import unittest
import networkx as nx

from openqaoa import RQAOA
from openqaoa.qaoa_components import Hamiltonian
from openqaoa.algorithms.rqaoa.rqaoa_utils import *
from openqaoa.problems import MaximumCut, QUBO
from openqaoa.backends import create_device

"""
Unittest based testing of current implementation of the RQAOA Algorithm
"""


class TestingRQAOA(unittest.TestCase):
    def test_find_parent(self):
        """
        Test of the find_parent function which backtracks the spin_map dictionary to obtain
        the final constraint used to eliminate a given spin.

        The test consists in finding the correct parents and factors of the constraint
        for a given example.
        """

        # Spin map example without correct final dependencies
        spin_map = dict(
            {
                (0, (1, 0)),
                (1, (-1, 3)),
                (2, (1, 5)),
                (3, (-1, 2)),
                (4, (1, 4)),
                (5, (1, 4)),
            }
        )

        # Solution to the problem
        parents = {0: 0, 1: 4, 2: 4, 3: 4, 4: 4, 5: 4}  # spin:parent_spin
        factors = [1, 1, 1, -1, 1, 1]

        comp_parents = {}
        comp_factors = []

        # For each spin compute parent and connecting factor and store them
        for spin in sorted(spin_map.keys()):
            comp_parent, comp_factor = find_parent(spin_map, spin)
            comp_parents.update({spin: comp_parent})
            comp_factors.append(comp_factor)

        # Test function result
        assert np.allclose(
            list(comp_parents.values()), list(parents.values())
        ), f"Computed parent spins are incorrect"
        assert np.allclose(
            comp_factors, factors
        ), f"Computed constraint factors are incorrect"

    def test_spin_mapping(self):
        """
        Test of the function that generates the spin_map dictionary containing all the elimination
        rules.

        The test consists in generating the correct spin_map and set of max terms and costs in
        its final form (accounting for the dependency that determines the elimination) for a
        given example.
        """

        ## Problem definition

        # Number of qubits
        n_qubits = 20

        # Terms and weights of the graph
        edges = [(i, j) for j in range(n_qubits) for i in range(j)]
        weights = [1 for _ in range(len(edges))]

        # Hyperparameters
        problem = QUBO(n=n_qubits, terms=edges, weights=weights)

        ## Testing

        # Example case of the maximal terms and values extracted from the expectation values
        max_terms_and_stats = dict(
            {
                (4,): 0.99,
                (0, 1): -0.89,
                (0, 9): 0.77,
                (1, 9): -0.73,
                (17,): -0.5,
                (2, 4): 0.32,
                (5, 4): 0.29,
                (16,): -0.8,
                (10,): 0.81,
                (16, 10): 0.4,
                (19, 14): -0.47,
            }
        )

        # Exact solution to the problem
        correct_spin_map = dict(
            {
                (0, (1, 0)),
                (1, (-1, 0)),
                (2, (1, None)),
                (3, (1, 3)),
                (4, (1, None)),
                (5, (1, None)),
                (6, (1, 6)),
                (7, (1, 7)),
                (8, (1, 8)),
                (9, (1, 0)),
                (10, (1, None)),
                (11, (1, 11)),
                (12, (1, 12)),
                (13, (1, 13)),
                (14, (1, 14)),
                (15, (1, 15)),
                (16, (-1, None)),
                (17, (-1, None)),
                (18, (1, 18)),
                (19, (-1, 14)),
            }
        )

        # Compute the spin_map and final constraints from the function
        spin_map = spin_mapping(problem, max_terms_and_stats)

        # Check both outputs contain the same number of keys as the correct solution
        assert len(correct_spin_map) == len(
            spin_map
        ), f"Computed spin_map has incorrect length"

        # Test the spin_map matches the correct solution
        for key in correct_spin_map.keys():
            assert (
                correct_spin_map[key][0] == spin_map[key][0]
            ), f"Computed spin_map contains incorrect factor"
            assert (
                correct_spin_map[key][1] == spin_map[key][1]
            ), f"Computed spin_map contains incorrect parent spin"

    def test_max_terms(self):
        """
        Test of the function that selects spin pairs or singlets are used for elimination,
        given the specific elimination number.

        The test consists in finding the correct pairs and singles to eliminate, according
        to the adaptive scheme, for a given example.
        """

        # Maximum number of singles/pairs to eliminate
        n_elim = 5

        # Set of single spin expectation values
        exp_vals_z = np.array([0.33, -0.21, -0.9, 0.06, -0.78])

        # Correlation matrix
        corr_mat = np.array(
            [
                [0.0, 0.01, -0.64, 0.69, 0.48],
                [0.0, 0.0, -0.99, -0.27, 0.03],
                [0.0, 0.0, 0.0, 1.0, -0.22],
                [0.0, 0.0, 0.0, 0.0, 0.37],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        # Correct solution
        max_tc = dict(
            {(2, 3): 1.0, (1, 2): -0.99, (2,): -0.9, (4,): -0.78, (0, 3): 0.69}
        )

        # Computed solution using the function
        comp_max_tc = max_terms(exp_vals_z, corr_mat, n_elim)

        # Confirm the computed solution has same number of items as correct one
        assert len(max_tc) == len(
            comp_max_tc
        ), f"Computed set of singlets/correlations contains incorrect number of elements"

        # Test the function has obtain the correct singlets/pairs with associated values
        for key in max_tc.keys():
            assert (
                max_tc[key] == comp_max_tc[key]
            ), f"Computed set of singlets/correlations contains incorrect values"

    def test_ada_max_terms(self):
        """
        Test of the function that adaptively selects spin pairs or singlets are used for elimination.

        The test consists in finding the correct pairs and singles to eliminate, according
        to the adaptive scheme, for a given example.
        """

        # Maximum number of singles/pairs allowed to be considered for elimination
        n_max = 3

        # Set of single spin expectation values
        exp_vals_z = np.array([0.33, -0.21, -0.9, 0.06, -0.19])

        # Correlation matrix
        corr_mat = np.array(
            [
                [0.0, 0.01, -0.64, 0.69, 0.48],
                [0.0, 0.0, -0.99, -0.27, 0.03],
                [0.0, 0.0, 0.0, 1.0, -0.22],
                [0.0, 0.0, 0.0, 0.0, 0.37],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        # Correct solution
        max_tc = dict({(2, 3): 1.0, (1, 2): -0.99, (2,): -0.9})

        # Computed solution using the function
        comp_max_tc = ada_max_terms(exp_vals_z, corr_mat, n_max)

        # Confirm the computed solution has same number of items as correct one
        assert len(max_tc) == len(
            comp_max_tc
        ), f"Computed set of singlets/correlations contains incorrect number of elements"

        # Test the function has obtain the correct singlets/pairs with associated values
        for key in max_tc.keys():
            assert (
                max_tc[key] == comp_max_tc[key]
            ), f"Computed set of singlets/correlations contains incorrect values"

    def test_final_solution(self):
        """
        Test the function that reconstructs the final solution by backtracking the
        elimination history and computing the energy of the final states.

        The test consists in reconstructing a set of states for a given elimination history
        amnd computing their energies.
        """

        ## Problem definition

        # Number of qubits
        n_qubits = 10

        # Terms and weights of the graph
        edges = [(i, i + 1) for i in range(n_qubits - 1)] + [(0, n_qubits - 1)] + [(0,)]
        weights = [1 for _ in range(len(edges))]

        # Hamiltonian
        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant=0)

        ## Testing

        # Trial elimination history and ouput of classical solver
        max_terms_and_stats_list = [
            [
                {"singlet": (1,), "bias": 1.0},
                {"pair": (0, 9), "correlation": -1.0},
            ],
            [
                {"pair": (0, 1), "correlation": 1.0},
                {"pair": (0, 7), "correlation": 1.0},
            ],
            [
                {"pair": (0, 1), "correlation": -1.0},
                {"singlet": (5,), "bias": 1.0},
            ],
            [{"pair": (0, 1), "correlation": 1.0}],
        ]

        classical_states = [[1, 0, 1]]

        # Correct solutions
        states = ["1010101010"]
        energies = [-11]
        correct_full_solution = dict(zip(states, energies))

        # Compute solutions
        full_solution = final_solution(
            max_terms_and_stats_list, classical_states, hamiltonian
        )

        # Test the computed solutions
        assert (
            correct_full_solution == full_solution
        ), f"Solution was not computed correctly"

    def test_problem_from_dict(self):
        """
        Test the function that computes a calssical Hamiltonian from a given graph, accounting for approriate
        labelling of the nodes and edges.

        The test consists in generating the correct QUBO problem for a given graph dictionary.
        """

        # Trial graph
        input_dict = dict(
            {
                (): 10,
                (1,): 1,
                (2,): -1,
                (6,): 4,
                (1, 2): 1,
                (2, 5): 2,
                (10, 14): 3,
                (6, 9): 4,
                (6, 14): 5,
                (5, 6): 6,
            }
        )

        # Correct hamiltonian
        correct_dict = dict(
            {
                (0,): 1,
                (1,): -1,
                (3,): 4,
                (0, 1): 1,
                (1, 2): 2,
                (2, 3): 6,
                (3, 4): 4,
                (3, 6): 5,
                (5, 6): 3,
            }
        )
        hamiltonian = Hamiltonian.classical_hamiltonian(
            list(correct_dict.keys()), list(correct_dict.values()), constant=10
        )
        hamiltonian_dict = {
            term.qubit_indices: coeff
            for term, coeff in zip(hamiltonian.terms, hamiltonian.coeffs)
        }

        # Compute hamiltonian from input graph
        comp_problem = problem_from_dict(input_dict)
        comp_hamiltonian = comp_problem.hamiltonian
        comp_hamiltonian_dict = {
            term.qubit_indices: coeff
            for term, coeff in zip(comp_hamiltonian.terms, comp_hamiltonian.coeffs)
        }

        # Test computed Hamiltonian contains the correct terms
        assert (
            hamiltonian_dict == comp_hamiltonian_dict
        ), f"Terms and coefficients in the computed Hamiltonian are incorrect"

        # Test computed Hamiltonian contains the correct terms
        assert np.allclose(
            hamiltonian.constant, comp_hamiltonian.constant
        ), f"Constant in the computed Hamiltonian is incorrect"

    def test_redefine_problem(self):
        """
        Test the function that computes the new QUBO for a reduced problem, given the original
        QUBO encoding the problem and a set of elimination rules via the spin_map.

        The test consists in computing the new Hamiltonian from a given one and a set of elimination rules.
        """

        ## Problem definition

        # Number of qubits
        n_qubits = 10

        # Edges and weights of the graph
        input_edges = [(i, i + 1) for i in range(n_qubits - 1)] + [(0, n_qubits - 1)]
        input_weights = [1 for _ in range(len(input_edges))]

        # Input problem
        input_problem = QUBO(n_qubits, input_edges, input_weights)

        # Input spin map (elimination rules)
        spin_map = dict(
            {
                (0, (1, 0)),
                (1, (-1, 0)),
                (2, (1, 2)),
                (3, (1, 3)),
                (4, (1, 2)),
                (5, (1, 2)),
                (6, (1, 6)),
                (7, (1, 7)),
                (8, (1, 8)),
                (9, (1, 0)),
            }
        )

        # Compute new problem
        comp_problem, _ = redefine_problem(input_problem, spin_map)

        # Compute the new hamiltonian
        comp_hamiltonian = comp_problem.hamiltonian

        ## Testing (Comparing the new hamiltonian with the correct one)

        # Correct edges, weights and hamiltonian for the reduced problem
        edges = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (0, 5)]
        weights = [-1, 2, 1, 1, 1, 1]

        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant=0)

        # Test computed Hamiltonian contains the correct terms
        assert (
            hamiltonian.terms == comp_hamiltonian.terms
        ), f"Terms in the computed Hamiltonian are incorrect"

        # Test computed Hamiltonian contains the correct terms
        assert np.allclose(
            hamiltonian.coeffs, comp_hamiltonian.coeffs
        ), f"Coefficients in the computed Hamiltonian are incorrect"

        # Test computed Hamiltonian contains the correct terms
        assert np.allclose(
            hamiltonian.constant, comp_hamiltonian.constant
        ), f"Constant in the computed Hamiltonian is incorrect"

    def test_redefine_problem_isolated_nodes(self):
        """
        Testing the redefine problem function for an edge case in which fixing of one spins leads to an isolated one. In the absense of biases, the only way to capture the isolated node is to keep track of eleiminatiosn and compare the old and new registers to it.

        The test recreates the 5-qubits MaxCut QUBO and the spin map and asserts that the new problem is correct.
        """
        test_qubo = QUBO(
            n=6,
            terms=[[0, 1], [2, 3], [2, 5], [3, 4], [4, 5]],
            weights=[2.0, 1.0, 1.0, 1.0, 1.0],
        )
        spin_map = {0: (1, 0), 1: (-1.0, 0), 2: (1, 2), 3: (1, 3), 4: (1, 4), 5: (1, 5)}
        new_problem, new_spin_map = redefine_problem(test_qubo, spin_map)

        assert new_problem.terms == [[0, 1], [0, 3], [1, 2], [2, 3]]

    def test_solution_for_vanishing_instances(self):
        """
        Test the function that computes the solution to the reduced problem in the rare case the problem vanishes after eliminations.
        The test consists of recreating three specific 4-nodes instances with different connectivity and weights, which can be the result of an RQAOA workflow, and computing their classical energy and ground states according to the function.
        """

        # first example in which 0 and 1 are anticorrelated while the other spins are arbitrary
        hamiltonian = Hamiltonian.classical_hamiltonian(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]], [2.0, 1.0, 1.0, 1.0, 1.0], 0
        )
        spin_map = {0: (1, None), 1: (-1.0, 0), 2: (1, None), 3: (1, None)}

        cl_energy, cl_ground_states = solution_for_vanishing_instances(
            hamiltonian, spin_map
        )

        assert cl_energy == -2.0
        assert cl_ground_states == ["1011"]

        # second example in which 0 and 2 are correlated while the other spins are arbitrary
        hamiltonian = Hamiltonian.classical_hamiltonian(
            [[0, 1], [0, 3], [2, 3], [1, 2], [0, 2]], [1.0, -1.0, 1.0, -1.0, -2.0], 0
        )
        spin_map = {0: (1, None), 1: (1, None), 2: (1.0, 0), 3: (1, None)}

        cl_energy, cl_ground_states = solution_for_vanishing_instances(
            hamiltonian, spin_map
        )

        assert cl_energy == -2.0
        assert cl_ground_states == ["1111"]

        # third example in which 0 and 3 are anticorrelated while the other spins are arbitrary
        hamiltonian = Hamiltonian.classical_hamiltonian(
            [[0, 1], [0, 3], [0, 2], [1, 3], [2, 3]], [1.0, 2.0, -1.0, 1.0, -1.0], 0
        )
        spin_map = {0: (1, None), 1: (1, None), 2: (1, None), 3: (-1.0, 0)}

        cl_energy, cl_ground_states = solution_for_vanishing_instances(
            hamiltonian, spin_map
        )

        assert cl_energy == -2.0
        assert cl_ground_states == ["1110"]

    def workflow_mockup(self, qubo, n_cutoff):
        r = RQAOA()
        r.set_rqaoa_parameters(n_cutoff=n_cutoff)
        r.set_circuit_properties(
            p=1, param_type="standard", init_type="ramp", mixer_hamiltonian="x"
        )

        # Define the device you want to run your problem on using the create_device() function
        device = create_device(location="local", name="analytical_simulator")
        r.set_device(device)

        # Set the classical method used to optimize over QAOA angles and its properties
        r.set_classical_optimizer(method="cobyla", maxiter=600)

        r.compile(qubo)
        r.optimize()
        return r.result

    def test_total_elimination_cutoff_1_whole_workflow(self):
        """
        Testing the whole RQAOA workflow for Solving the MaxCut on the Ring of Disagree with normalized weights and no biases.

        Recreating the specific qubo instance and running standard RQAOA with cutoff set to 1. This results in vanishing problem before reaching the cutoff, which tests both the proper isolation of nodes and the solution_for_vanishing_instances functions. RQAOA should be able to find the true ground state for this specific instance.
        """
        qubo = QUBO(
            n=8,
            terms=[[0, 7], [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]],
            weights=[1, 1 / 9, 2 / 9, 3 / 9, 5 / 9, 6 / 9, 7 / 9, 8 / 9],
        )

        opt_results = self.workflow_mockup(qubo, n_cutoff=1)

        assert opt_results["solution"] == {"10101010": -4.555555555555555}

    def test_isolated_nodes_whole_workflow(self):
        """
        Testing the whole RQAOA workflow for solving the MaxCut problem on random graph instances.

        Recreating a specific qubo and testing it with cutoff size set to 3, 2, and 1. RQAOA should be able to find the true ground state for this specific instance at all cutoffs. Note that the smaller the cutoff, the less of the degenerate solutions are recovered.
        """
        terms = [
            [0, 2],
            [0, 3],
            [0, 4],
            [0, 5],
            [0, 6],
            [0, 7],
            [0, 8],
            [0, 9],
            [0, 1],
            [1, 2],
            [1, 5],
            [1, 6],
            [2, 3],
            [2, 5],
            [2, 6],
            [2, 7],
            [8, 2],
            [9, 2],
            [3, 4],
            [3, 5],
            [3, 6],
            [3, 7],
            [8, 3],
            [9, 3],
            [1, 3],
            [4, 5],
            [4, 6],
            [4, 7],
            [8, 4],
            [9, 4],
            [1, 4],
            [5, 6],
            [5, 7],
            [9, 5],
            [6, 7],
            [8, 6],
            [9, 6],
            [8, 7],
            [1, 7],
            [8, 9],
            [8, 1],
            [1, 9],
        ]
        weights = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
        qubo = QUBO(n=10, terms=terms, weights=weights)

        opt_results_3 = self.workflow_mockup(qubo, n_cutoff=3)
        assert opt_results_3["solution"] == {"1011101000": -10.0, "0010101101": -10.0}
        assert opt_results_3["classical_output"]["optimal_states"] == ["110", "001"]

        opt_results_2 = self.workflow_mockup(qubo, n_cutoff=2)
        assert opt_results_2["solution"] == {"0010101101": -10.0, "1011101000": -10.0}
        assert opt_results_2["classical_output"]["optimal_states"] == ["00", "11"]

        opt_results_1 = self.workflow_mockup(qubo, n_cutoff=1)
        assert opt_results_1["solution"] == {"1011101000": -10.0}
        assert opt_results_1["classical_output"]["optimal_states"] == ["11"]


if __name__ == "__main__":
    unittest.main()
