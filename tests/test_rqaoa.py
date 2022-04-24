#   Copyright 2022 Entropica Labs
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import unittest

from openqaoa.qaoa_parameters import Hamiltonian
from openqaoa.rqaoa import *

"""
Unittest based testing of current implementation of the RQAOA Algorithm
"""

class TestingRQAOA(unittest.TestCase):

    def test_classical_exact_solution(self):
        """
        Test of the classical solver function which computes the solution of the problem when 
        the number of qubits reaches the cutoff value.

        The test consists in obtaining the solution for an unweighted ring graph.
        """
        ## Problem definition

        # Number of variables/qubits
        n_qubits = 10 

        # Terms and weights of the graph
        terms = [(i,i+1) for i in range(n_qubits-1)] + [(0,n_qubits-1)] # Ring structure
        weights = [1 for _ in range(len(terms))] # All weights equal to 1

        # Define Hamiltonian
        hamiltonian = Hamiltonian.classical_hamiltonian(terms, weights, constant = 0) 

        ## Testing

        # Exact solution to the problem
        solutions = [np.array([1,0,1,0,1,0,1,0,1,0]),np.array([0,1,0,1,0,1,0,1,0,1])]

        # Obtain solution from function
        _, comp_solutions = classical_exact_solution(hamiltonian)

        # Test function result
        assert np.allclose(solutions,comp_solutions), f'Computed solutions are incorrect'

    def test_find_parent(self):
        """
        Test of the find_parent function which backtracks the spin_map dictionary to obtain
        the final constraint used to eliminate a given spin.

        The test consists in finding the correct parents and factors of the constraint
        for a given example.
        """

        # Spin map example without correct final dependencies
        spin_map = dict({(0,(1,0)),(1,(-1,3)),(2,(1,5)),(3,(-1,2)),(4,(1,4)),(5,(1,4))})

        # Solution to the problem
        parents = {0:0,1:4,2:4,3:4,4:4,5:4} # spin:parent_spin
        factors = [1,1,1,-1,1,1]
        

        comp_parents = {}
        comp_factors = []

        # For each spin compute parent and connecting factor and store them
        for spin in sorted(spin_map.keys()):
            
            comp_parent, comp_factor = find_parent(spin_map,spin)
            comp_parents.update({spin:comp_parent})
            comp_factors.append(comp_factor)

        # Test function result
        assert np.allclose(list(comp_parents.values()),list(parents.values())), f'Computed parent spins are incorrect'
        assert np.allclose(comp_factors,factors), f'Computed constraint factors are incorrect'

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
        n_qubits = 10

        # Terms and weights of the graph
        terms = [(i,i+1) for i in range(n_qubits-1)] + [(0,n_qubits-1)] # Ring structure
        weights = [1 for _ in range(len(terms))] # All weights equal to 1

        # Hyperparameters
        hamiltonian = Hamiltonian.classical_hamiltonian(terms, weights, constant = 0) 

        ## Testing

        # Example case of the maximal terms and values extracted from the expectation values
        max_terms_and_stats = dict({(0, 1): -0.5, (0, 9): 0.32, (7,): -0.89, (2,4):0.73, (4,5):0.9})

        # Exact solution to the problem
        spin_map = dict({(0,(1,0)),(1,(-1,0)),(2,(1,2)),(3,(1,3)),(4,(1,2)),\
                (5,(1,2)),(6,(1,6)),(7,(-1,None)),(8,(1,8)),(9,(1,0))})

        new_max_tc = dict({(0, 1): -1, (2,4):1, (2,5): 1, (None,7):-1, (0, 9): 1}) # The function also returns final constraints

        # Compute the spin_map and final constraints from the function
        comp_spin_map, comp_new_max_tc = spin_mapping(hamiltonian,max_terms_and_stats)

        # Check both outputs contain the same number of keys as the correct solution
        assert len(spin_map) == len(comp_spin_map), f'Computed spin_map has incorrect length'
        assert len(new_max_tc) == len(comp_new_max_tc), f'Output max_terms_and_stats dictionary has incorrect length '
        
        # Test the spin_map matches the correct solution
        for key in spin_map.keys():
            assert spin_map[key][0] == comp_spin_map[key][0], f'Computed spin_map contains incorrect factor'
            assert spin_map[key][1] == comp_spin_map[key][1], f'Computed spin_map contains incorrect parent spin'
        
        # Test the final constraints match the correct solution
        for key in new_max_tc.keys():
            assert new_max_tc[key] == comp_new_max_tc[key], f'Output max_terms_and_stats contains incorrect factor'

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
        exp_vals_z = np.array([0.33,-0.21,-0.9,0.06,-0.78])
        
        # Correlation matrix
        corr_mat = np.array([[0.0,0.01,-0.64,0.69,0.48],\
                              [0.0,0.0,-0.99,-0.27,0.03],\
                              [0.0,0.0,0.0,1.0,-0.22],\
                              [0.0,0.0,0.0,0.0,0.37],\
                              [0.0,0.0,0.0,0.0,0.0]])
                              
        # Correct solution
        max_tc = dict({(2,3):1.0,(1,2):-0.99,(2,):-0.9,(4,):-0.78,(0,3):0.69}) 

        # Computed solution using the function
        comp_max_tc = max_terms(exp_vals_z, corr_mat, n_elim)

        # Confirm the computed solution has same number of items as correct one
        assert len(max_tc) == len(comp_max_tc), f'Computed set of singlets/correlations contains incorrect number of elements'

        # Test the function has obtain the correct singlets/pairs with associated values
        for key in max_tc.keys():
            assert max_tc[key] == comp_max_tc[key], f'Computed set of singlets/correlations contains incorrect values'

    def test_ada_max_terms(self):
        """
        Test of the function that adaptively selects spin pairs or singlets are used for elimination.

        The test consists in finding the correct pairs and singles to eliminate, according 
        to the adaptive scheme, for a given example.
        """

        # Maximum number of singles/pairs allowed to be considered for elimination
        n_max = 3

        # Set of single spin expectation values
        exp_vals_z = np.array([0.33,-0.21,-0.9,0.06,-0.19])
        
        # Correlation matrix
        corr_mat = np.array([[0.0,0.01,-0.64,0.69,0.48],\
                              [0.0,0.0,-0.99,-0.27,0.03],\
                              [0.0,0.0,0.0,1.0,-0.22],\
                              [0.0,0.0,0.0,0.0,0.37],\
                              [0.0,0.0,0.0,0.0,0.0]])
                              
        # Correct solution
        max_tc = dict({(2,3):1.0,(1,2):-0.99,(2,):-0.9}) 

        # Computed solution using the function
        comp_max_tc = ada_max_terms(exp_vals_z, corr_mat, n_max)

        # Confirm the computed solution has same number of items as correct one
        assert len(max_tc) == len(comp_max_tc), f'Computed set of singlets/correlations contains incorrect number of elements'

        # Test the function has obtain the correct singlets/pairs with associated values
        for key in max_tc.keys():
            assert max_tc[key] == comp_max_tc[key], f'Computed set of singlets/correlations contains incorrect values'

    def test_expectation_values_analytical(self):
        """
        Test of the function that computes singlet expectation values and correlations terms
        analytically for p = 1 and the function computing the full set of expectation values
        when analytical results can be obtained (p=1).

        NOTE: Correlations in the term_corr_analytical() and term_corr() functions are computed 
        as average value <Z_{i}Z_{j}>, meaning it includes the <Z_{i}><Z_{j}> contribution. 
        This is subtracted by default in the expectation_values() function.

        The tests consist in: computing expectation values for some example cases for the 
        first function, and a full set of expectation values for a given example. 
        """

        ## Problem definition

        # Number of qubits
        n_qubits = 4

        # Edges and weights of the graph
        pair_edges = [(i,i+1) for i in range(n_qubits-1)] + [(0,n_qubits-1)] 
        self_edges = [(i,) for i in range(n_qubits)]
        pair_weights = [1 for _ in range(len(pair_edges))] # All weights equal to 1
        self_weights = [(-1)**j for j in range(len(self_edges))]

        edges = pair_edges + self_edges
        weights = pair_weights + self_weights

        # Hamiltonian
        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant = 0)

        ## Testing

        # Spin and pair whose expectation values are computed
        spin = 0
        pair = (0,1)
        
        # Correct solution
        qaoa_angles_cases = {(0,0):(0,0),(np.pi,0):(0,0),(np.pi/4,np.pi/4):(0,0),\
            (np.pi/4,np.pi/8):(-np.sqrt(2)/4,-0.25),(np.pi/8,0):(0,0)} # (beta,gamma):(exp_val,corr)

        # Compute singlet expectation values and correlations for each set of angles
        for qaoa_angles in qaoa_angles_cases.keys():
            
            # Extract correct solution
            exp_val = np.round(qaoa_angles_cases[qaoa_angles][0],16)
            corr = np.round(qaoa_angles_cases[qaoa_angles][1],16)
            
            # Compute expectation values and correlations
            comp_exp_val = np.round(exp_val_single_spin_analytical(spin,hamiltonian,qaoa_angles),16)
            comp_corr = np.round(term_corr_analytical(pair,hamiltonian,qaoa_angles),16)

            # Test if computed results are correct
            assert np.allclose(exp_val,comp_exp_val), f'Incorrectly computed singlet expectation value'
            assert np.allclose(corr,comp_corr), f'Incorrectly computed correlation term'

        # Fix a set of angles for testing full set of expectation values and correlations
        fixed_angles = (np.pi/4,np.pi/8)

        # Correct solutions
        exp_val_list = np.array([-np.sqrt(2)/4,np.sqrt(2)/4,-np.sqrt(2)/4,np.sqrt(2)/4])
        corr_matrix = np.array([[0.0,-1/4,0,-1/4],\
                              [0.0,0.0,-1/4,0],\
                              [0.0,0.0,0.0,-1/4],\
                              [0.0,0.0,0.0,0.0]])

        corr_matrix -= np.outer(exp_val_list,exp_val_list)

        # Compute list of expectation values and correlation matrix
        comp_exp_val_list, comp_corr_matrix = expectation_values(variational_params = None,
                                                                 qaoa_results = {'final params' : fixed_angles},\
                                                                 qaoa_backend = None,\
                                                                 hamiltonian = hamiltonian,
                                                                 p = 1)

        # Test if computed results are correct
        assert np.allclose(exp_val_list,comp_exp_val_list), f'Computed set of singlet expectation values is incorrect'

        for j in range(len(comp_corr_matrix)):
            assert np.allclose(corr_matrix[j],comp_corr_matrix[j]), f'Computed correlation matrix is incorrect'
    
    def test_expectation_values(self):
        """
        Test of the function that computes singlet expectation values and correlations numerically through
        the QAOA output distribution of states.

        The test consist of computing the singlet expectation values and correlations for a given problem.
        The result is constrasted with the analytical result, whose implementation is tested in 
        test_expectation_values_analytical().
        """

        ## Problem definition

        # Number of qubits
        n_qubits = 10

        # Number of QAOA layers - necessary only for the definition of circuit parameters
        p = 1

        # Terms and weights of the graph
        edges = [(i,i+1) for i in range(n_qubits-1)] + [(0,n_qubits-1)]
        weights = [10 for _ in range(len(edges))]

        # Hyperparameters
        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant = 0)

        # Mixer Hamiltonian
        mixer_hamiltonian = X_mixer_hamiltonian(n_qubits)

        # Define circuit and variational parameters
        circuit_params = QAOACircuitParams(hamiltonian,mixer_hamiltonian, p = p)
        variational_params = create_qaoa_variational_params(circuit_params, params_type = 'standard', init_type = 'ramp')

        ## Testing

        # Perform QAOA and obtain expectation values numerically
        qaoa_backend = qaoa_cost_backend(circuit_params, device = 'vectorized', shots = None, qpu_params = None)
        qaoa_results = optimize_qaoa(qaoa_backend,variational_params,optimizer_dict = {'method':'cobyla','maxiter':200})
        
        num_exp_vals_z,num_corr_matrix = expectation_values(variational_params,qaoa_results, qaoa_backend, hamiltonian, p = p, analytical = False)

        # Analytical expectation values
        exp_vals_z, corr_matrix = expectation_values(variational_params,qaoa_results, qaoa_backend, hamiltonian, p = p)

        # Test if computed results are correct
        assert np.allclose(exp_vals_z,num_exp_vals_z), f'Computed singlet expectation values are incorrect'

        for j in range(len(num_corr_matrix)):
            assert np.allclose(corr_matrix[j],num_corr_matrix[j]), f'Computed correlation matrix is incorrect'

    def test_hamiltonian_from_graph(self):
        """
        Test the function that computes a calssical Hamiltonian from a given graph, accounting for approriate
        labelling of the nodes and edges.
        
        The test consists in generating the correct Hamiltonian for a given graph dictionary.
        """
        
        # Trial graph
        input_graph = dict({(1,2):1,(2,5):2,(10,14):3,(6,9):4,(6,14):5,(5,6):6})

        # Correct hamiltonian
        correct_graph = dict({(0,1):1,(1,2):2,(2,3):6,(3,4):4,(3,6):5,(5,6):3})
        hamiltonian = Hamiltonian.classical_hamiltonian(list(correct_graph.keys()),list(correct_graph.values()), constant = 0)
        hamiltonian_dict = {term.qubit_indices:coeff for term,coeff in zip(hamiltonian.terms,hamiltonian.coeffs)}

        # Compute hamiltonian from input graph
        comp_hamiltonian = hamiltonian_from_graph(input_graph)
        comp_hamiltonian_dict = {term.qubit_indices:coeff for term,coeff in zip(comp_hamiltonian.terms,comp_hamiltonian.coeffs)}


        # Test computed Hamiltonian contains the correct terms
        assert hamiltonian_dict == comp_hamiltonian_dict, f'Terms and coefficients in the computed Hamiltonian are incorrect'

        # Test computed Hamiltonian contains the correct terms
        # assert np.allclose(hamiltonian.coeffs,comp_hamiltonian.coeffs), f'Coefficients in the computed Hamiltonian are incorrect'

        # Test computed Hamiltonian contains the correct terms
        assert np.allclose(hamiltonian.constant,comp_hamiltonian.constant), f'Constant in the computed Hamiltonian is incorrect'

    def test_redefine_hamiltonian(self):
        """
        Test the function that computes the new Hamiltonian for a reduced problem, given the original
        Hamitlonian encoding the problem and a set of elimination rules via the spin_map.

        The test consists in computing the new Hamiltonian from a givne one and a set of elimination rules.
        """

        ## Problem definition

        # Number of qubits
        n_qubits = 10

        # Edges and weights of the graph
        input_edges = [(i,i+1) for i in range(n_qubits-1)] + [(0,n_qubits-1)]
        input_weights = [1 for _ in range(len(input_edges))]

        # Input hamiltonian
        input_hamiltonian = Hamiltonian.classical_hamiltonian(input_edges, input_weights, constant = 0)

        # Input spin map (elimination rules)
        spin_map = dict({(0,(1,0)),(1,(-1,0)),(2,(1,2)),(3,(1,3)),(4,(1,2)),\
                (5,(1,2)),(6,(1,6)),(7,(1,7)),(8,(1,8)),(9,(1,0))})

        ## Testing

        # Correct edges, weights and hamiltonian for the reduced problem
        edges = [(0,1),(1,2),(1,3),(3,4),(4,5),(0,5)]
        weights = [-1,2,1,1,1,1]

        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant = 0)

        # Compute new hamiltonian
        comp_hamiltonian = redefine_hamiltonian(input_hamiltonian, spin_map)

        # Test computed Hamiltonian contains the correct terms
        assert hamiltonian.terms == comp_hamiltonian.terms, f'Terms in the computed Hamiltonian are incorrect'

        # Test computed Hamiltonian contains the correct terms
        assert np.allclose(hamiltonian.coeffs,comp_hamiltonian.coeffs), f'Coefficients in the computed Hamiltonian are incorrect'

        # Test computed Hamiltonian contains the correct terms
        assert np.allclose(hamiltonian.constant,comp_hamiltonian.constant), f'Constant in the computed Hamiltonian is incorrect'

    def test_final_opt_strings(self):
        """
        Test the function that reconstructs the final output state by backtracking the
        elimination history.

        The test consists in reconstructing a state for a given elimination history.
        """

        # Trial elimination history and ouput of classical solver
        max_terms_and_stats_list = [{(0, 1): -1.0, (0, 9): -1.0}, {(0, 1): 1.0, (0, 7): 1.0}, 
                                        {(0, 1): -1.0, (0, 5): -1.0}, {(0, 1): 1.0}]

        classical_results = {'cost min': 0, 'opt strings': [np.array([0, 1, 0]), np.array([1, 0, 1])]}

        # Correct solutions
        solutions = [[0,1,0,1,0,1,0,1,0,1],[1,0,1,0,1,0,1,0,1,0]]

        # Compute solutions
        comp_solutions = final_opt_strings(max_terms_and_stats_list, classical_results)

        # Test the computed solutions
        assert np.allclose(solutions,comp_solutions), f'Solution backtracking process is incorrect'

    def test_final_energy(self):
        """
        Testing of the function which computes the energy of the states.

        The test consist in computing the energy of two given states for a ring-like geometry.
        """

        ## Problem definition

        # Number of qubits
        n_qubits = 10

        # Terms and weights of the graph
        edges = [(i,i+1) for i in range(n_qubits-1)] + [(0,n_qubits-1)]
        weights = [1 for _ in range(len(edges))]

        # Hamiltonian
        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant = 0)

        ## Testing

        # Trial states
        states = [[0,1,0,1,0,1,0,1,0,1],[1,0,1,0,1,0,1,0,1,0]]

        # Correct solution
        energies = [-10, -10]

        # Compute energies for the given trial states
        full_solution = final_energy(states,hamiltonian)
        comp_energies = list(full_solution.values())
        
        # Test if computed solution is correcrt
        assert np.allclose(energies,comp_energies), f'Computed energy is incorrect'

    def test_adaptive_rqaoa(self):
        """
        Test the overall wrapper of the Adaptive RQAOA algorithm.

        The test consists in solving an unweighted Ising ring problem in the presence of a very weak
        external field, such that the ground state solution remains the same as without it but we 
        ensure the system works in general. This is done for different elimination schemes, varying 
        the maximum number of allowed eliminations from 1 to 4. We consider a standard X mixer as a
        mixing Hamiltonian.
        """

        ## Problem definition

        # Number of qubits
        n_qubits = 12

        # Elimination schemes
        Nmax = [1,2,3,4]

        # Number of QAOA layers
        p = 1

        # Edges and weights of the graph
        pair_edges = [(i,i+1) for i in range(n_qubits-1)] + [(0,n_qubits-1)] 
        self_edges = [(i,) for i in range(n_qubits)]
        pair_weights = [1 for _ in range(len(pair_edges))] # All weights equal to 1
        self_weights = [10**(-4) for _ in range(len(self_edges))]

        edges = pair_edges + self_edges
        weights = pair_weights + self_weights

        # Cost and mixer Hamiltonians
        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant = 0)
        mixer_hamiltonian = X_mixer_hamiltonian(n_qubits)

        ## Testing

        # Correct solution
        exact_sol = {'101010101010': -12, '010101010101': -12}

        # Compute the solution for each elimination scheme
        for nmax in Nmax:
            
            # Compute Ada-RQAOA solution
            result = adaptive_rqaoa(hamiltonian, mixer_hamiltonian, p = p, n_max = nmax)
            full_sol = result['solution']

            # Test the number of solution states is correct
            assert len(exact_sol) == len(full_sol), f'The number of computed solutions is incorrect for nmax = {nmax}'

            # Test the assigned energy to each solution is correct
            for key in full_sol.keys():
                assert full_sol[key] == exact_sol[key], f'The computed energy of the state is incorrect'
    
    def test_custom_rqaoa(self):
        """
        Test the overall wrapper of the Custom RQAOA algorithm.

        The test consists in solving an unweighted Ising ring problem in the presence of a very weak
        external field, such that the ground state solution remains the same as without it but we 
        ensure the system works in general. This is done for different elimination schemes, for fixed
        elimination and for an input schedule. We consider a standard X mixer as a
        mixing Hamiltonian.
        """

        ## Problem definition

        # Number of qubits
        n_qubits = 12

        # Elimination schemes
        schedules = [1,[1,2,1,2,1]]

        # Number of QAOA layers
        p = 1

        # Edges and weights of the graph
        pair_edges = [(i,i+1) for i in range(n_qubits-1)] + [(0,n_qubits-1)] 
        self_edges = [(i,) for i in range(n_qubits)]
        pair_weights = [1 for _ in range(len(pair_edges))] # All weights equal to 1
        self_weights = [10**(-4) for _ in range(len(self_edges))]

        edges = pair_edges + self_edges
        weights = pair_weights + self_weights

        # Cost and mixer Hamiltonians
        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant = 0)
        mixer_hamiltonian = X_mixer_hamiltonian(n_qubits = n_qubits)

        ## Testing

        # Correct solution
        exact_sol = {'101010101010': -12, '010101010101': -12}

        # Compute the solution for each elimination scheme
        for schedule in schedules:
            
            # Compute RQAOA solution
            result = custom_rqaoa(hamiltonian, mixer_hamiltonian, p = p, steps = schedule)
            full_sol = result['solution']

            # Test the number of solution states is correct
            assert len(exact_sol) == len(full_sol), f'The number of computed solutions is incorrect'

            # Test the assigned energy to each solution is correct
            for key in full_sol.keys():
                assert full_sol[key] == exact_sol[key], f'The computed energy of the state is incorrect'

if __name__ == "__main__":
	unittest.main()


