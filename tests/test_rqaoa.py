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
        n_qubits = 20

        # Terms and weights of the graph
        edges = [(i,j) for j in range(n_qubits) for i in range(j)]
        weights = [1 for _ in range(len(edges))] 

        # Hyperparameters
        problem = QUBO(n= n_qubits, terms=edges, weights=weights)

        ## Testing

        # Example case of the maximal terms and values extracted from the expectation values
        max_terms_and_stats = dict({(4,):0.99,(0, 1): -0.89, (0, 9): 0.77, (1,9):-0.73,\
             (17,): -0.5, (2,4):0.32, (5,4):0.29,(16,):-0.8, (10,):0.81, (16,10):0.4, (19,14):-0.47})

        # Exact solution to the problem
        correct_spin_map = dict({(0,(1,0)),(1,(-1,0)),(2,(1,None)),(3,(1,3)),(4,(1,None)),\
                (5,(1,None)),(6,(1,6)),(7,(1,7)),(8,(1,8)),(9,(1,0)),(10,(1,None)),(11,(1,11)),(12,(1,12)),\
                (13,(1,13)),(14,(1,14)),(15,(1,15)),(16,(-1,None)),(17,(-1,None)),(18,(1,18)),(19,(-1,14))})

        # Compute the spin_map and final constraints from the function
        spin_map = spin_mapping(problem,max_terms_and_stats)

        # Check both outputs contain the same number of keys as the correct solution
        assert len(correct_spin_map) == len(spin_map), f'Computed spin_map has incorrect length'
        
        # Test the spin_map matches the correct solution
        for key in correct_spin_map.keys():
            assert correct_spin_map[key][0] == spin_map[key][0], f'Computed spin_map contains incorrect factor'
            assert correct_spin_map[key][1] == spin_map[key][1], f'Computed spin_map contains incorrect parent spin'

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
        edges = [(i,i+1) for i in range(n_qubits-1)] + [(0,n_qubits-1)]
        weights = [1 for _ in range(len(edges))]

        # Hamiltonian
        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant = 0)

        ## Testing

        # Trial elimination history and ouput of classical solver
        max_terms_and_stats_list = [{(0, 1): -1.0, (0, 9): -1.0}, {(0, 1): 1.0, (0, 7): 1.0}, 
                                        {(0, 1): -1.0, (0, 5): -1.0}, {(0, 1): 1.0}]

        classical_states = [[0, 1, 0],[1, 0, 1]]

        # Correct solutions
        states = ['0101010101','1010101010']
        energies = [-10, -10]
        correct_full_solution = dict(zip(states,energies))

        # Compute solutions
        full_solution = final_solution(max_terms_and_stats_list, classical_states, hamiltonian)

        # Test the computed solutions
        assert correct_full_solution == full_solution, f'Solution was not computed correctly'


if __name__ == "__main__":
	unittest.main()
 