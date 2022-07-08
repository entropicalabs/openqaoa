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
import networkx as nx
import unittest

from openqaoa.qaoa_parameters import Hamiltonian
from openqaoa.rqaoa import *
from openqaoa.problems.problem  import MinimumVertexCover

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
        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant = 0) 

        ## Testing

        # Example case of the maximal terms and values extracted from the expectation values
        max_terms_and_stats = dict({(4,):0.99,(0, 1): -0.89, (0, 9): 0.77, (1,9):-0.73,\
             (17,): -0.5, (2,4):0.32, (5,4):0.29,(16,):-0.8, (10,):0.81, (16,10):0.4, (19,14):-0.47})

        # Exact solution to the problem
        correct_spin_map = dict({(0,(1,0)),(1,(-1,0)),(2,(1,None)),(3,(1,3)),(4,(1,None)),\
                (5,(1,None)),(6,(1,6)),(7,(1,7)),(8,(1,8)),(9,(1,0)),(10,(1,None)),(11,(1,11)),(12,(1,12)),\
                (13,(1,13)),(14,(1,14)),(15,(1,15)),(16,(-1,None)),(17,(-1,None)),(18,(1,18)),(19,(-1,14))})

        # Compute the spin_map and final constraints from the function
        spin_map = spin_mapping(hamiltonian,max_terms_and_stats)

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

    def test_hamiltonian_from_dict(self):
        """
        Test the function that computes a calssical Hamiltonian from a given graph, accounting for approriate
        labelling of the nodes and edges.
        
        The test consists in generating the correct Hamiltonian for a given graph dictionary.
        """
        
        # Trial graph
        input_dict = dict({():10,(1,):1,(2,):-1,(6,):4,(1,2):1,(2,5):2,(10,14):3,(6,9):4,(6,14):5,(5,6):6})
        
        # Correct hamiltonian
        correct_dict = dict({(0,):1,(1,):-1,(3,):4,(0,1):1,(1,2):2,(2,3):6,(3,4):4,(3,6):5,(5,6):3})
        hamiltonian = Hamiltonian.classical_hamiltonian(list(correct_dict.keys()),list(correct_dict.values()), constant = 10)
        hamiltonian_dict = {term.qubit_indices:coeff for term,coeff in zip(hamiltonian.terms,hamiltonian.coeffs)}

        # Compute hamiltonian from input graph
        comp_hamiltonian = hamiltonian_from_dict(input_dict)
        comp_hamiltonian_dict = {term.qubit_indices:coeff for term,coeff in zip(comp_hamiltonian.terms,comp_hamiltonian.coeffs)}


        # Test computed Hamiltonian contains the correct terms
        assert hamiltonian_dict == comp_hamiltonian_dict, f'Terms and coefficients in the computed Hamiltonian are incorrect'

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
        comp_hamiltonian,comp_spin_map = redefine_hamiltonian(input_hamiltonian, spin_map)

        # Test computed Hamiltonian contains the correct terms
        assert hamiltonian.terms == comp_hamiltonian.terms, f'Terms in the computed Hamiltonian are incorrect'

        # Test computed Hamiltonian contains the correct terms
        assert np.allclose(hamiltonian.coeffs,comp_hamiltonian.coeffs), f'Coefficients in the computed Hamiltonian are incorrect'

        # Test computed Hamiltonian contains the correct terms
        assert np.allclose(hamiltonian.constant,comp_hamiltonian.constant), f'Constant in the computed Hamiltonian is incorrect'

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

    def test_adaptive_rqaoa(self):
        """
        Test the overall wrapper of the Adaptive RQAOA algorithm.

        The test consists in solving four examples. We consider a standard X mixer as a mixing Hamiltonian 
        for all cases. First, an unweighted Ising ring problem in the presence of a very weak
        external field, such that the ground state solution remains the same as without it but we 
        ensure the system works in general. Second, a Minimum Vertex Cover problem on a ring,
        which requires handling both types of eliminations and the consideration of isolated nodes 
        resulting from cancellation. Third and fourth, a Minimum Vertex Cover problem and an unweighted
        Ising Hamiltonian on a fully-connected graph to test the handling of isolated nodes and high
        degeneracies.

        The unsupported custom mixer connectivities exception is also tested.
        """
        # EXAMPLE 1

        ## Problem definition

        # Number of qubits
        n_qubits = 12

        # Elimination schemes
        Nmax = [1,2,3,4]
        n_cutoff = 5

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
        mixer = {'type':'x'}

        ## Testing

        # Correct solution
        exact_sol = {'101010101010': -12, '010101010101': -12}

        # Compute the solution for each elimination scheme
        for nmax in Nmax:
            
            # Compute Ada-RQAOA solution
            result = adaptive_rqaoa(hamiltonian, mixer, p = p, n_max = nmax, n_cutoff = n_cutoff)
            full_sol = result['solution']

            # Test the assigned energy to each solution is correct
            for key in full_sol.keys():
                assert full_sol[key] == exact_sol[key], f'The computed energy of the state is incorrect'


        # EXAMPLE 2

        ## Problem definition
        n_qubits = 10

        # Elimination schemes
        n_max = 1
        n_cutoff = 3

        # Number of QAOA layers
        p = 1

        # Ring graph
        G = nx.circulant_graph(n_qubits,[1])

        # Minimum vertex cover parameters
        field = 1.0
        penalty = 10

        # Define problem instance
        mvc = MinimumVertexCover(G,field = field,penalty = penalty).get_qubo_problem()

        # Cost and mixer Hamiltonians
        hamiltonian = Hamiltonian.classical_hamiltonian(mvc.terms, mvc.weights, mvc.constant)
        mixer = {'type':'x'}

        # Correct solution
        exact_sol = {'1010101010': 5, '0101010101': 5}

        # Compute Ada-RQAOA solution
        result = adaptive_rqaoa(hamiltonian, mixer, p = p, n_max = n_max, n_cutoff = n_cutoff)
        full_sol = result['solution']

        # Test the assigned energy to each solution is correct
        for key in full_sol.keys():
            assert full_sol[key] == exact_sol[key], f'The computed energy of the state is incorrect'

        # EXAMPLE 3

        ## Problem definition
        n_qubits = 10

        # Elimination schemes
        n_max = 4
        n_cutoff = 3

        # Number of QAOA layers
        p = 1

        # Ring graph
        G = nx.complete_graph(n_qubits)

        # Minimum vertex cover parameters
        field = 1.0
        penalty = 10

        # Define problem instance
        mvc = MinimumVertexCover(G,field = field,penalty = penalty).get_qubo_problem()

        # Cost and mixer Hamiltonians
        hamiltonian = Hamiltonian.classical_hamiltonian(mvc.terms, mvc.weights, mvc.constant)
        mixer = {'type':'x'}

        # Correct solution
        exact_sol = {'0111111111': 9, '1011111111': 9, '1101111111': 9, '1110111111': 9, '1111011111': 9,
        '1111101111': 9, '1111110111': 9,'1111111011': 9, '1111111101': 9,'1111111110': 9}

        # Compute Ada-RQAOA solution
        result = adaptive_rqaoa(hamiltonian, mixer, p = p, n_max = n_max, n_cutoff = n_cutoff)
        full_sol = result['solution']

        # Test the assigned energy to each solution is correct
        for key in full_sol.keys():
            assert full_sol[key] == exact_sol[key], f'The computed energy of the state is incorrect'

        # EXAMPLE 4

        ## Problem definition

        # Number of qubits
        n_qubits = 10

        # Elimination schemes
        Nmax = [1,2,3,4]
        n_cutoff = 3

        # Number of QAOA layers
        p = 1

        # Edges and weights of the graph
        edges = [(i,j) for j in range(n_qubits) for i in range(j)]
        weights = [1 for _ in range(len(edges))]

        # Cost and mixer Hamiltonians
        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant = 0)
        mixer = {'type':'x'}

        ## Testing

        # Correct solution
        exact_states = ['1111100000', '1111010000', '1110110000', '1101110000', '1011110000', '0111110000', '1111001000', '1110101000', '1101101000', '1011101000', '0111101000', '1110011000', '1101011000', '1011011000', '0111011000', '1100111000', '1010111000', '0110111000', '1001111000', '0101111000', '0011111000', '1111000100', '1110100100', '1101100100', '1011100100', '0111100100', '1110010100', '1101010100', '1011010100', '0111010100', '1100110100', '1010110100', '0110110100', '1001110100', '0101110100', '0011110100', '1110001100', '1101001100', '1011001100', '0111001100', '1100101100', '1010101100', '0110101100', '1001101100', '0101101100', '0011101100', '1100011100', '1010011100', '0110011100', '1001011100', '0101011100', '0011011100', '1000111100', '0100111100', '0010111100', '0001111100', '1111000010', '1110100010', '1101100010', '1011100010', '0111100010', '1110010010', '1101010010', '1011010010', '0111010010', '1100110010', '1010110010', '0110110010', '1001110010', '0101110010', '0011110010', '1110001010', '1101001010', '1011001010', '0111001010', '1100101010', '1010101010', '0110101010', '1001101010', '0101101010', '0011101010', '1100011010', '1010011010', '0110011010', '1001011010', '0101011010', '0011011010', '1000111010', '0100111010', '0010111010', '0001111010', '1110000110', '1101000110', '1011000110', '0111000110', '1100100110', '1010100110', '0110100110', '1001100110', '0101100110', '0011100110', '1100010110', '1010010110', '0110010110', '1001010110', '0101010110', '0011010110', '1000110110', '0100110110', '0010110110', '0001110110', '1100001110', '1010001110', '0110001110', '1001001110', '0101001110', '0011001110', '1000101110', '0100101110', '0010101110', '0001101110', '1000011110', '0100011110', '0010011110', '0001011110', '0000111110', '1111000001', '1110100001', '1101100001', '1011100001', '0111100001', '1110010001', '1101010001', '1011010001', '0111010001', '1100110001', '1010110001', '0110110001', '1001110001', '0101110001', '0011110001', '1110001001', '1101001001', '1011001001', '0111001001', '1100101001', '1010101001', '0110101001', '1001101001', '0101101001', '0011101001', '1100011001', '1010011001', '0110011001', '1001011001', '0101011001', '0011011001', '1000111001', '0100111001', '0010111001', '0001111001', '1110000101', '1101000101', '1011000101', '0111000101', '1100100101', '1010100101', '0110100101', '1001100101', '0101100101', '0011100101', '1100010101', '1010010101', '0110010101', '1001010101', '0101010101', '0011010101', '1000110101', '0100110101', '0010110101', '0001110101', '1100001101', '1010001101', '0110001101', '1001001101', '0101001101', '0011001101', '1000101101', '0100101101', '0010101101', '0001101101', '1000011101', '0100011101', '0010011101', '0001011101', '0000111101', '1110000011', '1101000011', '1011000011', '0111000011', '1100100011', '1010100011', '0110100011', '1001100011', '0101100011', '0011100011', '1100010011', '1010010011', '0110010011', '1001010011', '0101010011', '0011010011', '1000110011', '0100110011', '0010110011', '0001110011', '1100001011', '1010001011', '0110001011', '1001001011', '0101001011', '0011001011', '1000101011', '0100101011', '0010101011', '0001101011', '1000011011', '0100011011', '0010011011', '0001011011', '0000111011', '1100000111', '1010000111', '0110000111', '1001000111', '0101000111', '0011000111', '1000100111', '0100100111', '0010100111', '0001100111', '1000010111', '0100010111', '0010010111', '0001010111', '0000110111', '1000001111', '0100001111', '0010001111', '0001001111', '0000101111', '0000011111']

        exact_sol = {state:-5 for state in exact_states}

        # Compute the solution for each elimination scheme
        for nmax in Nmax:
            
            # Compute Ada-RQAOA solution
            result = adaptive_rqaoa(hamiltonian, mixer, p = p, n_max = nmax, n_cutoff = n_cutoff)
            full_sol = result['solution']

            # Test the assigned energy to each solution is correct
            for key in full_sol.keys():
                assert full_sol[key] == exact_sol[key], f'The computed energy of the state is incorrect'


        ## Exception case - custom connectivity fed into adaptive RQAOA
        n_qubits = 7
        edges = [(i,j) for j in range(n_qubits) for i in range(j)]
        weights = [1 for _ in range(len(edges))]
        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant = 0)

        # Generate custom mixer
        connectivity = [(i,i+1) for i in range(n_qubits-1)] + [(0,n_qubits-1)]
        mixer = {'type':'xy','connectivity': connectivity}

        # Attempt running custom RQAOA
        with self.assertRaises(NotImplementedError) as context:
            result = adaptive_rqaoa(hamiltonian, mixer)
            
        # Check exception message
        self.assertEqual("Custom mixer connectivities are not currently supported", str(context.exception))


    
    def test_custom_rqaoa(self):
        """
        Test the overall wrapper of the Custom RQAOA algorithm.

        The test consists in solving four examples. We consider a standard X mixer as a mixing Hamiltonian 
        for all cases. First, an unweighted Ising ring problem in the presence of a very weak
        external field, such that the ground state solution remains the same as without it but we 
        ensure the system works in general. Second, a Minimum Vertex Cover problem on a ring,
        which requires handling both types of eliminations and the consideration of isolated nodes 
        resulting from cancellation. Third and fourth, a Minimum Vertex Cover problem and an unweighted
        Ising Hamiltonian on a fully-connected graph to test the handling of isolated nodes and high
        degeneracies.

        The unsupported custom mixer connectivities exception is also tested.
        """
        # EXAMPLE 1

        ## Problem definition

        # Number of qubits
        n_qubits = 12

        # Elimination schemes
        schedules = [1,[1,2,1,2,7]]

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
        mixer = {'type':'x'}

        ## Testing

        # Correct solution
        exact_sol = {'101010101010': -12, '010101010101': -12}

        # Compute the solution for each elimination scheme
        for schedule in schedules:
            
            # Compute RQAOA solution
            result = custom_rqaoa(hamiltonian, mixer, p = p, steps = schedule)
            full_sol = result['solution']

            # Test the assigned energy to each solution is correct
            for key in full_sol.keys():
                assert full_sol[key] == exact_sol[key], f'The computed energy of the state is incorrect'


        # EXAMPLE 2

        ## Problem definition
        n_qubits = 10

        # Elimination scheme
        step = 1
        n_cutoff = 3

        # Number of QAOA layers
        p = 1

        # Ring graph
        G = nx.circulant_graph(n_qubits,[1])

        # Minimum vertex cover parameters
        field = 1.0
        penalty = 10

        # Define problem instance
        mvc = MinimumVertexCover(G,field = field,penalty = penalty).get_qubo_problem()

        # Cost and mixer Hamiltonians
        hamiltonian = Hamiltonian.classical_hamiltonian(mvc.terms, mvc.weights, mvc.constant)
        mixer = {'type':'x'}

        # Correct solution
        exact_sol = {'1010101010': 5, '0101010101': 5}

        # Compute Ada-RQAOA solution
        result = custom_rqaoa(hamiltonian, mixer, p = p, steps = step, n_cutoff = n_cutoff)
        full_sol = result['solution']

        # Test the assigned energy to each solution is correct
        for key in full_sol.keys():
            assert full_sol[key] == exact_sol[key], f'The computed energy of the state is incorrect'

        # EXAMPLE 3

        ## Problem definition
        n_qubits = 10

        # Elimination scheme
        step = 2
        n_cutoff = 3

        # Number of QAOA layers
        p = 1

        # Ring graph
        G = nx.complete_graph(n_qubits)

        # Minimum vertex cover parameters
        field = 1.0
        penalty = 10

        # Define problem instance
        mvc = MinimumVertexCover(G,field = field,penalty = penalty).get_qubo_problem()

        # Cost and mixer Hamiltonians
        hamiltonian = Hamiltonian.classical_hamiltonian(mvc.terms, mvc.weights, mvc.constant)
        mixer = {'type':'x'}

        # Correct solution
        exact_sol = {'0111111111': 9, '1011111111': 9, '1101111111': 9, '1110111111': 9, '1111011111': 9,
        '1111101111': 9, '1111110111': 9,'1111111011': 9, '1111111101': 9,'1111111110': 9}

        # Compute Ada-RQAOA solution
        result = custom_rqaoa(hamiltonian, mixer, p = p, steps = step, n_cutoff = n_cutoff)
        full_sol = result['solution']

        # Test the assigned energy to each solution is correct
        for key in full_sol.keys():
            assert full_sol[key] == exact_sol[key], f'The computed energy of the state is incorrect'


        # EXAMPLE 4

        ## Problem definition

        # Number of qubits
        n_qubits = 10

        # Elimination schemes
        schedules = [1,2,3]

        # Number of QAOA layers
        p = 1

        # Edges and weights of the graph
        edges = [(i,j) for j in range(n_qubits) for i in range(j)]
        weights = [1 for _ in range(len(edges))]

        # Cost and mixer Hamiltonians
        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant = 0)
        mixer = {'type':'x'}

        ## Testing

        # Correct solution
        exact_states = ['1111100000', '1111010000', '1110110000', '1101110000', '1011110000', '0111110000', '1111001000', '1110101000', '1101101000', '1011101000', '0111101000', '1110011000', '1101011000', '1011011000', '0111011000', '1100111000', '1010111000', '0110111000', '1001111000', '0101111000', '0011111000', '1111000100', '1110100100', '1101100100', '1011100100', '0111100100', '1110010100', '1101010100', '1011010100', '0111010100', '1100110100', '1010110100', '0110110100', '1001110100', '0101110100', '0011110100', '1110001100', '1101001100', '1011001100', '0111001100', '1100101100', '1010101100', '0110101100', '1001101100', '0101101100', '0011101100', '1100011100', '1010011100', '0110011100', '1001011100', '0101011100', '0011011100', '1000111100', '0100111100', '0010111100', '0001111100', '1111000010', '1110100010', '1101100010', '1011100010', '0111100010', '1110010010', '1101010010', '1011010010', '0111010010', '1100110010', '1010110010', '0110110010', '1001110010', '0101110010', '0011110010', '1110001010', '1101001010', '1011001010', '0111001010', '1100101010', '1010101010', '0110101010', '1001101010', '0101101010', '0011101010', '1100011010', '1010011010', '0110011010', '1001011010', '0101011010', '0011011010', '1000111010', '0100111010', '0010111010', '0001111010', '1110000110', '1101000110', '1011000110', '0111000110', '1100100110', '1010100110', '0110100110', '1001100110', '0101100110', '0011100110', '1100010110', '1010010110', '0110010110', '1001010110', '0101010110', '0011010110', '1000110110', '0100110110', '0010110110', '0001110110', '1100001110', '1010001110', '0110001110', '1001001110', '0101001110', '0011001110', '1000101110', '0100101110', '0010101110', '0001101110', '1000011110', '0100011110', '0010011110', '0001011110', '0000111110', '1111000001', '1110100001', '1101100001', '1011100001', '0111100001', '1110010001', '1101010001', '1011010001', '0111010001', '1100110001', '1010110001', '0110110001', '1001110001', '0101110001', '0011110001', '1110001001', '1101001001', '1011001001', '0111001001', '1100101001', '1010101001', '0110101001', '1001101001', '0101101001', '0011101001', '1100011001', '1010011001', '0110011001', '1001011001', '0101011001', '0011011001', '1000111001', '0100111001', '0010111001', '0001111001', '1110000101', '1101000101', '1011000101', '0111000101', '1100100101', '1010100101', '0110100101', '1001100101', '0101100101', '0011100101', '1100010101', '1010010101', '0110010101', '1001010101', '0101010101', '0011010101', '1000110101', '0100110101', '0010110101', '0001110101', '1100001101', '1010001101', '0110001101', '1001001101', '0101001101', '0011001101', '1000101101', '0100101101', '0010101101', '0001101101', '1000011101', '0100011101', '0010011101', '0001011101', '0000111101', '1110000011', '1101000011', '1011000011', '0111000011', '1100100011', '1010100011', '0110100011', '1001100011', '0101100011', '0011100011', '1100010011', '1010010011', '0110010011', '1001010011', '0101010011', '0011010011', '1000110011', '0100110011', '0010110011', '0001110011', '1100001011', '1010001011', '0110001011', '1001001011', '0101001011', '0011001011', '1000101011', '0100101011', '0010101011', '0001101011', '1000011011', '0100011011', '0010011011', '0001011011', '0000111011', '1100000111', '1010000111', '0110000111', '1001000111', '0101000111', '0011000111', '1000100111', '0100100111', '0010100111', '0001100111', '1000010111', '0100010111', '0010010111', '0001010111', '0000110111', '1000001111', '0100001111', '0010001111', '0001001111', '0000101111', '0000011111']

        exact_sol = {state:-5 for state in exact_states}

        for schedule in schedules:
            
            # Compute RQAOA solution
            result = custom_rqaoa(hamiltonian, mixer, p = p, steps = schedule)
            full_sol = result['solution']

            # Test the assigned energy to each solution is correct
            for key in full_sol.keys():
                assert full_sol[key] == exact_sol[key], f'The computed energy of the state is incorrect'

        ## Exception case - insufficient elimination steps fed into custom RQAOA
        n_qubits = 10
        edges = [(i,j) for j in range(n_qubits) for i in range(j)]
        weights = [1 for _ in range(len(edges))]
        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant = 0)
        mixer = {'type':'x'}
        n_cutoff = 5

        # Generate custom mixer
        steps = [1,2]

        # Attempt running custom RQAOA
        with self.assertRaises(AssertionError) as context:
            result = custom_rqaoa(hamiltonian, mixer, steps = steps, n_cutoff = n_cutoff)
            
        # Check exception message
        self.assertEqual(f"Schedule is incomplete, add {np.abs(n_qubits - n_cutoff) - sum(steps)} more eliminations", str(context.exception))


        ## Exception case - custom connectivity fed into custom RQAOA
        n_qubits = 7
        edges = [(i,j) for j in range(n_qubits) for i in range(j)]
        weights = [1 for _ in range(len(edges))]
        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant = 0)

        # Generate custom mixer
        connectivity = [(i,i+1) for i in range(n_qubits-1)] + [(0,n_qubits-1)]
        mixer = {'type':'xy','connectivity': connectivity}

        # Attempt running custom RQAOA
        with self.assertRaises(NotImplementedError) as context:
            result = custom_rqaoa(hamiltonian, mixer)
            
        # Check exception message
        self.assertEqual("Custom mixer connectivities are not currently supported", str(context.exception))

if __name__ == "__main__":
	unittest.main()
 