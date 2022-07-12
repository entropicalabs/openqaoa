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

import networkx as nx
import numpy as np
import itertools
import unittest

from openqaoa.devices import DeviceLocal
from openqaoa.utilities import *
from openqaoa.qaoa_parameters import PauliOp, Hamiltonian, QAOACircuitParams, create_qaoa_variational_params
from openqaoa.backends.qaoa_backend import get_qaoa_backend
from openqaoa.optimizers.qaoa_optimizer import get_optimizer
from openqaoa.problems.problem import MinimumVertexCover

"""
Unit test based testing of the utility functions
"""

class TestingUtilities(unittest.TestCase):

    def test_X_mixer_hamiltonian(self):
        """
        Tests the function that generates X mixer Hamiltonian.

        The test consists in generating X mixer Hamiltonians of different sizes.
        """

        # Set of sizes
        sizes = range(1,10)
        
        # Test for different sizes
        for n_qubits in sizes:
            
            # Define input coefficients
            input_coefficients = [-1 for _ in range(n_qubits)]

            # Define mixer Hamiltonian
            mixer_hamiltonian = X_mixer_hamiltonian(n_qubits,input_coefficients)

            # Extract mixer Hamiltonian attributes
            terms = mixer_hamiltonian.terms
            coefficients = mixer_hamiltonian.coeffs
            constant = mixer_hamiltonian.constant
            
            # Correct mixer hamiltonian attributes
            correct_terms = [PauliOp('X',(i,)) for i in range(n_qubits)]
            correct_coefficients = [-1 for _ in range(n_qubits)]
            correct_constant = 0

            # Test that the mixer Hamiltonian was correctly generated
            assert terms == correct_terms, f'The terms in the X mixer Hamiltonian were not generated correctly'
            assert np.allclose(coefficients,correct_coefficients), f'The coefficients in the X mixer Hamiltonian were not generated correctly'
            assert np.allclose(constant,correct_constant), f'The constant in the X mixer Hamiltonian was not generated correctly'

    def test_XY_mixer_hamiltonian(self):
        """
        Tests the function that generates the XY mixer Hamiltonian.

        The test consists in generating XY mixer Hamiltonians for different sizes and for the different available connectivities.
        The connectivities will be tested as input strings and as custom inputs
        """

        # Set of sizes
        sizes = range(2,10)

        # Set of default connectivities
        connectivities = ['full','chain','star']

        # String of Pauli objects contained in the XY mixer Hamiltonian
        terms_strings = ['XX','YY']

        # Set of uses
        input_type = ['string','custom']
        
        # Test for different sizes
        for n_qubits in sizes:
            
            # Test for different connectivities
            for connectivity in connectivities:

                # Test for different input types
                for input in input_type:

                    # Define connectivity explicit indexing
                    if connectivity == 'full':
                        terms_indices = list(itertools.combinations(range(n_qubits),2))

                    elif connectivity == 'chain':
                        terms_indices = [(i,i+1) for i in range(n_qubits-1)]

                    else:
                        terms_indices = [(0,i) for i in range(1,n_qubits)]
                
                    # Define input coefficients, account for two terms per connection (XX + YY)
                    input_coefficients = 2*[1 for _ in range(len(terms_indices))]

                    # Define mixer Hamiltonian
                    if input == 'string':
                        mixer_hamiltonian = XY_mixer_hamiltonian(n_qubits,connectivity,input_coefficients)

                    else:
                        mixer_hamiltonian = XY_mixer_hamiltonian(n_qubits,terms_indices,input_coefficients)

                    # Extract mixer Hamiltonian attributes
                    terms = mixer_hamiltonian.terms
                    coefficients = mixer_hamiltonian.coeffs
                    constant = mixer_hamiltonian.constant
                    
                    # Correct mixer hamiltonian attributes
                    correct_terms = [PauliOp(string,indices) for indices in terms_indices for string in terms_strings]
                    correct_coefficients = [1 for _ in range(len(correct_terms))]
                    correct_constant = 0
                    
                    # Test that the mixer Hamiltonian was correctly generated
                    assert terms == correct_terms, f'The terms in the XY mixer Hamiltonian were not generated correctly for {input} input'
                    assert np.allclose(coefficients,correct_coefficients), f'The coefficients in the XY mixer Hamiltonian were not generated correctly for {input} input'
                    assert np.allclose(constant,correct_constant), f'The constant in the XY mixer Hamiltonian was not generated correctly for {input} input'

        
        # Exception case - Insert an connectivity input name that is not supported

        # Supported connectivities
        supported_connectivities = ['full','chain','star']

        # Choose a wrong connectivity
        connectivity = 'bubbletea'

        # Attempt construction of Pauli operator 
        with self.assertRaises(ValueError) as context:
            mixer_hamiltonian = XY_mixer_hamiltonian(n_qubits,connectivity,input_coefficients)

        # Check exception message
        self.assertEqual(f"Please choose connection topology from {supported_connectivities}", str(context.exception))

    def test_get_mixer_hamiltonian(self):
        """
        Tests the generic function for retrieving mixer Hamiltonians.

        The test consists in retrieving mixing Hamiltonians from the different available types.
        """

        # Define number of qubits
        n_qubits = 10

        # Define a qubit connectivity for X, and a connectivity for XY mixers
        connectivities = [None,'star','full','chain']

        for connectivity in connectivities:

            # Define connectivity explicit indexing
            if connectivity == 'full':
                terms_indices = list(itertools.combinations(range(n_qubits),2))
                input_coefficients = 2*[1 for _ in range(len(terms_indices))]
                mixer_type = 'xy'

            elif connectivity == 'chain':
                terms_indices = [(i,i+1) for i in range(n_qubits-1)]
                input_coefficients = 2*[1 for _ in range(len(terms_indices))]
                mixer_type = 'xy'

            elif connectivity == 'star':
                terms_indices = [(0,i) for i in range(1,n_qubits)]
                input_coefficients = 2*[1 for _ in range(len(terms_indices))]
                mixer_type = 'xy'

            else:
                input_coefficients = [1 for _ in range(n_qubits)]
                mixer_type = 'x'

            # Retrieve Hamiltonian and attributes
            mixer_hamiltonian = get_mixer_hamiltonian(n_qubits,mixer_type,connectivity,input_coefficients)
            terms = mixer_hamiltonian.terms
            coefficients = mixer_hamiltonian.coeffs

            # Generate correct Hamiltonian and attributes
            if mixer_type == 'xy':
                correct_mixer_hamiltonian = XY_mixer_hamiltonian(n_qubits,connectivity,input_coefficients)

            else:
                correct_mixer_hamiltonian = X_mixer_hamiltonian(n_qubits,input_coefficients)
                
            correct_terms = correct_mixer_hamiltonian.terms
            correct_coefficients = correct_mixer_hamiltonian.coeffs

            # Test that the mixer Hamiltonian was correctly generated
            assert terms == correct_terms, f'The terms in the mixer Hamiltonian were not generated correctly for {connectivity} connectivity'
            assert np.allclose(coefficients,correct_coefficients), f'The coefficients in the mixer Hamiltonian were not generated correctly for {connectivity} connectivity'
             
    def test_graph_from_hamiltonian(self):
        """
        Tests the function that extracts the underlying graph from a Hamiltonian.

        The test consists in finding the graph for a given Hamiltonian.
        """

        # Number of qubits
        n_qubits = 10

        # Define hamiltonian
        terms = [PauliOp('ZZ',(i,j)) for j in range(n_qubits) for i in range(j)] + [PauliOp('Z',(i,)) for i in range(n_qubits)]
        coefficients = [0.5 for j in range(n_qubits) for _ in range(j)] + [-1 for _ in range(n_qubits)]
        constant = 0
        hamiltonian = Hamiltonian(terms,coefficients,constant)

        # Extract graph attributes from hamiltonian
        graph = graph_from_hamiltonian(hamiltonian)

        singlet_edges = list({(node,):graph.nodes[node]['weight'] for node in graph.nodes if graph.nodes[node].get('weight') is not None}.keys())
        pair_edges = list(nx.get_edge_attributes(graph, 'weight').keys())
        edges = singlet_edges + pair_edges

        pair_weights = list(nx.get_edge_attributes(graph, 'weight').values())
        singlet_weights = list({(node,):graph.nodes[node]['weight'] for node in graph.nodes if graph.nodes[node].get('weight') is not None}.values())
        weights = singlet_weights + pair_weights

        graph_dict = {edge:weight for edge,weight in zip(edges,weights)}

        # Correct graph attributes
        correct_edges = [term.qubit_indices for term in terms]
        correct_weights = [0.5 for _ in range(int(n_qubits*(n_qubits-1)/2))] + [-1 for _ in range(n_qubits)]

        correct_graph_dict = {edge:weight for edge,weight in zip(correct_edges,correct_weights)}

        # Test that the graph contains the correct number of edges
        assert len(graph_dict) == len(correct_graph_dict), f'An incorrect number of edges was generated'

        # Test that graph attributes were obtained correctly
        for edge in graph_dict.keys():
            assert np.allclose(graph_dict[edge], correct_graph_dict[edge]), f'Weights were not obtained correctly'

    def test_hamiltonian_from_graph(self):
        """
        Test the function that generates a classical Hamiltonian from a graph

        Tests
        """

        # Number of nodes
        n_qubits = 10

        # Define graph
        pair_edges = [(i,j) for j in range(n_qubits) for i in range(j)]
        singlet_edges = [(i,) for i in range(n_qubits)]
        edges = singlet_edges + pair_edges

        graph = nx.Graph()

        # Populate the graph with weighted nodes and weighted edges
        for edge in edges:
            
            # Weighted node
            if len(edge) == 1:
                graph.add_node(edge[0],weight = -1)
            
            # Weighted edge
            else:
                graph.add_edge(edge[0], edge[1], weight=0.5)

        # Generate Hamiltonian and extract attributes
        hamiltonian = hamiltonian_from_graph(graph)

        hamiltonian_dict = {term.qubit_indices:coeff for term,coeff in zip(hamiltonian.terms,hamiltonian.coeffs)}
        constant = hamiltonian.constant

        # Correct Hamiltonian attributes
        correct_terms = [PauliOp('Z',(i,)) for i in range(n_qubits)] + [PauliOp('ZZ',(i,j)) for j in range(n_qubits) for i in range(j)]
        correct_coefficients = [-1 for _ in range(n_qubits)] + [0.5 for j in range(n_qubits) for _ in range(j)]
        correct_hamiltonian_dict = {term.qubit_indices:coeff for term,coeff in zip(correct_terms,correct_coefficients)}
        correct_constant = 0

        assert hamiltonian_dict == correct_hamiltonian_dict, f'The terms and coefficients were not correctly generated'
        assert np.allclose(constant,correct_constant), f'The constant was not correctly generated'

    def test_random_k_regular_graph(self):
        """
        Test the function that generates random regular graphs.

        The test consists in generating graphs for different degrees.
        """

        # Number of nodes
        n_nodes = 6

        # Set of degrees
        degrees = [2,3,4,5]

        # Check for every degree
        for degree in degrees:

            # Check for weighted and unweighted
            for weighted in [False,True]:

                for biases in [False,True]:

                    # Generate graph
                    graph = random_k_regular_graph(degree,list(range(n_nodes)), weighted = weighted, biases = biases)

                    # Test it has the correct number of edges for a regular graph
                    assert np.allclose(len(list(graph.edges)), degree*n_nodes/2), f'The number of edges is not correct'

                    # Test graph is properly unweighted
                    if weighted is False:
                        assert np.allclose(list(nx.get_edge_attributes(graph, 'weight').values()),[1 for _ in range(int(degree*n_nodes/2))]), f'Graph is not unweighted'

    def test_random_classical_hamiltonian(self):
        """"
        Tests the function that generates a random classical Hamiltonian.
        
        The test consists in checking that the generated Hamiltonian is indeed classical (containing only 'Z' type pauils).
        """

        # Number of qubits
        sizes = list(range(2,10))

        # Test for different qubit numbers
        for n_qubits in sizes:

            hamiltonian = random_classical_hamiltonian(reg = list(range(n_qubits)))

            # Test the hamiltonian is a Hamiltonian object
            assert isinstance(hamiltonian,Hamiltonian), f'The object is not Hamiltonian object'

            # Check all terms are combination of 'Z' Paulis
            terms = hamiltonian.terms

            for term in terms:
                assert term.pauli_str  in ('Z','ZZ'), f'Found non-classical term'

    def test_ground_state_hamiltonian(self):
        """
        Tests the function that obtains the ground state and minimum energy of a Hamiltonian.

        The test consists in computing the ground state and minimum energy for given examples.
        """
        # Number of variables/qubits
        n_qubits = 10 

        # Terms and weights of the graph
        edges = [(i,i+1) for i in range(n_qubits-1)] + [(0,n_qubits-1)] # Ring structure
        weights = [1 for _ in range(len(edges))] # All weights equal to 1

        # Define Hamiltonian
        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant = 0) 

        ## Testing

        # Correct solutions to the problem
        correct_energy = -10
        correct_states = ['1010101010','0101010101']

        # Obtain solution from function
        energy, states = ground_state_hamiltonian(hamiltonian)
        
        # Test function result
        assert np.allclose(energy,correct_energy), f'Computed solutions are incorrect'
        assert states == correct_states , f'Computed solutions are incorrect'

    def test_bitstring_energy(self):
        """
        Test the function that computes the energy of a given string for a specificied 
        cost Hamiltonian.

        The test consists in computing the energy of a few strings given a specific Hamiltonian.
        """

        # Number of qubits
        n_qubits = 10

        # Edges and weights defining the problem graph
        edges = [(i,i+1) for i in range(n_qubits-1)] + [(0,n_qubits-1)]
        weights = [1 for _ in range(len(edges))]

        # Hamiltonian
        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant = 0)

        # Input states
        states = ['0101010101',[1,0,1,0,1,0,1,0,1,0]]

        # Correct energies
        correct_energy = -10

        # Test for each string
        for state in states:

            # Compute energies for the given trial states
            energy = bitstring_energy(hamiltonian,state)
            
            # Test computed solution is correcrt
            assert np.allclose(correct_energy,energy), f'Computed energy is incorrect'

    def test_energy_expectation(self):
        """
        Tests the function that computes the expectation value of a classical Hamiltonian
        for a dictionary containing measurement counts.

        The test consists in computing the expectation value from a given Hamiltonian
        and a given measurement counts dictionary.
        """

        # Number of qubits
        n_qubits = 10

        # Define edges and weights determining the problem graph
        edges = [(i,i+1) for i in range(n_qubits-1)] + [(0,n_qubits-1)]
        weights = [1 for _ in range(len(edges))]
        
        # Hamiltonian
        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant = 0)

        # Input measurement counts dictionary
        input_measurement_counts = {'0101010101':10,'1010101010':10,'0000000000':10,'1111111111':10,'1111111110':10}

        # Obtain energy expectation value
        energy = energy_expectation(hamiltonian,input_measurement_counts)
        
        # Correct energy
        correct_energy = 1.2
        
        # Test energy was computed correctly
        assert np.allclose(energy,correct_energy), f'The energy expectation value was not computed correctly'
    
    def test_energy_spectrum_hamiltonian(self):
        """
        Tests the function that computes the exact energy spectrum from a given classical Hamiltonian.

        The test consists in computing the spectrum of a given classical Hamiltonian.
        """

        # Number of qubits
        n_qubits = 3

        # Edges and weights defining the classical Hamiltonian
        edges = [(i,i+1) for i in range(n_qubits-1)] + [(0,n_qubits-1)]
        weights = [1 for _ in range(len(edges))]
        constant = 10

        # Hamiltonian
        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant)

        # Extract energy spectrum and sort it
        energies = energy_spectrum_hamiltonian(hamiltonian)
        energies.sort()
        # Correct energies
        correct_energies = [9,9,9,9,9,9,13,13]

        # Test computed energies are correct
        assert np.allclose(energies,correct_energies), f'Energy spectrum was not computed correctly'

    def test_low_energy_states(self):
        """
        Test the function that retrieves the energy eigenstates which are below
        a certain threshold percentage away from the ground state energy.

        The test consists in computing the low energy states for a given Hamiltonian
        and a given threshold percentage.
        """

        # Define threshold
        threshold_percentage = 3/10

        # Define Hamiltonian
        terms = [PauliOp('ZZ',(0,1)),PauliOp('ZZ',(1,2)),PauliOp('ZZ',(0,2))]
        coeffs = [1,2,3]
        constant = 0

        hamiltonian = Hamiltonian(terms,coeffs,constant)

        # Obtain low energy states and the low energy threshold
        low_energy_threshold,states = low_energy_states(hamiltonian,threshold_percentage)

        # Correct low energy state and low energy threshold
        correct_low_energy_threshold = -1
        correct_states = ['100','011','001','110']

        # Test that low energy threshold and states were correctly retrieved
        assert np.allclose(low_energy_threshold,correct_low_energy_threshold), f'Low energy threshold was not computed correctly'
        assert set(states) == set(correct_states), f'Low energy states were not computed correctly'

    def test_low_energy_states_overlap(self):
        """
        Test the function that computes the overlap between the low energy states, defined
        by a threshold percentage from the gound state energy, and a given input state, expressed
        in terms of a probability dictionary.
        
        The test consists in computing the overlap for a given Hamiltonian and a given input state.
        """

        # Define threshold
        threshold_percentage = 3/10

        # Define Hamiltonian
        terms = [PauliOp('ZZ',(0,1)),PauliOp('ZZ',(1,2)),PauliOp('ZZ',(0,2))]
        coeffs = [1,2,3]
        constant = 0

        hamiltonian = Hamiltonian(terms,coeffs,constant)

        # Input state probability dictionary
        prob_dict = {'000':20,'001':10,'010':10,'100':10,'011':20,'101':10,'110':10,'111':10}

        # Obtain low energy states overlap
        overlap = low_energy_states_overlap(hamiltonian,threshold_percentage,prob_dict)

        # Correct overlap
        correct_overlap = 0.5

        # Test overlap has been generated correctly
        assert np.allclose(overlap,correct_overlap), f'The overlap was not computed correctly'

    def test_ring_of_disagrees(self):
        """
        Tests the function that constructs the ring of disagrees, as decribed in (https://arxiv.org/abs/1411.4028).

        The test consists in generating rings of disagrees of different sizes.
        """

        # Set of sizes
        sizes = range(3,10)

        # Uniform case
        for n_qubits in sizes:
            
            # Generate ring of disagrees and extract attributes
            rod_hamiltonian = ring_of_disagrees(reg = list(range(n_qubits)))

            rod_terms = rod_hamiltonian.terms
            rod_coefficients = rod_hamiltonian.coeffs
            rod_constant = rod_hamiltonian.constant

            # Correct ring of disagrees hamiltonian
            correct_rod_terms = [PauliOp('ZZ',(i,i+1)) for i in range(n_qubits-1)] + [PauliOp('ZZ',(0,n_qubits-1))]
            correct_rod_coefficients = [0.5 for _ in range(len(correct_rod_terms))]
            correct_rod_constant = -n_qubits*0.5

            # Test the ring of disagrees Hamiltonian was properly generated
            assert rod_terms == correct_rod_terms, f'The terms in the uniform ROD Hamiltonian were not generated correctly'
            assert np.allclose(rod_coefficients,correct_rod_coefficients), f'The coefficients in the uniform ROD Hamiltonian were not generated correctly'
            assert np.allclose(rod_constant,correct_rod_constant), f'The constant in the uniform ROD Hamiltonian was not generated correctly'

    def test_exp_val_hamiltonian_termwise_analytical(self):
        """
        Test of the function that computes singlet expectation values and correlations terms
        analytically for p = 1 and the function computing the full set of expectation values
        when analytical results can be obtained (p=1).

        NOTE: Correlations in the exp_val_pair_analytical() and exp_val_pair() functions are computed 
        as average value <Z_{i}Z_{j}>, meaning it includes the <Z_{i}><Z_{j}> contribution. 
        This is subtracted by default in the exp_val_hamiltonian_termwise() function.

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
            comp_exp_val = np.round(exp_val_single_analytical(spin,hamiltonian,qaoa_angles),16)
            comp_corr = np.round(exp_val_pair_analytical(pair,hamiltonian,qaoa_angles),16)

            # Test if computed results are correct
            assert np.allclose(exp_val,comp_exp_val), f'Incorrectly computed singlet expectation value'
            assert np.allclose(corr,comp_corr), f'Incorrectly computed correlation term'

        # Fix a set of angles for testing full set of expectation values and correlations
        fixed_angles = [np.pi/4,np.pi/8]

        # Correct solutions
        exp_val_list = np.array([-np.sqrt(2)/4,np.sqrt(2)/4,-np.sqrt(2)/4,np.sqrt(2)/4])
        corr_matrix = np.array([[0.0,-1/4,0,-1/4],\
                              [0.0,0.0,-1/4,0],\
                              [0.0,0.0,0.0,-1/4],\
                              [0.0,0.0,0.0,0.0]])

        corr_matrix -= np.outer(exp_val_list,exp_val_list)

        # Compute list of expectation values and correlation matrix
        comp_exp_val_list, comp_corr_matrix = exp_val_hamiltonian_termwise(variational_params = None,
                                                                        qaoa_backend = None,
                                                                        hamiltonian = hamiltonian,
                                                                        p = 1, 
                                                                        mixer_type='x',
                                                                        qaoa_optimized_angles = fixed_angles,
                                                                        analytical=True)

        # Test if computed results are correct
        assert np.allclose(exp_val_list,comp_exp_val_list), f'Computed set of singlet expectation values is incorrect'

        for j in range(len(comp_corr_matrix)):
            assert np.allclose(corr_matrix[j],comp_corr_matrix[j]), f'Computed correlation matrix is incorrect'
    
    def test_exp_val_hamiltonian_termwise(self):
        """
        Test of the function that computes singlet expectation values and correlations numerically through
        the QAOA output distribution of states.

        The test consist of computing the singlet expectation values and correlations for a given problem.
        The result is constrasted with the analytical result, whose implementation is tested in 
        test_exp_val_hamiltonian_termwise_analytical().
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
        hamiltonian = Hamiltonian.classical_hamiltonian(edges, weights, constant = 10)

        # Mixer Hamiltonian
        mixer_hamiltonian = X_mixer_hamiltonian(n_qubits)

        # Define circuit and variational parameters
        circuit_params = QAOACircuitParams(hamiltonian,mixer_hamiltonian, p = p)
        variational_params = create_qaoa_variational_params(circuit_params, params_type = 'standard', init_type = 'ramp')

        ## Testing

        # Perform QAOA and obtain expectation values numerically
        qaoa_backend = get_qaoa_backend(circuit_params, device = DeviceLocal('vectorized'))
        optimizer = get_optimizer(qaoa_backend, variational_params, optimizer_dict = {'method':'cobyla','maxiter':200})
        optimizer()
        qaoa_results = optimizer.qaoa_result
        
        qaoa_results_optimized = qaoa_results.optimized 
        qaoa_optimized_angles = qaoa_results_optimized['optimized angles']
        qaoa_optimized_counts = qaoa_results.get_counts(qaoa_results_optimized['optimized measurement outcomes'])
        num_exp_vals_z, num_corr_matrix = exp_val_hamiltonian_termwise(variational_params, 
            qaoa_backend, hamiltonian, 'x', p, qaoa_optimized_angles, qaoa_optimized_counts, analytical=False)

        # Analytical expectation values
        exp_vals_z, corr_matrix = exp_val_hamiltonian_termwise(variational_params, 
            qaoa_backend, hamiltonian, 'x', p, qaoa_optimized_angles, qaoa_optimized_counts, analytical=True)


        # Test if computed results are correct
        assert np.allclose(exp_vals_z,num_exp_vals_z), f'Computed singlet expectation values are incorrect'

        for j in range(len(num_corr_matrix)):
            assert np.allclose(corr_matrix[j],num_corr_matrix[j]), f'Computed correlation matrix is incorrect'

    def test_energy_expectation_analytical(self):
        """
        Tests the function that computed the expectation value of the Hamiltonian from an
        analytical function valid for a single layer QAOA Ansatz with the X mixer.

        The test consists in computing the expectation value of an example Hamiltonian for different angles.
        """
        ## Problem definition - Minimum Vertex Cover on a Ring

        # Number of qubits
        n_qubits = 6

        # Edges of the graph
        edges = [(i,i+1) for i in range(n_qubits-1)] + [(0,n_qubits-1)]

        # Define graph and add edges
        G = nx.Graph()
        G.add_edges_from(edges)

        # QUBO instance of the problem
        field = 1.0
        penalty = 10.0
        mvc = MinimumVertexCover(G, field=field, penalty=penalty).get_qubo_problem()

        # Minimum Vertex Cover Hamiltonian
        hamiltonian = Hamiltonian.classical_hamiltonian(mvc.terms, mvc.weights, mvc.constant)

        # Set of angles on which to compute the energy
        angle_set = [(0,0),(np.pi/4,0),(2*np.pi,np.pi),(np.pi/8,np.pi/2)]

        # Test correct and computed energies
        for angles in angle_set:
            b,g = angles

            correct_energy = n_qubits*(-np.sin(2*b)*np.sin(2*g*field)*np.cos(g*penalty/2) + \
            (1/2)*np.sin(2*b)**2*np.cos(g*penalty/2)**2*(1 - np.cos(2*g*field)) - \
                np.sin(4*b)*np.cos(g*field)*np.sin(g*penalty/2)*np.cos(g*penalty/2) ) + mvc.constant

            energy = energy_expectation_analytical(angles,hamiltonian)
            
            assert energy == correct_energy, f'Computed energy {energy} is not equal to correct value {correct_energy} for (beta,gamma) = {(b,g)}'

    def test_flip_counts(self):
        """
        Tests the function that returns a probability dictionary but with flipped keys.

        The test consists in flipping the keys for a given set of examples dictionaries.
        """
        
        # Input dictionaries
        input_dicts = [{'0':2/3, '1':1/3}, {'00':0,'01':0,'10':1/2,'11':1/2},
                                {'000':0,'001':1/3,'010':0,'100':0,'011':1/3,'101':0,'110':1/3,'111':0}]

        # Flip the keys of the dictionaries
        dicts = [flip_counts(input) for input in input_dicts]

        # Correct dictionaries
        correct_dicts = [{'0':2/3, '1':1/3}, {'00':0,'01':1/2,'10':0,'11':1/2},
                                {'000':0,'001':0,'010':0,'100':1/3,'011':1/3,'101':0,'110':1/3,'111':0}]


        # Test that fictionary keys have been flipped correctly
        assert dicts == correct_dicts, f'Dictionary key flipping has not been performed correctly'
        
    def test_qaoa_probabilities(self):
        """
        Tests the function that generates a qiskit-style probability dictionary from a statevector.

        The test consist in retrieving the probability dictionary from a set of example statevectors.
        """

        # State vectors
        state_vecs = [np.array(([np.sqrt(2)/np.sqrt(3),-1j/np.sqrt(3)]),dtype = complex),
                    np.array(([-1/2,-1/2,1j/2,-1j/2]),dtype = complex),
                    np.array(([1,0,0,0,0,0,0,0]),dtype = complex)]

        # Compute probability dictionaries
        prob_dicts = [qaoa_probabilities(state_vec) for state_vec in state_vecs]
    
        # Correct probability dictionaries
        correct_prob_dicts = [{'0':2/3, '1':1/3}, {'00':1/4,'01':1/4,'10':1/4,'11':1/4},
                                {'000':1,'001':0,'010':0,'100':0,'011':0,'101':0,'110':0,'111':0}]

        # Test that each probability dictionary has been generated correctly
        for idx,prob_dict in enumerate(prob_dicts):
            
            # Check each string has the same probability associated
            for string in prob_dict.keys():
                assert np.allclose(prob_dicts[idx][string],correct_prob_dicts[idx][string]), f'Probablity have not been generated correctly'
        
if __name__ == "__main__":
	unittest.main()
