import random
import numpy as np
from copy import deepcopy
from typing import Dict

from .basebackend import VQABaseBackend

from .cost_function import cost_function

from ..qaoa_components.ansatz_constructor.gates import (
    X
)

from qiskit.circuit.library import (XGate,)



from ..qaoa_components import Hamiltonian

from ..utilities import exp_val_hamiltonian_termwise, energy_expectation, exp_val_pair, exp_val_single, negate_counts_dictionary

class BaseWrapper(VQABaseBackend):
    def __init__(self, backend):
        self.backend = backend
        
    def __getattr__(self, name):
        return getattr(self.backend, name)
    
    def expectation(self, *args, **kwargs):
        return self.backend.expectation(*args, **kwargs)
    
    def expectation_w_uncertainty(self, *args, **kwargs):
        return self.backend.expectation_w_uncertainty(*args, **kwargs)
    
    def exact_solution(self, *args, **kwargs):
        return self.backend.exact_solution(*args, **kwargs)
    

class SPAMTwirlingWrapper(BaseWrapper):
    def __init__(self, backend, n_batches):
        super().__init__(backend)
        self.n_batches = n_batches
        
        #self.lambda_singles = ... 
        # self.calibration_factors = # TODO
    
    def get_counts(self, params, n_shots = None):
        '''
        Modified function to...
            divide into batches
            change the abstract circuit (deepcopy) according to the schedule, s.
            get the counts and classically negate them
            combine all batches into a count dict under BFA
        '''
        # list of integers whose binary representation signifies which qubits or be flipped at every batch 
        s_list = [] 
        for _ in range(0, self.n_batches):
            s_list.append(random.getrandbits(self.backend.n_qubits)) 
        #s_list = [3, 0, 1, 2] # TESTING ONLY, can be specified by the user 
        #s_list = [1, 1, 1, 1]
        
        n_shots = self.backend.n_shots if n_shots == None else n_shots
        n_shots_batch = int(n_shots / self.n_batches)
        
        counts = {}
        
        for batch in range(0, self.n_batches):
            #print("batch ", batch)
            s = s_list[batch]
            s_binary = format(s, 'b').zfill(self.backend.n_qubits) # convert to binary
            arr = np.fromiter(s_binary, dtype=int)
            negated_qubits = np.where(arr == 1)[0] # where the syndrome has a 1

            circuit_to_append = self.backend.gate_applicator.create_quantum_circuit(self.backend.n_qubits)
            
            for negated_qubit in negated_qubits:
                circuit_to_append = self.backend.gate_applicator.apply_1q_fixed_gate(qiskit_gate=XGate, qubit_1=negated_qubit, circuit=circuit_to_append)
                #circuit_to_append = self.backend.gate_applicator.apply_gate(X, negated_qubit, circuit_to_append)
            
            self.backend.append_state = circuit_to_append
            #print(self.backend.append_state)
        
            counts_batch = self.backend.get_counts(params, n_shots_batch) # should call the original get_counts of the specific backend
            
            negated_counts = negate_counts_dictionary(counts_dictionary = counts_batch, s = s)
            
            # Add to the final counts dict
            for key in negated_counts:
                if key in counts:
                    counts[key] = counts[key] + negated_counts[key]
                else:
                    counts.update([(key, negated_counts[key])])
        
        
        self.measurement_outcomes = counts
        
        return counts
    
    def expectation_value_spam_twirled(self, counts: Dict, hamiltonian: Hamiltonian, lambdas_single: list, lambdas_double: list):
        """
        need to have access to some calibration data here for correcting
        """
        # Initialize the z expectation values and correlation matrix with 0s
        n_qubits = hamiltonian.n_qubits
        exp_vals_z = np.zeros(n_qubits)
        corr_matrix = np.zeros((n_qubits, n_qubits))
        
        terms = [term.qubit_indices for term in hamiltonian.terms]
        coeffs = [coeff for coeff in hamiltonian.coeffs]
        hamiltonian_as_dict = dict(zip(terms,coeffs))
        
        energy = 0

        # Compute expectation values and correlations of terms present in the Hamiltonian
        for term in hamiltonian.terms:

            # If bias term compute expectation value
            if len(term) == 1:
                i = term.qubit_indices[0]
                exp_vals_z[i] = exp_val_single(i, counts)
                exp_vals_z[i] /= lambdas_single[i]

                energy += exp_vals_z[i] * self.qaoa_descriptor.cost_single_qubit_coeffs[i] 

            # If two-body term compute correlation
            elif len(term) == 2:
                i, j = term.qubit_indices
                corr_matrix[i][j] = exp_val_pair((i, j), counts)
                corr_matrix[i][j] /= lambdas_double[i][j]
                
                energy += corr_matrix[i][j] * hamiltonian_as_dict[(i,j)]  
                
            # If constant term, ignore
            if len(term) == 0:
                continue

        energy += hamiltonian.constant

        return energy
    


    def expectation(self, params, n_shots=None) -> float:
        '''
        use the total counts under BFA to compute expectation values Zi and ZiZj
        correct these expectation values with the calibration data, lambda_i, lambda_ij
        combine all corrected expectation values into the energy = cost fn to be given to the optimizer every time it calls expectation 
        '''
        #print(self.qaoa_descriptor)
        counts = self.get_counts(params, n_shots)
        
        hamiltonian = self.backend.qaoa_descriptor.cost_hamiltonian
        
        calibration_data = counts  # if empty circuit, otherwise should coem from the calibration data
        
        #TODO change, a bit hacky
        mixer_type = self.qaoa_descriptor.mixer_qubits_singles[1] or self.qaoa_descriptor.mixer_qubits_pairs[1:2]

        # TODO: lambdas should be dictionaries
        lambda_singles, lambda_pairs = exp_val_hamiltonian_termwise(variational_params = params,
                                                           qaoa_backend = self.backend,
                                                           hamiltonian = hamiltonian,
                                                           mixer_type = mixer_type,
                                                           p = self.qaoa_descriptor.p,
                                                           qaoa_optimized_counts = counts,
                                                           analytical = False,
                                                          )

        lambda_pairs += np.outer(lambda_singles, lambda_singles)
        #print(np.array2string(lambda_singles, separator=","))
        #print(np.array2string(lambda_pairs, separator=","))

        #cost = cost_function(counts, hamiltonian, self.cvar_alpha)
        #cost = energy_expectation(self.qaoa_descriptor.cost_hamiltonian, counts)
        
            
        cost = self.expectation_value_spam_twirled(counts, hamiltonian, lambda_singles, lambda_pairs)
            
            
        return cost
            
        
    
    
    