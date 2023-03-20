import random
import numpy as np
from copy import deepcopy
from typing import Dict

from .basebackend import VQABaseBackend

from .cost_function import cost_function

from ..qaoa_components.ansatz_constructor.gatemap import (
    XGateMap
)

from ..qaoa_components import Hamiltonian

from ..utilities import exp_val_hamiltonian_termwise, energy_expectation, exp_val_pair, exp_val_single

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
        #print("negating schedule ", s_list)
        #s_list = [3, 0, 1, 2] # TESTING ONLY, can be specified by the user 
        #s_list = [1, 1, 1, 1]
        
        n_shots = self.backend.n_shots if n_shots == None else n_shots
        n_shots_batch = int(n_shots / self.n_batches)
        
        counts = {}
        
        for batch in range(0, self.n_batches):
            #print("batch ", batch)
            s = s_list[batch]
            
            negated_counts = {}
           
            s_binary = format(s, 'b').zfill(self.backend.n_qubits) # convert to binary
            arr = np.fromiter(s_binary, dtype=int)
            negated_qubits = np.where(arr == 1)[0] # where the syndrome has a 1
            #print("qubits to negate ", negated_qubits)

            # create a new copy of the initial abstract circuit
            append_abstract_circuit = []

            for negated_qubit in negated_qubits:
                append_abstract_circuit.append(XGateMap(qubit_1 = negated_qubit))
            
            self.backend.append_state = self.backend.from_abstract_to_real(append_abstract_circuit)
        
            counts_batch = self.backend.get_counts(params, n_shots_batch) # should call the original get_counts of the specific backend
            #print(" batch counts before negating ", counts_batch)
            
            # consider putting this in utilities.py, similar to permuted_counts with SWAP gates (?)
            for key in counts_batch.keys():   
                negated_key = s ^ int(key, 2)  # bitwise XOR to classically negate randomly chosen qubits, specified by s
                negated_counts.update([(format(negated_key, 'b').zfill(self.backend.n_qubits), counts_batch[key])])  # make sure that the key is of the correct length 
            
            #print("negated counts ", negated_counts)
            
            # Add to the final counts dict
            for key in negated_counts:
                if key in counts:
                    counts[key] = counts[key] + negated_counts[key]
                else:
                    counts.update([(key, negated_counts[key])])
        
        #print("final counts ", counts)
        
        self.measurement_outcomes = counts
        
        return counts
    
    def expectation_value_spam_twirled(self, counts: Dict, hamiltonian: Hamiltonian, lambdas_single: list, lambdas_double: list):
        """
        need to have access to the lambdas here for correcting

        TODO use energy_expectation(hamiltonian: Hamiltonian, measurement_counts: dict) instead
        """
        print("Hamiltonian coeffs ", hamiltonian.coeffs) # TODO how are these ordered?

        # Initialize the z expectation values and correlation matrix with 0s
        n_qubits = hamiltonian.n_qubits
        exp_vals_z = np.zeros(n_qubits)
        corr_matrix = np.zeros((n_qubits, n_qubits))


        energy = 0

        # Compute expectation values and correlations of terms present in the Hamiltonian
        for term in hamiltonian.terms:

            # If bias term compute expectation value
            if len(term) == 1:
                i = term.qubit_indices[0]
                exp_vals_z[i] = exp_val_single(i, counts)
                #print("Exp value ", exp_vals_z[i])
                exp_vals_z[i] /= lambdas_single[i]
                #print("Exp value ", exp_vals_z[i])

                energy += exp_vals_z[i] * self.qaoa_descriptor.cost_single_qubit_coeffs[i] # TODO include the coefficientes of the hamiltonian
                #print("energy ", energy)

            # If two-body term compute correlation
            elif len(term) == 2:
                i, j = term.qubit_indices
                corr_matrix[i][j] = exp_val_pair((i, j), counts)
                #print("Corr matrix ", corr_matrix[i][j])
                #print("Lambdas double ", lambdas_double[i][j])

                corr_matrix[i][j] /= lambdas_double[i][j]

                #print("Corr matrix ", corr_matrix[i][j])
                energy += corr_matrix[i][j]  * self.qaoa_descriptor.cost_pair_qubit_coeffs[i]  # TODO that's wrong, or buggy at least, should be a double index
                #print("energy ", energy)

            # If constant term, ignore
            if len(term) == 0:
                continue

        # Remove expectation value contribution from the correlations
        corr_matrix -= np.outer(exp_vals_z, exp_vals_z)

        energy += hamiltonian.constant

        #print("Energy is ", energy)

        return energy
    


    def expectation(self, params, n_shots=None) -> float:
        '''
        use the total counts under BFA to compute expectation values Zi and ZiZj
        correct these expectation values with the calibration data, lambda_i, lambda_ij
        combine all corrected expectation values into the energy = cost fn to be given to the optimizer every time it calls expectation 
        '''
        print(self.qaoa_descriptor)
        print(self.qaoa_descriptor.cost_single_qubit_coeffs)
        
        counts = self.get_counts(params, n_shots)
        
        hamiltonian = self.backend.qaoa_descriptor.cost_hamiltonian
        
        calibration = False
        
        if calibration:
            lambdas_single, lambdas_double = exp_val_hamiltonian_termwise(variational_params = params,
                                                               qaoa_backend = self.backend,
                                                               hamiltonian = hamiltonian,
                                                               mixer_type = "X",  # TODO
                                                               p = self.qaoa_descriptor.p,
                                                               qaoa_optimized_counts = counts,
                                                               analytical = False,
                                                              )
            
            lambdas_double += np.outer(lambdas_single, lambdas_single)
            #print(lambda_single, lambda_double)
            print(np.array2string(lambdas_single, separator=","))
            print(np.array2string(lambdas_double, separator=","))
            
            cost = cost_function(counts, hamiltonian, self.cvar_alpha)
            
        else:
            #cost = energy_expectation(self.qaoa_descriptor.cost_hamiltonian, counts)
            
            lambdas_single = [0.72038 ,0.73956 ,0.659624,0.      ,0.      ,0.      ,0.      ,0.      ,
 0.      ]
            
            lambdas_double = [[0.      ,0.536244,0.48054 ,0.491818,0.476336,0.      ,0.      ,0.      ,
                              0.      ],
                             [0.      ,0.      ,0.487656,0.      ,0.      ,0.      ,0.527784,0.      ,
                              0.      ],
                             [0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.478438,
                              0.      ],
                             [0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,
                              0.492004],
                             [0.      ,0.      ,0.      ,0.      ,0.      ,0.480378,0.      ,0.47979 ,
                              0.      ],
                             [0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.52819 ,0.      ,
                              0.      ],
                             [0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,
                              0.      ],
                             [0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,
                              0.      ],
                             [0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,0.      ,
                              0.      ]]
            
            # TODO doesn't work for weights
            cost = self.expectation_value_spam_twirled(counts, hamiltonian, lambdas_single, lambdas_double)
            
            
        return cost
            
        
    
    
    