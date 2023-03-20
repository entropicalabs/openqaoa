import random
import numpy as np
from copy import deepcopy

from .basebackend import VQABaseBackend

from .cost_function import cost_function

from ..qaoa_components.ansatz_constructor.gatemap import (
    XGateMap
)

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
        
        
    def expectation(self, params, n_shots=None) -> float:
        '''
        use the total counts under BFA to compute expectation values Zi and ZiZj
        correct these expectation values with the calibration data, lambda_i, lambda_ij
        combine all corrected expectation values into the energy = cost fn to be given to the optimizer every time it calls expectation 
        '''
        
        counts = self.get_counts(params, n_shots)
        

        # TODO change the way we compute the cost function to be termwise, not stringwise
        cost = cost_function(
            counts, self.qaoa_descriptor.cost_hamiltonian, self.cvar_alpha
        )
        return cost
            
        
    
    
    