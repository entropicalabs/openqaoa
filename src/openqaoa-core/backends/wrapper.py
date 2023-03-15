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
    

class TwirlingWrapper(BaseWrapper):
    def __init__(self, backend, n_batches):
        super().__init__(backend)
        self.n_batches = n_batches
        
    
    def change_abstract_circuit(self, abstract_circuit, s):
        s = format(s, 'b').zfill(self.backend.n_qubits) # convert to binary
        arr = np.fromiter(s, dtype=int)
        negated_qubits = np.where(arr == 1)[0] # where the syndrome has a 1
        print("qubits to negate ", negated_qubits)

        for negated_qubit in negated_qubits:
            abstract_circuit.append(XGateMap(qubit_1 = negated_qubit))

        return abstract_circuit
        
    def expectation(self, params, n_shots=None) -> float:
        '''
        divide into batches
        change the abstract circuit (deepcopy) according to the schedule, s.
        get the counts and classically negate them
        combine all batches into a count dict under BFA
        use this dict to compute expectation values Zi and ZiZj
        correct these expectation values with the calibration data, lambda_i, lambda_ij
        combine all corrected expectation values into the energy = cost fn to be given to the optimizer every time it calls expectation 
        '''
        
        
        # list of integers whose binary representation signifies which qubits or be flipped at every batch 
        s_list = [] 
        for _ in range(0, self.n_batches):
            s_list.append(random.getrandbits(self.backend.n_qubits)) 
        s_list = [3, 0, 1, 2] # TESTING ONLY, can be specified by the user 
        #s_list = [1, 1, 1, 1]
        
        n_shots_batch = int(self.backend.n_shots / self.n_batches)
        
        counts = {}
        
        initial_abstract_circuit = deepcopy(self.backend.abstract_circuit)
        
        for batch in range(0, self.n_batches):
            print("batch ", batch)
            s = s_list[batch]
            
            negated_counts = {}
            
            
            
            
            # TODO change the abstract circuit here
            
            print(self.backend.abstract_circuit)
            self.backend.abstract_circuit = self.change_abstract_circuit(initial_abstract_circuit, s) # maybe as an attribute ?
            
            
            
            
            
            counts_batch = self.get_counts(params, n_shots_batch)
            
            # consider putting this in utilities.py, similar to permuted_counts with SWAP gates (?)
            for key in counts_batch.keys():   
                negated_key = s ^ int(key, 2)  # bitwise XOR to classically negate randomly chosen qubits, specified by s
                negated_counts.update([(format(negated_key, 'b').zfill(self.backend.n_qubits), counts_batch[key])])  # make sure that the key is of the correct length 
            
            print("negated counts ", negated_counts)
            
            # Add to the final counts dict
            for key in negated_counts:
                if key in counts:
                    counts[key] = counts[key] + negated_counts[key]
                else:
                    counts.update([(key, negated_counts[key])])
        
        print("final counts ", counts)

        # TODO change the way we compute the cost function to be termwise, not stringwise
        cost = cost_function(
            counts, self.qaoa_descriptor.cost_hamiltonian, self.cvar_alpha
        )
        return cost
            
        
    
    
    