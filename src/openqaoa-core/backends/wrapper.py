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

from ..utilities import exp_val_hamiltonian_termwise, energy_expectation, exp_val_pair, exp_val_single, negate_counts_dictionary, calculate_calibration_factors

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
        
        # should come externally in the same way as n_batches and as a json
        calibration_data = {'001001111': 53, '001010111': 53, '010110110': 47, '111001101': 7, '011101011': 13, '011011110': 8, '100011111': 10, '111101100': 7, '000111011': 47, '000101111': 49, '101110011': 14, '110010111': 12, '110100011': 75, '100111011': 8, '111000011': 57, '001110011': 59, '111001100': 36, '100110110': 52, '101111010': 9, '101011111': 2, '110001101': 32, '000111101': 35, '111010001': 48, '011101001': 49, '111011000': 47, '001110110': 41, '100111111': 1, '000111111': 9, '011111100': 5, '011111000': 55, '110011100': 34, '011110001': 48, '111100100': 40, '011001110': 30, '010111011': 11, '101011010': 33, '011000111': 49, '001101011': 50, '011101010': 46, '001111001': 40, '101001011': 55, '101001001': 231, '011101110': 8, '010011011': 50, '100011101': 31, '011000101': 209, '100010011': 327, '101110010': 54, '100011011': 43, '110000011': 336, '100110001': 297, '001011001': 195, '110110010': 54, '011001001': 271, '011001011': 44, '101111000': 38, '011001100': 163, '001110100': 178, '000111110': 42, '000100111': 285, '100001011': 287, '100001111': 34, '011110110': 7, '110010110': 55, '010101101': 44, '100111101': 10, '010110101': 67, '000011101': 196, '101000110': 276, '110110000': 301, '011010101': 29, '010110100': 218, '001101110': 44, '001101111': 11, '000110101': 245, '011010100': 207, '011001010': 239, '101011000': 201, '100101011': 50, '111000001': 293, '100111010': 36, '010001110': 228, '001010000': 6809, '100001010': 1404, '100111100': 26, '000010111': 256, '111100000': 288, '010010001': 1557, '001000001': 8289, '100010111': 50, '000001001': 7127, '101000000': 7454, '101001100': 192, '010001111': 42, '100110101': 32, '110000110': 286, '000101101': 245, '000010001': 7930, '000100101': 1326, '011100111': 10, '100100011': 355, '010110001': 324, '010001101': 190, '101001101': 22, '110111110': 3, '111110000': 49, '101101011': 15, '001110000': 1306, '110001011': 58, '111010010': 41, '100010100': 1231, '101100011': 60, '110000100': 1293, '000111010': 264, '101010011': 50, '111010100': 29, '000100001': 8697, '000110111': 51, '001011000': 1064, '111011110': 1, '111100010': 51, '000100110': 1351, '001101000': 1247, '001100011': 332, '000110000': 7548, '110010101': 51, '011011101': 14, '110001110': 46, '100101000': 1243, '111101010': 9, '100101110': 36, '110100000': 1637, '001001001': 1321, '000000011': 9036, '101100101': 58, '110100010': 349, '010011000': 1127, '001100110': 228, '010001010': 1473, '010000101': 1324, '001100010': 1549, '100001000': 6385, '100011001': 198, '111001001': 42, '100010001': 1585, '001000000': 38524, '000100100': 6511, '000101100': 1000, '010011101': 50, '000110100': 1196, '010101010': 314, '011010010': 257, '100011110': 35, '100101100': 195, '001000110': 1163, '000101010': 1494, '010101110': 36, '000000101': 6859, '110111000': 48, '010010011': 316, '100100100': 1340, '010000111': 307, '011001101': 27, '010101011': 66, '011001000': 1147, '110101001': 41, '000010000': 38952, '000110110': 247, '000011110': 178, '111000110': 40, '101001000': 1138, '001000010': 7682, '010000000': 41706, '010000110': 1331, '110000101': 277, '101101000': 218, '010001100': 1034, '000000010': 44598, '101100000': 1488, '101101101': 8, '110000000': 8047, '001010100': 1059, '100101111': 17, '000011011': 275, '001011110': 23, '000001010': 7724, '000101000': 7133, '100000100': 6821, '010000100': 6524, '100011100': 192, '011000000': 7088, '100000010': 8307, '100010000': 7279, '011000010': 1456, '101000100': 1195, '100000111': 288, '000100000': 42781, '000111100': 170, '011110100': 39, '001100000': 7787, '000001011': 1579, '101000111': 51, '000100010': 8727, '001100111': 62, '100110000': 1456, '100100010': 1623, '101011001': 50, '010110000': 1488, '000001000': 35886, '101010000': 1264, '000011000': 6289, '001010101': 222, '000110011': 330, '000000111': 1417, '110111100': 10, '100100001': 1795, '100011000': 1125, '101111001': 7, '001001101': 195, '100101101': 47, '100101001': 263, '010001000': 6926, '110010000': 1460, '001011100': 203, '001111010': 47, '010101001': 288, '100001101': 201, '010010010': 1511, '010000001': 8334, '001100101': 242, '000001100': 5538, '001011010': 241, '110010100': 245, '011110010': 56, '000010011': 1579, '100000001': 8547, '001011011': 51, '001110010': 270, '100000011': 1669, '010100001': 1558, '100100111': 56, '110010010': 266, '001010110': 191, '111001110': 7, '101010101': 47, '111001000': 199, '010100010': 1750, '101010100': 224, '100100000': 8511, '011110000': 266, '101001010': 231, '000011100': 933, '001100100': 1192, '000000100': 34019, '000000000': 218681, '011000011': 300, '000000001': 44185, '111100110': 11, '011100010': 312, '100010010': 1424, '010110010': 292, '111001010': 39, '000100011': 1734, '001001100': 938, '110001001': 241, '011100101': 42, '100000000': 40894, '001001000': 6349, '010000011': 1743, '100110100': 249, '000001111': 211, '100001100': 1081, '011100100': 213, '000010110': 1159, '100100101': 259, '100001001': 1239, '100000110': 1462, '010011100': 158, '000101001': 1378, '111101000': 40, '010001011': 288, '010111000': 226, '101101100': 32, '010010110': 260, '101101010': 47, '111010000': 247, '011100000': 1394, '100110010': 270, '001110101': 43, '101000010': 1403, '010100000': 8145, '011100110': 61, '110100100': 259, '000110001': 1493, '000111000': 1183, '110101000': 262, '001110001': 255, '101100010': 262, '000011010': 1315, '111000101': 43, '100111000': 222, '110010011': 74, '110001010': 265, '110011000': 229, '001001011': 274, '101000011': 311, '010001001': 1349, '111000010': 285, '101100100': 254, '000111001': 219, '010101000': 1379, '001100001': 1733, '011011000': 197, '110101100': 45, '011100001': 272, '101001110': 43, '110000001': 1653, '100010101': 255, '011010111': 5, '000000110': 7018, '010010101': 264, '111000000': 1372, '110011001': 46, '111000100': 214, '001111100': 29, '010010000': 7379, '100100110': 284, '100111001': 41, '101110000': 267, '010100100': 1271, '001101101': 31, '010000010': 8574, '011010000': 1306, '010101100': 211, '100011010': 225, '110100001': 341, '000001101': 1092, '110001100': 230, '010010100': 1159, '110101010': 71, '010111100': 34, '101000101': 268, '000010100': 6151, '000101011': 282, '001001110': 185, '100010110': 238, '101111101': 3, '001000101': 1272, '110100101': 49, '100110011': 70, '001000100': 5818, '001101010': 285, '011101000': 221, '110000010': 1693, '110110100': 51, '110001000': 1238, '001001010': 1343, '000010101': 1212, '011000001': 1464, '010100111': 45, '100001110': 242, '101100001': 335, '001010011': 243, '001000111': 272, '001111000': 217, '110011010': 50, '110001111': 7, '101010010': 256, '101010001': 292, '100101010': 277, '110101101': 11, '011000110': 200, '110110101': 8, '001101001': 253, '001000011': 1557, '011000100': 1099, '101011100': 32, '001101100': 174, '001010001': 1395, '001010010': 1338, '011010001': 278, '101000001': 1677, '011101100': 38, '000011001': 1239, '110010001': 317, '010011001': 262, '000101110': 210, '000001110': 1168, '011011010': 51, '101100110': 44, '011101101': 10, '010100011': 320, '110100110': 56, '010100110': 275, '110000111': 69, '010011010': 253, '000110010': 1502, '100000101': 1402, '010110011': 62, '000010010': 7750, '011010011': 55, '010100101': 254, '111001011': 5, '111110100': 8, '111011011': 5, '101011101': 9, '110011110': 8, '011010110': 44, '101011011': 7, '011011001': 41, '111110001': 11, '011110101': 15, '011110011': 7, '111100001': 68, '110100111': 6, '111100011': 9, '100110111': 9, '010010111': 54, '111011100': 6, '001110111': 14, '101110101': 7, '110111010': 10, '110101011': 6, '101101001': 59, '101010110': 38, '011110111': 4, '110110011': 9, '111110010': 8, '110111101': 3, '010111001': 47, '101110001': 49, '110011011': 8, '010110111': 8, '001011101': 31, '010111010': 50, '110110001': 57, '111100101': 6, '111000111': 7, '111110011': 3, '011111110': 2, '111110110': 1, '010011110': 30, '010111101': 5, '101110110': 2, '110111001': 4, '110110111': 1, '111100111': 1, '011001111': 7, '101110100': 46, '011100011': 59, '011111010': 6, '101100111': 13, '010011111': 5, '111111010': 2, '011011011': 11, '110011101': 8, '001111101': 4, '101101110': 6, '111010101': 13, '001111110': 4, '000011111': 38, '100111110': 6, '111011010': 4, '111011001': 4, '101001111': 10, '111010111': 3, '111010011': 9, '011011111': 1, '011011100': 25, '001111011': 13, '101111011': 1, '010101111': 5, '011111001': 6, '111010110': 7, '001011111': 8, '111011101': 1, '111101011': 2, '110101110': 7, '111101001': 5, '101110111': 3, '101111110': 2, '111111000': 2, '110101111': 2, '111001111': 3, '101010111': 6, '101111100': 3, '110110110': 5, '011111101': 1, '101101111': 2, '101011110': 4, '010111111': 2, '111101101': 1, '010111110': 1, '011111011': 3, '111111100': 1, '111110101': 1, '110011111': 1, '111101110': 3, '111110111': 1, '001111111': 1}
        
        # compute here so that we do it only once
        self.calibration_factors = calculate_calibration_factors(self.backend.qaoa_descriptor.cost_hamiltonian, calibration_data)  
        
        print(self.calibration_factors)
    
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
    
    def expectation_value_spam_twirled(self, counts: Dict, hamiltonian: Hamiltonian, calibration_factors: dict):
        """
        
        """
         
        
        terms = [term.qubit_indices for term in hamiltonian.terms]
        coeffs = [coeff for coeff in hamiltonian.coeffs]
        hamiltonian_as_dict = dict(zip(terms,coeffs))
        
        print(self.qaoa_descriptor.cost_single_qubit_coeffs[0])
        print(hamiltonian_as_dict)
        print(hamiltonian_as_dict[(0,)]) 
        
        energy = 0

        # Compute expectation values and correlations of terms present in the Hamiltonian
        for term in hamiltonian.terms:

            # If bias term compute expectation value
            if len(term) == 1:
                i = term.qubit_indices[0]
                exp_vals_z = exp_val_single(i, counts)
                exp_vals_z /= calibration_factors[(i,)]

                energy += exp_vals_z * hamiltonian_as_dict[(i,)]  

            # If two-body term compute correlation
            elif len(term) == 2:
                i, j = term.qubit_indices
                exp_vals_zz = exp_val_pair((i, j), counts)
                exp_vals_zz /= calibration_factors[(i,j)]
                
                energy += exp_vals_zz * hamiltonian_as_dict[(i,j)]  
                
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
        counts = self.get_counts(params, n_shots)
        
        print(counts)
        
        
        lambdas_single, lambdas_double = exp_val_hamiltonian_termwise(
                                                               hamiltonian = self.backend.qaoa_descriptor.cost_hamiltonian,
                                                               mixer_type = "X",  # TODO
                                                               p = self.qaoa_descriptor.p,
                                                               qaoa_optimized_counts = counts,
                                                               analytical = False,
                                                              )
            
        lambdas_double += np.outer(lambdas_single, lambdas_single)
        print(np.array2string(lambdas_single, separator=","))
        print(np.array2string(lambdas_double, separator=","))

        #print("counts empty circuit ", counts)
        
        cost = self.expectation_value_spam_twirled(counts, self.backend.qaoa_descriptor.cost_hamiltonian, self.calibration_factors)
            
        return cost
            
        
    
    
    