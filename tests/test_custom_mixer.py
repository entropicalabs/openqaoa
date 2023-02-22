import unittest

import networkx as nx

from openqaoa.qaoa_components.ansatz_constructor import RXGateMap, RZXGateMap, RXXGateMap
from openqaoa.qaoa_components import QAOADescriptor, create_qaoa_variational_params
from openqaoa.backends import create_device
from openqaoa.optimizers import get_optimizer
from openqaoa.backends.qaoa_backend import get_qaoa_backend
from openqaoa.utilities import quick_create_mixer_for_topology
from openqaoa.problems import MinimumVertexCover


class TestingCustomMixer(unittest.TestCase):
    
    def setUp(self):
        
        
        nodes = 6
        edge_probability = 0.7
        g = nx.generators.fast_gnp_random_graph(n=nodes,p=edge_probability, seed=34)
        mini_cov = MinimumVertexCover(g, field = 1., penalty = 1.)
        self.PROBLEM_QUBO = mini_cov.qubo
        
        # Case with Mixer with 1 AND 2-qubit terms
        self.MANUAL_GATEMAP_LIST, self.MANUAL_COEFFS = [RXGateMap(0), RZXGateMap(0, 1), RZXGateMap(0, 2), 
                                                        RZXGateMap(0, 3), RZXGateMap(0, 4), RXGateMap(4), 
                                                        RZXGateMap(0, 5), RXXGateMap(1, 2)], [1., 1., 1., 1., 1., 1., 1., 1.]
        self.MANUAL_SEQUENCE = [0, 0, 1, 2, 3, 1, 4, 5]
        
        zx_gatemap_list, zx_gatemap_coeffs = quick_create_mixer_for_topology(RZXGateMap, 6, qubit_connectivity='star')
        xx_gatemap_list, xx_gatemap_coeffs = quick_create_mixer_for_topology(RXXGateMap, 6, qubit_connectivity='full')
        
        zx_gatemap_list.extend(xx_gatemap_list)
        zx_gatemap_coeffs.extend(xx_gatemap_coeffs)
        
        # Case with Multiple types of 2-qubit gates
        self.COMPLICATED_GATEMAP_LIST, self.COMPLICATED_COEFFS = zx_gatemap_list, zx_gatemap_coeffs
        self.COMPLICATED_SEQUENCE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        
        self.TESTING_GATEMAPS = [[self.MANUAL_GATEMAP_LIST, self.MANUAL_COEFFS, self.MANUAL_SEQUENCE], 
                                 [self.COMPLICATED_GATEMAP_LIST, self.COMPLICATED_COEFFS,self.COMPLICATED_SEQUENCE]]
    
    def test_custom_mixer_basic_workflow(self):
        
        """
        Check that using custom mixers works.
        Custom Mixers are only available in Manual mode.
        """
        
        for each_gatemap_list, each_gatemap_coeffs, _ in self.TESTING_GATEMAPS:
        
            custom_mixer_block_gatemap = each_gatemap_list
            custom_mixer_block_coeffs = each_gatemap_coeffs

            qaoa_descriptor = QAOADescriptor(self.PROBLEM_QUBO.hamiltonian, 
                                               custom_mixer_block_gatemap, p=1, 
                                               mixer_coeffs=custom_mixer_block_coeffs)
            device_local = create_device(location='local', name='qiskit.shot_simulator')
            variate_params = create_qaoa_variational_params(qaoa_descriptor, 'standard', 'rand')
            backend_local = get_qaoa_backend(qaoa_descriptor, device_local, n_shots=500)
            optimizer = get_optimizer(backend_local, variate_params, {'method': 'cobyla', 
                                                                      'maxiter': 10})
            optimizer.optimize()
    
    def test_mixer_block_properties_sequence(self):
        
        """
        The custom mixers should have sequences that are correct.
        The sequence values are based on the position of the gate in the block 
        relative to other gates of the same qubit count.
        """
        
        for each_gatemap_list, each_gatemap_coeffs, correct_seq in self.TESTING_GATEMAPS:
            
            gatemap_list_sequence = []
            one_qubit_count = 0
            two_qubit_count = 0
            
            for each_gatemap in each_gatemap_list:
                
                if each_gatemap.gate_label.n_qubits == 1:
                    
                    gatemap_list_sequence.append(one_qubit_count)
                    one_qubit_count += 1
                    
                elif each_gatemap.gate_label.n_qubits == 2:
                    
                    gatemap_list_sequence.append(two_qubit_count)
                    two_qubit_count += 1
            
            # Test Equality between hand-written and programmatic assignment
            self.assertEqual(gatemap_list_sequence, correct_seq)
            
            qaoa_descriptor = QAOADescriptor(self.PROBLEM_QUBO.hamiltonian, 
                                             each_gatemap_list, p=1, 
                                             mixer_coeffs=each_gatemap_coeffs)
            
            
            descriptor_mixer_seq = [each_mixer_gatemap.gate_label.sequence for each_mixer_gatemap in qaoa_descriptor.mixer_block]
            
            # Test Equality between OQ and hand-written sequence
            self.assertEqual(descriptor_mixer_seq, correct_seq)
        
        
    
if __name__ == '__main__':
    unittest.main()