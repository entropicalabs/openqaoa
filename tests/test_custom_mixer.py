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
        
        # Case with Mixer with 1 AND 2-qubit terms
        self.MANUAL_GATEMAP_LIST, self.MANUAL_COEFFS = [RXGateMap(0), RZXGateMap(0, 1), RZXGateMap(0, 2), 
                                                        RZXGateMap(0, 3), RZXGateMap(0, 4), RXGateMap(4), 
                                                        RZXGateMap(0, 5), RXXGateMap(1, 2)], [1., 1., 1., 1., 1., 1., 1., 1.]
        
        zx_gatemap_list, zx_gatemap_coeffs = quick_create_mixer_for_topology(RZXGateMap, 6, qubit_connectivity='star')
        xx_gatemap_list, xx_gatemap_coeffs = quick_create_mixer_for_topology(RXXGateMap, 6, qubit_connectivity='full')
        
        zx_gatemap_list.extend(xx_gatemap_list)
        zx_gatemap_coeffs.extend(xx_gatemap_coeffs)

        self.COMPLICATED_GATEMAP_LIST, self.COMPLICATED_COEFFS = zx_gatemap_list, zx_gatemap_coeffs
        
        self.TESTING_GATEMAPS = [[self.MANUAL_GATEMAP_LIST, self.MANUAL_COEFFS], 
                                 [self.COMPLICATED_GATEMAP_LIST, self.COMPLICATED_COEFFS]]
    
    def test_custom_mixer_basic_workflow(self):
        
        """
        Check that using custom mixers works.
        Custom Mixers are only available in Manual mode.
        """
        
        nodes = 6
        edge_probability = 0.7
        g = nx.generators.fast_gnp_random_graph(n=nodes,p=edge_probability, seed=34)
        mini_cov = MinimumVertexCover(g, field = 1., penalty = 1.)
        mini_cov_qubo = mini_cov.qubo
        
        for each_gatemap_list, each_gatemap_coeffs in self.TESTING_GATEMAPS:
        
            custom_mixer_block_gatemap = each_gatemap_list
            custom_mixer_block_coeffs = each_gatemap_coeffs

            qaoa_descriptor = QAOADescriptor(mini_cov_qubo.hamiltonian, 
                                               custom_mixer_block_gatemap, p=1, 
                                               mixer_coeffs=custom_mixer_block_coeffs)
            device_local = create_device(location='local', name='qiskit.shot_simulator')
            variate_params = create_qaoa_variational_params(qaoa_descriptor, 'standard', 'rand')
            backend_local = get_qaoa_backend(qaoa_descriptor, device_local, n_shots=500)
            optimizer = get_optimizer(backend_local, variate_params, {'method': 'cobyla', 
                                                                      'maxiter': 10})
            optimizer.optimize()
        
if __name__ == '__main__':
    unittest.main()