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


"""
Test the plot functionalities in qaoa_parameters 
"""
# import the standard modules from python
import numpy as np
import unittest
import matplotlib.pyplot as plt

# import the OpenQAOA Parameterisation classes manually: Manual Mode
from openqaoa.qaoa_parameters.qaoa_params import create_qaoa_variational_params
from openqaoa.qaoa_parameters.extendedparams import QAOAVariationalExtendedParams
from openqaoa.qaoa_parameters.standardparams import QAOAVariationalStandardParams

# import the other OpenQAOA modules required for this example
from openqaoa.qaoa_parameters import PauliOp, Hamiltonian, QAOACircuitParams
from openqaoa.utilities import X_mixer_hamiltonian

from openqaoa.problems.problem import MaximumCut


# Create a hamiltonian problem on 15 qubits and create Circuit Params Class
hamiltonian1 = MaximumCut.random_instance(n_nodes=15, edge_probability = 0.9).get_qubo_problem().hamiltonian 
mixer_hamiltonian1 = X_mixer_hamiltonian(n_qubits = 15)
qaoa_circuit_params1 = QAOACircuitParams(cost_hamiltonian = hamiltonian1, mixer_block = mixer_hamiltonian1, p=4)

# Create a hamiltonian on 3 qubits with 2 coupling terms and 1 bias term and create Circuit Params Class
Term1 = PauliOp('ZZ', (0, 1)) 
Term2 = PauliOp('ZZ', (0, 2))
Term3 = PauliOp('Z', (0, ))

hamiltonian2 = Hamiltonian([Term1, Term2, Term3], [0.7, 1.2, -0.5], 0.0)
mixer_hamiltonian2 = X_mixer_hamiltonian(n_qubits = 3)
qaoa_circuit_params2 = QAOACircuitParams(cost_hamiltonian = hamiltonian2, mixer_block = mixer_hamiltonian2, p=4)



class TestingQAOAParametersPlots(unittest.TestCase):

    def _plot(self, qaoa_circuit_params, params_type, init_type = 'rand'):
        #generic function to create the plots
        print(f'Example of {params_type} params')
        if params_type in ['fourier', 'fourier_extended', 'fourier_w_bias']:
            params = create_qaoa_variational_params(qaoa_circuit_params = qaoa_circuit_params, params_type = params_type, init_type = init_type, q=2)
        else:
            params = create_qaoa_variational_params(qaoa_circuit_params = qaoa_circuit_params, params_type = params_type, init_type = init_type)
        params.plot()

    def test_QAOAVariationalStandardParamsPlots(self):        # standard params
        self._plot(qaoa_circuit_params1, 'standard')

    def test_QAOAVariationalStandardBiasParamsPlots(self):    # standard_w_bias params
        self._plot(qaoa_circuit_params2, 'standard_w_bias')

    def test_QAOAVariationalExtendedParamsPlots(self):        # extended params
        self._plot(qaoa_circuit_params1, 'extended')
        self._plot(qaoa_circuit_params2, 'extended')    
        
    def test_QAOAVariationalFourierParamsPlots(self):        # fourier params
        self._plot(qaoa_circuit_params2, 'fourier')    
        
    def test_QAOAVariationalFourierBiasParamsPlots(self):        # fourier_w_bias params
        self._plot(qaoa_circuit_params2, 'fourier_w_bias')  
        
    def test_QAOAVariationalFourierExtendedParamsPlots(self):        # fourier_extended params
        self._plot(qaoa_circuit_params1, 'fourier_extended')
        self._plot(qaoa_circuit_params2, 'fourier_extended')   
        
    def test_QAOAVariationalAnnealingParamsPlots(self):        # annealing params
        self._plot(qaoa_circuit_params2, 'annealing', init_type = 'ramp')  



if __name__ == "__main__":
    unittest.main()

