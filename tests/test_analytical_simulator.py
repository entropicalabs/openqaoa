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


import unittest
import numpy as np

from openqaoa.backends.simulators.qaoa_analytical_sim import QAOABackendAnalyticalSimulator 
#from openqaoa.utilities import random_k_regular_graph
from openqaoa.workflows.optimizer import QAOA, RQAOA
from openqaoa.problems.problem import QUBO, MaximumCut
from openqaoa.devices import create_device
from openqaoa.utilities import X_mixer_hamiltonian, ring_of_disagrees
from openqaoa.qaoa_parameters.baseparams import QAOACircuitParams
from openqaoa.qaoa_parameters import QAOAVariationalStandardParams

"""
A set of tests for obtaining the expectation value of a given quantum circuit given by the analytical formula, as derived in arXiv:2011.13420v2
"""
def Disagrees_SetUp(n_qubits, p, mixer_hamil):
    """
    Helper function for the tests below
    """

    register = range(n_qubits)
    cost_hamil = ring_of_disagrees(register)
    

    betas = [np.pi/8]
    gammas = [np.pi/4]

    qaoa_circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p)
    variational_params_std = QAOAVariationalStandardParams(qaoa_circuit_params,
                                                           betas,
                                                           gammas)
    # Get the part of the Hamiltonian proportional to the identity

    return register, cost_hamil, qaoa_circuit_params, variational_params_std


class TestingQAOABackendAnalyticalSimulator(unittest.TestCase):
    """
    Unittest based testing of QAOABackendAnalyticalSimulator
    """

    def test_expectation(self):
        """
        Testing if the analytical formula returns the energy expectation value for a given beta and gamma on an easy problem (Ring of disagree). 
        """
        n_qubits = 8
        p = 1
        mixer_hamil = X_mixer_hamiltonian(n_qubits)
        register, cost_hamil, qaoa_circuit_params, variate_params = Disagrees_SetUp(n_qubits, p, mixer_hamil)

        backend_analytical = QAOABackendAnalyticalSimulator(qaoa_circuit_params)
        exp_val = backend_analytical.expectation(variate_params)

        # Check correct expecation value
        assert np.isclose(exp_val, -6)
        
    def test_p_not_1_fails(self):
        """
        Testing if the analytical backend fails if the number of layers, p, is different than 1. 
        """
        try:
            n_qubits = 8
            p = 2
            mixer_hamil = X_mixer_hamiltonian(n_qubits)
            register, cost_hamil, qaoa_circuit_params, variate_params = Disagrees_SetUp(n_qubits, p, mixer_hamil)
            backend_analytical = QAOABackendAnalyticalSimulator(qaoa_circuit_params)
            exception = False
        except:
            exception = True
            
        assert exception, "p != 1 didn't fail."
        
    def test_different_mixer_fails(self):
        """
        Testing if the analytical backend fails if the mixer hamiltonian is different than X. 
        """
        try:
            n_qubits = 8
            p = 1
            mixer_hamil = XY_mixer_hamiltonian(n_qubits)
            register, cost_hamil, qaoa_circuit_params, variate_params = Disagrees_SetUp(n_qubits, p, mixer_hamil)
            backend_analytical = QAOABackendAnalyticalSimulator(qaoa_circuit_params)
            exception = False
        except:
            exception = True
            
        assert exception, "XY mixer Hamiltonian didn't fail."
        
    def test_end_to_end_rqaoa(self):
        pass
        
    def test_exact_solution(self):
        """
        NOTE:Since the implementation of exact solution is backend agnostic
            Checking it once should be okay. 

        Nevertheless, for the sake of completeness it will be tested for all backend
        instances.

        """

        n_qubits = 8
        register = range(n_qubits)
        p = 1

        correct_energy = -8
        correct_config = [0, 1, 0, 1, 0, 1, 0, 1]

        # The tests pass regardless of the value of betas and gammas is this correct?
        betas = [np.pi/8]
        gammas = [np.pi/4]

        cost_hamiltonian = ring_of_disagrees(register)
        mixer_hamiltonian = X_mixer_hamiltonian(n_qubits)
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = QAOAVariationalStandardParams(qaoa_circuit_params,
                                                               betas, gammas)

        backend_analytical = QAOABackendAnalyticalSimulator(qaoa_circuit_params)
        # exact solution is defined as the property of the cost function
        energy_vec, config_vec = backend_analytical.exact_solution

        assert np.isclose(energy_vec, correct_energy)

        config_vec = [config.tolist() for config in config_vec]

        assert correct_config in config_vec
        

        
'''    
def run_Rqaoa_experiment():  
    
    g = random_k_regular_graph(degree=3, nodes=range(8), weighted=True, biases=False) 

    # Define te problem and translate it into a binary Qubo.
    maxcut_prob = MaximumCut(g)
    maxcut_qubo = maxcut_prob.get_qubo_problem()

    
    # Define the RQAOA object (default rqaoa_type = 'adaptive')
    r = RQAOA()

    # Set parameters for RQAOA, in this case we fix the n_max to 1 (default), the final cutoff value to 3
    r.set_rqaoa_parameters(steps=1, n_cutoff=3)

    ## Setting up the QAOA properties
    init_beta = 0.42
    init_gamma = 0.42

    r.set_circuit_properties(p=1, init_type='custom', variational_params_dict={"betas":[init_beta], "gammas":[init_gamma]}, mixer_hamiltonian='x')

    # Define the device you want to run your problem on using the create_device() function - Here we choose the local wavefunction simulator supported by OpenQAOA
    device = create_device(location='local', name='analytical_simulator')
    r.set_device(device)
    
    # Set the classical method used to optimiza over QAOA angles and its properties
    r.set_classical_optimizer(method="COBYLA", optimizer_options = {"stepsize" : 10**(-10)}, tol=10**(-1), maxfev=1, maxiter=1, optimization_progress=False, cost_progress=False, parameter_log=False)

   
    # Compile problem instance on RQAOA, just like with QAOA
    r.compile(maxcut_qubo)
    
    r.optimize()
    
    print(r.results)
    
    
run_Rqaoa_experiment()    
'''

if __name__ == "__main__":
    unittest.main()
