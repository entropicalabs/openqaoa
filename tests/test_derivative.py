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

import warnings
import numpy as np
import unittest

# OpenQAOA imports
from openqaoa.backends.simulators.qaoa_vectorized import QAOAvectorizedBackendSimulator
from openqaoa.qaoa_parameters import Hamiltonian, create_qaoa_variational_params
from openqaoa.qaoa_parameters.baseparams import QAOACircuitParams
from openqaoa.utilities import X_mixer_hamiltonian
from openqaoa.optimizers.logger_vqa import Logger
from openqaoa.derivative_functions import derivative

"""
Unittest based testing of derivative computations.
"""


class TestQAOACostBaseClass(unittest.TestCase):

    '''
    def test_gradient_agreement(self):
        "Test agreement between gradients computed from finite difference, parameter shift and SPS (all gates sampled) for weighted and unweighted graphs at several parameters."
        
        # unweighted graph
        terms = [[0,1], [0,2], [1,3], [2]]
        weights = [1, 1, 1, 1]
        register = [0, 1, 2, 3]
        p = 2
        nqubits = 4

        cost_hamiltonian = Hamiltonian.classical_hamiltonian(
            terms, weights, constant=0)
        mixer_hamiltonian = X_mixer_hamiltonian(nqubits)
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = create_qaoa_variational_params(
            qaoa_circuit_params, 'standard', 'ramp')
        backend_vectorized = QAOAvectorizedBackendSimulator(
            qaoa_circuit_params, prepend_state=None, append_state=None, init_hadamard=True)
        
        grad_stepsize = 0.00000001
        gradient_ps = backend_vectorized.derivative_function(variational_params_std, 'gradient', 'param_shift')
        gradient_fd = backend_vectorized.derivative_function(variational_params_std, 'gradient', 'finite_difference', {'stepsize': grad_stepsize})
        gradient_sps = backend_vectorized.derivative_function(variational_params_std, 'gradient', 'stoch_param_shift', {'stepsize':grad_stepsize, 'n_beta_single':-1, 'n_beta_pair':-1, 'n_gamma_pair':-1, 'n_gamma_single':-1})

        params = [[0,0,0,0], [1,1,1,1], [np.pi/2, np.pi/2, np.pi/2, np.pi/2]]

        for param in params:
            grad_fd = gradient_fd(param)
            grad_ps = gradient_ps(param)
            grad_sps = gradient_sps(param)

            for i, grad in enumerate(grad_fd): 
                assert np.isclose(grad, grad_ps[i], rtol=1e-05, atol=1e-05)
                assert np.isclose(grad, grad_sps[i], rtol=1e-05, atol=1e-05)

        # weighted graph with bias
        terms = [[0,1], [1,2], [0,3], [2], [1]]
        weights = [1, 1.1, 1.5, 2, -0.8]
        register = [0, 1, 2, 3]
        p = 2
        nqubits = 4

        cost_hamiltonian = Hamiltonian.classical_hamiltonian(
            terms, weights, constant=0.8)
        mixer_hamiltonian = X_mixer_hamiltonian(nqubits)
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = create_qaoa_variational_params(
            qaoa_circuit_params, 'standard', 'ramp')
        backend_vectorized = QAOAvectorizedBackendSimulator(
            qaoa_circuit_params, prepend_state=None, append_state=None, init_hadamard=True)
        
        grad_stepsize = 0.00000001
        gradient_ps = backend_vectorized.derivative_function(variational_params_std, 'gradient', 'param_shift')
        gradient_fd = backend_vectorized.derivative_function(variational_params_std, 'gradient', 'finite_difference', {'stepsize': grad_stepsize})
        gradient_sps = backend_vectorized.derivative_function(variational_params_std, 'gradient', 'stoch_param_shift', {'stepsize':grad_stepsize, 'n_beta_single':-1, 'n_beta_pair':-1, 'n_gamma_pair':-1, 'n_gamma_single':-1})
    
        params = [[0,0,0,0], [1,1,1,1], [np.pi/2, np.pi/2, np.pi/2, np.pi/2]]

        for param in params:
            grad_fd = gradient_fd(param)
            grad_ps = gradient_ps(param)
            grad_sps = gradient_sps(param)

            for i, grad in enumerate(grad_fd): 
                assert np.isclose(grad, grad_ps[i], rtol=1e-05, atol=1e-05)
                assert np.isclose(grad, grad_sps[i], rtol=1e-05, atol=1e-05)
    '''
    
    def setUp(self):
        
        self.log = Logger({'func_evals': 
                           {
                               'history_update_bool': False, 
                               'best_update_string': 'HighestOnly'
                           }, 
                           'jac_func_evals': 
                           {
                               'history_update_bool': False, 
                               'best_update_string': 'HighestOnly'
                           }
                          }, 
                          {
                              'root_nodes': ['func_evals', 'jac_func_evals'],
                              'best_update_structure': []
                          })
        
        self.log.log_variables({'func_evals': 0})
        self.log.log_variables({'jac_func_evals': 0})

    def test_gradient_computation(self):
        "Test gradient computation by param. shift, finite difference, and SPS (all gates sampled) on barbell graph."

        # Analytical cost expression : C(b,g) = -sin(4b)*sin(2g)
        terms = [[0, 1]]
        weights = [1]
        register = [0, 1]
        p = 1
        nqubits = 2

        cost_hamiltonian = Hamiltonian.classical_hamiltonian(
            terms, weights, constant=0)
        mixer_hamiltonian = X_mixer_hamiltonian(nqubits)
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = create_qaoa_variational_params(
            qaoa_circuit_params, 'standard', 'ramp')
        backend_vectorized = QAOAvectorizedBackendSimulator(
            qaoa_circuit_params, prepend_state=None, append_state=None, init_hadamard=True)

        grad_stepsize = 0.00000001
        gradient_ps = derivative(backend_vectorized, variational_params_std, 
                                 self.log, 'gradient', 'param_shift')
        gradient_fd = derivative(backend_vectorized, variational_params_std, 
                                 self.log, 'gradient', 'finite_difference',
                                 {'stepsize': grad_stepsize})
        gradient_sps = derivative(backend_vectorized, variational_params_std, 
                                  self.log, 'gradient', 'stoch_param_shift', 
                                  {'stepsize':grad_stepsize, 'n_beta':-1, 'n_gamma_pair':-1, 'n_gamma_single':-1})

        test_points = [[0, 0], [np.pi/2, np.pi/3], [1, 2]]

        for point in test_points:
            beta, gamma = point[0], point[1]

            dCdb = -4*np.cos(4*beta)*np.sin(2*gamma)
            dCdg = -2*np.sin(4*beta)*np.cos(2*gamma)

            assert np.isclose(dCdb, gradient_ps(point)[0], rtol=1e-05, atol=1e-05)
            assert np.isclose(dCdg, gradient_ps(point)[1], rtol=1e-05, atol=1e-05)

            assert np.isclose(dCdb, gradient_fd(
                point)[0], rtol=1e-05, atol=1e-05)
            assert np.isclose(dCdg, gradient_fd(
                point)[1], rtol=1e-05, atol=1e-05)

            assert np.isclose(dCdb, gradient_sps(point)[0], rtol=1e-05, atol=1e-05)
            assert np.isclose(dCdg, gradient_sps(point)[1], rtol=1e-05, atol=1e-05)

    def test_hessian_computation(self):
        "Test Hessian computation by finite difference on barbell graph"

        # Analytical cost expression : C(b,g) = -sin(4b)*sin(2g)
        terms = [[0, 1]]
        weights = [1]
        register = [0, 1]
        p = 1
        nqubits = 2

        cost_hamiltonian = Hamiltonian.classical_hamiltonian(
            terms, weights, constant=0)
        mixer_hamiltonian = X_mixer_hamiltonian(nqubits)
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = create_qaoa_variational_params(
            qaoa_circuit_params, 'standard', 'ramp')
        backend_vectorized = QAOAvectorizedBackendSimulator(
            qaoa_circuit_params, prepend_state=None, append_state=None, init_hadamard=True)

        hessian_fd = derivative(backend_vectorized, variational_params_std, 
                                self.log,'hessian', 'finite_difference', 
                                {'stepsize': 0.0001})

        test_points = [[0, 0], [np.pi/2, np.pi/3], [1, 2]]

        for point in test_points:
            beta, gamma = point[0], point[1]

            dCdbb = 16*np.sin(4*beta)*np.sin(2*gamma)
            dCdbg = -8*np.cos(4*beta)*np.cos(2*gamma)
            dCdgb = dCdbg
            dCdgg = 4*np.sin(4*beta)*np.sin(2*gamma)

            assert np.isclose(dCdbb, hessian_fd(
                point)[0][0], rtol=1e-05, atol=1e-05)
            assert np.isclose(dCdbg, hessian_fd(
                point)[0][1], rtol=1e-05, atol=1e-05)
            assert np.isclose(dCdgb, hessian_fd(
                point)[1][0], rtol=1e-05, atol=1e-05)
            assert np.isclose(dCdgg, hessian_fd(
                point)[1][1], rtol=1e-05, atol=1e-05)


    def test_SPS_sampling(self):
        "Test that SPS samples all gates when (n_beta, n_gamma_pair, n_gamma_single) is (-1, -1, -1), on barbell graph."
        
        # Analytical cost expression : C(b,g) = -sin(4b)*sin(2g)
        terms = [[0, 1]]
        weights = [1]
        register = [0, 1]
        p = 1
        nqubits = 2

        cost_hamiltonian = Hamiltonian.classical_hamiltonian(
            terms, weights, constant=0)
        mixer_hamiltonian = X_mixer_hamiltonian(nqubits)
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = create_qaoa_variational_params(
            qaoa_circuit_params, 'standard', 'ramp')
        backend_vectorized = QAOAvectorizedBackendSimulator(
            qaoa_circuit_params, prepend_state=None, append_state=None, init_hadamard=True)

        grad_stepsize = 0.00000001
        gradient_sps1 = derivative(backend_vectorized, variational_params_std, self.log, 'gradient', 'stoch_param_shift', {'stepsize':grad_stepsize, 'n_beta_single':-1, 'n_beta_pair':-1, 'n_gamma_pair':-1, 'n_gamma_single':-1})
        gradient_sps2 = derivative(backend_vectorized, variational_params_std, self.log, 'gradient', 'stoch_param_shift', {'stepsize':grad_stepsize, 'n_beta_single':2, 'n_beta_pair':-1, 'n_gamma_pair':1, 'n_gamma_single':0})
        
        test_points = [[0,0], [np.pi/2, np.pi/3], [1,2]]
        
        for point in test_points:
            beta, gamma = point[0], point[1]
            
            assert np.isclose(gradient_sps1(point)[0], gradient_sps2(point)[0], rtol=1e-05, atol=1e-05)
            assert np.isclose(gradient_sps1(point)[1], gradient_sps2(point)[1], rtol=1e-05, atol=1e-05)



if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=PendingDeprecationWarning)
        unittest.main()
