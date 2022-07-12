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
from scipy.optimize._minimize import MINIMIZE_METHODS

from openqaoa.qaoa_parameters import create_qaoa_variational_params, QAOACircuitParams, PauliOp, Hamiltonian
from openqaoa.utilities import X_mixer_hamiltonian
from openqaoa.backends.qaoa_backend import get_qaoa_backend
from openqaoa.devices import create_device
from openqaoa.optimizers import get_optimizer
from openqaoa.derivative_functions import derivative
from openqaoa.optimizers.logger_vqa import Logger
from openqaoa.qfim import qfim

"""
Unittest based testing of custom optimizers.
"""


class TestQAOACostBaseClass(unittest.TestCase):
    
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
                           },
                           'qfim_func_evals': 
                           {
                               'history_update_bool': False, 
                               'best_update_string': 'HighestOnly'
                           }
                          }, 
                          {
                              'root_nodes': ['func_evals', 'jac_func_evals', 
                                             'qfim_func_evals'], 
                              'best_update_structure': []
                          })
        
        self.log.log_variables({'func_evals': 0, 'jac_func_evals': 0, 'qfim_func_evals': 0})

    def test_scipy_optimizers_global(self):
        " Check that final value of all scipy MINIMIZE_METHODS optimizers agrees with pre-computed optimized value."

        # Create problem instance, cost function, and gradient functions
        cost_hamil = Hamiltonian([PauliOp('ZZ', (0, 1)), PauliOp('ZZ', (1, 2)), PauliOp(
            'ZZ', (0, 3)), PauliOp('Z', (2,)), PauliOp('Z', (1,))], [1, 1.1, 1.5, 2, -0.8], 0.8)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=4)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=1)
        device = create_device('local','vectorized')
        backend_obj_vectorized = get_qaoa_backend(circuit_params,device)
        variate_params = create_qaoa_variational_params(
            circuit_params, 'standard', 'ramp')

        niter = 5
        stepsize = 0.001
        y_precomp = [-2.4345058914425626, -2.5889608823632795, -2.588960865651421, -2.5889608823632786, -2.5889608823632795, -2.588960882363273, -2.5889608823632786,
                     0.7484726235465329, -2.588960882363272, -2.588960882363281, -2.5889608823632786, -2.5889608823632795, -2.5889608823632786, -2.5889608823632786]
        optimizer_dicts = []
        for method in MINIMIZE_METHODS:
            optimizer_dicts.append(
                {'method': method, 'maxiter': niter, 'tol': 10**(-9)})

        for i, optimizer_dict in enumerate(optimizer_dicts):

            optimizer_dict['jac'] = derivative(backend_obj_vectorized, 
                variate_params, self.log, 'gradient', 'finite_difference')
            optimizer_dict['hess'] = derivative(backend_obj_vectorized, 
                variate_params, self.log, 'hessian', 'finite_difference')

            # Optimize
            vector_optimizer = get_optimizer(
                backend_obj_vectorized, variate_params, optimizer_dict=optimizer_dict)
            vector_optimizer()

            y_opt = vector_optimizer.qaoa_result.intermediate['intermediate cost']

            assert np.isclose(y_precomp[i], y_opt[-1], rtol=1e-04,
                              atol=1e-04), f"{optimizer_dict['method']} failed the test."

    def test_gradient_optimizers_global(self):
        " Check that final value of all implemented gradient optimizers agrees with pre-computed optimized value."

        cost_hamil = Hamiltonian([PauliOp('ZZ', (0, 1)), PauliOp('ZZ', (1, 2)), PauliOp(
            'ZZ', (0, 3)), PauliOp('Z', (2,)), PauliOp('Z', (1,))], [1, 1.1, 1.5, 2, -0.8], 0.8)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=4)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=1)
        device = create_device('local','vectorized')
        backend_obj_vectorized = get_qaoa_backend(circuit_params,device)
        variate_params = create_qaoa_variational_params(
            circuit_params, 'standard', 'ramp')
        niter = 10
        stepsize = 0.001

        # pre-computed final optimized costs
        y_precomp = [-2.4212581335011456, -2.4246393953483825, -
                     2.47312715451289, -2.5031221706241906]

        optimizer_dicts = []
        optimizer_dicts.append({'method': 'vgd', 'maxiter': niter,
                               'stepsize': stepsize, 'tol': 10**(-9), 'jac': 'finite_difference'})
        optimizer_dicts.append({'method': 'newton', 'maxiter': niter, 'stepsize': stepsize,
                               'tol': 10**(-9), 'jac': 'finite_difference', 'hess': 'finite_difference'})
        optimizer_dicts.append({'method': 'natural_grad_descent', 'maxiter': niter,
                               'stepsize': 0.01, 'tol': 10**(-9), 'jac': 'finite_difference'})
        optimizer_dicts.append({'method': 'rmsprop', 'maxiter': niter, 'stepsize': stepsize,
                               'tol': 10**(-9), 'jac': 'finite_difference', 'decay': 0.9, 'eps': 1e-07})

        for i, optimizer_dict in enumerate(optimizer_dicts):

            # Optimize
            vector_optimizer = get_optimizer(
                backend_obj_vectorized, variate_params, optimizer_dict=optimizer_dict)
            vector_optimizer()

            y_opt = vector_optimizer.qaoa_result.intermediate['intermediate cost']

            assert np.isclose(y_precomp[i], y_opt[-1], rtol=1e-04,
                              atol=1e-04), f"{optimizer_dict['method']} method failed the test."

    def test_gradient_descent_step(self):
        '''
        Check that implemented gradient descent takes the first two steps correctly.
        '''

        cost_hamil = Hamiltonian([PauliOp('ZZ', (0, 1)), PauliOp('ZZ', (1, 2)), PauliOp(
            'ZZ', (0, 3)), PauliOp('Z', (2,)), PauliOp('Z', (1,))], [1, 1.1, 1.5, 2, -0.8], 0.8)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=4)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=2)
        device = create_device('local','vectorized')
        backend_obj_vectorized = get_qaoa_backend(circuit_params,device)
        variate_params = create_qaoa_variational_params(
            circuit_params, 'standard', 'ramp')
        niter = 5
        grad_stepsize = 0.0001
        stepsize = 0.001

        params_array = variate_params.raw().copy()
        jac = derivative(backend_obj_vectorized, variate_params, self.log, 
                         'gradient', 'finite_difference', 
                         {'stepsize': grad_stepsize})

        # Optimize
        vector_optimizer = get_optimizer(backend_obj_vectorized, variate_params, optimizer_dict={
                                         'method': 'vgd', 'maxiter': niter, 'stepsize': stepsize, 'tol': 10**(-9), 'jac': jac})
        vector_optimizer()
        y_opt = vector_optimizer.qaoa_result.intermediate['intermediate cost'][1:4]

        # Stepwise optimize
        def step(x0):
            x1 = x0 - stepsize*jac(x0)

            variate_params.update_from_raw(x1)

            return [x1, np.real(backend_obj_vectorized.expectation(variate_params))]

        x0 = params_array
        variate_params.update_from_raw(x0)
        y0 = backend_obj_vectorized.expectation(variate_params)
        [x1, y1] = step(x0)
        [x2, y2] = step(x1)

        y = [y0, y1, y2]

        for i, yi in enumerate(y):
            assert np.isclose(yi, y_opt[i], rtol=1e-05, atol=1e-05)

    def test_newton_step(self):
        '''
        Check that implemented Newton descent takes the first two steps correctly.
        '''

        cost_hamil = Hamiltonian([PauliOp('ZZ', (0, 1)), PauliOp('ZZ', (1, 2)), PauliOp(
            'ZZ', (0, 3)), PauliOp('Z', (2,)), PauliOp('Z', (1,))], [1, 1.1, 1.5, 2, -0.8], 0.8)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=4)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=2)
        variate_params = create_qaoa_variational_params(
            circuit_params, 'standard', 'ramp')
        device = create_device('local','vectorized')
        backend_obj_vectorized = get_qaoa_backend(circuit_params,device)
        niter = 5
        grad_stepsize = 0.0001
        stepsize = 0.001

        params_array = variate_params.raw().copy()
        jac = derivative(backend_obj_vectorized, variate_params, self.log, 
                         'gradient', 'finite_difference', 
                         {'stepsize': grad_stepsize})
        hess = derivative(backend_obj_vectorized, variate_params, self.log, 
                          'hessian', 'finite_difference', 
                          {'stepsize': grad_stepsize})

        # Optimize
        vector_optimizer = get_optimizer(backend_obj_vectorized, variate_params, optimizer_dict={
                                         'method': 'newton', 'maxiter': niter, 'stepsize': stepsize, 'tol': 10**(-9), 'jac': jac, 'hess': hess})
        vector_optimizer()
        y_opt = vector_optimizer.qaoa_result.intermediate['intermediate cost'][1:4]

        # Stepwise optimize
        def step(x0):
            scaled_gradient = np.linalg.solve(hess(x0), jac(x0))
            x1 = x0 - stepsize*scaled_gradient
            variate_params.update_from_raw(x1)
            return [x1, np.real(backend_obj_vectorized.expectation(variate_params))]

        x0 = params_array
        variate_params.update_from_raw(x0)
        y0 = backend_obj_vectorized.expectation(variate_params)
        [x1, y1] = step(x0)
        [x2, y2] = step(x1)

        y = [y0, y1, y2]

        for i, yi in enumerate(y):
            assert np.isclose(yi, y_opt[i], rtol=1e-05, atol=1e-05)

    def test_natural_gradient_descent_step(self):
        '''
        Check that implemented natural gradient descent takes the first two steps correctly.
        '''

        cost_hamil = Hamiltonian([PauliOp('ZZ', (0, 1)), PauliOp('ZZ', (1, 2)), PauliOp(
            'ZZ', (0, 3)), PauliOp('Z', (2,)), PauliOp('Z', (1,))], [1, 1.1, 1.5, 2, -0.8], 0.8)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=4)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=2)
        device = create_device('local','vectorized')
        backend_obj_vectorized = get_qaoa_backend(circuit_params,device)
        variate_params = create_qaoa_variational_params(
            circuit_params, 'standard', 'ramp')
        niter = 5
        grad_stepsize = 0.0001
        stepsize = 0.001

        params_array = variate_params.raw().copy()
        jac = derivative(backend_obj_vectorized, variate_params, self.log, 
                         'gradient', 'finite_difference', 
                         {'stepsize': grad_stepsize})

        # Optimize
        vector_optimizer = get_optimizer(backend_obj_vectorized, variate_params, optimizer_dict={
                                         'method': 'natural_grad_descent', 'maxiter': niter, 'stepsize': stepsize, 'tol': 10**(-9), 'jac': jac})
        vector_optimizer()
        y_opt = vector_optimizer.qaoa_result.intermediate['intermediate cost'][1:4]

        # Stepwise optimize
        def step(x0):
            qfim_ = qfim(backend_obj_vectorized, variate_params, self.log)
            scaled_gradient = np.linalg.solve(qfim_(x0), jac(x0))
            x1 = x0 - stepsize*scaled_gradient
            variate_params.update_from_raw(x1)
            return [x1, np.real(backend_obj_vectorized.expectation(variate_params))]

        x0 = params_array
        variate_params.update_from_raw(x0)
        y0 = backend_obj_vectorized.expectation(variate_params)
        [x1, y1] = step(x0)
        [x2, y2] = step(x1)

        y = [y0, y1, y2]

        for i, yi in enumerate(y):
            assert np.isclose(yi, y_opt[i], rtol=1e-05, atol=1e-05)

    def test_rmsprop_step(self):
        '''
        Check that implemented RMSProp takes the first two steps correctly.
        '''

        cost_hamil = Hamiltonian([PauliOp('ZZ', (0, 1)), PauliOp('ZZ', (1, 2)), PauliOp(
            'ZZ', (0, 3)), PauliOp('Z', (2,)), PauliOp('Z', (1,))], [1, 1.1, 1.5, 2, -0.8], 0.8)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=4)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=2)
        device = create_device('local','vectorized')
        backend_obj_vectorized = get_qaoa_backend(circuit_params,device)
        variate_params = create_qaoa_variational_params(
            circuit_params, 'standard', 'ramp')
        niter = 5
        grad_stepsize = 0.0001
        stepsize = 0.001

        params_array = variate_params.raw().copy()
        jac = derivative(backend_obj_vectorized, variate_params, self.log, 
                         'gradient', 'finite_difference', 
                         {'stepsize': grad_stepsize})

        decay = 0.9
        eps = 1e-07

        # Optimize
        vector_optimizer = get_optimizer(backend_obj_vectorized, variate_params, optimizer_dict={
                                         'method': 'rmsprop', 'maxiter': niter, 'stepsize': stepsize, 'tol': 10**(-9), 'jac': jac, 'decay': decay, 'eps': eps})
        vector_optimizer()
        y_opt = vector_optimizer.qaoa_result.intermediate['intermediate cost'][1:4]

        # Stepwise optimize
        def step(x0, sqgrad0):
            sqgrad = decay*sqgrad0 + (1-decay)*jac(x0)**2
            x1 = x0 - stepsize*jac(x0)/(np.sqrt(sqgrad) + eps)
            variate_params.update_from_raw(x1)
            return [x1, np.real(backend_obj_vectorized.expectation(variate_params)), sqgrad0]

        x0 = params_array
        variate_params.update_from_raw(x0)
        y0 = backend_obj_vectorized.expectation(variate_params)
        sqgrad0 = jac(x0)**2
        [x1, y1, sqgrad1] = step(x0, sqgrad0)
        [x2, y2, sqgrad2] = step(x1, sqgrad1)

        y = [y0, y1, y2]

        for i, yi in enumerate(y):
            assert np.isclose(yi, y_opt[i], rtol=1e-05, atol=1e-05)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=PendingDeprecationWarning)
        unittest.main()
