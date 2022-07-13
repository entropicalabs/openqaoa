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

from argparse import SUPPRESS
from threading import local
from openqaoa.utilities import X_mixer_hamiltonian, XY_mixer_hamiltonian
from openqaoa.workflows.optimizer import QAOA, RQAOA
from openqaoa.backends.qaoa_backend import (DEVICE_NAME_TO_OBJECT_MAPPER,
                                            DEVICE_ACCESS_OBJECT_MAPPER)
from openqaoa.devices import create_device,SUPPORTED_LOCAL_SIMULATORS, DeviceLocal, DevicePyquil, DeviceQiskit
from openqaoa.qaoa_parameters import (Hamiltonian, QAOACircuitParams, QAOAVariationalStandardParams, QAOAVariationalStandardWithBiasParams, 
QAOAVariationalExtendedParams, QAOAVariationalFourierParams, 
QAOAVariationalFourierExtendedParams, QAOAVariationalFourierWithBiasParams)
from openqaoa.backends.simulators.qaoa_pyquil_sim import QAOAPyQuilWavefunctionSimulatorBackend
from openqaoa.backends.simulators.qaoa_qiskit_sim import QAOAQiskitBackendShotBasedSimulator, QAOAQiskitBackendStatevecSimulator
from openqaoa.backends.simulators.qaoa_vectorized import QAOAvectorizedBackendSimulator
from openqaoa.optimizers.qaoa_optimizer import available_optimizers
from openqaoa.optimizers.training_vqa import ScipyOptimizer, CustomScipyGradientOptimizer
import unittest
import networkx as nw
import pytest
import numpy as np

from openqaoa.problems.problem import MinimumVertexCover

ALLOWED_LOCAL_SIMUALTORS = SUPPORTED_LOCAL_SIMULATORS
LOCAL_DEVICES = ALLOWED_LOCAL_SIMUALTORS + ['6q-qvm', 'Aspen-11']


class TestingVanillaQAOA(unittest.TestCase):

    """
    Unit test based testing of the QAOA workflow class
    """

    def test_vanilla_qaoa_default_values(self):
        
        q = QAOA()
        assert q.circuit_properties.p == 1
        assert q.circuit_properties.param_type == 'standard'
        assert q.circuit_properties.init_type == 'ramp'
        assert q.device.device_location == 'local'
        assert q.device.device_name == 'vectorized'

    def test_end_to_end_vectorized(self):
        
        g = nw.circulant_graph(6, [1])
        vc = MinimumVertexCover(g, field =1.0, penalty=10).get_qubo_problem()

        q = QAOA()
        q.set_classical_optimizer(optimization_progress = True)
        q.compile(vc)
        q.optimize()

        result = q.results.most_probable_states['solutions_bitstrings'][0]
        assert '010101' == result or '101010' == result

    def test_set_device_local(self):
        """"
        Check that all local devices are correctly initialised
        """
        q = QAOA()
        for d in q.local_simulators:
            q.set_device(create_device(location='local', name=d))
            assert type(q.device) == DeviceLocal
            assert q.device.device_name == d
            assert q.device.device_location == 'local'

    def test_set_device_cloud(self):
        """"
        Check that all QPU-provider related devices are correctly initialised
        """
        q = QAOA()
        q.set_device(create_device('qcs', 
                                name='6q-qvm',
                                **{'as_qvm':True, 'execution_timeout' : 10, 'compiler_timeout':10}))
        assert type(q.device) == DevicePyquil
        assert q.device.device_name == '6q-qvm'
        assert q.device.device_location ==  'qcs'


        q.set_device(create_device('ibmq', 
                                name='place_holder',
                                **{"api_token": "**",
                                "hub": "***", 
                                "group": "***", 
                                "project": "***"}))
        assert type(q.device) == DeviceQiskit
        assert q.device.device_name == 'place_holder'
        assert q.device.device_location ==  'ibmq'

    def test_compile_before_optimise(self):
        """
        Assert that compilation has to be called before optimisation
        """    
        g = nw.circulant_graph(6, [1])
        # vc = MinimumVertexCover(g, field =1.0, penalty=10).get_qubo_problem()

        q = QAOA()
        q.set_classical_optimizer(optimization_progress = True)

        self.assertRaises(ValueError, lambda: q.optimize())
            
    def test_cost_hamil(self):
        
        g = nw.circulant_graph(6, [1])
        problem = MinimumVertexCover(g, field =1.0, penalty=10)
        qubo_problem = problem.get_qubo_problem()
        
        test_hamil = Hamiltonian.classical_hamiltonian(terms = qubo_problem.terms, 
                                                       coeffs = qubo_problem.weights, 
                                                       constant = qubo_problem.constant)

        q = QAOA()
        
        q.compile(problem = qubo_problem)
        
        self.assertEqual(q.cost_hamil.expression, test_hamil.expression)
        self.assertEqual(q.circuit_params.cost_hamiltonian.expression, 
                         test_hamil.expression)
        
    def test_set_circuit_properties_fourier_q(self):
        
        """
        The value of q should be None if the param_type used is not fourier.
        Else if param_type is fourier, fourier_extended or fourier_w_bias, it
        should be the value of q, if it is provided.
        """
        
        fourier_param_types = ['fourier', 'fourier_extended', 'fourier_w_bias']
        
        q = QAOA()
        
        for each_param_type in fourier_param_types:
            q.set_circuit_properties(param_type = each_param_type, q = 1)
            self.assertEqual(q.circuit_properties.q, 1)
        
        q.set_circuit_properties(param_type = "standard", q = 1)
        
        self.assertEqual(q.circuit_properties.q, None)
        
    def test_set_circuit_properties_annealing_time_linear_ramp_time(self):
        
        """
        Check that linear_ramp_time and annealing_time are updated appropriately 
        as the value of p is changed. 
        """
        
        q = QAOA()
        
        q.set_circuit_properties(p=3)
        
        self.assertEqual(q.circuit_properties.annealing_time, 0.7*3)
        self.assertEqual(q.circuit_properties.linear_ramp_time, 0.7*3)
        
        q.set_circuit_properties(p=2)
        
        self.assertEqual(q.circuit_properties.annealing_time, 0.7*2)
        self.assertEqual(q.circuit_properties.linear_ramp_time, 0.7*2)
        
            
    def test_set_circuit_properties_circuit_params_mixer_x(self):
        
        nodes = 6
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes,p=edge_probability)
        problem = MinimumVertexCover(g, field =1.0, penalty=10)

        q = QAOA()
        q.set_circuit_properties(mixer_hamiltonian = 'x', p = 2)

        q.compile(problem = problem.get_qubo_problem())

        self.assertEqual(type(q.circuit_params), QAOACircuitParams)
        self.assertEqual(q.circuit_params.p, 2)

        mixer_hamil = X_mixer_hamiltonian(n_qubits = nodes)

        self.assertEqual(q.mixer_hamil.expression, mixer_hamil.expression)
        self.assertEqual(q.circuit_params.mixer_hamiltonian.expression, 
                         mixer_hamil.expression)
            
    def test_set_circuit_properties_circuit_params_mixer_xy(self):
        
        g_c = nw.circulant_graph(6, [1])
        g_f = nw.complete_graph(6)
        # A 5-sided star graoh requires 6 qubit. (Center Qubit of the pattern)
        g_s = nw.star_graph(5)
        problems = [MinimumVertexCover(g_c, field =1.0, penalty=10), 
                   MinimumVertexCover(g_f, field =1.0, penalty=10), 
                   MinimumVertexCover(g_s, field =1.0, penalty=10)]
        qubit_connectivity_name = ['chain', 'full', 'star']
        
        for i in range(3):
            q = QAOA()
            q.set_circuit_properties(mixer_hamiltonian = 'xy', 
                                     mixer_qubit_connectivity = qubit_connectivity_name[i],
                                     p = 2)

            q.compile(problem = problems[i].get_qubo_problem())

            self.assertEqual(type(q.circuit_params), QAOACircuitParams)
            self.assertEqual(q.circuit_params.p, 2)

            mixer_hamil = XY_mixer_hamiltonian(n_qubits = 6, qubit_connectivity = qubit_connectivity_name[i])
            
            self.assertEqual(q.mixer_hamil.expression, mixer_hamil.expression)
            self.assertEqual(q.circuit_params.mixer_hamiltonian.expression, 
                             mixer_hamil.expression)
        
    def test_set_circuit_properties_variate_params(self):
        
        """
        Ensure that the Varitional Parameter Object created based on the input string , param_type, is correct.
        
        TODO: Check if q=None is the appropriate default.
        """
        
        param_type_names = ['standard', 'standard_w_bias', 'extended', 
                            'fourier', 'fourier_extended', 'fourier_w_bias']
        object_types = [QAOAVariationalStandardParams, 
                        QAOAVariationalStandardWithBiasParams, 
                        QAOAVariationalExtendedParams, QAOAVariationalFourierParams, 
                        QAOAVariationalFourierExtendedParams, 
                        QAOAVariationalFourierWithBiasParams]
        
        nodes = 6
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes,p=edge_probability)
        problem = MinimumVertexCover(g, field =1.0, penalty=10)
        
        for i in range(len(object_types)):

            q = QAOA()
            q.set_circuit_properties(param_type = param_type_names[i], q=1)

            q.compile(problem = problem.get_qubo_problem())
            
            self.assertEqual(type(q.variate_params), object_types[i])
            
    def test_set_circuit_properties_change(self):
        
        """
        Ensure that once a property has beefn changed via set_circuit_properties.
        The attribute has been appropriately updated. 
        Updating all attributes at the same time.
        """
        
#         default_pairings = {'param_type': 'standard', 
#                             'init_type': 'ramp', 
#                             'qubit_register': [], 
#                             'p': 1, 
#                             'q': None, 
#                             'annealing_time': 0.7, 
#                             'linear_ramp_time': 0.7, 
#                             'variational_params_dict': {}, 
#                             'mixer_hamiltonian': 'x', 
#                             'mixer_qubit_connectivity': None, 
#                             'mixer_coeffs': None, 
#                             'seed': None}
        
        q = QAOA()

        # TODO: Some weird error related to the initialisation of QAOA here
#         for each_key, each_value in default_pairings.items():
#             print(each_key, getattr(q.circuit_properties, each_key), each_value)
#             self.assertEqual(getattr(q.circuit_properties, each_key), each_value)
            
        update_pairings = {'param_type': 'fourier', 
                           'init_type': 'rand', 
                           'qubit_register': [0, 1], 
                           'p': 2, 
                           'q': 2, 
                           'annealing_time': 1.0, 
                           'linear_ramp_time': 1.0, 
                           'variational_params_dict': {'key': 'value'}, 
                           'mixer_hamiltonian': 'xy', 
                           'mixer_qubit_connectivity': 'chain', 
                           'mixer_coeffs': [0.1, 0.2], 
                           'seed': 45}
        
        q.set_circuit_properties(**update_pairings)
        
        for each_key, each_value in update_pairings.items():
            self.assertEqual(getattr(q.circuit_properties, each_key), each_value)
            
    def test_set_circuit_properties_rejected_values(self):
        
        """
        Some properties of CircuitProperties Object return a ValueError if the specified property has not been whitelisted in the code. 
        This checks that the ValueError is raised if the argument is not whitelisted.
        """
        
        q = QAOA()
        
        self.assertRaises(ValueError, lambda: q.set_circuit_properties(param_type = 'wrong name'))
        self.assertRaises(ValueError, lambda: q.set_circuit_properties(init_type = 'wrong name'))
        self.assertRaises(ValueError, lambda: q.set_circuit_properties(mixer_hamiltonian = 'wrong name'))
        self.assertRaises(ValueError, lambda: q.set_circuit_properties(p = -1))
            
    def test_set_backend_properties_change(self):
        
        """
        Ensure that once a property has been changed via set_backend_properties.
        The attribute has been appropriately updated.
        Updating all attributes at the same time.
        """
        
        default_pairings = {'prepend_state': None, 
                            'append_state': None, 
                            'init_hadamard': True, 
                            'n_shots': 100, 
                            'cvar_alpha': 1.}
        
        q = QAOA()
        
        for each_key, each_value in default_pairings.items():
            self.assertEqual(getattr(q.backend_properties, each_key), each_value)
            
        update_pairings = {'prepend_state': [[0, 0]], 
                           'append_state': [[0, 0]], 
                           'init_hadamard': False, 
                           'n_shots': 10, 
                           'cvar_alpha': .5}
        
        q.set_backend_properties(**update_pairings)
        
        for each_key, each_value in update_pairings.items():
            self.assertEqual(getattr(q.backend_properties, each_key), each_value)
            
    def test_set_backend_properties_check_backend_vectorized(self):
        
        """
        Check if the backend returned by set_backend_properties is correct
        Based on the input device.
        Also Checks if defaults from workflows are used in the backend.
        """
        
        nodes = 6
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes,p=edge_probability)
        problem = MinimumVertexCover(g, field =1.0, penalty=10)
        qubo_problem = problem.get_qubo_problem()
        
        q = QAOA()
        q.set_device(create_device(location = 'local', name = 'vectorized'))
        q.compile(problem = qubo_problem)
        
        self.assertEqual(type(q.backend), QAOAvectorizedBackendSimulator)
        
        self.assertEqual(q.backend.init_hadamard, True)
        self.assertEqual(q.backend.prepend_state, None)
        self.assertEqual(q.backend.append_state, None)
        self.assertEqual(q.backend.cvar_alpha, 1)
        
        self.assertRaises(AttributeError, lambda: q.backend.n_shots)
        
    def test_set_backend_properties_check_backend_vectorized_w_custom(self):
        
        """
        Check if the backend returned by set_backend_properties is correct
        Based on the input device.
        Uses custom values for attributes in backend_properties and checks if the
        backend object responds appropriately.
        """
        
        nodes = 3
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes,p=edge_probability)
        problem = MinimumVertexCover(g, field =1.0, penalty=10)
        qubo_problem = problem.get_qubo_problem()
        
        q = QAOA()
        q.set_device(create_device(location = 'local', name = 'vectorized'))
        
        prepend_state_rand = np.random.rand(2**3)
        append_state_rand = np.eye(2**3)
        
        update_pairings = {'prepend_state': prepend_state_rand, 
                           'append_state': append_state_rand, 
                           'init_hadamard': False, 
                           'n_shots': 10, 
                           'cvar_alpha': 1}
        
        q.set_backend_properties(**update_pairings)
        
        q.compile(problem = qubo_problem)
        
        self.assertEqual(type(q.backend), QAOAvectorizedBackendSimulator)

        self.assertEqual(q.backend.init_hadamard, False)
        self.assertEqual((q.backend.prepend_state == prepend_state_rand).all(), True)
        self.assertEqual((q.backend.append_state == append_state_rand).all(), True)
        self.assertEqual(q.backend.cvar_alpha, 1)
        
        self.assertRaises(AttributeError, lambda: q.backend.n_shots)
        
    def test_set_backend_properties_check_backend_vectorized_error_values(self):
        
        """
        If the values provided from the workflows are incorrect, we should
        receive the appropriate error messages from the vectorized backend.
        
        Checks:
        Incorrect size of prepend state and append state.
        """
        
        nodes = 3
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes,p=edge_probability)
        problem = MinimumVertexCover(g, field =1.0, penalty=10)
        qubo_problem = problem.get_qubo_problem()
        
        q = QAOA()
        q.set_device(create_device(location = 'local', name = 'vectorized'))
        
        prepend_state_rand = np.random.rand(2**2)
        
        update_pairings = {'prepend_state': prepend_state_rand, 
                           'append_state': None}
        
        q.set_backend_properties(**update_pairings)
        
        self.assertRaises(ValueError, lambda : q.compile(problem = qubo_problem))
        
        q = QAOA()
        q.set_device(create_device(location = 'local', name = 'vectorized'))
        
        append_state_rand = np.random.rand(2**2, 2**2)
        
        update_pairings = {'prepend_state': None, 
                           'append_state': append_state_rand}
        
        q.set_backend_properties(**update_pairings)
        
        self.assertRaises(ValueError, lambda : q.compile(problem = qubo_problem))
        
    def test_set_backend_properties_check_backend_qiskit_qasm(self):
        
        """
        Check if the backend returned by set_backend_properties is correct
        Based on the input device. For qiskit qasm simulator.
        Also Checks if defaults from workflows are used in the backend.
        """
        
        nodes = 6
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes,p=edge_probability)
        problem = MinimumVertexCover(g, field =1.0, penalty=10)
        qubo_problem = problem.get_qubo_problem()
        
        q = QAOA()
        q.set_device(create_device(location = 'local', name = 'qiskit.qasm_simulator'))
        q.compile(problem = qubo_problem)
        
        self.assertEqual(type(q.backend), QAOAQiskitBackendShotBasedSimulator)
        
        self.assertEqual(q.backend.init_hadamard, True)
        self.assertEqual(q.backend.prepend_state, None)
        self.assertEqual(q.backend.append_state, None)
        self.assertEqual(q.backend.cvar_alpha, 1)
        self.assertEqual(q.backend.n_shots, 100)
        
    def test_set_backend_properties_check_backend_qiskit_statevector(self):
        
        """
        Check if the backend returned by set_backend_properties is correct
        Based on the input device. For qiskit statevector simulator.
        Also Checks if defaults from workflows are used in the backend.
        """
        
        nodes = 6
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes,p=edge_probability)
        problem = MinimumVertexCover(g, field =1.0, penalty=10)
        qubo_problem = problem.get_qubo_problem()
        
        q = QAOA()
        q.set_device(create_device(location = 'local', name = 'qiskit.statevector_simulator'))
        q.compile(problem = qubo_problem)
        
        self.assertEqual(type(q.backend), QAOAQiskitBackendStatevecSimulator)
        
        self.assertEqual(q.backend.init_hadamard, True)
        self.assertEqual(q.backend.prepend_state, None)
        self.assertEqual(q.backend.append_state, None)
        self.assertEqual(q.backend.cvar_alpha, 1)
        
        self.assertRaises(AttributeError, lambda: q.backend.n_shots)
        
    def test_set_backend_properties_check_backend_pyquil_statevector(self):
        
        """
        Check if the backend returned by set_backend_properties is correct
        Based on the input device. For pyquil statevector simulator.
        Also Checks if defaults from workflows are used in the backend.
        """
        
        nodes = 6
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes,p=edge_probability)
        problem = MinimumVertexCover(g, field =1.0, penalty=10)
        qubo_problem = problem.get_qubo_problem()
        
        q = QAOA()
        q.set_device(create_device(location = 'local', name = 'pyquil.statevector_simulator'))
        q.compile(problem = qubo_problem)
        
        self.assertEqual(type(q.backend), QAOAPyQuilWavefunctionSimulatorBackend)
        
        self.assertEqual(q.backend.init_hadamard, True)
        self.assertEqual(q.backend.prepend_state, None)
        self.assertEqual(q.backend.append_state, None)
        self.assertEqual(q.backend.cvar_alpha, 1)
        
        self.assertRaises(AttributeError, lambda: q.backend.n_shots)
        
    def test_set_classical_optimizer_defaults(self):
        
        """
        Check if the fields in the default classical_optimizer dict are correct
        """
        
        default_pairings = {'optimize': True, 'method': 'cobyla', 
                            'maxiter': 100, 'jac': None, 'hess': None, 
                            'constraints': None, 'bounds': None, 'tol': None, 
                            'optimizer_options': None, 'jac_options': None, 
                            'hess_options': None, 'optimization_progress': False, 
                            'cost_progress': True, 'parameter_log': True, 
                            }
        
        q = QAOA()
        
        for each_key, each_value in default_pairings.items():
            self.assertEqual(getattr(q.classical_optimizer, each_key), each_value)
            
            if each_value != None:
                self.assertEqual(q.classical_optimizer.asdict()[each_key], each_value)
            
    def test_set_classical_optimizer_jac_hess_casing(self):
        
        """
        jac and hess should be in lower case if it is a string.
        """
        
        q = QAOA()
        q.set_classical_optimizer(jac = 'JaC', hess = 'HeSS')
        
        self.assertEqual(q.classical_optimizer.jac, 'jac')
        self.assertEqual(q.classical_optimizer.hess, 'hess')
        
    def test_set_classical_optimizer_method_selectors(self):
        
        """
        Different methods would return different Optimizer classes.
        Check that the correct class is returned.
        """
        
        nodes = 6
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes,p=edge_probability)
        problem = MinimumVertexCover(g, field =1.0, penalty=10)
        qubo_problem = problem.get_qubo_problem()
        
        for each_method in available_optimizers()['scipy']:
            q = QAOA()
            q.set_classical_optimizer(method = each_method, jac='grad_spsa')
            q.compile(problem = qubo_problem)
            
            self.assertEqual(isinstance(q.optimizer, ScipyOptimizer), True)
            self.assertEqual(isinstance(q.optimizer, CustomScipyGradientOptimizer), False)
            
        for each_method in available_optimizers()['custom_scipy_gradient']:
            q = QAOA()
            q.set_classical_optimizer(method = each_method, jac='grad_spsa', 
                                      hess='finite_difference')
            q.compile(problem = qubo_problem)
            
            self.assertEqual(isinstance(q.optimizer, ScipyOptimizer), False)
            self.assertEqual(isinstance(q.optimizer, CustomScipyGradientOptimizer), True)

class TestingRQAOA(unittest.TestCase):
    """
    Unit test based testing of the RQAOA workflow class
    """

    def test_rqaoa_default_values(self):
        """
        Tests all default values are correct
        """
        r = RQAOA()

        assert isinstance(r.qaoa,QAOA)
        assert r.qaoa.circuit_properties.p == 1
        assert r.qaoa.circuit_properties.param_type == 'standard'
        assert r.qaoa.circuit_properties.init_type == 'ramp'
        assert r.qaoa.device.device_location == 'local'
        assert r.qaoa.device.device_name == 'vectorized'
        assert r.rqaoa_parameters.rqaoa_type == 'adaptive'
        assert r.rqaoa_parameters.n_cutoff == 5
        assert r.rqaoa_parameters.n_max == 1
        assert r.rqaoa_parameters.steps == 1

    def test_end_to_end_vectorized(self):
        """
        Test the full workflow with vectorized backend.
        """
        
        g = nw.circulant_graph(6, [1])
        vc = MinimumVertexCover(g, field =1.0, penalty=10).get_qubo_problem()

        r = RQAOA()
        r.compile(vc)
        r.optimize()

        # Computed solution
        sol_states = list(r.result['solution'].keys())

        # Correct solution
        exact_sol_states = ['101010','010101']

        # Check computed solutions are among the correct ones
        for sol in sol_states:
            assert sol in exact_sol_states

if __name__ == '__main__':
    unittest.main()
