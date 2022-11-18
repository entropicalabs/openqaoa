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
from openqaoa.optimizers.training_vqa import ScipyOptimizer, CustomScipyGradientOptimizer, PennyLaneOptimizer
import unittest
import networkx as nw
import numpy as np

from openqaoa.problems.problem import MinimumVertexCover, QUBO

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
                                **{"hub": "***", 
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
        
        """
        Checks if the X mixer created by the X_mixer_hamiltonian method and the automated methods in workflows do the same thing.
        
        For each qubit, there should be 1 RXGateMap per layer of p.
        """
        
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
        
        self.assertEqual(len(q.circuit_params.mixer_qubits_singles), 6)
        self.assertEqual(len(q.circuit_params.mixer_qubits_pairs), 0)
        for each_gatemap_name in q.circuit_params.mixer_qubits_singles:    
            self.assertEqual(each_gatemap_name, 'RXGateMap')

        for j in range(2):
            for i in range(6):
                self.assertEqual(q.circuit_params.mixer_block[j][i].qubit_1, i)

    def test_set_circuit_properties_circuit_params_mixer_xy(self):
        
        """
        Checks if the XY mixer created by the XY_mixer_hamiltonian method and the automated methods in workflows do the same thing.
        
        Depending on the qubit connectivity selected. (chain, full or star)
        For each pair of connected qubits, there should be 1 RXXGateMap and RYYGateMap per layer of p.
        """
        
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
            
            self.assertEqual(len(q.circuit_params.mixer_qubits_singles), 0)
            for i in range(0, len(q.circuit_params.mixer_qubits_pairs), 2):
                self.assertEqual(q.circuit_params.mixer_qubits_pairs[i], 'RXXGateMap')
                self.assertEqual(q.circuit_params.mixer_qubits_pairs[i+1], 'RYYGateMap')
        
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
            self.assertEqual(isinstance(q.optimizer, PennyLaneOptimizer), False)
            
        for each_method in available_optimizers()['custom_scipy_gradient']:
            q = QAOA()
            q.set_classical_optimizer(method = each_method, jac='grad_spsa', 
                                      hess='finite_difference')
            q.compile(problem = qubo_problem)
            
            self.assertEqual(isinstance(q.optimizer, ScipyOptimizer), False)
            self.assertEqual(isinstance(q.optimizer, CustomScipyGradientOptimizer), True)
            self.assertEqual(isinstance(q.optimizer, PennyLaneOptimizer), False)
            
        for each_method in available_optimizers()['custom_scipy_pennylane']:
            q = QAOA()
            q.set_classical_optimizer(method = each_method, jac='grad_spsa')
            q.compile(problem = qubo_problem)
            
            self.assertEqual(isinstance(q.optimizer, ScipyOptimizer), False)
            self.assertEqual(isinstance(q.optimizer, CustomScipyGradientOptimizer), False)
            self.assertEqual(isinstance(q.optimizer, PennyLaneOptimizer), True)

class TestingRQAOA(unittest.TestCase):
    """
    Unit test based testing of the RQAOA workflow class
    """

    def _test_default_values(self, x):
        """
        General function to check default values of rqaoa and qaoa
        """

        # circuit_properties
        cp = x.circuit_properties
        assert cp.param_type == 'standard'
        assert cp.init_type == 'ramp'
        assert cp.p == 1
        assert cp.q == None
        assert cp.mixer_hamiltonian == 'x'

        # device
        d = x.device 
        assert d.device_location == 'local'
        assert d.device_name == 'vectorized' 

    def test_rqaoa_default_values(self):
        """
        Tests all default values are correct
        """
        r = RQAOA()

        assert r.rqaoa_parameters.rqaoa_type == 'custom'
        assert r.rqaoa_parameters.n_cutoff == 5
        assert r.rqaoa_parameters.n_max == 1
        assert r.rqaoa_parameters.steps == 1
        assert r.rqaoa_parameters.original_hamiltonian == None
        assert r.rqaoa_parameters.counter == 0

        self._test_default_values(r)

    def test_rqaoa_compile_and_qoao_default_values(self):
        """
        Test creation of the qaoa object and its default values
        """
        r = RQAOA()
        r.compile(QUBO.random_instance(n=7))

        self._test_default_values(r._q)
        

    def _run_rqaoa(self, type, problem, n_cutoff=5, eliminations=1, p=1, param_type='standard', mixer='x', method='cobyla', maxiter=15, name_device='qiskit.statevector_simulator'):

        r = RQAOA()
        qiskit_device = create_device(location='local', name=name_device)
        r.set_device(qiskit_device)
        if type == 'adaptive':
            r.set_rqaoa_parameters(n_cutoff = n_cutoff, n_max=eliminations, rqaoa_type=type)
        else:
            r.set_rqaoa_parameters(n_cutoff = n_cutoff, steps=eliminations, rqaoa_type=type)
        r.set_circuit_properties(p=p, param_type=param_type, mixer_hamiltonian=mixer)
        r.set_backend_properties(prepend_state=None, append_state=None)
        r.set_classical_optimizer(method=method, maxiter=maxiter, optimization_progress=True, cost_progress=True, parameter_log=True)
        r.compile(problem)
        r.optimize()

        return r.results.get_solution()

    def test_rqaoa_optimize_multiple_times(self):
        """
        Test that the rqaoa can be compiled multiple times
        """
        graph = nw.circulant_graph(10, [1])
        problem = MinimumVertexCover(graph, field=1, penalty=10).get_qubo_problem()

        r = RQAOA()
        r.compile(problem)
        for _ in range(5): r.optimize()

    def test_example_1_adaptive_custom(self):

        # Number of qubits
        n_qubits = 12

        # Elimination schemes
        Nmax = [1,2,3,4]
        schedules = [1,[1,2,1,2,7]]
        n_cutoff = 5

        # Edges and weights of the graph
        pair_edges = [(i,i+1) for i in range(n_qubits-1)] + [(0,n_qubits-1)] 
        self_edges = [(i,) for i in range(n_qubits)]
        pair_weights = [1 for _ in range(len(pair_edges))] # All weights equal to 1
        self_weights = [10**(-4) for _ in range(len(self_edges))]

        edges = pair_edges + self_edges
        weights = pair_weights + self_weights
        problem = QUBO(n_qubits, edges, weights)

        # list of solutions of rqaoa
        solutions = []

        # run RQAOA and append solution
        for nmax in Nmax:
            solutions.append(self._run_rqaoa('adaptive', problem, n_cutoff, nmax))

        for schedule in schedules:
            solutions.append(self._run_rqaoa('custom', problem, n_cutoff, schedule))

        # Correct solution
        exact_soutions = {'101010101010': -12, '010101010101': -12}

        # Check computed solutions are among the correct ones
        for solution in solutions:
            for key in solution:
                assert solution[key] == exact_soutions[key]

    def test_example_2_adaptive_custom(self):

        # Elimination scheme
        n_cutoff = 3

        # Define problem instance (Ring graph 10 qubits)
        graph = nw.circulant_graph(10, [1])
        problem = MinimumVertexCover(graph, field=1, penalty=10).get_qubo_problem()

        # run RQAOA and append solution in list
        solutions = []
        solutions.append(self._run_rqaoa('adaptive', problem, n_cutoff))
        solutions.append(self._run_rqaoa('custom', problem, n_cutoff))

        # Correct solution
        exact_soutions = {'1010101010': 5, '0101010101': 5}

        # Check computed solutions are among the correct ones
        for solution in solutions:
            for key in solution:
                assert solution[key] == exact_soutions[key]

    def test_example_3_adaptive_custom(self):

        # Elimination scheme
        step = 2
        nmax = 4
        n_cutoff = 3

        # Define problem instance (Ring graph 10 qubits)
        graph = nw.complete_graph(10)
        problem = MinimumVertexCover(graph, field=1, penalty=10).get_qubo_problem()

        # run RQAOA and append solution in list
        solutions = []
        solutions.append(self._run_rqaoa('adaptive', problem, n_cutoff, nmax))
        solutions.append(self._run_rqaoa('custom', problem, n_cutoff, step))

        # Correct solution
        exact_soutions = {'0111111111': 9, '1011111111': 9, '1101111111': 9, '1110111111': 9, '1111011111': 9,
        '1111101111': 9, '1111110111': 9,'1111111011': 9, '1111111101': 9,'1111111110': 9}

        # Check computed solutions are among the correct ones
        for solution in solutions:
            for key in solution:
                assert solution[key] == exact_soutions[key]

    def test_example_4_adaptive_custom(self):

        # Number of qubits
        n_qubits = 10

        # Elimination schemes
        Nmax = [1,2,3,4]
        schedules = [1,2,3]
        n_cutoff = 3

        # Edges and weights of the graph
        edges = [(i,j) for j in range(n_qubits) for i in range(j)]
        weights = [1 for _ in range(len(edges))]

        problem = QUBO(n_qubits, edges, weights) 

        # list of solutions of rqaoa
        solutions = []

        # run RQAOA and append solution
        for nmax in Nmax:
            solutions.append(self._run_rqaoa('adaptive', problem, n_cutoff, nmax))

        for schedule in schedules:
            solutions.append(self._run_rqaoa('custom', problem, n_cutoff, schedule))

        # Correct solution
        exact_states = ['1111100000', '1111010000', '1110110000', '1101110000', '1011110000', '0111110000', '1111001000', '1110101000', '1101101000', '1011101000', '0111101000', '1110011000', '1101011000', '1011011000', '0111011000', '1100111000', '1010111000', '0110111000', '1001111000', '0101111000', '0011111000', '1111000100', '1110100100', '1101100100', '1011100100', '0111100100', '1110010100', '1101010100', '1011010100', '0111010100', '1100110100', '1010110100', '0110110100', '1001110100', '0101110100', '0011110100', '1110001100', '1101001100', '1011001100', '0111001100', '1100101100', '1010101100', '0110101100', '1001101100', '0101101100', '0011101100', '1100011100', '1010011100', '0110011100', '1001011100', '0101011100', '0011011100', '1000111100', '0100111100', '0010111100', '0001111100', '1111000010', '1110100010', '1101100010', '1011100010', '0111100010', '1110010010', '1101010010', '1011010010', '0111010010', '1100110010', '1010110010', '0110110010', '1001110010', '0101110010', '0011110010', '1110001010', '1101001010', '1011001010', '0111001010', '1100101010', '1010101010', '0110101010', '1001101010', '0101101010', '0011101010', '1100011010', '1010011010', '0110011010', '1001011010', '0101011010', '0011011010', '1000111010', '0100111010', '0010111010', '0001111010', '1110000110', '1101000110', '1011000110', '0111000110', '1100100110', '1010100110', '0110100110', '1001100110', '0101100110', '0011100110', '1100010110', '1010010110', '0110010110', '1001010110', '0101010110', '0011010110', '1000110110', '0100110110', '0010110110', '0001110110', '1100001110', '1010001110', '0110001110', '1001001110', '0101001110', '0011001110', '1000101110', '0100101110', '0010101110', '0001101110', '1000011110', '0100011110', '0010011110', '0001011110', '0000111110', '1111000001', '1110100001', '1101100001', '1011100001', '0111100001', '1110010001', '1101010001', '1011010001', '0111010001', '1100110001', '1010110001', '0110110001', '1001110001', '0101110001', '0011110001', '1110001001', '1101001001', '1011001001', '0111001001', '1100101001', '1010101001', '0110101001', '1001101001', '0101101001', '0011101001', '1100011001', '1010011001', '0110011001', '1001011001', '0101011001', '0011011001', '1000111001', '0100111001', '0010111001', '0001111001', '1110000101', '1101000101', '1011000101', '0111000101', '1100100101', '1010100101', '0110100101', '1001100101', '0101100101', '0011100101', '1100010101', '1010010101', '0110010101', '1001010101', '0101010101', '0011010101', '1000110101', '0100110101', '0010110101', '0001110101', '1100001101', '1010001101', '0110001101', '1001001101', '0101001101', '0011001101', '1000101101', '0100101101', '0010101101', '0001101101', '1000011101', '0100011101', '0010011101', '0001011101', '0000111101', '1110000011', '1101000011', '1011000011', '0111000011', '1100100011', '1010100011', '0110100011', '1001100011', '0101100011', '0011100011', '1100010011', '1010010011', '0110010011', '1001010011', '0101010011', '0011010011', '1000110011', '0100110011', '0010110011', '0001110011', '1100001011', '1010001011', '0110001011', '1001001011', '0101001011', '0011001011', '1000101011', '0100101011', '0010101011', '0001101011', '1000011011', '0100011011', '0010011011', '0001011011', '0000111011', '1100000111', '1010000111', '0110000111', '1001000111', '0101000111', '0011000111', '1000100111', '0100100111', '0010100111', '0001100111', '1000010111', '0100010111', '0010010111', '0001010111', '0000110111', '1000001111', '0100001111', '0010001111', '0001001111', '0000101111', '0000011111']
        exact_soutions = {state:-5 for state in exact_states}

        # Check computed solutions are among the correct ones
        for solution in solutions:
            for key in solution:
                assert solution[key] == exact_soutions[key]


    
    # TODO : do we want to test qaoa properties through rqaoa?


if __name__ == '__main__':
    unittest.main()
