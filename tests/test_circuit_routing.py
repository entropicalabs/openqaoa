# unit testing for circuit routing functionality in OQ
import unittest
import numpy as np
from typing import List, Callable, Optional

from openqaoa import QAOA
from openqaoa.qaoa_components import create_qaoa_variational_params, QAOADescriptor, PauliOp, Hamiltonian
from openqaoa.utilities import X_mixer_hamiltonian
from openqaoa.backends import QAOAvectorizedBackendSimulator, create_device
from openqaoa.problems import NumberPartition, QUBO, Knapsack
from openqaoa_pyquil.backends import DevicePyquil, QAOAPyQuilQPUBackend
from openqaoa.backends.devices_core import DeviceBase
from openqaoa.qaoa_components.ansatz_constructor.gatemap import SWAPGateMap
from qiskit import IBMQ

class TestingQAOAPyquilQVM_QR(unittest.TestCase):
    
    """Tests pyquil backend compatibility with routing_function.
    
    For all of these tests, qvm and quilc must be running.
    """
    
    def test_no_swap(self):
        """
        Tests that QAOADescriptor with a trivial `routing_function` input (with no swaps) returns identical 
        results as QAOADescriptor with no `routing_function` input, by comparing output of seeded QVM run.
        Different values of p, arguments, and cost hamiltonian coefficients are tested.

        """

        def routing_function_test1(device, problem_to_solve):

            # tuples ordered from 0,n, both SWAP and ising gates
            gate_list_indices = [[0,1]]

            # True for SWAP
            swap_mask = [False]

            # {QPU: (0 to n index)}
            initial_physical_to_logical_mapping = {0:0, 1:1}

            # 0 to n, permuted
            final_mapping = [0,1]

            return gate_list_indices, swap_mask, initial_physical_to_logical_mapping, final_mapping


        args_lst = [[np.pi/8, np.pi/4], [np.pi/3.5, np.pi/3],[np.pi/8, np.pi/4], [np.pi/3.5, np.pi/3], [1,2,3,4], [np.pi/8, np.pi/4, np.pi/8, np.pi/4], [1,2,3,4], [np.pi/8, np.pi/4, np.pi/8, np.pi/4]]
        p_lst = [1,1,1,1,2,2,2,2]
        cost_hamil_lst = [ Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1,1], 1),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1,1], 1),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [0.5,0,2], 0.7),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1.2,1], 0),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1,1], 1),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1,1], 1),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [0.5,0,2], 0.7),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1.2,1], 0)
                         ]
        shots = 2
        seed = 1
        
        device_pyquil = DevicePyquil(device_name = "2q-qvm", as_qvm=True, execution_timeout = 5, compiler_timeout=5)
        device_pyquil.quantum_computer.qam.random_seed = seed

        for i in range(len(p_lst)):

            p = p_lst[i]
            args = args_lst[i]
            cost_hamil = cost_hamil_lst[i]

            # With routing
            mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, routing_function=routing_function_test1, p=p)
            variate_params = create_qaoa_variational_params(qaoa_descriptor,'standard','ramp')

            variate_params.update_from_raw(args)
            backend_obj_pyquil = QAOAPyQuilQPUBackend(qaoa_descriptor = qaoa_descriptor, device = device_pyquil, prepend_state = None, append_state = None, init_hadamard = True, cvar_alpha = 1, n_shots=shots)
            expt_pyquil_w_qr = backend_obj_pyquil.expectation(variate_params)

            # No routing
            mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
            variate_params = create_qaoa_variational_params(qaoa_descriptor,'standard','ramp')

            variate_params.update_from_raw(args)
            backend_obj_pyquil = QAOAPyQuilQPUBackend(qaoa_descriptor = qaoa_descriptor, device = device_pyquil, prepend_state = None, append_state = None, init_hadamard = True, cvar_alpha = 1, n_shots=shots)
            expt_pyquil_no_qr = backend_obj_pyquil.expectation(variate_params)

            self.assertAlmostEqual(expt_pyquil_w_qr, expt_pyquil_no_qr)
            
    def test_cancelled_swap(self):
        """
        Tests that QAOADescriptor with a trivial `routing_function` input (with two swaps that cancel each other) returns identical 
        results as QAOADescriptor with no `routing_function` input, by comparing output of seeded QVM run.
        Different values of p, arguments, and cost hamiltonian coefficients are tested.

        """

        def routing_function_test1(device, problem_to_solve):

            # tuples ordered from 0,n, both SWAP and ising gates
            gate_list_indices = [[0,1],[0,1],[0,1]]

            # True for SWAP
            swap_mask = [True,True,False]

            # {QPU: (0 to n index)}
            initial_physical_to_logical_mapping = {0:0, 1:1}

            # 0 to n, permuted
            final_mapping = [0,1]

            return gate_list_indices, swap_mask, initial_physical_to_logical_mapping, final_mapping


        args_lst = [[np.pi/8, np.pi/4], [np.pi/3.5, np.pi/3],[np.pi/8, np.pi/4], [np.pi/3.5, np.pi/3], [1,2,3,4], [np.pi/8, np.pi/4, np.pi/8, np.pi/4], [1,2,3,4], [np.pi/8, np.pi/4, np.pi/8, np.pi/4]]
        p_lst = [1,1,1,1,2,2,2,2]
        cost_hamil_lst = [ Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1,1], 1),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1,1], 1),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [0.5,0,2], 0.7),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1.5,1.2,1], 0),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1,1], 1),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1,1], 1),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [0.5,0,2], 0.7),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,5.2,1], 0)
                         ]
        shots = 3
        seed = 4

        device_pyquil = DevicePyquil(device_name = "2q-qvm", as_qvm=True, execution_timeout = 5, compiler_timeout=5)
        device_pyquil.quantum_computer.qam.random_seed = seed

        for i in range(len(p_lst)):

            p = p_lst[i]
            args = args_lst[i]
            cost_hamil = cost_hamil_lst[i]

            # With routing
            mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, routing_function=routing_function_test1, p=p)
            variate_params = create_qaoa_variational_params(qaoa_descriptor,'standard','ramp')

            variate_params.update_from_raw(args)
            backend_obj_pyquil = QAOAPyQuilQPUBackend(qaoa_descriptor = qaoa_descriptor, device = device_pyquil, prepend_state = None, append_state = None, init_hadamard = True, cvar_alpha = 1, n_shots=shots)
            expt_pyquil_w_qr = backend_obj_pyquil.expectation(variate_params)

            # No routing
            mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
            variate_params = create_qaoa_variational_params(qaoa_descriptor,'standard','ramp')

            variate_params.update_from_raw(args)
            backend_obj_pyquil = QAOAPyQuilQPUBackend(qaoa_descriptor = qaoa_descriptor, device = device_pyquil, prepend_state = None, append_state = None, init_hadamard = True, cvar_alpha = 1, n_shots=shots)
            expt_pyquil_no_qr = backend_obj_pyquil.expectation(variate_params)

            self.assertAlmostEqual(expt_pyquil_w_qr, expt_pyquil_no_qr)
            
    def test_simplest_swap(self):
        """
        Tests that QAOADescriptor with a trivial `routing_function` input (with no swaps) returns identical 
        results as QAOADescriptor with no `routing_function` input, by comparing output of seeded QVM run.
        Different values of p, arguments, and cost hamiltonian coefficients are tested.
        
        Note : Even with a fixed seed, insertion of swaps changes measurement statistics. 
        Final assertion is therefore only up to a tolerance, chosen by eyeballing results for a chosen seed.

        """

        def routing_function_test1(device, problem_to_solve):

            # tuples ordered from 0,n, both SWAP and ising gates
            gate_list_indices = [[0,1],[0,1]]

            # True for SWAP
            swap_mask = [True,False]

            # {QPU: (0 to n index)}
            initial_physical_to_logical_mapping = {0:0, 1:1}

            # 0 to n, permuted
            final_mapping = [1,0]

            return gate_list_indices, swap_mask, initial_physical_to_logical_mapping, final_mapping


        args_lst = [[np.pi/8, np.pi/4], [np.pi/3.5, np.pi/3],[np.pi/8, np.pi/4], [np.pi/3.5, np.pi/3], [1,2,3,4], [np.pi/8, np.pi/4, np.pi/8, np.pi/4], [1,2,3,4], [np.pi/8, np.pi/4, np.pi/8, np.pi/4]]
        p_lst = [1,1,1,1,2,2,2,2]
        cost_hamil_lst = [ Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1,1], 1),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1,1], 1),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [0.5,0,2], 0.7),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1.5,1.2,1], 0),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1,1], 1),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1,1], 1),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [0.5,0,2], 0.7),
                          Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,5.2,1], 0)
                         ]
        shots = 10
        seed = 4

        device_pyquil = DevicePyquil(device_name = "2q-qvm", as_qvm=True, execution_timeout = 10, compiler_timeout=10)
        device_pyquil.quantum_computer.qam.random_seed = seed

        for i in range(len(p_lst)):

            p = p_lst[i]
            args = args_lst[i]
            cost_hamil = cost_hamil_lst[i]

            # With routing
            mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, routing_function=routing_function_test1, p=p)
            variate_params = create_qaoa_variational_params(qaoa_descriptor,'standard','ramp')

            variate_params.update_from_raw(args)
            backend_obj_pyquil = QAOAPyQuilQPUBackend(qaoa_descriptor = qaoa_descriptor, device = device_pyquil, prepend_state = None, append_state = None, init_hadamard = True, cvar_alpha = 1, n_shots=shots)
            expt_pyquil_w_qr = backend_obj_pyquil.expectation(variate_params)

            # No routing
            mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
            variate_params = create_qaoa_variational_params(qaoa_descriptor,'standard','ramp')

            variate_params.update_from_raw(args)
            backend_obj_pyquil = QAOAPyQuilQPUBackend(qaoa_descriptor = qaoa_descriptor, device = device_pyquil, prepend_state = None, append_state = None, init_hadamard = True, cvar_alpha = 1, n_shots=shots)
            expt_pyquil_no_qr = backend_obj_pyquil.expectation(variate_params)

            # Note : Even with a fixed seed, insertion of swaps changes measurement statistics. 
            # Final assertion is therefore only up to a tolerance, chosen by eyeballing results for a chosen seed.
            self.assertAlmostEqual(expt_pyquil_w_qr, expt_pyquil_no_qr, delta = 1)
            
            
            
    def test_different_topologies(self):
        """
        Tests QAOADescriptor with different devices.
        results as QAOADescriptor with no `routing_function` input, by comparing output of seeded QVM run.
        Different values of p, arguments, and cost hamiltonian coefficients are tested.

        """
        
        def routing_function_test1(device, problem_to_solve):

            # tuples ordered from 0,n, both SWAP and ising gates
            gate_list_indices = [[0,1],[1,0],[0,1]]

            # True for SWAP
            swap_mask = [True,True,False]

            # {QPU: (0 to n index)}
            initial_physical_to_logical_mapping = {0:0, 1:1}

            # 0 to n, permuted
            final_mapping = [0,1]

            return gate_list_indices, swap_mask, initial_physical_to_logical_mapping, final_mapping


        args_lst = [[np.pi/8, np.pi/4], [np.pi/8, np.pi/4, 1, 2], [1,2,3,4]]
        p_lst = [1,2,2]
        cost_hamil_lst = [ Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1.5,2], 0.5),
                           Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1.5,2], 0.5),
                           Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1.5,2], 0.5)
                         ]

        device_name_lst = ['2q-qvm', '3q-qvm', 'Aspen-M-3']

        shots = 2
        seed = 1

        for i in range(len(p_lst)):

            p = p_lst[i]
            args = args_lst[i]
            cost_hamil = cost_hamil_lst[i]

            device_pyquil = DevicePyquil(device_name = device_name_lst[i], as_qvm=True, execution_timeout = 5, compiler_timeout=5)
            device_pyquil.quantum_computer.qam.random_seed = seed

            # With routing
            mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, routing_function=routing_function_test1, p=p)
            variate_params = create_qaoa_variational_params(qaoa_descriptor,'standard','ramp')

            variate_params.update_from_raw(args)
            backend_obj_pyquil = QAOAPyQuilQPUBackend(qaoa_descriptor = qaoa_descriptor, device = device_pyquil, prepend_state = None, append_state = None, init_hadamard = True, cvar_alpha = 1, n_shots=shots)
            expt_pyquil_w_qr = backend_obj_pyquil.expectation(variate_params)

            # No routing
            mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
            variate_params = create_qaoa_variational_params(qaoa_descriptor,'standard','ramp')

            variate_params.update_from_raw(args)
            backend_obj_pyquil = QAOAPyQuilQPUBackend(qaoa_descriptor = qaoa_descriptor, device = device_pyquil, prepend_state = None, append_state = None, init_hadamard = True, cvar_alpha = 1, n_shots=shots)
            expt_pyquil_no_qr = backend_obj_pyquil.expectation(variate_params)

            self.assertAlmostEqual(expt_pyquil_w_qr, expt_pyquil_no_qr)


class ExpectedRouting:
    
        def __init__(
            self, 
            device_name, 
            device_location, 
            problem_to_solve, 
            initial_mapping,
            gate_indices_list,
            swap_mask,
            initial_physical_to_logical_mapping,
            final_logical_qubit_order
        ):
            self.device_name = device_name
            self.device_location = device_location
            self.problem_to_solve = problem_to_solve
            self.initial_mapping = initial_mapping

            self.gate_indices_list = gate_indices_list
            self.swap_mask = swap_mask
            self.initial_physical_to_logical_mapping = initial_physical_to_logical_mapping
            self.final_logical_qubit_order = final_logical_qubit_order

        def values_input(self):
            return self.device_name, self.device_location, self.problem_to_solve, self.initial_mapping

        def values_return(self):
            return self.gate_indices_list, self.swap_mask, self.initial_physical_to_logical_mapping, self.final_logical_qubit_order
    
IBM_OSLO_1 = ExpectedRouting(
    device_name='ibm_oslo',
    device_location='ibmq',
    problem_to_solve=[(0, 1), (2, 3), (2, 4), (3, 4), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)],
    initial_mapping=None,
    gate_indices_list= [[3, 5], [1, 5], [4, 5], [2, 6], [0, 6], [4, 6], [4, 6], [4, 5], [0, 6], [1, 5], [4, 5], [4, 5], [4, 6], [0, 6], [4, 6], [1, 5], [4, 6], [4, 5]],
    swap_mask=[True, False, False, True, False, False, True, False, False, True, False, True, False, True, False, True, True, False],
    initial_physical_to_logical_mapping={6: 0, 0: 1, 4: 2, 2: 3, 3: 4, 1: 5, 5: 6},
    final_logical_qubit_order=[4, 6, 1, 5, 0, 3, 2]
)


class TestingQubitRouting(unittest.TestCase):

    def __routing_function_mock(self,
        device: DeviceBase,
        problem_to_solve: List[List[int]],
        initial_mapping: Optional[List[int]] = None
    ):
        """
        function that imitates the routing function for testing purposes.
        """
        if IBM_OSLO_1.values_input() == (device.device_name, device.device_location, problem_to_solve, initial_mapping):
            return IBM_OSLO_1.values_return()


    def __compare_results(self, expected: ExpectedRouting, qaoa):
        """
        function that compares the expected and actual results of the routing function.
        """
        backend, result, qubo = qaoa.backend, qaoa.result, qaoa.problem
        
        assert backend.n_qubits == len(expected.final_logical_qubit_order), \
        """Number of qubits in the circuit is not equal to the number of qubits given by routing"""

        assert backend.problem_qubits == qubo.n, \
        f"""Number of nodes in problem is not equal to backend.problem_qubits, 
        is '{backend.problem_qubits }' but should be '{qubo.n}'"""

        assert len(list(result.optimized['measurement_outcomes'].keys())[0]) == qubo.n, \
        "The number of qubits in the optimized circuit is not equal to the number of qubits in the problem."

        # check that swap gates are applied in the correct position
        swap_mask_new = []
        for gate in backend.abstract_circuit:
            if not gate.gate_label.n_qubits == 1:
                swap_mask_new.append(isinstance(gate, SWAPGateMap))
        assert swap_mask_new == IBM_OSLO_1.swap_mask, "Swap gates are not in the correct position"

        # check that the correct qubits are used in the gates
        gate_indices_list_new  = []
        for gate in backend.abstract_circuit:
            if not gate.gate_label.n_qubits == 1:
                gate_indices_list_new.append([gate.qubit_1, gate.qubit_2])
        assert gate_indices_list_new == IBM_OSLO_1.gate_indices_list, "The qubits used in the gates are not correct"

        

    def test_qubit_routing(self):

        IBMQ.load_account()

        qubo = Knapsack.random_instance(n_items=3, seed=20).qubo

        qpu_credentials ={
            "hub": "ibm-q",
            "group": "open", 
            "project": "main"
        }

        qiskit_cloud = create_device(location='ibmq', name='ibm_oslo', **qpu_credentials, as_emulator=True)

        q_qiskit = QAOA()
        q_qiskit.set_device(qiskit_cloud)
        q_qiskit.set_circuit_properties(p=1, param_type='standard', init_type='rand', mixer_hamiltonian='x')
        q_qiskit.set_backend_properties(prepend_state=None, append_state=None)
        q_qiskit.set_classical_optimizer(method='nelder-mead', maxiter=1, cost_progress=True, parameter_log=True, optimization_progress=True)
        q_qiskit.compile(qubo, routing_function=self.__routing_function_mock)
        q_qiskit.optimize()

        # check that the results and circuit are as expected
        self.__compare_results(IBM_OSLO_1, q_qiskit)
            
            
if __name__ == '__main__':
    unittest.main()