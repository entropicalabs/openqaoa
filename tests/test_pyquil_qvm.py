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
from pyquil import Program, quilbase
from pyquil.gates import RX, RY, RZ

from openqaoa.qaoa_parameters import create_qaoa_variational_params, QAOACircuitParams, PauliOp, Hamiltonian
from openqaoa.utilities import X_mixer_hamiltonian
from openqaoa.devices import DevicePyquil
from openqaoa.backends import QAOAPyQuilQPUBackend
from openqaoa.backends.simulators.qaoa_vectorized import QAOAvectorizedBackendSimulator

class TestingQAOACostPyquilQVM(unittest.TestCase):
    
    """This Object tests the QAOA Cost PyQuil QPU object, which is tasked with the
    creation and execution of a QAOA circuit for the selected QPU provider and
    backend. `as_qvm` is set to be True throughout.
    
    For all of these tests, qvm and quilc must be running.
    """
    
    def test_connection(self):
        
        """
        Checks if connection to qvm and quilc is successful.
        TODO : improve test
        """
        
        # Check connection to qvm
        device_pyquil = DevicePyquil(device_name = "2q-qvm", as_qvm=True, execution_timeout = 3, compiler_timeout=3)
        
        # Check connection to quilc compiler
        program = Program().inst(RX(np.pi, 0))
        device_pyquil.quantum_computer.compiler.quil_to_native_quil(program)

        pass
    
    def test_active_reset(self):
        
        """
        Test if active_reset works fine.
        Check for RESET instruction in parametric circuit when active_reset = True / False
        """
        
        device_pyquil = DevicePyquil(device_name = "2q-qvm", as_qvm=True, execution_timeout = 3, compiler_timeout=3)
        cost_hamil = Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,))], [1,2], 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=1)
        
        backend_obj_pyquil = QAOAPyQuilQPUBackend(circuit_params = circuit_params, device = device_pyquil, prepend_state = None, append_state = None, init_hadamard = True, cvar_alpha = 1, n_shots=1, active_reset = True)
        assert 'RESET' in [str(instr) for instr in backend_obj_pyquil.parametric_circuit]
        
        backend_obj_pyquil = QAOAPyQuilQPUBackend(circuit_params = circuit_params, device = device_pyquil, prepend_state = None, append_state = None, init_hadamard = True, cvar_alpha = 1, n_shots=1, active_reset = False)
        assert 'RESET' not in [str(instr) for instr in backend_obj_pyquil.parametric_circuit]
        
    def test_rewiring(self):
        
        """
        Test if rewiring works fine.
        
        """

        device_pyquil = DevicePyquil(device_name = "2q-qvm", as_qvm=True, execution_timeout = 3, compiler_timeout=3)
        cost_hamil = Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,))], [1,2], 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=1)
        
        # Test if error is raised correctly
        self.assertRaises(ValueError, lambda : QAOAPyQuilQPUBackend(circuit_params = circuit_params, device = device_pyquil, prepend_state = None, append_state = None, init_hadamard = True, cvar_alpha = 1, n_shots=1, rewiring = 'illegal string')) 
        
        # Test when rewiring = 'PRAGMA INITIAL_REWIRING "NAIVE"'
        backend_obj_pyquil = QAOAPyQuilQPUBackend(circuit_params = circuit_params, device = device_pyquil, prepend_state = None, append_state = None, init_hadamard = True, cvar_alpha = 1, n_shots=1, rewiring = 'PRAGMA INITIAL_REWIRING "NAIVE"')
        assert 'PRAGMA INITIAL_REWIRING "NAIVE"' in [str(instr) for instr in backend_obj_pyquil.parametric_circuit]
        
        # Test when rewiring = 'PRAGMA INITIAL_REWIRING "PARTIAL"'
        backend_obj_pyquil = QAOAPyQuilQPUBackend(circuit_params = circuit_params, device = device_pyquil, prepend_state = None, append_state = None, init_hadamard = True, cvar_alpha = 1, n_shots=1, rewiring = 'PRAGMA INITIAL_REWIRING "PARTIAL"')
        assert 'PRAGMA INITIAL_REWIRING "PARTIAL"' in [str(instr) for instr in backend_obj_pyquil.parametric_circuit]
        
        
    def test_qaoa_pyquil_expectation(self):
    
        """
        Checks if expectation value agrees with known values. Since angles are selected such that the final state is one of the computational basis states, shots do not matter (there is no statistical variance).
        """

        device_pyquil = DevicePyquil(device_name = "2q-qvm", as_qvm=True, execution_timeout = 3, compiler_timeout=3)

        # Without interaction terms
        cost_hamil = Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,))], [1,1], 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=1)
        variate_params = create_qaoa_variational_params(circuit_params,'standard','ramp')

        args = [np.pi/4, np.pi/4] # beta, gamma
        variate_params.update_from_raw(args)
        
        backend_obj_pyquil = QAOAPyQuilQPUBackend(circuit_params = circuit_params, device = device_pyquil, prepend_state = None, append_state = None, init_hadamard = True, cvar_alpha = 1, n_shots=1)
        backend_obj_pyquil.expectation(variate_params)

        assert np.isclose(backend_obj_pyquil.expectation(variate_params), -1)
        
    def test_qaoa_pyquil_gate_names(self):
    
        """
        Checks if names of gates are correct, and no. of measurement gates match the no. of qubits.
        """

        device_pyquil = DevicePyquil(device_name = "2q-qvm", as_qvm=True, execution_timeout = 3, compiler_timeout=3)

        # Without interaction terms
        cost_hamil = Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,))], [1,1], 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=1)
        backend_obj_pyquil = QAOAPyQuilQPUBackend(circuit_params = circuit_params, device = device_pyquil, prepend_state = None, append_state = None, init_hadamard = True, cvar_alpha = 1, n_shots=1)

        gate_names = [instr.name for instr in backend_obj_pyquil.parametric_circuit if type(instr) == quilbase.Gate]
        assert gate_names == ['RZ','RX','RZ','RX','RZ','RX','RZ','RX', 'RZ', 'RZ', 'RX', 'RX']

        measurement_gate_no = len([instr for instr in backend_obj_pyquil.parametric_circuit if type(instr) == quilbase.Measurement])
        assert measurement_gate_no == 2

        # With interaction terms
        cost_hamil = Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1,1], 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=1)
        backend_obj_pyquil = QAOAPyQuilQPUBackend(circuit_params = circuit_params, device = device_pyquil, prepend_state = None, append_state = None, init_hadamard = True, cvar_alpha = 1, n_shots=1)

        gate_names = [instr.name for instr in backend_obj_pyquil.parametric_circuit if type(instr) == quilbase.Gate]
        assert gate_names == ['RZ', 'RX', 'RZ', 'RX', 'RZ', 'RX', 'RZ', 'RX', 'RZ', 'RZ', 'RZ', 'RZ', 'CPHASE', 'RX', 'RX']

        measurement_gate_no = len([instr for instr in backend_obj_pyquil.parametric_circuit if type(instr) == quilbase.Measurement])
        assert measurement_gate_no == 2

    def test_circuit_init_hadamard(self):

        """
        Checks correctness of circuit for the argument `init_hadamard`.
        """

        device_pyquil = DevicePyquil(device_name = "2q-qvm", as_qvm=True, execution_timeout = 3, compiler_timeout=3)

        # With hadamard
        cost_hamil = Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1,1], 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=1)
        pyquil_backend = QAOAPyQuilQPUBackend(device_pyquil, circuit_params, 
                                              n_shots = 10, prepend_state = None, 
                                              append_state = None, init_hadamard = True, cvar_alpha = 1)

        assert ['RZ', 'RX', 'RZ', 'RX', 'RZ', 'RX', 'RZ', 'RX', 'RZ', 'RZ', 'RZ', 'RZ', 'CPHASE', 'RX', 'RX'] == [instr.name for instr in pyquil_backend.parametric_circuit if type(instr) == quilbase.Gate]

        # Without hadamard
        cost_hamil = Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1,1], 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=1)
        pyquil_backend = QAOAPyQuilQPUBackend(device_pyquil, circuit_params, 
                                              n_shots = 10, prepend_state = None, 
                                              append_state = None, init_hadamard = False, cvar_alpha = 1)

        assert ['RZ', 'RZ', 'RZ', 'RZ', 'CPHASE', 'RX', 'RX'] == [instr.name for instr in pyquil_backend.parametric_circuit if type(instr) == quilbase.Gate]

    def test_circuit_append_state(self):

        """
        Checks correctness of circuit for the argument `append_state`.
        """

        device_pyquil = DevicePyquil(device_name = "2q-qvm", as_qvm=True, execution_timeout = 3, compiler_timeout=3)

        # With append_state
        cost_hamil = Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1,1], 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=1)

        append_circuit = Program().inst(RX(np.pi, 0), RY(np.pi/2, 1), RZ(np.pi, 0))


        pyquil_backend = QAOAPyQuilQPUBackend(device_pyquil, circuit_params, 
                                              n_shots = 10, prepend_state = None, 
                                              append_state = append_circuit, init_hadamard = False, cvar_alpha = 1)

        assert ['RZ', 'RZ', 'RZ', 'RZ', 'CPHASE', 'RX', 'RX', 'RX', 'RY', 'RZ'] == [instr.name for instr in pyquil_backend.parametric_circuit if type(instr) == quilbase.Gate]

    def test_circuit_prepend_state(self):

        """
        Checks correctness of circuit for the argument `prepend_state`.
        """

        device_pyquil = DevicePyquil(device_name = "2q-qvm", as_qvm=True, execution_timeout = 3, compiler_timeout=3)

        # With prepend_state
        cost_hamil = Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1,1], 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=1)

        prepend_circuit = Program().inst(RX(np.pi, 0), RY(np.pi/2, 1), RZ(np.pi, 0))


        pyquil_backend = QAOAPyQuilQPUBackend(device_pyquil, circuit_params, 
                                              n_shots = 10, prepend_state = prepend_circuit, 
                                              append_state = None, init_hadamard = False, cvar_alpha = 1)

        assert ['RX', 'RY', 'RZ', 'RZ', 'RZ', 'RZ', 'RZ', 'CPHASE', 'RX', 'RX'] == [instr.name for instr in pyquil_backend.parametric_circuit if type(instr) == quilbase.Gate]
        
        # Test if error is raised correctly
        prepend_circuit = Program().inst(RX(np.pi, 0), RY(np.pi/2, 1), RZ(np.pi, 2))
        self.assertRaises(AssertionError, lambda : QAOAPyQuilQPUBackend(circuit_params = circuit_params, device = device_pyquil, prepend_state = prepend_circuit, append_state = None, init_hadamard = True, cvar_alpha = 1, n_shots=1)) 
        
    def test_pyquil_vectorized_agreement(self):

        """
        Checks correctness of expectation values with vectorized backend, up to a tolerance of delta = std.dev.
        """

        # Without interaction terms
        device_pyquil = DevicePyquil(device_name = "2q-qvm", as_qvm=True, execution_timeout = 3, compiler_timeout=3)

        cost_hamil = Hamiltonian([PauliOp('Z',(0,)), PauliOp('Z',(1,)), PauliOp('ZZ',(0,1))], [1,1,1], 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=1)
        variate_params = create_qaoa_variational_params(circuit_params,'standard','ramp')
        args = [np.pi/8, np.pi/4] # beta, gamma

        variate_params.update_from_raw(args)
        backend_obj_pyquil = QAOAPyQuilQPUBackend(circuit_params = circuit_params, device = device_pyquil, prepend_state = None, append_state = None, init_hadamard = True, cvar_alpha = 1, n_shots=100)
        expt_pyquil = backend_obj_pyquil.expectation(variate_params)
        
        variate_params.update_from_raw(args)
        backend_obj_vectorized = QAOAvectorizedBackendSimulator(circuit_params, prepend_state=None, append_state=None, init_hadamard=True)
        expt_vec, std_dev_vec = backend_obj_vectorized.expectation_w_uncertainty(variate_params)

        self.assertAlmostEqual(expt_vec, expt_pyquil, delta = std_dev_vec)
    

if __name__ == '__main__':
    unittest.main()
