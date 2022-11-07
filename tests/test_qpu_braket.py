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
import json
import numpy as np
from braket.circuits import Circuit
import pytest

from openqaoa.qaoa_parameters import PauliOp, Hamiltonian, QAOACircuitParams
from openqaoa.qaoa_parameters.standardparams import QAOAVariationalStandardParams
from openqaoa.devices import DeviceAWS
from openqaoa.backends.qpus.qaoa_braket_qpu import QAOAAWSQPUBackend
from openqaoa.utilities import X_mixer_hamiltonian


class TestingQAOABraketQPUBackend(unittest.TestCase):

    """This Object tests the QAOA Braket QPU Backend objects, which is tasked with the
    creation and execution of a QAOA circuit for the selected QPU provider and
    backend.

    For all of these tests, credentials.json MUST be filled with the appropriate
    credentials. If unsure about to correctness of the current input credentials
    , please run test_qpu_auth.py. 
    """


    @pytest.mark.qpu
    def setUp(self):
        
        try:
            opened_f = open('./tests/credentials.json', 'r')
        except FileNotFoundError:
            opened_f = open('credentials.json', 'r')
                
        with opened_f as f:
            json_obj = json.load(f)['AWS']
            self.AWS_ACCESS_KEY_ID = json_obj['AWS_ACCESS_KEY_ID']
            self.AWS_SECRET_ACCESS_KEY = json_obj['AWS_SECRET_ACCESS_KEY']
            self.AWS_REGION = json_obj['AWS_REGION']
            self.S3_BUCKET_NAME = json_obj['S3_BUCKET_NAME']

        if self.AWS_ACCESS_KEY_ID == "None":
            raise ValueError(
                "Please provide an appropriate AWS ACCESS KEY ID in crendentials.json.")
        elif self.AWS_SECRET_ACCESS_KEY == "None":
            raise ValueError(
                "Please provide an appropriate AWS SECRET ACCESS KEY name in crendentials.json.")
        elif self.AWS_REGION == "None":
            raise ValueError(
                "Please provide an appropriate AWS REGION name in crendentials.json.")
        elif self.S3_BUCKET_NAME == "None":
            raise ValueError(
                "Please provide an appropriate S3 BUCKET NAME name in crendentials.json.")


    @pytest.mark.qpu
    def test_circuit_angle_assignment_qpu_backend(self):
        """
        A tests that checks if the circuit created by the AWS Backend
        has the appropriate angles assigned before the circuit is executed.
        Checks the circuit created on the AWS Backend.
        """

        nqubits = 3
        p = 2
        weights = [1, 1, 1]
        gammas = [0, 1/8*np.pi]
        betas = [1/2*np.pi, 3/8*np.pi]
        shots = 10000

        cost_hamil = Hamiltonian([PauliOp('ZZ', (0, 1)), PauliOp('ZZ', (1, 2)),
                                  PauliOp('ZZ', (0, 2))], weights, 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(circuit_params,
                                                       betas, gammas)

        aws_device = DeviceAWS("SV1", self.AWS_ACCESS_KEY_ID, 
                                  self.AWS_SECRET_ACCESS_KEY, self.AWS_REGION, 
                                  self.S3_BUCKET_NAME)

        aws_backend = QAOAAWSQPUBackend(circuit_params, aws_device, 
                                        shots, None, None, False, 1.)
        qpu_circuit = aws_backend.qaoa_circuit(variate_params)

        # Standard Decomposition
        main_circuit = Circuit()
        main_circuit.cnot(0, 1)
        main_circuit.rz(1, 2*gammas[0])
        main_circuit.cnot(0, 1)
        main_circuit.cnot(1, 2)
        main_circuit.rz(2, 2*gammas[0])
        main_circuit.cnot(1, 2)
        main_circuit.cnot(0, 2)
        main_circuit.rz(2, 2*gammas[0])
        main_circuit.cnot(0, 2)
        main_circuit.rx(0, -2*betas[0])
        main_circuit.rx(1, -2*betas[0])
        main_circuit.rx(2, -2*betas[0])
        main_circuit.cnot(0, 1)
        main_circuit.rz(1, 2*gammas[1])
        main_circuit.cnot(0, 1)
        main_circuit.cnot(1, 2)
        main_circuit.rz(2, 2*gammas[1])
        main_circuit.cnot(1, 2)
        main_circuit.cnot(0, 2)
        main_circuit.rz(2, 2*gammas[1])
        main_circuit.cnot(0, 2)
        main_circuit.rx(0, -2*betas[1])
        main_circuit.rx(1, -2*betas[1])
        main_circuit.rx(2, -2*betas[1])
        main_circuit.probability()

        self.assertEqual(main_circuit, qpu_circuit)


    @pytest.mark.qpu
    def test_circuit_angle_assignment_qpu_backend_w_hadamard(self):
        """
        Checks for consistent if init_hadamard is set to True.
        """

        nqubits = 3
        p = 2
        weights = [1, 1, 1]
        gammas = [0, 1/8*np.pi]
        betas = [1/2*np.pi, 3/8*np.pi]
        shots = 10000

        cost_hamil = Hamiltonian([PauliOp('ZZ', (0, 1)), PauliOp('ZZ', (1, 2)),
                                  PauliOp('ZZ', (0, 2))], weights, 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(circuit_params,
                                                       betas, gammas)

        aws_device = DeviceAWS("SV1", self.AWS_ACCESS_KEY_ID, 
                                  self.AWS_SECRET_ACCESS_KEY, self.AWS_REGION, 
                                  self.S3_BUCKET_NAME)

        aws_backend = QAOAAWSQPUBackend(circuit_params, aws_device, 
                                        shots, None, None, True, 1.)
        qpu_circuit = aws_backend.qaoa_circuit(variate_params)

        # Standard Decomposition
        main_circuit = Circuit()
        main_circuit.h(0)
        main_circuit.h(1)
        main_circuit.h(2)
        main_circuit.cnot(0, 1)
        main_circuit.rz(1, 2*gammas[0])
        main_circuit.cnot(0, 1)
        main_circuit.cnot(1, 2)
        main_circuit.rz(2, 2*gammas[0])
        main_circuit.cnot(1, 2)
        main_circuit.cnot(0, 2)
        main_circuit.rz(2, 2*gammas[0])
        main_circuit.cnot(0, 2)
        main_circuit.rx(0, -2*betas[0])
        main_circuit.rx(1, -2*betas[0])
        main_circuit.rx(2, -2*betas[0])
        main_circuit.cnot(0, 1)
        main_circuit.rz(1, 2*gammas[1])
        main_circuit.cnot(0, 1)
        main_circuit.cnot(1, 2)
        main_circuit.rz(2, 2*gammas[1])
        main_circuit.cnot(1, 2)
        main_circuit.cnot(0, 2)
        main_circuit.rz(2, 2*gammas[1])
        main_circuit.cnot(0, 2)
        main_circuit.rx(0, -2*betas[1])
        main_circuit.rx(1, -2*betas[1])
        main_circuit.rx(2, -2*betas[1])
        main_circuit.probability()

        self.assertEqual(main_circuit, qpu_circuit)


    @pytest.mark.qpu
    def test_prepend_circuit(self):
        """
        Checks if prepended circuit has been prepended correctly.
        """

        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [1/8*np.pi]
        betas = [1/8*np.pi]
        shots = 10000

        # Prepended Circuit
        prepend_circuit = Circuit()
        prepend_circuit.x(0)
        prepend_circuit.x(1)
        prepend_circuit.x(2)

        cost_hamil = Hamiltonian([PauliOp('ZZ', (0, 1)), PauliOp('ZZ', (1, 2)),
                                  PauliOp('ZZ', (0, 2))], weights, 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(circuit_params,
                                                       betas, gammas)

        aws_device = DeviceAWS("SV1", self.AWS_ACCESS_KEY_ID, 
                                  self.AWS_SECRET_ACCESS_KEY, self.AWS_REGION, 
                                  self.S3_BUCKET_NAME)

        aws_backend = QAOAAWSQPUBackend(circuit_params, aws_device, 
                                        shots, prepend_circuit, None, True, 1.)
        qpu_circuit = aws_backend.qaoa_circuit(variate_params)

        # Standard Decomposition
        main_circuit = Circuit()
        main_circuit.x(0)
        main_circuit.x(1)
        main_circuit.x(2)
        main_circuit.h(0)
        main_circuit.h(1)
        main_circuit.h(2)
        main_circuit.cnot(0, 1)
        main_circuit.rz(1, 2*gammas[0])
        main_circuit.cnot(0, 1)
        main_circuit.cnot(1, 2)
        main_circuit.rz(2, 2*gammas[0])
        main_circuit.cnot(1, 2)
        main_circuit.cnot(0, 2)
        main_circuit.rz(2, 2*gammas[0])
        main_circuit.cnot(0, 2)
        main_circuit.rx(0, -2*betas[0])
        main_circuit.rx(1, -2*betas[0])
        main_circuit.rx(2, -2*betas[0])
        main_circuit.probability()

        self.assertEqual(main_circuit, qpu_circuit)


    @pytest.mark.qpu
    def test_append_circuit(self):
        """
        Checks if appended circuit is appropriately appended to the back of the
        QAOA Circuit.
        """

        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [1/8*np.pi]
        betas = [1/8*np.pi]
        shots = 10000

        # Appended Circuit
        append_circuit = Circuit()
        append_circuit.x(0)
        append_circuit.x(1)
        append_circuit.x(2)

        cost_hamil = Hamiltonian([PauliOp('ZZ', (0, 1)), PauliOp('ZZ', (1, 2)),
                                  PauliOp('ZZ', (0, 2))], weights, 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(circuit_params,
                                                       betas, gammas)

        aws_device = DeviceAWS("SV1", self.AWS_ACCESS_KEY_ID, 
                                  self.AWS_SECRET_ACCESS_KEY, self.AWS_REGION, 
                                  self.S3_BUCKET_NAME)

        aws_backend = QAOAAWSQPUBackend(circuit_params, aws_device, 
                                        shots, None, append_circuit, True, 1.)
        qpu_circuit = aws_backend.qaoa_circuit(variate_params)

        # Standard Decomposition
        main_circuit = Circuit()
        main_circuit.h(0)
        main_circuit.h(1)
        main_circuit.h(2)
        main_circuit.cnot(0, 1)
        main_circuit.rz(1, 2*gammas[0])
        main_circuit.cnot(0, 1)
        main_circuit.cnot(1, 2)
        main_circuit.rz(2, 2*gammas[0])
        main_circuit.cnot(1, 2)
        main_circuit.cnot(0, 2)
        main_circuit.rz(2, 2*gammas[0])
        main_circuit.cnot(0, 2)
        main_circuit.rx(0, -2*betas[0])
        main_circuit.rx(1, -2*betas[0])
        main_circuit.rx(2, -2*betas[0])
        main_circuit.x(0)
        main_circuit.x(1)
        main_circuit.x(2)
        main_circuit.probability()

        self.assertEqual(main_circuit, qpu_circuit)


    @pytest.mark.qpu
    def test_exceptions_in_init(self):
        
        """
        Testing the Exceptions in the init function of the QAOAAWSQPUBackend
        """
        
        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [1/8*np.pi]
        betas = [1/8*np.pi]
        shots = 10000

        cost_hamil = Hamiltonian([PauliOp('ZZ', (0, 1)), PauliOp('ZZ', (1, 2)),
                                  PauliOp('ZZ', (0, 2))], weights, 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(circuit_params,
                                                       betas, gammas)

        aws_device = DeviceAWS('', '', '', '', '')
        
        try:
            QAOAAWSQPUBackend(circuit_params, aws_device, 
                                 shots, None, None, True, 1.)
        except Exception as e:
            self.assertEqual(str(e), 'Error connecting to AWS.')
        
        
        self.assertRaises(Exception, QAOAAWSQPUBackend, (circuit_params, 
                                                            aws_device, 
                                                            shots, None, None, 
                                                            True, 1.))
        
        aws_device = DeviceAWS("SV1", self.AWS_ACCESS_KEY_ID, 
                               self.AWS_SECRET_ACCESS_KEY, self.AWS_REGION, 
                               self.S3_BUCKET_NAME)
        
        try:
            QAOAAWSQPUBackend(circuit_params, aws_device, 
                                 shots, None, None, True, 1.)
        except Exception as e:
            self.assertEqual(str(e), 'Connection to AWS was made. Error connecting to the specified backend.')
        
        
        self.assertRaises(Exception, QAOAAWSQPUBackend, (circuit_params, 
                                                            aws_device, 
                                                            shots, None, None, 
                                                            True, 1.))


    @pytest.mark.qpu
    def test_correct_device_creation(self):
        
        device_map = {'us-east-1': 
                      {'IonQ Device': 'arn:aws:braket:::device/qpu/ionq/ionQdevice'}, 
                      'eu-west-2': 
                      {'Lucy': 'arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy'}, 
                      'us-west-1': 
                      {'SV1': 'arn:aws:braket:::device/quantum-simulator/amazon/sv1',
                       'Aspen-M-2': 'arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-2'}
                     }
        
        for each_region, devices_dict in device_map.items():
        
            for each_key, each_value in devices_dict.items():

                aws_device = DeviceAWS(each_key, self.AWS_ACCESS_KEY_ID, 
                                       self.AWS_SECRET_ACCESS_KEY, each_region, 
                                       self.S3_BUCKET_NAME)

                aws_device.check_connection()
                
                print(each_key, aws_device.device_arn, '/n')

                self.assertEqual(aws_device.device_arn, each_value)
     
        
#     def test_remote_integration_qpu_run(self):
#         """
#         Test Actual QPU Workflow. Checks if the expectation value is returned
#         after the circuit run.
#         """

#         nqubits = 3
#         p = 1
#         weights = [1, 1, 1]
#         gammas = [[1/8*np.pi]]
#         betas = [[1/8*np.pi]]
#         shots = 10000

#         cost_hamil = Hamiltonian([PauliOp('ZZ', (0, 1)), PauliOp('ZZ', (1, 2)),
#                                   PauliOp('ZZ', (0, 2))], weights, 1)
#         mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
#         circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=p)
#         variate_params = QAOAVariationalStandardParams(circuit_params,
#                                                        betas,
#                                                        gammas)
#         aws_device = DeviceAWS("SV1", self.AWS_ACCESS_KEY_ID, 
#                                self.AWS_SECRET_ACCESS_KEY, self.AWS_REGION, 
#                                self.S3_BUCKET_NAME)

#         aws_backend = QAOAAWSQPUBackend(circuit_params, aws_device, 
#                                            shots, None, None, False)
#         aws_expectation = aws_backend.expectation(variate_params)
        
#         self.assertEqual(type(aws_expectation.item()), float)


if __name__ == '__main__':
    unittest.main()
