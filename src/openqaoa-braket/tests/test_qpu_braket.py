import unittest
from unittest.mock import Mock, MagicMock
import numpy as np
import pytest

from braket.circuits import Circuit

from openqaoa.qaoa_components import (
    PauliOp,
    Hamiltonian,
    QAOADescriptor,
    create_qaoa_variational_params,
    QAOAVariationalStandardParams,
)
from openqaoa.utilities import X_mixer_hamiltonian
from openqaoa.problems import NumberPartition
from openqaoa_braket.backends import DeviceAWS, QAOAAWSQPUBackend


class TestingQAOABraketQPUBackend(unittest.TestCase):

    """This Object tests the QAOA Braket QPU Backend objects, which is tasked with the
    creation and execution of a QAOA circuit for the selected QPU provider and
    backend.

    These tests require authentication through the AWS CLI.
    """

    @pytest.mark.braket_api
    def test_circuit_angle_assignment_qpu_backend(self):
        """
        A tests that checks if the circuit created by the AWS Backend
        has the appropriate angles assigned before the circuit is executed.
        Checks the circuit created on the AWS Backend.
        """

        nqubits = 3
        p = 2
        weights = [1, 1, 1]
        gammas = [0, 1 / 8 * np.pi]
        betas = [1 / 2 * np.pi, 3 / 8 * np.pi]
        shots = 100

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        aws_device = DeviceAWS("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

        aws_backend = QAOAAWSQPUBackend(
            qaoa_descriptor, aws_device, shots, None, None, False, 1.0
        )
        qpu_circuit = aws_backend.qaoa_circuit(variate_params)

        # Standard Decomposition
        main_circuit = Circuit()
        main_circuit.cnot(0, 1)
        main_circuit.rz(1, 2 * gammas[0])
        main_circuit.cnot(0, 1)
        main_circuit.cnot(1, 2)
        main_circuit.rz(2, 2 * gammas[0])
        main_circuit.cnot(1, 2)
        main_circuit.cnot(0, 2)
        main_circuit.rz(2, 2 * gammas[0])
        main_circuit.cnot(0, 2)
        main_circuit.rx(0, -2 * betas[0])
        main_circuit.rx(1, -2 * betas[0])
        main_circuit.rx(2, -2 * betas[0])
        main_circuit.cnot(0, 1)
        main_circuit.rz(1, 2 * gammas[1])
        main_circuit.cnot(0, 1)
        main_circuit.cnot(1, 2)
        main_circuit.rz(2, 2 * gammas[1])
        main_circuit.cnot(1, 2)
        main_circuit.cnot(0, 2)
        main_circuit.rz(2, 2 * gammas[1])
        main_circuit.cnot(0, 2)
        main_circuit.rx(0, -2 * betas[1])
        main_circuit.rx(1, -2 * betas[1])
        main_circuit.rx(2, -2 * betas[1])
        main_circuit.probability()

        self.assertEqual(main_circuit, qpu_circuit)

    @pytest.mark.braket_api
    def test_circuit_angle_assignment_qpu_backend_w_hadamard(self):
        """
        Checks for consistent if init_hadamard is set to True.
        """

        nqubits = 3
        p = 2
        weights = [1, 1, 1]
        gammas = [0, 1 / 8 * np.pi]
        betas = [1 / 2 * np.pi, 3 / 8 * np.pi]
        shots = 100

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        aws_device = DeviceAWS("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

        aws_backend = QAOAAWSQPUBackend(
            qaoa_descriptor, aws_device, shots, None, None, True, 1.0
        )
        qpu_circuit = aws_backend.qaoa_circuit(variate_params)

        # Standard Decomposition
        main_circuit = Circuit()
        main_circuit.h(0)
        main_circuit.h(1)
        main_circuit.h(2)
        main_circuit.cnot(0, 1)
        main_circuit.rz(1, 2 * gammas[0])
        main_circuit.cnot(0, 1)
        main_circuit.cnot(1, 2)
        main_circuit.rz(2, 2 * gammas[0])
        main_circuit.cnot(1, 2)
        main_circuit.cnot(0, 2)
        main_circuit.rz(2, 2 * gammas[0])
        main_circuit.cnot(0, 2)
        main_circuit.rx(0, -2 * betas[0])
        main_circuit.rx(1, -2 * betas[0])
        main_circuit.rx(2, -2 * betas[0])
        main_circuit.cnot(0, 1)
        main_circuit.rz(1, 2 * gammas[1])
        main_circuit.cnot(0, 1)
        main_circuit.cnot(1, 2)
        main_circuit.rz(2, 2 * gammas[1])
        main_circuit.cnot(1, 2)
        main_circuit.cnot(0, 2)
        main_circuit.rz(2, 2 * gammas[1])
        main_circuit.cnot(0, 2)
        main_circuit.rx(0, -2 * betas[1])
        main_circuit.rx(1, -2 * betas[1])
        main_circuit.rx(2, -2 * betas[1])
        main_circuit.probability()

        self.assertEqual(main_circuit, qpu_circuit)

    @pytest.mark.braket_api
    def test_prepend_circuit(self):
        """
        Checks if prepended circuit has been prepended correctly.
        """

        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [1 / 8 * np.pi]
        betas = [1 / 8 * np.pi]
        shots = 100

        # Prepended Circuit
        prepend_circuit = Circuit()
        prepend_circuit.x(0)
        prepend_circuit.x(1)
        prepend_circuit.x(2)

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        aws_device = DeviceAWS("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

        aws_backend = QAOAAWSQPUBackend(
            qaoa_descriptor, aws_device, shots, prepend_circuit, None, True, 1.0
        )
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
        main_circuit.rz(1, 2 * gammas[0])
        main_circuit.cnot(0, 1)
        main_circuit.cnot(1, 2)
        main_circuit.rz(2, 2 * gammas[0])
        main_circuit.cnot(1, 2)
        main_circuit.cnot(0, 2)
        main_circuit.rz(2, 2 * gammas[0])
        main_circuit.cnot(0, 2)
        main_circuit.rx(0, -2 * betas[0])
        main_circuit.rx(1, -2 * betas[0])
        main_circuit.rx(2, -2 * betas[0])
        main_circuit.probability()

        self.assertEqual(main_circuit, qpu_circuit)

    @pytest.mark.braket_api
    def test_append_circuit(self):
        """
        Checks if appended circuit is appropriately appended to the back of the
        QAOA Circuit.
        """

        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [1 / 8 * np.pi]
        betas = [1 / 8 * np.pi]
        shots = 100

        # Appended Circuit
        append_circuit = Circuit()
        append_circuit.x(0)
        append_circuit.x(1)
        append_circuit.x(2)

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        aws_device = DeviceAWS("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

        aws_backend = QAOAAWSQPUBackend(
            qaoa_descriptor, aws_device, shots, None, append_circuit, True, 1.0
        )
        qpu_circuit = aws_backend.qaoa_circuit(variate_params)

        # Standard Decomposition
        main_circuit = Circuit()
        main_circuit.h(0)
        main_circuit.h(1)
        main_circuit.h(2)
        main_circuit.cnot(0, 1)
        main_circuit.rz(1, 2 * gammas[0])
        main_circuit.cnot(0, 1)
        main_circuit.cnot(1, 2)
        main_circuit.rz(2, 2 * gammas[0])
        main_circuit.cnot(1, 2)
        main_circuit.cnot(0, 2)
        main_circuit.rz(2, 2 * gammas[0])
        main_circuit.cnot(0, 2)
        main_circuit.rx(0, -2 * betas[0])
        main_circuit.rx(1, -2 * betas[0])
        main_circuit.rx(2, -2 * betas[0])
        main_circuit.x(0)
        main_circuit.x(1)
        main_circuit.x(2)
        main_circuit.probability()

        self.assertEqual(main_circuit, qpu_circuit)

    @pytest.mark.braket_api
    def test_prepend_exception(self):
        """
        Test that the error catching for a prepend ciruit larger than the problem
        circuit is invalid
        """

        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [1 / 8 * np.pi]
        betas = [1 / 8 * np.pi]
        shots = 100

        # Prepended Circuit
        prepend_circuit = Circuit()
        prepend_circuit.x(0)
        prepend_circuit.x(1)
        prepend_circuit.x(2)
        prepend_circuit.x(3)

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)

        # Try to create the variate params
        _ = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        aws_device = DeviceAWS("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

        try:
            # Try to create the AWS backend
            _ = QAOAAWSQPUBackend(
                qaoa_descriptor, aws_device, shots, prepend_circuit, None, True, 1.0
            )
        except Exception as e:
            self.assertEqual(
                str(e), "Cannot attach a bigger circuit to the QAOA routine"
            )

    @pytest.mark.braket_api
    def test_exceptions_in_init(self):
        """
        Testing the Exceptions in the init function of the QAOAAWSQPUBackend
        """

        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [1 / 8 * np.pi]
        betas = [1 / 8 * np.pi]
        shots = 100

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)

        # Try instantiating the variate params
        _ = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        # If the user's aws credentials is not correct.
        mock_device = Mock()
        mock_device.configure_mock(
            **{
                "check_connection.return_value": False,
                "provider_connected.return_value": False,
                "qpu_connected.return_value": None,
                "n_qubits": 3,
            }
        )

        try:
            QAOAAWSQPUBackend(
                qaoa_descriptor, mock_device, shots, None, None, True, 1.0
            )
        except Exception as e:
            self.assertEqual(str(e), "Error connecting to AWS.")

        # Wrong arn string name
        aws_device = DeviceAWS(
            "arn:aws:braket:::device/quantum-simulator/amazon/invalid_backend_arn"
        )

        try:
            QAOAAWSQPUBackend(qaoa_descriptor, aws_device, shots, None, None, True, 1.0)
        except Exception as e:
            self.assertEqual(
                str(e),
                "Connection to AWS was made. Error connecting to the specified backend.",
            )

        # No device specified
        aws_device = DeviceAWS("")

        try:
            QAOAAWSQPUBackend(qaoa_descriptor, aws_device, shots, None, None, True, 1.0)
        except Exception as e:
            self.assertEqual(
                str(e), "Connection to AWS was made. A device name was not specified."
            )

        # Correct device arn (Errorless)
        aws_device = DeviceAWS("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

        QAOAAWSQPUBackend(qaoa_descriptor, aws_device, shots, None, None, True, 1.0)

    @pytest.mark.braket_api
    def test_remote_qubit_overflow(self):
        """
        If the user creates a circuit that is larger than the maximum circuit size
        that is supported by the QPU. An Exception should be raised with the
        appropriate error message alerting the user to the error.
        """

        shots = 100

        set_of_numbers = np.random.randint(1, 10, 100).tolist()
        qubo = NumberPartition(set_of_numbers).qubo

        mixer_hamil = X_mixer_hamiltonian(n_qubits=6)
        qaoa_descriptor = QAOADescriptor(qubo.hamiltonian, mixer_hamil, p=1)
        variate_params = create_qaoa_variational_params(
            qaoa_descriptor, "standard", "rand"
        )

        aws_device = DeviceAWS("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

        try:
            braket_backend = QAOAAWSQPUBackend(
                qaoa_descriptor, aws_device, shots, None, None, True, 1.0
            )
            braket_backend.expectation(variate_params)
        except Exception as e:
            self.assertEqual(
                str(e),
                "There are lesser qubits on the device than the number of qubits required for the circuit.",
            )

    @pytest.mark.sim
    def test_remote_integration_qpu_run(self):
        """
        Run a toy example in manual mode to make sure everything works as
        expected for a remote backend
        """

        shots = 100

        set_of_numbers = np.random.randint(1, 10, 10).tolist()
        qubo = NumberPartition(set_of_numbers).qubo

        mixer_hamil = X_mixer_hamiltonian(n_qubits=6)
        qaoa_descriptor = QAOADescriptor(qubo.hamiltonian, mixer_hamil, p=1)
        variate_params = create_qaoa_variational_params(
            qaoa_descriptor, "standard", "rand"
        )

        aws_device = DeviceAWS("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

        braket_backend = QAOAAWSQPUBackend(
            qaoa_descriptor, aws_device, shots, None, None, True, 1.0
        )
        braket_backend.expectation(variate_params)

    # Test to be improved, to be run when OQ starts supporting LocalSimulators from braket
    # @pytest.mark.sim
    # def test_braket_measurement_instructions(self):
    #     """
    #     When using the routing functionality within OpenQAOA, the
    #     final order of qubits is changed. This function tests for
    #     the functionality in the braket backend to reset the order
    #     of qubits in the measurement instructions.

    #     The goal is compare the output of a routed circuit to that of
    #     a non-routed circuit on the SV1 simulator to test the correctness
    #     of the output.
    #     """
    #     routing_function = MagicMock(
    #         return_value=(
    #             [
    #                 [5, 7],
    #                 [0, 4],
    #                 [2, 6],
    #                 [1, 6],
    #                 [3, 7],
    #                 [1, 3],
    #                 [0, 5],
    #                 [2, 4],
    #                 [1, 6],
    #                 [1, 3],
    #                 [2, 6],
    #                 [2, 4],
    #                 [0, 4],
    #                 [2, 6],
    #                 [3, 7],
    #                 [5, 7],
    #                 [1, 3],
    #                 [0, 5],
    #                 [5, 7],
    #                 [0, 4],
    #                 [0, 4],
    #                 [2, 4],
    #                 [5, 7],
    #                 [0, 5],
    #                 [3, 7],
    #                 [1, 6],
    #                 [1, 3],
    #                 [2, 6],
    #                 [2, 4],
    #                 [2, 6],
    #                 [3, 7],
    #                 [1, 3],
    #                 [1, 6],
    #                 [1, 3],
    #                 [2, 6],
    #                 [0, 5],
    #                 [5, 7],
    #                 [0, 4],
    #                 [5, 7],
    #                 [0, 4],
    #                 [0, 5],
    #             ],
    #             [
    #                 False,
    #                 False,
    #                 False,
    #                 False,
    #                 False,
    #                 False,
    #                 False,
    #                 False,
    #                 True,
    #                 False,
    #                 False,
    #                 True,
    #                 False,
    #                 False,
    #                 True,
    #                 False,
    #                 False,
    #                 True,
    #                 False,
    #                 False,
    #                 True,
    #                 False,
    #                 True,
    #                 False,
    #                 False,
    #                 True,
    #                 False,
    #                 False,
    #                 True,
    #                 False,
    #                 True,
    #                 False,
    #                 True,
    #                 False,
    #                 False,
    #                 True,
    #                 False,
    #                 False,
    #                 True,
    #                 True,
    #                 False,
    #             ],
    #             {1: 0, 5: 1, 3: 2, 6: 3, 2: 4, 0: 5, 4: 6, 7: 7},
    #             [3, 6, 7, 4, 0, 2, 1, 5],
    #         )
    #     )
    #     n_shots = 100
    #     # generate a random set of 8 integers
    #     set_of_numbers = [9, 4, 4, 7, 3, 1, 1, 9]
    #     qubo = NumberPartition(set_of_numbers).qubo

    #     mixer_hamil = X_mixer_hamiltonian(n_qubits=8)
    #     aws_device = DeviceAWS("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

    #     qaoa_descriptor_routed = QAOADescriptor(
    #         qubo.hamiltonian, mixer_hamil, p=1, routing_function=routing_function
    #     )
    #     variate_params = create_qaoa_variational_params(
    #         qaoa_descriptor_routed, "standard", "rand"
    #     )
    #     braket_backend_routed = QAOAAWSQPUBackend(
    #         qaoa_descriptor_routed, aws_device, n_shots, None, None, True, 1.0
    #     )

    #     qaoa_descriptor = QAOADescriptor(qubo.hamiltonian, mixer_hamil, p=1)
    #     braket_backend = QAOAAWSQPUBackend(
    #         qaoa_descriptor, aws_device, n_shots, None, None, True, 1.0
    #     )

    #     list_angles = [[1.0, 0.0]]#, [0.8, 0.2], [0.6, 0.4], [0.4, 0.6]]
    #     for angles in list_angles:
    #         variate_params.update_from_raw(angles)
    #         # Run the circuit without routing
    #         non_routed_exp = braket_backend.expectation(variate_params)
    #         # Run the circuit with routing
    #         routed_exp = braket_backend_routed.expectation(variate_params)
    #         self.assertAlmostEqual(non_routed_exp, routed_exp, places=2)


if __name__ == "__main__":
    unittest.main()
