import time
import unittest
from unittest.mock import Mock
import json
import numpy as np
import pytest
import subprocess
from collections import OrderedDict

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.tools.monitor import job_monitor
from qiskit_ibm_provider.job.exceptions import IBMJobError
from qiskit.converters import circuit_to_dag, dag_to_circuit

from openqaoa.qaoa_components import (
    create_qaoa_variational_params,
    PauliOp,
    Hamiltonian,
    QAOADescriptor,
    QAOAVariationalStandardParams,
)
from openqaoa_qiskit.backends import (
    DeviceQiskit,
    QAOAQiskitQPUBackend,
    QAOAQiskitBackendStatevecSimulator,
)
from openqaoa.utilities import X_mixer_hamiltonian
from openqaoa.problems import NumberPartition
from openqaoa import QAOA, create_device


def remove_idle_qwires(circ):
    """Function that removes qubits that are not actred upon in a QuantumCircuit

    Args:
        circ (QuantumCircuit): quantum circuit that has unused qubits

    Returns:
        QuantumCircuit: quantum circuit without unused qubits
    """
    dag = circuit_to_dag(circ)

    idle_wires = list(dag.idle_wires())
    for w in idle_wires:
        dag._remove_idle_wire(w)
        dag.qubits.remove(w)

    dag.qregs = OrderedDict()

    return dag_to_circuit(dag)


class TestingQAOAQiskitQPUBackend(unittest.TestCase):

    """This Object tests the QAOA Qiskit QPU Backend objects, which is tasked with the
    creation and execution of a QAOA circuit for the selected QPU provider and
    backend.

    IBMQ Account has to be saved locally to run these tests.
    """

    @pytest.mark.api
    def setUp(self):
        """
        Define the credentials
        """
        self.HUB = "ibm-q"
        self.GROUP = "open"
        self.PROJECT = "main"

    @pytest.mark.api
    def test_circuit_angle_assignment_qpu_backend(self):
        """
        A tests that checks if the circuit created by the Qiskit Backend
        has the appropriate angles assigned before the circuit is executed.
        Checks the circuit created on both IBM QPU Backends.
        """

        nqubits = 3
        p = 2
        weights = [1, 1, 1]
        gammas = [0, 1 / 8 * np.pi]
        betas = [1 / 2 * np.pi, 3 / 8 * np.pi]
        shots = 10000

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        qiskit_device = DeviceQiskit(
            "ibmq_qasm_simulator", self.HUB, self.GROUP, self.PROJECT
        )

        qiskit_backend = QAOAQiskitQPUBackend(
            qaoa_descriptor, qiskit_device, shots, None, None, False
        )
        qpu_circuit = qiskit_backend.qaoa_circuit(variate_params)

        # Standard Decomposition
        main_circuit = QuantumCircuit(3)
        main_circuit.cx(0, 1)
        main_circuit.rz(2 * gammas[0], 1)
        main_circuit.cx(0, 1)
        main_circuit.cx(1, 2)
        main_circuit.rz(2 * gammas[0], 2)
        main_circuit.cx(1, 2)
        main_circuit.cx(0, 2)
        main_circuit.rz(2 * gammas[0], 2)
        main_circuit.cx(0, 2)
        main_circuit.rx(-2 * betas[0], 0)
        main_circuit.rx(-2 * betas[0], 1)
        main_circuit.rx(-2 * betas[0], 2)
        main_circuit.cx(0, 1)
        main_circuit.rz(2 * gammas[1], 1)
        main_circuit.cx(0, 1)
        main_circuit.cx(1, 2)
        main_circuit.rz(2 * gammas[1], 2)
        main_circuit.cx(1, 2)
        main_circuit.cx(0, 2)
        main_circuit.rz(2 * gammas[1], 2)
        main_circuit.cx(0, 2)
        main_circuit.rx(-2 * betas[1], 0)
        main_circuit.rx(-2 * betas[1], 1)
        main_circuit.rx(-2 * betas[1], 2)

        qpu_circuit.remove_final_measurements(inplace=True)
        qpu_circuit = remove_idle_qwires(qpu_circuit)
        qpu_circuit_operator = Operator(qpu_circuit)
        main_circuit_operator = Operator(main_circuit)

        assert qpu_circuit_operator.equiv(main_circuit_operator)

    @pytest.mark.api
    def test_circuit_angle_assignment_qpu_backend_w_hadamard(self):
        """
        Checks for consistent if init_hadamard is set to True.
        """

        nqubits = 3
        p = 2
        weights = [1, 1, 1]
        gammas = [0, 1 / 8 * np.pi]
        betas = [1 / 2 * np.pi, 3 / 8 * np.pi]
        shots = 10000

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        qiskit_device = DeviceQiskit(
            "ibmq_qasm_simulator", self.HUB, self.GROUP, self.PROJECT
        )

        qiskit_backend = QAOAQiskitQPUBackend(
            qaoa_descriptor, qiskit_device, shots, None, None, True
        )
        qpu_circuit = qiskit_backend.qaoa_circuit(variate_params)

        # Standard Decomposition
        main_circuit = QuantumCircuit(3)
        main_circuit.h([0, 1, 2])
        main_circuit.cx(0, 1)
        main_circuit.rz(2 * gammas[0], 1)
        main_circuit.cx(0, 1)
        main_circuit.cx(1, 2)
        main_circuit.rz(2 * gammas[0], 2)
        main_circuit.cx(1, 2)
        main_circuit.cx(0, 2)
        main_circuit.rz(2 * gammas[0], 2)
        main_circuit.cx(0, 2)
        main_circuit.rx(-2 * betas[0], 0)
        main_circuit.rx(-2 * betas[0], 1)
        main_circuit.rx(-2 * betas[0], 2)
        main_circuit.cx(0, 1)
        main_circuit.rz(2 * gammas[1], 1)
        main_circuit.cx(0, 1)
        main_circuit.cx(1, 2)
        main_circuit.rz(2 * gammas[1], 2)
        main_circuit.cx(1, 2)
        main_circuit.cx(0, 2)
        main_circuit.rz(2 * gammas[1], 2)
        main_circuit.cx(0, 2)
        main_circuit.rx(-2 * betas[1], 0)
        main_circuit.rx(-2 * betas[1], 1)
        main_circuit.rx(-2 * betas[1], 2)

        qpu_circuit.remove_final_measurements(inplace=True)
        qpu_circuit = remove_idle_qwires(qpu_circuit)
        qpu_circuit_operator = Operator(qpu_circuit)
        main_circuit_operator = Operator(main_circuit)

        assert qpu_circuit_operator.equiv(main_circuit_operator)

    @pytest.mark.api
    def test_prepend_circuit(self):
        """
        Checks if prepended circuit has been prepended correctly.
        """

        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [1 / 8 * np.pi]
        betas = [1 / 8 * np.pi]
        shots = 10000

        # Prepended Circuit
        prepend_circuit = QuantumCircuit(3)
        prepend_circuit.x([0, 1, 2])

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        qiskit_device = DeviceQiskit(
            "ibmq_qasm_simulator", self.HUB, self.GROUP, self.PROJECT
        )

        qiskit_backend = QAOAQiskitQPUBackend(
            qaoa_descriptor, qiskit_device, shots, prepend_circuit, None, True
        )
        qpu_circuit = qiskit_backend.qaoa_circuit(variate_params)

        # Standard Decomposition
        main_circuit = QuantumCircuit(3)
        main_circuit.x([0, 1, 2])
        main_circuit.h([0, 1, 2])
        main_circuit.cx(0, 1)
        main_circuit.rz(2 * gammas[0], 1)
        main_circuit.cx(0, 1)
        main_circuit.cx(1, 2)
        main_circuit.rz(2 * gammas[0], 2)
        main_circuit.cx(1, 2)
        main_circuit.cx(0, 2)
        main_circuit.rz(2 * gammas[0], 2)
        main_circuit.cx(0, 2)
        main_circuit.rx(-2 * betas[0], 0)
        main_circuit.rx(-2 * betas[0], 1)
        main_circuit.rx(-2 * betas[0], 2)

        qpu_circuit.remove_final_measurements(inplace=True)
        qpu_circuit = remove_idle_qwires(qpu_circuit)
        qpu_circuit_operator = Operator(qpu_circuit)
        main_circuit_operator = Operator(main_circuit)

        assert qpu_circuit_operator.equiv(main_circuit_operator)

    @pytest.mark.api
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
        shots = 10000

        # Appended Circuit
        append_circuit = QuantumCircuit(3)
        append_circuit.x([0, 1, 2])

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        qiskit_device = DeviceQiskit(
            "ibmq_qasm_simulator", self.HUB, self.GROUP, self.PROJECT
        )

        qiskit_backend = QAOAQiskitQPUBackend(
            qaoa_descriptor, qiskit_device, shots, None, append_circuit, True
        )
        qpu_circuit = qiskit_backend.qaoa_circuit(variate_params)

        # Standard Decomposition
        main_circuit = QuantumCircuit(3)
        main_circuit.h([0, 1, 2])
        main_circuit.cx(0, 1)
        main_circuit.rz(2 * gammas[0], 1)
        main_circuit.cx(0, 1)
        main_circuit.cx(1, 2)
        main_circuit.rz(2 * gammas[0], 2)
        main_circuit.cx(1, 2)
        main_circuit.cx(0, 2)
        main_circuit.rz(2 * gammas[0], 2)
        main_circuit.cx(0, 2)
        main_circuit.rx(-2 * betas[0], 0)
        main_circuit.rx(-2 * betas[0], 1)
        main_circuit.rx(-2 * betas[0], 2)
        main_circuit.x([0, 1, 2])

        qpu_circuit.remove_final_measurements(inplace=True)
        qpu_circuit = remove_idle_qwires(qpu_circuit)
        qpu_circuit_operator = Operator(qpu_circuit)
        main_circuit_operator = Operator(main_circuit)

        assert qpu_circuit_operator.equiv(main_circuit_operator)

    @pytest.mark.api
    def test_expectations_in_init(self):
        """
        Testing the Exceptions in the init function of the QiskitQPUShotBasedBackend
        """

        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [1 / 8 * np.pi]
        betas = [1 / 8 * np.pi]
        shots = 10000

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)

        # Check the creation of the varitional parms
        _ = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        # We mock the potential Exception that could occur in the Device class
        qiskit_device = DeviceQiskit("", "", "", "")
        qiskit_device._check_provider_connection = Mock(return_value=False)

        try:
            QAOAQiskitQPUBackend(
                qaoa_descriptor, qiskit_device, shots, None, None, True
            )
        except Exception as e:
            self.assertEqual(str(e), "Error connecting to IBMQ.")

        self.assertRaises(
            Exception,
            QAOAQiskitQPUBackend,
            (qaoa_descriptor, qiskit_device, shots, None, None, True),
        )

        qiskit_device = DeviceQiskit(
            device_name="",
            hub=self.HUB,
            group=self.GROUP,
            project=self.PROJECT,
        )

        try:
            QAOAQiskitQPUBackend(
                qaoa_descriptor, qiskit_device, shots, None, None, True
            )
        except Exception as e:
            self.assertEqual(
                str(e),
                "Connection to IBMQ was made. Error connecting to the specified backend.",
            )

        self.assertRaises(
            Exception,
            QAOAQiskitQPUBackend,
            qaoa_descriptor,
            qiskit_device,
            shots,
            None,
            None,
            True,
        )

    @pytest.mark.sim
    def test_remote_integration_sim_run(self):
        """
        Checks if Remote IBM QASM Simulator is similar/close to Local IBM
        Statevector Simulator.
        This test also serves as an integration test for the IBMQPU Backend.

        This test takes a long time to complete.
        """

        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [[0], [1 / 8 * np.pi], [0], [1 / 8 * np.pi]]
        betas = [[0], [0], [1 / 8 * np.pi], [1 / 8 * np.pi]]
        shots = 10000

        for i in range(4):
            cost_hamil = Hamiltonian(
                [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
                weights,
                1,
            )
            mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
            variate_params = QAOAVariationalStandardParams(
                qaoa_descriptor, betas[i], gammas[i]
            )

            qiskit_device = DeviceQiskit(
                "ibmq_qasm_simulator", self.HUB, self.GROUP, self.PROJECT
            )

            qiskit_backend = QAOAQiskitQPUBackend(
                qaoa_descriptor, qiskit_device, shots, None, None, False
            )
            qiskit_expectation = qiskit_backend.expectation(variate_params)

            qiskit_statevec_backend = QAOAQiskitBackendStatevecSimulator(
                qaoa_descriptor, None, None, False
            )
            qiskit_statevec_expectation = qiskit_statevec_backend.expectation(
                variate_params
            )

            acceptable_delta = 0.05 * qiskit_statevec_expectation
            self.assertAlmostEqual(
                qiskit_expectation, qiskit_statevec_expectation, delta=acceptable_delta
            )

    @pytest.mark.api
    def test_remote_qubit_overflow(self):
        """
        If the user creates a circuit that is larger than the maximum circuit size
        that is supported by the QPU. An Exception should be raised with the
        appropriate error message alerting the user to the error.
        """

        shots = 100

        set_of_numbers = np.random.randint(1, 10, 8).tolist()
        qubo = NumberPartition(set_of_numbers).qubo

        mixer_hamil = X_mixer_hamiltonian(n_qubits=8)
        qaoa_descriptor = QAOADescriptor(qubo.hamiltonian, mixer_hamil, p=1)

        # Check the creation of the varitional parms
        _ = create_qaoa_variational_params(qaoa_descriptor, "standard", "rand")

        qiskit_device = DeviceQiskit("ibm_kyoto", self.HUB, self.GROUP, self.PROJECT)

        try:
            QAOAQiskitQPUBackend(
                qaoa_descriptor, qiskit_device, shots, None, None, True
            )
        except Exception as e:
            self.assertEqual(
                str(e),
                "There are lesser qubits on the device than the number of qubits required for the circuit.",
            )

    @pytest.mark.qpu
    def test_integration_on_emulator(self):
        """
        Test Emulated QPU Workflow. Checks if the expectation value is returned
        after the circuit run.
        """

        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [[1 / 8 * np.pi]]
        betas = [[1 / 8 * np.pi]]
        shots = 10000

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)
        qiskit_device = DeviceQiskit(
            "ibm_kyoto",
            self.HUB,
            self.GROUP,
            self.PROJECT,
            as_emulator=True,
        )

        qiskit_backend = QAOAQiskitQPUBackend(
            qaoa_descriptor, qiskit_device, shots, None, None, False
        )
        qiskit_expectation = qiskit_backend.expectation(variate_params)

        self.assertEqual(type(qiskit_expectation.item()), float)

    @pytest.mark.qpu
    def test_remote_integration_qpu_run(self):
        """
        Test Actual QPU Workflow. Checks if the expectation value is returned
        after the circuit run.
        """

        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [[1 / 8 * np.pi]]
        betas = [[1 / 8 * np.pi]]
        shots = 10000

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)
        qiskit_device = DeviceQiskit("ibm_kyoto", self.HUB, self.GROUP, self.PROJECT)

        qiskit_backend = QAOAQiskitQPUBackend(
            qaoa_descriptor, qiskit_device, shots, None, None, False
        )
        circuit = qiskit_backend.qaoa_circuit(variate_params)
        job = qiskit_backend.device.backend_device.run(
            circuit, shots=qiskit_backend.n_shots
        )

        # check if the cirucit is validated by IBMQ servers when submitted for execution
        # check the status of the job and keep retrying until its completed or queued
        while job.status().name not in ["DONE", "CANCELLED", "ERROR"]:
            # if the job is queued, cancel the job
            if job.status().name == "QUEUED":
                try:
                    job.cancel()
                except IBMJobError as e:
                    print(f"Job cancellation issue on IBMs end: {e}")
            else:
                time.sleep(1)

        assert job.status().name in ["DONE", "CANCELLED"]


if __name__ == "__main__":
    unittest.main()
