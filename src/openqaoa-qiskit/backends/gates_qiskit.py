from typing import Callable

from qiskit.circuit.library import (
    XGate,
    RXGate,
    RYGate,
    RZGate,
    CXGate,
    CZGate,
    RXXGate,
    RZXGate,
    RZZGate,
    RYYGate,
    CRZGate,
)
from qiskit import QuantumCircuit, QuantumRegister

from openqaoa.qaoa_components.ansatz_constructor.rotationangle import RotationAngle
import openqaoa.qaoa_components.ansatz_constructor.gates as gates_core


class QiskitGateApplicator(gates_core.GateApplicator):
    QISKIT_OQ_GATE_MAPPER = {
        gates_core.X.__name__: XGate,
        gates_core.RZ.__name__: RZGate,
        gates_core.RX.__name__: RXGate,
        gates_core.RY.__name__: RYGate,
        gates_core.CX.__name__: CXGate,
        gates_core.CZ.__name__: CZGate,
        gates_core.RXX.__name__: RXXGate,
        gates_core.RZX.__name__: RZXGate,
        gates_core.RZZ.__name__: RZZGate,
        gates_core.RYY.__name__: RYYGate,
        gates_core.CPHASE.__name__: CRZGate,
    }

    library = "qiskit"

    def create_quantum_circuit(self, n_qubits) -> QuantumCircuit:
        """
        Function which creates and empty circuit specific to the qiskit backend.
        Needed for SPAM twirling but more general than this.
        """
        qureg = QuantumRegister(n_qubits)
        parametric_circuit = QuantumCircuit(qureg)
        return parametric_circuit

    def gate_selector(self, gate: gates_core.Gate) -> Callable:
        selected_qiskit_gate = QiskitGateApplicator.QISKIT_OQ_GATE_MAPPER[gate.__name__]
        return selected_qiskit_gate

    @staticmethod
    def apply_1q_rotation_gate(
        qiskit_gate,
        qubit_1: int,
        rotation_object: RotationAngle,
        circuit: QuantumCircuit,
    ) -> QuantumCircuit:
        circuit.append(qiskit_gate(rotation_object.rotation_angle), [qubit_1], [])
        return circuit

    @staticmethod
    def apply_2q_rotation_gate(
        qiskit_gate,
        qubit_1: int,
        qubit_2: int,
        rotation_object: RotationAngle,
        circuit: QuantumCircuit,
    ) -> QuantumCircuit:
        circuit.append(
            qiskit_gate(rotation_object.rotation_angle), [qubit_1, qubit_2], []
        )
        return circuit

    @staticmethod
    def apply_1q_fixed_gate(
        qiskit_gate, qubit_1: int, circuit: QuantumCircuit
    ) -> QuantumCircuit:
        circuit.append(qiskit_gate(), [qubit_1], [])
        return circuit

    @staticmethod
    def apply_2q_fixed_gate(
        qiskit_gate, qubit_1: int, qubit_2: int, circuit: QuantumCircuit
    ) -> QuantumCircuit:
        circuit.append(qiskit_gate(), [qubit_1, qubit_2], [])
        return circuit

    def apply_gate(self, gate: gates_core.Gate, *args):
        selected_qiskit_gate = self.gate_selector(gate)
        if gate.n_qubits == 1:
            if hasattr(gate, "rotation_object"):
                # *args must be of the following format -- (qubit_1,rotation_object,circuit)
                return self.apply_1q_rotation_gate(selected_qiskit_gate, *args)
            else:
                # *args must be of the following format -- (qubit_1,circuit)
                return self.apply_1q_fixed_gate(selected_qiskit_gate, *args)
        elif gate.n_qubits == 2:
            if hasattr(gate, "rotation_object"):
                # *args must be of the following format -- (qubit_1,qubit_2,rotation_object,circuit)
                return self.apply_2q_rotation_gate(selected_qiskit_gate, *args)
            else:
                # *args must be of the following format -- (qubit_1,qubit_2,circuit)
                return self.apply_2q_fixed_gate(selected_qiskit_gate, *args)
        else:
            raise ValueError("Only 1 and 2-qubit gates are supported.")
