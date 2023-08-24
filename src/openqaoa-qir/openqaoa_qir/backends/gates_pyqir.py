from openqaoa.qaoa_components.ansatz_constructor import gates, RotationAngle
from pyqir import BasicQisBuilder, SimpleModule
from typing import Callable


class QIRGateApplicator(gates.GateApplicator):
    library = "PYQIR"

    def __init__(self, qaoa_module: SimpleModule, qis_builder: BasicQisBuilder):
        self.qaoa_module = qaoa_module
        self.qis_builder = qis_builder

        self.QIS_OQ_MAPPER = {
            gates.CX.__name__: self.qis_builder.cx,
            gates.CZ.__name__: self.qis_builder.cz,
            gates.RX.__name__: self.qis_builder.rx,
            gates.RY.__name__: self.qis_builder.ry,
            gates.RZ.__name__: self.qis_builder.rz,
        }

    def gate_selector(self, gate: gates.Gate, qis_builder: BasicQisBuilder) -> Callable:
        selected_qiskit_gate = self.QIS_OQ_MAPPER[gate.__name__]
        return selected_qiskit_gate

    def apply_1q_rotation_gate(
        self, qis_gate, qubit_1, rotation_object: RotationAngle, ckt=None
    ) -> None:
        qubit_1 = self.qaoa_module.qubits[qubit_1]
        qis_gate(rotation_object.rotation_angle, qubit_1)

    def apply_2q_rotation_gate(
        self, qis_gate, qubit_1, qubit_2, rotation_object: RotationAngle, ckt=None
    ) -> None:
        qubit_1 = self.qaoa_module.qubits[qubit_1]
        qubit_2 = self.qaoa_module.qubits[qubit_2]
        qis_gate(rotation_object.rotation_angle, qubit_1, qubit_2)

    def apply_1q_fixed_gate(self, qis_gate, qubit_1: int, ckt=None) -> None:
        qubit_1 = self.qaoa_module.qubits[qubit_1]
        qis_gate(qubit_1)

    def apply_2q_fixed_gate(
        self, qis_gate, qubit_1: int, qubit_2: int, ckt=None
    ) -> None:
        qubit_1 = self.qaoa_module.qubits[qubit_1]
        qubit_2 = self.qaoa_module.qubits[qubit_2]
        qis_gate(qubit_1, qubit_2)

    def apply_gate(self, gate: gates.Gate, *args):
        selected_qiskit_gate = self.gate_selector(gate, self.qis_builder)
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
