from typing import Callable

from braket.circuits import gates, Circuit

from openqaoa.qaoa_components.ansatz_constructor.rotationangle import RotationAngle
import openqaoa.qaoa_components.ansatz_constructor.gates as gates_core

class BraketGateApplicator(gates_core.GateApplicator):

    BRAKET_OQ_GATE_MAPPER = {
        gates_core.RZ.__name__: gates.Rz.rz,
        gates_core.RX.__name__: gates.Rx.rx,
        gates_core.RY.__name__: gates.Ry.ry,
        gates_core.CX.__name__: gates.CNot.cnot,
        gates_core.CZ.__name__: gates.CZ.cz,
        gates_core.RXX.__name__: gates.XX.xx,
        gates_core.RZZ.__name__: gates.ZZ.zz,
        gates_core.RYY.__name__: gates.YY.yy,
        gates_core.CPHASE.__name__: gates.CPhaseShift.cphaseshift,
        gates_core.RiSWAP.__name__: gates.XY.xy
    }

    library = 'braket'

    def gate_selector(self, gate: gates_core.Gate) -> Callable:
        selected_braket_gate = BraketGateApplicator.BRAKET_OQ_GATE_MAPPER[gate.__name__]
        return selected_braket_gate

    @staticmethod
    def apply_1q_rotation_gate(
        braket_gate,
        qubit_1: int,
        rotation_object: RotationAngle,
        circuit: Circuit
    ) -> Circuit:
        circuit += braket_gate(
            qubit_1, 
            rotation_object.rotation_angle
        )
        return circuit

    @staticmethod
    def apply_2q_rotation_gate(
        braket_gate,
        qubit_1: int,
        qubit_2: int,
        rotation_object: RotationAngle,
        circuit: Circuit
    ) -> Circuit:
        circuit += braket_gate(
            qubit_1, 
            qubit_2, 
            rotation_object.rotation_angle
        )
        return circuit

    @staticmethod
    def apply_1q_fixed_gate(
        braket_gate,
        qubit_1: int,
        circuit: Circuit
    ) -> Circuit:
        circuit += braket_gate(qubit_1)
        return circuit

    @staticmethod
    def apply_2q_fixed_gate(
        braket_gate,
        qubit_1: int,
        qubit_2: int,
        circuit: Circuit
    ) -> Circuit:
        circuit += braket_gate(qubit_1, qubit_2)
        return circuit

    def apply_gate(self, gate: gates_core.Gate, *args) -> Circuit:
        selected_braket_gate = self.gate_selector(gate)
        if gate.n_qubits == 1:
            if hasattr(gate, 'rotation_object'):
                # *args must be of the following format -- (qubit_1,rotation_object,circuit)
                return self.apply_1q_rotation_gate(selected_braket_gate, *args)
            else:
                # *args must be of the following format -- (qubit_1,circuit)
                return self.apply_1q_fixed_gate(selected_braket_gate, *args)
        elif gate.n_qubits == 2:
            if hasattr(gate, 'rotation_object'):
                # *args must be of the following format -- (qubit_1,qubit_2,rotation_object,circuit)
                return self.apply_2q_rotation_gate(selected_braket_gate, *args)
            else:
                # *args must be of the following format -- (qubit_1,qubit_2,circuit)
                return self.apply_2q_fixed_gate(selected_braket_gate, *args)
        else:
            raise ValueError("Only 1 and 2-qubit gates are supported.")