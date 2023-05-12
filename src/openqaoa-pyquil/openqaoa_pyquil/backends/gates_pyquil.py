from typing import Callable

from pyquil import Program, gates
from pyquil.quilatom import QubitPlaceholder

from openqaoa.qaoa_components.ansatz_constructor.rotationangle import RotationAngle
import openqaoa.qaoa_components.ansatz_constructor.gates as gates_core


class PyquilGateApplicator(gates_core.GateApplicator):
    PYQUIL_OQ_GATE_MAPPER = {
        gates_core.X.__name__: gates.X,
        gates_core.RZ.__name__: gates.RZ,
        gates_core.RX.__name__: gates.RX,
        gates_core.RY.__name__: gates.RY,
        gates_core.CX.__name__: gates.CNOT,
        gates_core.CZ.__name__: gates.CZ,
        gates_core.CPHASE.__name__: gates.CPHASE,
        gates_core.RiSWAP.__name__: gates.XY,
    }

    library = "pyquil"

    def create_quantum_circuit(self, n_qubits) -> Program:
        """
        Function which creates and empty circuit specific to the pyquil backend.
        Needed for SPAM twirling but can be used more generally.
        """
        return Program()

    def gate_selector(self, gate: gates_core.Gate) -> Callable:
        selected_pyquil_gate = PyquilGateApplicator.PYQUIL_OQ_GATE_MAPPER[gate.__name__]
        return selected_pyquil_gate

    @staticmethod
    def apply_1q_rotation_gate(
        pyquil_gate, qubit_1: int, rotation_object: RotationAngle, circuit: Program
    ) -> Program:
        circuit += pyquil_gate(rotation_object.rotation_angle, qubit_1)
        return circuit

    @staticmethod
    def apply_2q_rotation_gate(
        pyquil_gate,
        qubit_1: int,
        qubit_2: int,
        rotation_object: RotationAngle,
        circuit: Program,
    ) -> Program:
        circuit += pyquil_gate(rotation_object.rotation_angle, qubit_1, qubit_2)
        return circuit

    @staticmethod
    def apply_1q_fixed_gate(pyquil_gate, qubit_1: int, circuit: Program) -> Program:
        circuit += pyquil_gate(qubit_1)
        return circuit

    @staticmethod
    def apply_2q_fixed_gate(
        pyquil_gate, qubit_1: int, qubit_2: int, circuit: Program
    ) -> Program:
        circuit += pyquil_gate(qubit_1, qubit_2)
        return circuit

    def apply_gate(self, gate: gates_core.Gate, *args):
        selected_pyquil_gate = self.gate_selector(gate)
        if gate.n_qubits == 1:
            if hasattr(gate, "rotation_object"):
                # *args must be of the following format -- (qubit_1,rotation_object,circuit)
                return self.apply_1q_rotation_gate(selected_pyquil_gate, *args)
            else:
                # *args must be of the following format -- (qubit_1,circuit)
                return self.apply_1q_fixed_gate(selected_pyquil_gate, *args)
        elif gate.n_qubits == 2:
            if hasattr(gate, "rotation_object"):
                # *args must be of the following format -- (qubit_1,qubit_2,rotation_object,circuit)
                return self.apply_2q_rotation_gate(selected_pyquil_gate, *args)
            else:
                # *args must be of the following format -- (qubit_1,qubit_2,circuit)
                return self.apply_2q_fixed_gate(selected_pyquil_gate, *args)
        else:
            raise ValueError("Only 1 and 2-qubit gates are supported.")
