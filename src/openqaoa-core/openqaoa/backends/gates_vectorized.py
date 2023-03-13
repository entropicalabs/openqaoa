from typing import Callable
from ..qaoa_components.ansatz_constructor import gates as gates_core
from ..qaoa_components.ansatz_constructor.rotationangle import RotationAngle

class VectorizedGateApplicator(gates_core.GateApplicator):

    VECTORIZED_OQ_GATE_MAPPER = lambda x: {
        gates_core.RZ.__name__: x.apply_rz,
        gates_core.RX.__name__: x.apply_rx,
        gates_core.RY.__name__: x.apply_ry,
        gates_core.RXX.__name__: x.apply_rxx,
        gates_core.RZX.__name__: x.apply_rzx,
        gates_core.RZZ.__name__: x.apply_rzz,
        gates_core.RYY.__name__: x.apply_ryy,
        gates_core.RYZ.__name__: x.apply_ryz,
        gates_core.RiSWAP.__name__: x.apply_rxy
    }

    library = 'vectorized'

    def gate_selector(self, gate:gates_core.Gate, vectorized_backend: 'QAOAvectorizedBackendSimulator'):
        selected_gate = VectorizedGateApplicator.VECTORIZED_OQ_GATE_MAPPER(vectorized_backend)[gate.__name__]
        return selected_gate

    @staticmethod
    def apply_1q_rotation_gate(
        vectorized_gate: Callable,
        qubit_1: int,
        rotation_object: RotationAngle
    ):
        vectorized_gate(
            qubit_1,
            rotation_object.rotation_angle
        )

    @staticmethod
    def apply_2q_rotation_gate(
        vectorized_gate: Callable,
        qubit_1: int,
        qubit_2: int,
        rotation_object: RotationAngle
    ):
        vectorized_gate(
            qubit_1,
            qubit_2,
            rotation_object.rotation_angle
        )

    @staticmethod
    def apply_1q_fixed_gate(
        vectorized_gate: Callable,
        qubit_1: int
    ):
        vectorized_gate(
            qubit_1
        )

    @staticmethod
    def apply_2q_fixed_gate(
        vectorized_gate: Callable,
        qubit_1: int,
        qubit_2: int
    ):
        vectorized_gate(
            qubit_1,
            qubit_2
        )

    def apply_gate(self, gate: gates_core.Gate, *args):
        selected_vector_gate = self.gate_selector(gate, args[-1])
        # Remove argument from tuple
        args=args[:-1]
        if gate.n_qubits == 1:
            if hasattr(gate, 'rotation_object'):
                # *args must be of the following format -- (qubit_1,rotation_object)
                self.apply_1q_rotation_gate(selected_vector_gate, *args)
            else:
                # *args must be of the following format -- (qubit_1)
                self.apply_1q_fixed_gate(selected_vector_gate, *args)
        elif gate.n_qubits == 2:
            if hasattr(gate, 'rotation_object'):
                # *args must be of the following format -- (qubit_1,qubit_2,rotation_object)
                self.apply_2q_rotation_gate(selected_vector_gate, *args)
            else:
                # *args must be of the following format -- (qubit_1,qubit_2)
                self.apply_2q_fixed_gate(selected_vector_gate, *args)
        else:
            raise ValueError("Error applying the requested gate. Please check in the input")