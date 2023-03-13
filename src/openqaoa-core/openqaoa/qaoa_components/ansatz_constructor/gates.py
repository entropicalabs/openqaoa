from typing import Callable
from abc import ABC, abstractmethod
import numpy as np
from .rotationangle import RotationAngle


class GateApplicator(ABC):
    @abstractmethod
    def apply_gate(self, gate) -> Callable:
        """
        Apply gate to the circuit
        """
        pass


class Gate(ABC):
    def __init__(self, applicator: GateApplicator):
        self.applicator = applicator

    @abstractmethod
    def apply_gate(self, ckt):
        pass


class OneQubitGate(Gate):
    def __init__(self, applicator: GateApplicator, qubit_1: int):
        super().__init__(applicator)
        self.qubit_1 = qubit_1
        self.n_qubits = 1

    def apply_gate(self, ckt):
        return self.applicator.apply_gate(self, self.qubit_1, ckt)


class OneQubitRotationGate(OneQubitGate):
    def __init__(
        self, applicator: GateApplicator, qubit_1: int, rotation_object: RotationAngle
    ):
        super().__init__(applicator, qubit_1)
        self.rotation_object = rotation_object

    def apply_gate(self, ckt):
        return self.applicator.apply_gate(self, self.qubit_1, self.rotation_object, ckt)


class TwoQubitGate(Gate):
    def __init__(self, applicator: GateApplicator, qubit_1: int, qubit_2: int):
        super().__init__(applicator)
        self.qubit_1 = qubit_1
        self.qubit_2 = qubit_2
        self.n_qubits = 2

    def apply_gate(self, ckt):
        return self.applicator.apply_gate(self, self.qubit_1, self.qubit_2, ckt)


class TwoQubitRotationGate(TwoQubitGate):
    def __init__(
        self,
        applicator: GateApplicator,
        qubit_1: int,
        qubit_2: int,
        rotation_object: RotationAngle,
    ):
        super().__init__(applicator, qubit_1, qubit_2)
        self.rotation_object = rotation_object

    def apply_gate(self, ckt):
        return self.applicator.apply_gate(
            self, self.qubit_1, self.qubit_2, self.rotation_object, ckt
        )

    def apply_vector_gate(self, input_obj):
        input_obj.apply_rzz(
            self.qubit_1, self.qubit_2, self.rotation_object.rotation_angle
        )


class RZ(OneQubitRotationGate):
    __name__ = "RZ"


class RY(OneQubitRotationGate):
    __name__ = "RY"


class RX(OneQubitRotationGate):
    __name__ = "RX"


class CZ(TwoQubitGate):
    __name__ = "CZ"


class CX(TwoQubitGate):
    __name__ = "CX"


class RXX(TwoQubitRotationGate):
    __name__ = "RXX"


class RYY(TwoQubitRotationGate):
    __name__ = "RYY"


class RZZ(TwoQubitRotationGate):
    __name__ = "RZZ"


class RXY(TwoQubitRotationGate):
    __name__ = "RXY"


class RZX(TwoQubitRotationGate):
    __name__ = "RZX"


class RYZ(TwoQubitRotationGate):
    __name__ = "RYZ"


class CPHASE(TwoQubitRotationGate):
    __name__ = "CPHASE"


class RiSWAP(TwoQubitRotationGate):
    __name__ = "RiSWAP"