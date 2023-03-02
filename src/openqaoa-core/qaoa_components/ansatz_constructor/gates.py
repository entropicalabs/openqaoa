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

    def apply_vector_gate(self, input_obj):
        input_obj.apply_rz(self.qubit_1, self.rotation_object.rotation_angle)


class TwoQubitGate(Gate):
    def __init__(self, applicator: GateApplicator, qubit_1: int, qubit_2: int):
        super().__init__(applicator)
        self.qubit_1 = qubit_1
        self.qubit_2 = qubit_2
        self.n_qubits = 2

    def apply_gate(self, ckt):
        return self.applicator.apply_gate(self, self.qubit_1, self.qubit_2, ckt)

    def apply_vector_gate(self, input_obj):
        raise NotImplemented()


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


# from abc import ABC, abstractmethod
# from typing import List, Tuple
# import numpy as np

# from qiskit import QuantumCircuit as qkQuantumCircuit
# from qiskit.circuit.library import (
#     RXGate,
#     RYGate,
#     RZGate,
#     CXGate,
#     CZGate,
#     RXXGate,
#     RZXGate,
#     RZZGate,
#     RYYGate,
#     CRZGate,
# )
# from pyquil import Program as quilProgram
# from pyquil import gates as quilgates
# from pyquil.quilatom import QubitPlaceholder as quilQubitPlaceholder
# from .rotationangle import RotationAngle
# from braket.circuits import gates as braketgates
# from braket.circuits import Circuit


# class Gate(ABC):
#     def __init__(self, ibm_gate, pyquil_gate, braket_gate, vector_gate):
#         self.ibm_gate = ibm_gate
#         self.pyquil_gate = pyquil_gate
#         self.braket_gate = braket_gate
#         self.vector_gate = vector_gate

#     @abstractmethod
#     def apply_ibm_gate(self, circuit):
#         pass

#     @abstractmethod
#     def apply_pyquil_gate(self, circuit):
#         pass

#     @abstractmethod
#     def apply_braket_gate(self, circuit):
#         pass

#     @abstractmethod
#     def apply_vector_gate(self, circuit):
#         pass


# class OneQubitGate(Gate):
#     def apply_ibm_gate(
#         self,
#         qubit_idx: int,
#         rotation_angle_obj: RotationAngle,
#         circuit: qkQuantumCircuit,
#     ):
#         if self.ibm_gate is not None:
#             circuit.append(
#                 self.ibm_gate(rotation_angle_obj.rotation_angle), [qubit_idx], []
#             )
#         else:
#             raise NotImplementedError()
#         return circuit

#     def apply_pyquil_gate(
#         self,
#         qubit_idx: quilQubitPlaceholder,
#         rotation_angle_obj: RotationAngle,
#         program: quilProgram,
#     ):
#         if self.pyquil_gate is not None:
#             program += self.pyquil_gate(rotation_angle_obj.rotation_angle, qubit_idx)
#         else:
#             raise NotImplementedError()
#         return program

#     def apply_braket_gate(
#         self, qubit_idx: int, rotation_angle_obj: RotationAngle, circuit: Circuit
#     ):
#         if self.braket_gate is not None:
#             circuit += self.braket_gate(qubit_idx, rotation_angle_obj.rotation_angle)
#         else:
#             raise NotImplementedError()
#         return circuit

#     def apply_vector_gate(self, qubit_idx, rotation_angle_obj, input_obj):
#         return NotImplementedError(
#             "Implement this method for each supporting gate class"
#         )


# class RY(OneQubitGate):
#     def __init__(self):

#         ibm_gate = RYGate
#         pyquil_gate = quilgates.RY
#         braket_gate = braketgates.Ry.ry
#         vector_gate = None

#         super().__init__(
#             ibm_gate=ibm_gate,
#             pyquil_gate=pyquil_gate,
#             braket_gate=braket_gate,
#             vector_gate=vector_gate,
#         )

#     def apply_vector_gate(self, qubit_idx, rotation_angle_obj, input_obj):
#         input_obj.apply_ry(qubit_idx, rotation_angle_obj.rotation_angle)


# class RX(OneQubitGate):
#     def __init__(self):

#         ibm_gate = RXGate
#         pyquil_gate = quilgates.RX
#         braket_gate = braketgates.Rx.rx
#         vector_gate = None

#         super().__init__(
#             ibm_gate=ibm_gate,
#             pyquil_gate=pyquil_gate,
#             braket_gate=braket_gate,
#             vector_gate=vector_gate,
#         )

#     def apply_vector_gate(self, qubit_idx, rotation_angle_obj, input_obj):
#         input_obj.apply_rx(qubit_idx, rotation_angle_obj.rotation_angle)


# class RZ(OneQubitGate):
#     def __init__(self):

#         ibm_gate = RZGate
#         pyquil_gate = quilgates.RZ
#         braket_gate = braketgates.Rz.rz
#         vector_gate = None

#         super().__init__(
#             ibm_gate=ibm_gate,
#             pyquil_gate=pyquil_gate,
#             braket_gate=braket_gate,
#             vector_gate=vector_gate,
#         )

#     def apply_vector_gate(self, qubit_idx, rotation_angle_obj, input_obj):
#         input_obj.apply_rz(qubit_idx, rotation_angle_obj.rotation_angle)


# class TwoQubitGate(Gate):
#     def apply_ibm_gate(self, qubit_indices: List[int], circuit: qkQuantumCircuit):
#         if self.ibm_gate is not None:
#             circuit.append(self.ibm_gate(), qubit_indices, [])
#         else:
#             raise NotImplementedError()
#         return circuit

#     def apply_pyquil_gate(
#         self, qubit_indices: List[quilQubitPlaceholder], program: quilProgram
#     ):
#         if self.pyquil_gate is not None:
#             program += self.pyquil_gate(qubit_indices[0], qubit_indices[1])
#         else:
#             raise NotImplementedError()
#         return program

#     def apply_braket_gate(self, qubit_indices: List[int], circuit: Circuit):

#         if self.braket_gate is not None:
#             circuit += self.braket_gate(qubit_indices[0], qubit_indices[1])
#         else:
#             raise NotImplementedError()
#         return circuit

#     def apply_vector_gate(self, qubit_indices, input_obj):
#         return NotImplementedError(
#             "Implement this method for each supporting gate class"
#         )


# class CZ(TwoQubitGate):
#     def __init__(self):

#         ibm_gate = CZGate
#         pyquil_gate = quilgates.CZ
#         braket_gate = braketgates.CZ.cz
#         vector_gate = None

#         super().__init__(
#             ibm_gate=ibm_gate,
#             pyquil_gate=pyquil_gate,
#             braket_gate=braket_gate,
#             vector_gate=vector_gate,
#         )

#     def apply_vector_gate(self):

#         raise NotImplementedError("This gate is not yet supported")


# class CX(TwoQubitGate):
#     def __init__(self, mode: str = "CX"):

#         ibm_gate = CXGate
#         pyquil_gate = quilgates.CNOT
#         braket_gate = braketgates.CNot.cnot
#         vector_gate = None

#         super().__init__(
#             ibm_gate=ibm_gate,
#             pyquil_gate=pyquil_gate,
#             braket_gate=braket_gate,
#             vector_gate=vector_gate,
#         )
#         self.mode = mode

#     def _in_XY(self, qubit_indices) -> List[Tuple[Gate, List]]:
#         qubit_1 = qubit_indices[0]
#         qubit_2 = qubit_indices[1]
#         return [
#             (RX, [qubit_2, RotationAngle(lambda x: x, [], np.pi / 2)]),
#             (RZ, [qubit_1, RotationAngle(lambda x: x, [], -np.pi / 2)]),
#             (RZ, [qubit_1, RotationAngle(lambda x: x, [], np.pi / 2)]),
#             (RiSWAP, [[qubit_1, qubit_2, RotationAngle(lambda x: x, [], np.pi)]]),
#             (RX, [qubit_1, RotationAngle(lambda x: x, [], np.pi / 2)]),
#             (RiSWAP, [[qubit_1, qubit_2, RotationAngle(lambda x: x, [], np.pi)]]),
#             (RZ, [qubit_2, RotationAngle(lambda x: x, [], np.pi / 2)]),
#         ]

#     def _in_CZ(self, qubit_indices) -> List[Tuple[Gate, List]]:
#         qubit_1 = qubit_indices[0]
#         qubit_2 = qubit_indices[1]
#         return [
#             (RY, [qubit_2, RotationAngle(lambda x: x, [], np.pi / 2)]),
#             (RX, [qubit_2, RotationAngle(lambda x: x, [], np.pi)]),
#             (CZ, [[qubit_1, qubit_2]]),
#             (RY, [qubit_2, RotationAngle(lambda x: x, [], np.pi / 2)]),
#             (RX, [qubit_2, RotationAngle(lambda x: x, [], np.pi)]),
#         ]

#     def apply_ibm_gate(
#         self, qubit_indices: List[int], circuit: qkQuantumCircuit
#     ) -> qkQuantumCircuit:

#         if self.mode == "CX":
#             circuit.cx(qubit_indices[0], qubit_indices[1])
#         elif self.mode == "CZ":
#             for each_object, init_params in self._in_CZ(qubit_indices):
#                 circuit = each_object().apply_ibm_gate(*init_params, circuit)

#         return circuit

#     def apply_vector_gate(self, input_obj):

#         raise NotImplementedError("This gate is not yet supported")


# class TwoQubitGateWithAngle(TwoQubitGate):
#     def apply_ibm_gate(
#         self,
#         qubit_indices: List[int],
#         rotation_angle_obj: RotationAngle,
#         circuit: qkQuantumCircuit,
#     ):
#         if self.ibm_gate is not None:
#             circuit.append(
#                 self.ibm_gate(rotation_angle_obj.rotation_angle), qubit_indices, []
#             )
#         else:
#             raise NotImplementedError()
#         return circuit

#     def apply_pyquil_gate(
#         self,
#         qubit_indices: List[quilQubitPlaceholder],
#         rotation_angle_obj: RotationAngle,
#         program: quilProgram,
#     ):
#         if self.pyquil_gate is not None:
#             program += self.pyquil_gate(
#                 rotation_angle_obj.rotation_angle, qubit_indices[0], qubit_indices[1]
#             )
#         else:
#             raise NotImplementedError()
#         return program

#     def apply_braket_gate(self, qubit_indices, rotation_angle_obj, circuit):

#         if self.braket_gate is not None:
#             circuit += self.braket_gate(
#                 qubit_indices[0], qubit_indices[1], rotation_angle_obj.rotation_angle
#             )
#         else:
#             raise NotImplementedError()
#         return circuit

#     def apply_vector_gate(self, qubit_indices, rotation_angle_obj, input_obj):
#         return NotImplementedError(
#             "Implement this method for each supporting gate class"
#         )


# class RXX(TwoQubitGateWithAngle):
#     def __init__(self):

#         ibm_gate = RXXGate
#         pyquil_gate = None
#         braket_gate = braketgates.XX.xx
#         vector_gate = None

#         super(TwoQubitGate, self).__init__(
#             ibm_gate=ibm_gate,
#             pyquil_gate=pyquil_gate,
#             braket_gate=braket_gate,
#             vector_gate=vector_gate,
#         )

#     def apply_vector_gate(self, qubit_indices, rotation_angle_obj, input_obj):
#         input_obj.apply_rxx(
#             qubit_indices[0], qubit_indices[1], rotation_angle_obj.rotation_angle
#         )


# class RYY(TwoQubitGateWithAngle):
#     def __init__(self):

#         ibm_gate = RYYGate
#         pyquil_gate = None
#         braket_gate = braketgates.YY.yy
#         vector_gate = None

#         super(TwoQubitGate, self).__init__(
#             ibm_gate=ibm_gate,
#             pyquil_gate=pyquil_gate,
#             braket_gate=braket_gate,
#             vector_gate=vector_gate,
#         )

#     def apply_vector_gate(self, qubit_indices, rotation_angle_obj, input_obj):
#         input_obj.apply_ryy(
#             qubit_indices[0], qubit_indices[1], rotation_angle_obj.rotation_angle
#         )


# class RZZ(TwoQubitGateWithAngle):
#     def __init__(self):

#         ibm_gate = RZZGate
#         pyquil_gate = None
#         braket_gate = braketgates.ZZ.zz
#         vector_gate = None

#         super(TwoQubitGate, self).__init__(
#             ibm_gate=ibm_gate,
#             pyquil_gate=pyquil_gate,
#             braket_gate=braket_gate,
#             vector_gate=vector_gate,
#         )

#     def apply_vector_gate(self, qubit_indices, rotation_angle_obj, input_obj):
#         input_obj.apply_rzz(
#             qubit_indices[0], qubit_indices[1], rotation_angle_obj.rotation_angle
#         )


# class RXY(TwoQubitGateWithAngle):
#     def __init__(self):

#         ibm_gate = None
#         pyquil_gate = None
#         braket_gate = None
#         vector_gate = None

#         super(TwoQubitGate, self).__init__(
#             ibm_gate=ibm_gate,
#             pyquil_gate=pyquil_gate,
#             braket_gate=braket_gate,
#             vector_gate=vector_gate,
#         )

#     def apply_vector_gate(self, qubit_indices, rotation_angle_obj, input_obj):
#         input_obj.apply_rxy(
#             qubit_indices[0], qubit_indices[1], rotation_angle_obj.rotation_angle
#         )


# class RZX(TwoQubitGateWithAngle):
#     def __init__(self):

#         ibm_gate = RZXGate
#         pyquil_gate = None
#         braket_gate = None
#         vector_gate = None

#         super(TwoQubitGate, self).__init__(
#             ibm_gate=ibm_gate,
#             pyquil_gate=pyquil_gate,
#             braket_gate=braket_gate,
#             vector_gate=vector_gate,
#         )

#     def apply_ibm_gate(
#         self,
#         qubit_indices: List[int],
#         rotation_angle_obj: RotationAngle,
#         circuit: qkQuantumCircuit,
#     ):

#         circuit.rzx(
#             rotation_angle_obj.rotation_angle, qubit_indices[0], qubit_indices[1]
#         )
#         return circuit

#     def apply_vector_gate(self, qubit_indices, rotation_angle_obj, input_obj):
#         input_obj.apply_rzx(
#             qubit_indices[0], qubit_indices[1], rotation_angle_obj.rotation_angle
#         )


# class RYZ(TwoQubitGateWithAngle):
#     def __init__(self):

#         ibm_gate = None
#         pyquil_gate = None
#         braket_gate = None
#         vector_gate = None

#         super(TwoQubitGate, self).__init__(
#             ibm_gate=ibm_gate,
#             pyquil_gate=pyquil_gate,
#             braket_gate=braket_gate,
#             vector_gate=vector_gate,
#         )

#     def apply_vector_gate(self, qubit_indices, rotation_angle_obj, input_obj):
#         input_obj.apply_ryz(
#             qubit_indices[0], qubit_indices[1], rotation_angle_obj.rotation_angle
#         )


# class CPHASE(TwoQubitGateWithAngle):
#     def __init__(self):

#         ibm_gate = CRZGate
#         pyquil_gate = quilgates.CPHASE
#         braket_gate = braketgates.CPhaseShift.cphaseshift
#         vector_gate = None

#         super(TwoQubitGate, self).__init__(
#             ibm_gate=ibm_gate,
#             pyquil_gate=pyquil_gate,
#             braket_gate=braket_gate,
#             vector_gate=vector_gate,
#         )

#     def _vector_gate(self):

#         raise NotImplementedError("This gate is not yet supported")


# class RiSWAP(TwoQubitGateWithAngle):
#     def __init__(self):

#         ibm_gate = None
#         pyquil_gate = quilgates.XY
#         braket_gate = braketgates.XY.xy
#         vector_gate = None

#         super(TwoQubitGate, self).__init__(
#             ibm_gate=ibm_gate,
#             pyquil_gate=pyquil_gate,
#             braket_gate=braket_gate,
#             vector_gate=vector_gate,
#         )

#     def _vector_gate(self):

#         raise NotImplementedError("This gate is not yet supported")
