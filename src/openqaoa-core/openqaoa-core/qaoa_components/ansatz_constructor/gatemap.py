from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Union
from copy import deepcopy

from .operators import Hamiltonian
from .gates import *
from .rotationangle import RotationAngle
from .gatemaplabel import GateMapLabel, GateMapType


class GateMap(ABC):
    def __init__(self, qubit_1: int):
        self.qubit_1 = qubit_1
        self.gate_label = None

    def decomposition(self, decomposition_type: str) -> List[Tuple]:
        try:
            return getattr(self, "_decomposition_" + decomposition_type)
        except Exception as e:
            print(e, "\nReturning default decomposition.")
            return getattr(self, "_decomposition_standard")

    @property
    @abstractmethod
    def _decomposition_standard(self) -> List[Tuple]:
        pass


class SWAPGateMap(GateMap):
    def __init__(self, qubit_1: int, qubit_2: int):

        super().__init__(qubit_1)
        self.qubit_2 = qubit_2
        self.gate_label = GateMapLabel(n_qubits=2, gatemap_type=GateMapType.FIXED)

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [
            (CX, [self.qubit_1, self.qubit_2]),
            (CX, [self.qubit_2, self.qubit_1]),
            (CX, [self.qubit_1, self.qubit_2]),
        ]

    @property
    def _decomposition_standard2(self) -> List[Tuple]:

        return [
            (
                RZ,
                [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (
                RZ,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            # X gate decomposition
            (RZ, [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RX,
                [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RZ, [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RX,
                [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, -np.pi / 2)],
            ),
            # X gate decomposition
            (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RX,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RX,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, -np.pi / 2)],
            ),
            (
                RiSWAP,
                [
                    self.qubit_1,
                    self.qubit_2,
                    RotationAngle(lambda x: x, self.gate_label, np.pi),
                ],
            ),
            (CZ, [self.qubit_1, self.qubit_2]),
            # X gate decomposition
            (RZ, [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RX,
                [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RZ, [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RX,
                [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, -np.pi / 2)],
            ),
            # X gate decomposition
            (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RX,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RX,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, -np.pi / 2)],
            ),
        ]


class RotationGateMap(GateMap):
    def __init__(self, qubit_1: int):

        super().__init__(qubit_1)
        self.angle_value = None
        self.gate_label = GateMapLabel(n_qubits=1)

    @property
    def _decomposition_trivial(self) -> List[Tuple]:
        return self._decomposition_standard


class RYGateMap(RotationGateMap):
    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [
            (
                RY,
                [
                    self.qubit_1,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            )
        ]


class RXGateMap(RotationGateMap):
    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [
            (
                RX,
                [
                    self.qubit_1,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            )
        ]


class RZGateMap(RotationGateMap):
    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [
            (
                RZ,
                [
                    self.qubit_1,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            )
        ]


class TwoQubitRotationGateMap(RotationGateMap):
    def __init__(self, qubit_1: int, qubit_2: int):

        super().__init__(qubit_1)
        self.qubit_2 = qubit_2
        self.gate_label = GateMapLabel(n_qubits=2)

    @property
    def _decomposition_trivial(self) -> List[Tuple]:

        low_level_gate = eval(type(self).__name__.strip("GateMap"))
        return [
            (
                low_level_gate,
                [
                    self.qubit_1,
                    self.qubit_2,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            )
        ]


class RXXGateMap(TwoQubitRotationGateMap):
    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [
            (
                RY,
                [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RX, [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RY,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RX, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (CX, [self.qubit_1, self.qubit_2]),
            (
                RZ,
                [
                    self.qubit_2,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            ),
            (CX, [self.qubit_1, self.qubit_2]),
            (
                RY,
                [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RX, [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RY,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RX, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
        ]


class RXYGateMap(TwoQubitRotationGateMap):
    @property
    def _decomposition_standard(self) -> List[Tuple]:

        raise NotImplementedError()


class RYYGateMap(TwoQubitRotationGateMap):
    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [
            (
                RX,
                [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (
                RX,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (CX, [self.qubit_1, self.qubit_2]),
            (
                RZ,
                [
                    self.qubit_2,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            ),
            (CX, [self.qubit_1, self.qubit_2]),
            (
                RY,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, -np.pi / 2)],
            ),
            (
                RX,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, -np.pi / 2)],
            ),
        ]


class RZXGateMap(TwoQubitRotationGateMap):
    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [
            (
                RY,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RX, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (CX, [self.qubit_1, self.qubit_2]),
            (
                RZ,
                [
                    self.qubit_2,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            ),
            (CX, [self.qubit_1, self.qubit_2]),
            (
                RY,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RX, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
        ]


class RZZGateMap(TwoQubitRotationGateMap):
    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [
            (CX, [self.qubit_1, self.qubit_2]),
            (
                RZ,
                [
                    self.qubit_2,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            ),
            (CX, [self.qubit_1, self.qubit_2]),
        ]

    @property
    def _decomposition_standard2(self) -> List[Tuple]:

        return [
            (
                RZ,
                [
                    self.qubit_1,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            ),
            (
                RZ,
                [
                    self.qubit_2,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            ),
            (
                CPHASE,
                [
                    self.qubit_1,
                    self.qubit_2,
                    RotationAngle(lambda x: -2 * x, self.gate_label, self.angle_value),
                ],
            ),
        ]


class RYZGateMap(TwoQubitRotationGateMap):
    @property
    def _decomposition_standard(self) -> List[Tuple]:

        raise NotImplementedError()


class RiSWAPGateMap(TwoQubitRotationGateMap):
    """
    Parameterised-iSWAP gate
    """

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        total_decomp = RXXGateMap.decomposition
        total_decomp.extend(RYYGateMap.decomposition)
        return total_decomp

    @property
    def _decomposition_standard2(self) -> List[Tuple]:

        return [
            (
                RiSWAP,
                [
                    self.qubit_1,
                    self.qubit_2,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            )
        ]


class RotationGateMapFactory(object):

    PAULI_OPERATORS = ["X", "Y", "Z", "XX", "ZX", "ZZ", "XY", "YY", "YZ"]
    GATE_GENERATOR_GATEMAP_MAPPER = {
        term: eval(f"R{term}GateMap") for term in PAULI_OPERATORS
    }

    def rotationgatemap_list_from_hamiltonian(
        hamil_obj: Hamiltonian, gatemap_type: GateMapType = None
    ) -> List[RotationGateMap]:
        """
        Constructs a list of Rotation GateMaps from the input Hamiltonian Object.

        Parameters
        ----------
        hamil_obj: Hamiltonian
            Hamiltonian object to construct the circuit from
        gatemap_type: GateMapType
            Gatemap type constructed
        """

        pauli_terms = hamil_obj.terms
        output_gates = []

        one_qubit_count = 0
        two_qubit_count = 0

        for each_term in pauli_terms:
            if each_term.pauli_str in RotationGateMapFactory.PAULI_OPERATORS:
                pauli_str = each_term.pauli_str
                qubit_indices = each_term.qubit_indices
            elif each_term.pauli_str[::-1] in RotationGateMapFactory.PAULI_OPERATORS:
                pauli_str = each_term.pauli_str[::-1]
                qubit_indices = each_term.qubit_indices[::-1]
            else:
                raise ValueError("Hamiltonian contains non-Pauli terms")

            try:
                gate_class = RotationGateMapFactory.GATE_GENERATOR_GATEMAP_MAPPER[
                    pauli_str
                ]
            except Exception:
                raise Exception("Generating gates from Hamiltonian terms failed")

            if len(each_term.qubit_indices) == 2:
                gate = gate_class(qubit_indices[0], qubit_indices[1])
                gate.gate_label.update_gatelabel(
                    new_application_sequence=two_qubit_count,
                    new_gatemap_type=gatemap_type,
                )
                output_gates.append(gate)
                two_qubit_count += 1
            elif len(each_term.qubit_indices) == 1:
                gate = gate_class(qubit_indices[0])
                gate.gate_label.update_gatelabel(
                    new_application_sequence=one_qubit_count,
                    new_gatemap_type=gatemap_type,
                )
                output_gates.append(gate)
                one_qubit_count += 1

        return output_gates

    def gatemaps_layer_relabel(
        gatemap_list: List[GateMap], new_layer_number: int
    ) -> List[RotationGateMap]:
        """
        Reconstruct a new gatemap list from a list of RotationGateMap Objects with the input
        layer number in the gate_label attribute.

        Parameters
        ----------
        gatemap_list: `List[RotationGateMap]
            The list of GateMap objects whose labels need to be udpated
        """
        output_gate_list = []

        for each_gatemap in gatemap_list:
            new_gatemap = deepcopy(each_gatemap)
            new_gatemap.gate_label.update_gatelabel(new_layer_number=new_layer_number)
            output_gate_list.append(new_gatemap)

        return output_gate_list
