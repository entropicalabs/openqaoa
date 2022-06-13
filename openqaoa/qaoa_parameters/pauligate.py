#   Copyright 2022 Entropica Labs
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from abc import ABC, abstractmethod
# from types import NoneType
import numpy as np
from typing import List, Tuple, Union

from .operators import Hamiltonian
from .lowlevelgate import *
from .rotationangle import RotationAngle


class GateMap(ABC):

    def __init__(self, qubit_1: int):

        self.qubit_1 = qubit_1

    def decomposition(self, decomposition_type: str) -> List[Tuple]:

        try:
            return getattr(self, '_decomposition_'+decomposition_type)
        except Exception as e:
            print(e, '\nReturning default decomposition.')
            return getattr(self, '_decomposition_standard')

    @property
    @abstractmethod
    def _decomposition_standard(self) -> List[Tuple]:
        pass


class SWAPGate(GateMap):

    def __init__(self, qubit_1: int, qubit_2: int):

        super().__init__(qubit_1)
        self.qubit_2 = qubit_2

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [(CX, [[self.qubit_1, self.qubit_2]]),
                (CX, [[self.qubit_2, self.qubit_1]]),
                (CX, [[self.qubit_1, self.qubit_2]])]
    
    @property
    def _decomposition_standard2(self) -> List[Tuple]:

        return [(RZ, [self.qubit_1, np.pi/2]),
                (RZ, [self.qubit_2, np.pi/2]),
                
                # X gate decomposition
                (RZ, [self.qubit_1, np.pi]),
                (RX, [self.qubit_1, np.pi/2]),
                (RZ, [self.qubit_1, np.pi]),
                (RX, [self.qubit_1, -np.pi/2]),
                
                # X gate decomposition
                (RZ, [self.qubit_2, np.pi]),
                (RX, [self.qubit_2, np.pi/2]),
                (RZ, [self.qubit_2, np.pi]),
                (RX, [self.qubit_2, -np.pi/2]),
                
                (RiSWAP, [[self.qubit_1, self.qubit_2, np.pi]]),
                (CZ, [[self.qubit_1, self.qubit_2]]),
                
                # X gate decomposition
                (RZ, [self.qubit_1, np.pi]),
                (RX, [self.qubit_1, np.pi/2]),
                (RZ, [self.qubit_1, np.pi]),
                (RX, [self.qubit_1, -np.pi/2]),
                
                # X gate decomposition
                (RZ, [self.qubit_2, np.pi]),
                (RX, [self.qubit_2, np.pi/2]),
                (RZ, [self.qubit_2, np.pi]),
                (RX, [self.qubit_2, -np.pi/2])]

    
class PauliGate(GateMap):

    def __init__(self, qubit_1: int, pauli_label: List):

        super().__init__(qubit_1)

        self.pauli_label = pauli_label
        self.rotation_angle = None

    @property
    def pauli_label(self) -> List:

        return self._pauli_label

    @pauli_label.setter
    def pauli_label(self, input_label: List) -> None:

        gate_type = ['1q']
        gate_type.extend(input_label)
        self._pauli_label = gate_type
    
    @property
    def _decomposition_trivial(self) -> List[Tuple]:
        return self._decomposition_standard


class RYPauliGate(PauliGate):

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [(RY, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  self.rotation_angle)])]


class RXPauliGate(PauliGate):

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [(RX, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  self.rotation_angle)])]


class RZPauliGate(PauliGate):

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [(RZ, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  self.rotation_angle)])]


class TwoPauliGate(PauliGate):

    def __init__(self, qubit_1: int, qubit_2: int, pauli_index: int):

        super().__init__(qubit_1, pauli_index)
        self.qubit_2 = qubit_2

    @property
    def pauli_label(self) -> List:

        return self._pauli_label

    @pauli_label.setter
    def pauli_label(self, input_label: List) -> None:

        gate_type = ['2q']
        gate_type.extend(input_label)
        self._pauli_label = gate_type

    @property
    def _decomposition_trivial(self) -> List[Tuple]:

        low_level_gate = eval(type(self).__name__.strip('PauliGate'))
        return [(low_level_gate, [[self.qubit_1, self.qubit_2],
                                  RotationAngle(lambda x: x, self.pauli_label,
                                  self.rotation_angle)])]


class RXXPauliGate(TwoPauliGate):

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [(RY, [self.qubit_1, np.pi/2]),
                (RX, [self.qubit_1, np.pi]),
                (RY, [self.qubit_2, np.pi/2]),
                (RX, [self.qubit_2, np.pi]),
                (CX, [[self.qubit_1, self.qubit_2]]),
                (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  self.rotation_angle)]),
                (CX, [[self.qubit_1, self.qubit_2]]),
                (RY, [self.qubit_1, np.pi/2]),
                (RX, [self.qubit_1, np.pi]),
                (RY, [self.qubit_2, np.pi/2]),
                (RX, [self.qubit_2, np.pi])]


class RXYPauliGate(TwoPauliGate):

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        raise NotImplementedError()


class RYYPauliGate(TwoPauliGate):

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [(RX, [self.qubit_1, np.pi/2]),
                (RX, [self.qubit_2, np.pi/2]),
                (CX, [[self.qubit_1, self.qubit_2]]),
                (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  self.rotation_angle)]),
                (CX, [[self.qubit_1, self.qubit_2]]),
                (RY, [self.qubit_2, -np.pi/2]),
                (RX, [self.qubit_2, -np.pi/2])]


class RZXPauliGate(TwoPauliGate):

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [(RY, [self.qubit_2, np.pi/2]),
                (RX, [self.qubit_2, np.pi]),
                (CX, [[self.qubit_1, self.qubit_2]]),
                (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  self.rotation_angle)]),
                (CX, [[self.qubit_1, self.qubit_2]]),
                (RY, [self.qubit_2, np.pi/2]),
                (RX, [self.qubit_2, np.pi])]


class RZZPauliGate(TwoPauliGate):

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [(CX, [[self.qubit_1, self.qubit_2]]),
                (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  self.rotation_angle)]),
                (CX, [[self.qubit_1, self.qubit_2]])]

    @property
    def _decomposition_standard2(self) -> List[Tuple]:

        return [(RZ, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  self.rotation_angle)]),
                (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  self.rotation_angle)]),
                (CPHASE, [[self.qubit_1, self.qubit_2],
                 RotationAngle(lambda x: -2*x, self.pauli_label, self.rotation_angle)])]


class RYZPauliGate(TwoPauliGate):

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        raise NotImplementedError()


class RiSWAPPauliGate(TwoPauliGate):
    """
    Parameterised-iSWAP gate
    """

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        total_decomp = RXXPauliGate.decomposition
        total_decomp.extend(RYYPauliGate.decomposition)
        return total_decomp

    @property
    def _decomposition_standard2(self) -> List[Tuple]:

        return [(RiSWAP, [[self.qubit_1, self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                                     self.rotation_angle)]])]


class PauliGateFactory(object):

    def convert_hamiltonian_to_pauli_gates(hamil_obj: Hamiltonian,
                                           input_label: List) -> List[PauliGate]:

        pauli_terms = hamil_obj.terms

        output_gates = []

        one_qubit_count = 0
        two_qubit_count = 0

        for each_term in pauli_terms:
            if each_term.pauli_str in ['XX', 'XZ', 'ZX', 'ZZ', 'XY', 'YX', 'YY', 'YZ', 'ZY']:
                if each_term.pauli_str == 'XX':
                    output_gates.append(RXXPauliGate(each_term.qubit_indices[0],
                                                     each_term.qubit_indices[1],
                                                     [*input_label, two_qubit_count]))
                elif each_term.pauli_str == 'XZ':
                    output_gates.append(RZXPauliGate(each_term.qubit_indices[1],
                                                     each_term.qubit_indices[0],
                                                     [*input_label, two_qubit_count]))
                elif each_term.pauli_str == 'ZX':
                    output_gates.append(RZXPauliGate(each_term.qubit_indices[0],
                                                     each_term.qubit_indices[1],
                                                     [*input_label, two_qubit_count]))
                elif each_term.pauli_str == 'ZZ':
                    output_gates.append(RZZPauliGate(each_term.qubit_indices[0],
                                                     each_term.qubit_indices[1],
                                                     [*input_label, two_qubit_count]))
                elif each_term.pauli_str == 'XY':
                    output_gates.append(RXYPauliGate(each_term.qubit_indices[0],
                                                     each_term.qubit_indices[1],
                                                     [*input_label, two_qubit_count]))
                elif each_term.pauli_str == 'YX':
                    output_gates.append(RXYPauliGate(each_term.qubit_indices[1],
                                                     each_term.qubit_indices[0],
                                                     [*input_label, two_qubit_count]))
                elif each_term.pauli_str == 'YY':
                    output_gates.append(RYYPauliGate(each_term.qubit_indices[0],
                                                     each_term.qubit_indices[1],
                                                     [*input_label, two_qubit_count]))
                elif each_term.pauli_str == 'ZY':
                    output_gates.append(RYZPauliGate(each_term.qubit_indices[1],
                                                     each_term.qubit_indices[0],
                                                     [*input_label, two_qubit_count]))
                elif each_term.pauli_str == 'YZ':
                    output_gates.append(RYZPauliGate(each_term.qubit_indices[0],
                                                     each_term.qubit_indices[1],
                                                     [*input_label, two_qubit_count]))
                two_qubit_count += 1

            elif each_term.pauli_str in ['X', 'Y', 'Z']:

                if each_term.pauli_str == 'X':
                    output_gates.append(RXPauliGate(each_term.qubit_indices[0],
                                                    [*input_label, one_qubit_count]))
                elif each_term.pauli_str == 'Y':
                    output_gates.append(RYPauliGate(each_term.qubit_indices[0],
                                                    [*input_label, one_qubit_count]))
                elif each_term.pauli_str == 'Z':
                    output_gates.append(RZPauliGate(each_term.qubit_indices[0],
                                                    [*input_label, one_qubit_count]))
                one_qubit_count += 1

        return output_gates
