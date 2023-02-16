from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

from .operators import Hamiltonian
from .gates import *
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


class SWAPGateMap(GateMap):

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

        return [(RZ, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi/2)]),
                (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi/2)]),
                
                # X gate decomposition
                (RZ, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi)]),
                (RX, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi/2)]),
                (RZ, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi)]),
                (RX, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  -np.pi/2)]),
                
                # X gate decomposition
                (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi)]),
                (RX, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi/2)]),
                (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi)]),
                (RX, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  -np.pi/2)]),
                
                (RiSWAP, [[self.qubit_1, self.qubit_2,
                            RotationAngle(lambda x: x, self.pauli_label,np.pi)]]),
                (CZ, [[self.qubit_1, self.qubit_2]]),
                
                # X gate decomposition
                (RZ, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi)]),
                (RX, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi/2)]),
                (RZ, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi)]),
                (RX, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  -np.pi/2)]),
                
                # X gate decomposition
                (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi)]),
                (RX, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi/2)]),
                (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi)]),
                (RX, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  -np.pi/2)])]

    
class RotationGateMap(GateMap):

    def __init__(self, qubit_1: int, pauli_label: List = []):

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


class RYGateMap(RotationGateMap):

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [(RY, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  self.rotation_angle)])]


class RXGateMap(RotationGateMap):

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [(RX, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  self.rotation_angle)])]


class RZGateMap(RotationGateMap):

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [(RZ, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  self.rotation_angle)])]


class TwoQubitRotationGateMap(RotationGateMap):

    def __init__(self, qubit_1: int, qubit_2: int, pauli_label: List = []):

        super().__init__(qubit_1, pauli_label)
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

        low_level_gate = eval(type(self).__name__.strip('GateMap'))
        return [(low_level_gate, [[self.qubit_1, self.qubit_2],
                                  RotationAngle(lambda x: x, self.pauli_label,
                                  self.rotation_angle)])]


class RXXGateMap(TwoQubitRotationGateMap):

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [(RY, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi/2)]),
                (RX, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi)]),
                (RY, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi/2)]),
                (RX, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi)]),
                (CX, [[self.qubit_1, self.qubit_2]]),
                (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  self.rotation_angle)]),
                (CX, [[self.qubit_1, self.qubit_2]]),
                (RY, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi/2)]),
                (RX, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi)]),
                (RY, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi/2)]),
                (RX, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi)])]


class RXYGateMap(TwoQubitRotationGateMap):

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        raise NotImplementedError()


class RYYGateMap(TwoQubitRotationGateMap):

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [(RX, [self.qubit_1, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi/2)]),
                (RX, [self.qubit_2,RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi/2)]),
                (CX, [[self.qubit_1, self.qubit_2]]),
                (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  self.rotation_angle)]),
                (CX, [[self.qubit_1, self.qubit_2]]),
                (RY, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  -np.pi/2)]),
                (RX, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  -np.pi/2)])]


class RZXGateMap(TwoQubitRotationGateMap):

    @property
    def _decomposition_standard(self) -> List[Tuple]:

        return [(RY, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi/2)]),
                (RX, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi)]),
                (CX, [[self.qubit_1, self.qubit_2]]),
                (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  self.rotation_angle)]),
                (CX, [[self.qubit_1, self.qubit_2]]),
                (RY, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi/2)]),
                (RX, [self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                  np.pi)])]


class RZZGateMap(TwoQubitRotationGateMap):

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

        return [(RiSWAP, [[self.qubit_1, self.qubit_2, RotationAngle(lambda x: x, self.pauli_label,
                                                                     self.rotation_angle)]])]


class RotationGateMapFactory(object):

    def convert_hamiltonian_to_gate_maps(hamil_obj: Hamiltonian,
                                           input_label: List) -> List[RotationGateMap]:
        
        """
        Converts a Hamiltonian Object into a List of RotationGateMap Objects.
        """

        pauli_terms = hamil_obj.terms

        output_gates = []

        one_qubit_count = 0
        two_qubit_count = 0

        for each_term in pauli_terms:
            if each_term.pauli_str in ['XX', 'XZ', 'ZX', 'ZZ', 'XY', 'YX', 'YY', 'YZ', 'ZY']:
                if each_term.pauli_str == 'XX':
                    output_gates.append(RXXGateMap(each_term.qubit_indices[0],
                                                     each_term.qubit_indices[1],
                                                     [*input_label, two_qubit_count]))
                elif each_term.pauli_str == 'XZ':
                    output_gates.append(RZXGateMap(each_term.qubit_indices[1],
                                                     each_term.qubit_indices[0],
                                                     [*input_label, two_qubit_count]))
                elif each_term.pauli_str == 'ZX':
                    output_gates.append(RZXGateMap(each_term.qubit_indices[0],
                                                     each_term.qubit_indices[1],
                                                     [*input_label, two_qubit_count]))
                elif each_term.pauli_str == 'ZZ':
                    output_gates.append(RZZGateMap(each_term.qubit_indices[0],
                                                     each_term.qubit_indices[1],
                                                     [*input_label, two_qubit_count]))
                elif each_term.pauli_str == 'XY':
                    output_gates.append(RXYGateMap(each_term.qubit_indices[0],
                                                     each_term.qubit_indices[1],
                                                     [*input_label, two_qubit_count]))
                elif each_term.pauli_str == 'YX':
                    output_gates.append(RXYGateMap(each_term.qubit_indices[1],
                                                     each_term.qubit_indices[0],
                                                     [*input_label, two_qubit_count]))
                elif each_term.pauli_str == 'YY':
                    output_gates.append(RYYGateMap(each_term.qubit_indices[0],
                                                     each_term.qubit_indices[1],
                                                     [*input_label, two_qubit_count]))
                elif each_term.pauli_str == 'ZY':
                    output_gates.append(RYZGateMap(each_term.qubit_indices[1],
                                                     each_term.qubit_indices[0],
                                                     [*input_label, two_qubit_count]))
                elif each_term.pauli_str == 'YZ':
                    output_gates.append(RYZGateMap(each_term.qubit_indices[0],
                                                     each_term.qubit_indices[1],
                                                     [*input_label, two_qubit_count]))
                two_qubit_count += 1

            elif each_term.pauli_str in ['X', 'Y', 'Z']:

                if each_term.pauli_str == 'X':
                    output_gates.append(RXGateMap(each_term.qubit_indices[0],
                                                    [*input_label, one_qubit_count]))
                elif each_term.pauli_str == 'Y':
                    output_gates.append(RYGateMap(each_term.qubit_indices[0],
                                                    [*input_label, one_qubit_count]))
                elif each_term.pauli_str == 'Z':
                    output_gates.append(RZGateMap(each_term.qubit_indices[0],
                                                    [*input_label, one_qubit_count]))
                one_qubit_count += 1

        return output_gates
    
    def remap_gate_map_labels(gatemap_list: List[RotationGateMap], input_label: List) -> List[RotationGateMap]:
        
        """
        Recreates a list of RotationGateMap Objects with the appropriately 
        assigned pauli_label attribute.
        """
        
        output_gates = []

        one_qubit_count = 0
        two_qubit_count = 0
        
        for each_gatemap in gatemap_list:
            
            if each_gatemap.pauli_label[0] == '1q':
                each_gatemap.pauli_label = [*input_label, one_qubit_count]
                one_qubit_count += 1
                
            elif each_gatemap.pauli_label[0] == '2q':
                each_gatemap.pauli_label = [*input_label, two_qubit_count]
                two_qubit_count += 1
                
            output_gates.append(each_gatemap)

        return output_gates