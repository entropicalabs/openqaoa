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
from typing import List, Tuple, Optional, Union
from attr import ib
from cirq import quil
import numpy as np

from qiskit import QuantumCircuit as qkQuantumCircuit
# from qiskit.circuit import Parameter as qkParameter
from qiskit.circuit.library import (RXGate, RYGate, RZGate, CXGate, CZGate,
                                    RXXGate, RZXGate, RZZGate, RYYGate, CRZGate)
from pyquil import Program as quilProgram
from pyquil import gates as quilgates
from pyquil.quilatom import QubitPlaceholder as quilQubitPlaceholder
from .rotationangle import RotationAngle


class LowLevelGate(ABC):

    def __init__(self, ibm_gate, pyquil_gate, braket_gate, vector_gate):
        self.ibm_gate = ibm_gate
        self.pyquil_gate = pyquil_gate
        self.braket_gate = braket_gate
        self.vector_gate = vector_gate
    
    @abstractmethod
    def apply_ibm_gate(self,circuit):
        pass
    
    @abstractmethod
    def apply_pyquil_gate(self,circuit):     
        pass
    
    @abstractmethod
    def apply_braket_gate(self,circuit):     
        pass
        
    @abstractmethod
    def apply_vector_gate(self,circuit): 
        pass
    
    
class OneQubitGate(LowLevelGate):

    def apply_ibm_gate(self, 
                       qubit_idx: int,
                       rotation_angle_obj: RotationAngle,
                       circuit: qkQuantumCircuit):                       
        if self.ibm_gate is not None:
            circuit.append(self.ibm_gate(rotation_angle_obj.rotation_angle), [qubit_idx], [])
        else:
            raise NotImplementedError()
        return circuit
    
    def apply_pyquil_gate(self,
                          qubit_idx: quilQubitPlaceholder,
                          rotation_angle_obj: RotationAngle,
                          program: quilProgram):
        if self.pyquil_gate is not None:
            program += self.pyquil_gate(rotation_angle_obj.rotation_angle, qubit_idx)
        else:
            raise NotImplementedError()
        return program
    
    def apply_braket_gate(self, qubit_idx, rotation_angle_obj, circuit):
        if self.braket_gate is not None:
            pass #TODO: Implement the gate
        else: 
            raise NotImplementedError()

    def apply_vector_gate(self, qubit_idx, rotation_angle_obj, input_obj):
        return NotImplementedError('Implement this method for each supporting gate class')
        
    # @abstractmethod
    # def apply_vector_gate(self, qubit_idx, rotation_angle_obj, input_obj):
        
    #     pass
    #     input_obj.self.vector_gate(qubit_idx, rotation_angle_obj.rotation_angle)

    # def __init__(self, qubit_index: Union[int,object], rotation_angle_obj: RotationAngle):
        
    #     self.qubit_1 = qubit_index
    #     self.gate_label = rotation_angle_obj.pauli_label
    #     self.rotation_angle_obj = rotation_angle_obj
        
        
class RY(OneQubitGate):

    def __init__(self):
        
        ibm_gate = RYGate
        pyquil_gate = quilgates.RY
        braket_gate = None
        vector_gate = None

        super().__init__(ibm_gate=ibm_gate, pyquil_gate=pyquil_gate,
                         braket_gate=braket_gate, vector_gate=vector_gate)
            
    def apply_vector_gate(self, qubit_idx, rotation_angle_obj, input_obj):
        input_obj.apply_ry(qubit_idx, rotation_angle_obj.rotation_angle)


class RX(OneQubitGate):
        
    def __init__(self):

        ibm_gate = RXGate
        pyquil_gate = quilgates.RX
        braket_gate = None
        vector_gate = None

        super().__init__(ibm_gate=ibm_gate, pyquil_gate=pyquil_gate,
                         braket_gate=braket_gate, vector_gate=vector_gate)
    # def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
    #     circuit.rx(self.rotation_angle_obj.rotation_angle, self.qubit_1)
    #     return circuit
    
    # def _pyquil_gate(self, program: quilProgram) -> quilProgram:
    
    #     program += quilgates.RX(self.rotation_angle_obj.rotation_angle,
    #                             self.qubit_1)
    #     return program
        
    # def _braket_gate(self):
        
    #     raise NotImplementedError()
        
    def apply_vector_gate(self, qubit_idx, rotation_angle_obj, input_obj):
        input_obj.apply_rx(qubit_idx, rotation_angle_obj.rotation_angle)

        
        
class RZ(OneQubitGate):

    def __init__(self):

        ibm_gate = RZGate
        pyquil_gate = quilgates.RZ
        braket_gate = None
        vector_gate = None

        super().__init__(ibm_gate=ibm_gate, pyquil_gate=pyquil_gate,
                         braket_gate=braket_gate, vector_gate=vector_gate)
    # def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
    #     circuit.rz(self.rotation_angle_obj.rotation_angle, self.qubit_1)
    #     return circuit
    
    # def _pyquil_gate(self, program: quilProgram) -> quilProgram:

    #     program += quilgates.RZ(self.rotation_angle_obj.rotation_angle,
    #                             self.qubit_1)
    #     return program
        
    # def _braket_gate(self):
        
    #     raise NotImplementedError()
        
    def apply_vector_gate(self, qubit_idx, rotation_angle_obj, input_obj):
        input_obj.apply_rz(qubit_idx, rotation_angle_obj.rotation_angle)

        
        
class TwoQubitGate(LowLevelGate):
    
    def apply_ibm_gate(self, 
                       qubit_indices: List[int],
                       circuit: qkQuantumCircuit):                       
        if self.ibm_gate is not None:
            circuit.append(self.ibm_gate(), qubit_indices, [])
        else: 
            raise NotImplementedError()
        return circuit
    
    def apply_pyquil_gate(self,
                          qubit_indices: List[quilQubitPlaceholder],
                          program: quilProgram):
        if self.pyquil_gate is not None:
            program += self.pyquil_gate(qubit_indices[0], qubit_indices[1])
        else:
            raise NotImplementedError()
        return program
    
    def apply_braket_gate(self, qubit_indices, circuit):
        
        raise NotImplementedError()

    def apply_vector_gate(self,qubit_indices,input_obj):
        return NotImplementedError('Implement this method for each supporting gate class')
    

    # def __init__(self, qubit_indices: List[int]):
        
    #     self.qubit_1 = qubit_indices[0]
    #     self.qubit_2 = qubit_indices[1]


class CZ(TwoQubitGate):

    def __init__(self):
        
        ibm_gate = CZGate
        pyquil_gate = quilgates.CZ
        braket_gate = None
        vector_gate = None

        super().__init__(ibm_gate=ibm_gate, pyquil_gate=pyquil_gate,
                         braket_gate=braket_gate, vector_gate=vector_gate)
    # def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
    #     circuit.cz(self.qubit_1, self.qubit_2)
    #     return circuit
    
    # def _pyquil_gate(self, program: quilProgram) -> quilProgram:
        
    #     program += quilgates.CZ(self.qubit_1, self.qubit_2)
    #     return program
        
    # def _braket_gate(self):
        
    #     raise NotImplementedError()
        
    def apply_vector_gate(self):
        
        raise NotImplementedError('This gate is not yet supported')

        
class CX(TwoQubitGate):

    def __init__(self, mode: str = 'CX'):
        
        ibm_gate = CXGate
        pyquil_gate = quilgates.CNOT
        braket_gate = None
        vector_gate = None

        super().__init__(ibm_gate=ibm_gate, pyquil_gate=pyquil_gate,
                         braket_gate=braket_gate, vector_gate=vector_gate)
        self.mode = mode
        
    @property
    def _in_XY(self) -> List[Tuple[LowLevelGate, List]]:
    
        return [(RX, [self.qubit_2, RotationAngle(lambda x: x, [], np.pi/2)]), 
                (RZ, [self.qubit_1, RotationAngle(lambda x: x, [], -np.pi/2)]), 
                (RZ, [self.qubit_1, RotationAngle(lambda x: x, [], np.pi/2)]), 
                (RiSWAP, [[self.qubit_1, self.qubit_2, RotationAngle(lambda x: x, [], np.pi)]]), 
                (RX, [self.qubit_1, RotationAngle(lambda x: x, [], np.pi/2)]), 
                (RiSWAP, [[self.qubit_1, self.qubit_2, RotationAngle(lambda x: x, [], np.pi)]]), 
                (RZ, [self.qubit_2, RotationAngle(lambda x: x, [], np.pi/2)])]
    
    @property
    def _in_CZ(self) -> List[Tuple[LowLevelGate, List]]:
        
        return [(RY, [self.qubit_2, RotationAngle(lambda x: x, [], np.pi/2)]), 
                (RX, [self.qubit_2, RotationAngle(lambda x: x, [], np.pi)]), 
                (CZ, [[self.qubit_1, self.qubit_2]]), 
                (RY, [self.qubit_2, RotationAngle(lambda x: x, [], np.pi/2)]), 
                (RX, [self.qubit_2, RotationAngle(lambda x: x, [], np.pi)])]
    
    # def apply_gate(self, circuit, circuit_library: str, mode: Optional[str] = 'CX'):
        
    #     if circuit_library == 'ibm':
    #         return self._ibm_gate(circuit, mode)
    #     elif circuit_library == 'pyquil':
    #         return self._pyquil_gate(circuit, mode)
    #     elif circuit_library == 'braket':
    #         return self._braket_gate(circuit, mode)
    #     elif circuit_library == 'vector':
    #         return self._vector_gate(circuit)
    
    def apply_ibm_gate(self,
                       qubit_indices: List[int],
                       circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
        if self.mode == 'CX':
            circuit.cx(qubit_indices[0], qubit_indices[1])
        elif self.mode == 'CZ':
            for each_object, init_params in self._in_CZ:
                circuit = each_object().apply_ibm_gate(*init_params,circuit)
        
        return circuit
    
    # def apply_pyquil_gate(self, program: quilProgram, mode: str) -> quilProgram:
        
    #     program += quilgates.CNOT(self.qubit_1,self.qubit_2)
    #     return program
        
    # def apply_braket_gate(self):
        
    #     raise NotImplementedError()
        
    def apply_vector_gate(self, input_obj):
        
        raise NotImplementedError('This gate is not yet supported')
    

class TwoQubitGateWithAngle(TwoQubitGate):
    
    def apply_ibm_gate(self, 
                       qubit_indices: List[int],
                       rotation_angle_obj: RotationAngle,
                       circuit: qkQuantumCircuit):                       
        if self.ibm_gate is not None:
            circuit.append(self.ibm_gate(rotation_angle_obj.rotation_angle), qubit_indices, [])
        else: 
            raise NotImplementedError()
        return circuit
    
    def apply_pyquil_gate(self,
                          qubit_indices: List[quilQubitPlaceholder],
                          rotation_angle_obj: RotationAngle,
                          program: quilProgram):
        if self.pyquil_gate is not None:
            program += self.pyquil_gate(rotation_angle_obj.rotation_angle, qubit_indices[0], qubit_indices[1])
        else: 
            raise NotImplementedError()
        return program
    
    def apply_braket_gate(self, qubit_indices, rotation_angle_obj, circuit):
        
        raise NotImplementedError()

    def apply_vector_gate(self, qubit_indices, rotation_angle_obj, input_obj):
        return NotImplementedError('Implement this method for each supporting gate class')
    

    # def __init__(self, qubit_indices: List[int], rotation_angle_obj: RotationAngle):
        
    #     super().__init__(qubit_indices)
    #     self.gate_label = rotation_angle_obj.pauli_label
    #     self.rotation_angle_obj = rotation_angle_obj
        
        
class RXX(TwoQubitGateWithAngle):

    def __init__(self):

        ibm_gate = RXXGate
        pyquil_gate = None
        braket_gate = None
        vector_gate = None

        super(TwoQubitGate,self).__init__(ibm_gate=ibm_gate, pyquil_gate=pyquil_gate,
                                          braket_gate=braket_gate, vector_gate=vector_gate)

    # def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
    #     circuit.rxx(self.rotation_angle_obj.rotation_angle,
    #                             self.qubit_1,
    #                             self.qubit_2)
    #     return circuit
        
    # def _pyquil_gate(self):
        
    #     raise NotImplementedError()
        
    # def _braket_gate(self):
        
    #     raise NotImplementedError()
        
    def apply_vector_gate(self, qubit_indices, rotation_angle_obj, input_obj):
        input_obj.apply_rxx(qubit_indices[0], qubit_indices[1], rotation_angle_obj.rotation_angle)
        
        
class RYY(TwoQubitGateWithAngle):

    def __init__(self):

        ibm_gate = RYYGate
        pyquil_gate = None
        braket_gate = None
        vector_gate = None

        super(TwoQubitGate,self).__init__(ibm_gate=ibm_gate, pyquil_gate=pyquil_gate,
                         braket_gate=braket_gate, vector_gate=vector_gate)
    
    # def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
    #     circuit.ryy(self.rotation_angle_obj.rotation_angle,
    #                             self.qubit_1,
    #                             self.qubit_2)
    #     return circuit
        
    # def _pyquil_gate(self):
        
    #     raise NotImplementedError()
        
    # def _braket_gate(self):
        
    #     raise NotImplementedError()
        
    def apply_vector_gate(self, qubit_indices, rotation_angle_obj, input_obj):
        input_obj.apply_ryy(qubit_indices[0], qubit_indices[1], rotation_angle_obj.rotation_angle)
    
    
class RZZ(TwoQubitGateWithAngle):

    def __init__(self):

        ibm_gate = RZZGate
        pyquil_gate = None
        braket_gate = None
        vector_gate = None

        super(TwoQubitGate,self).__init__(ibm_gate=ibm_gate, pyquil_gate=pyquil_gate,
                         braket_gate=braket_gate, vector_gate=vector_gate)

        
    # def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
    #     circuit.rzz(self.rotation_angle_obj.rotation_angle,
    #                             self.qubit_1,
    #                             self.qubit_2)
    #     return circuit
    
    # def _pyquil_gate(self) -> quilProgram:
        
    #     raise NotImplementedError()


    # def _braket_gate(self):
        
    #     raise NotImplementedError()
        
    def apply_vector_gate(self, qubit_indices, rotation_angle_obj, input_obj):
        input_obj.apply_rzz(qubit_indices[0], qubit_indices[1], rotation_angle_obj.rotation_angle)
    
class RXY(TwoQubitGateWithAngle):

    def __init__(self):

        ibm_gate = None
        pyquil_gate = None
        braket_gate = None
        vector_gate = None

        super(TwoQubitGate,self).__init__(ibm_gate=ibm_gate, pyquil_gate=pyquil_gate,
                         braket_gate=braket_gate, vector_gate=vector_gate)

    # def _ibm_gate(self):
        
    #     raise NotImplementedError()
    
    # def _pyquil_gate(self):
        
    #     raise NotImplementedError()
        
    # def _braket_gate(self):
        
    #     raise NotImplementedError()
        
    def apply_vector_gate(self, qubit_indices, rotation_angle_obj, input_obj):
        input_obj.apply_rxy(qubit_indices[0], qubit_indices[1], rotation_angle_obj.rotation_angle)
    
class RXZ(TwoQubitGateWithAngle):
    
    def __init__(self):

        ibm_gate = RZXGate
        pyquil_gate = None
        braket_gate = None
        vector_gate = None

        super(TwoQubitGate,self).__init__(ibm_gate=ibm_gate, pyquil_gate=pyquil_gate,
                         braket_gate=braket_gate, vector_gate=vector_gate)

    def apply_ibm_gate(self, 
                       qubit_indices: List[int],
                       rotation_angle_obj: RotationAngle,
                       circuit: qkQuantumCircuit):                       
        
        circuit.rzx(rotation_angle_obj.rotation_angle,
                    qubit_indices[1],
                    qubit_indices[0])
        return circuit
    
    # def _pyquil_gate(self):
        
    #     raise NotImplementedError()
        
    # def _braket_gate(self):
        
    #     raise NotImplementedError()
        
    def apply_vector_gate(self, qubit_indices, rotation_angle_obj, input_obj):
        input_obj.apply_rxz(qubit_indices[0], qubit_indices[1], rotation_angle_obj.rotation_angle)

    
class RYZ(TwoQubitGateWithAngle):

    def __init__(self):

        ibm_gate = None
        pyquil_gate = None
        braket_gate = None
        vector_gate = None

        super(TwoQubitGate,self).__init__(ibm_gate=ibm_gate, pyquil_gate=pyquil_gate,
                         braket_gate=braket_gate, vector_gate=vector_gate)

    # def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
    #     raise NotImplementedError()
    
    # def _pyquil_gate(self):
        
    #     raise NotImplementedError()
        
    # def _braket_gate(self):
        
    #     raise NotImplementedError()
        
    def apply_vector_gate(self, qubit_indices, rotation_angle_obj, input_obj):
        input_obj.apply_ryz(qubit_indices[0], qubit_indices[1], rotation_angle_obj.rotation_angle)


class CPHASE(TwoQubitGateWithAngle):

    def __init__(self):

        ibm_gate = CRZGate
        pyquil_gate = quilgates.CPHASE
        braket_gate = None
        vector_gate = None

        super(TwoQubitGate,self).__init__(ibm_gate=ibm_gate, pyquil_gate=pyquil_gate,
                         braket_gate=braket_gate, vector_gate=vector_gate)

    
    # def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
    #     circuit.crz(self.rotation_angle_obj.rotation_angle,
    #                             self.qubit_1,
    #                             self.qubit_2)
    #     return circuit
    
    # def _pyquil_gate(self, program: quilProgram) -> quilProgram:
        
    #     program += quilgates.CPHASE(self.rotation_angle_obj.rotation_angle,
    #                                 self.qubit_1, self.qubit_2)
    #     return program

    # def _braket_gate(self):
        
    #     raise NotImplementedError()
            
    def _vector_gate(self):
        
        raise NotImplementedError('This gate is not yet supported')

        
class RiSWAP(TwoQubitGateWithAngle):

    def __init__(self):

        ibm_gate = None
        pyquil_gate = quilgates.XY
        braket_gate = None
        vector_gate = None

        super(TwoQubitGate,self).__init__(ibm_gate=ibm_gate, pyquil_gate=pyquil_gate,
                         braket_gate=braket_gate, vector_gate=vector_gate)

    
    # def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
    #     raise NotImplementedError()
        
    # def _pyquil_gate(self, program: quilProgram) -> quilProgram:
        
    #     program += quilgates.XY(self.rotation_angle_obj.rotation_angle,
    #                             self.qubit_1, self.qubit_2)
    #     return program
        
    # def _braket_gate(self):
        
    #     raise NotImplementedError()
        
    def _vector_gate(self):
        
        raise NotImplementedError('This gate is not yet supported')