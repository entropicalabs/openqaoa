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
from typing import List, Tuple, Optional
import numpy as np

from qiskit import QuantumCircuit as qkQuantumCircuit
from pyquil import Program as quilProgram
from pyquil import gates as quilgates
from .rotationangle import RotationAngle


class LowLevelGate(ABC):
    
    def apply_gate(self, circuit, circuit_library: str):
        
        if circuit_library == 'ibm':
            return self._ibm_gate(circuit)
        elif circuit_library == 'pyquil':
            return self._pyquil_gate(circuit)
        elif circuit_library == 'braket':
            return self._braket_gate(circuit)
        elif circuit_library == 'vector':
            return self._vector_gate(circuit)
    
    @abstractmethod
    def _ibm_gate(self):
        
        pass
    
    @abstractmethod
    def _pyquil_gate(self):
        
        pass
    
    @abstractmethod
    def _braket_gate(self):
        
        pass
        
    @abstractmethod
    def _vector_gate(self):
        
        pass
    
    
class OneQubitGate(LowLevelGate):
    
    def __init__(self, qubit_index: int, rotation_angle_obj: RotationAngle):
        
        self.qubit_1 = qubit_index
        self.gate_label = rotation_angle_obj.pauli_label
        self.rotation_angle_obj = rotation_angle_obj
        
        
class RY(OneQubitGate):
        
    def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
        circuit.ry(self.rotation_angle_obj.rotation_angle, self.qubit_1)
        return circuit
    
    def _pyquil_gate(self, program: quilProgram) -> quilProgram:

        program += quilgates.RY(self.rotation_angle_obj.rotation_angle,
                                self.qubit_1)
        return program
        
    def _braket_gate(self):
        
        raise NotImplementedError()
        
    def _vector_gate(self, input_obj):
        input_obj.apply_ry(self.qubit_1, self.rotation_angle_obj.rotation_angle)


class RX(OneQubitGate):
        
    def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
        circuit.rx(self.rotation_angle_obj.rotation_angle, self.qubit_1)
        return circuit
    
    def _pyquil_gate(self, program: quilProgram) -> quilProgram:
    
        program += quilgates.RX(self.rotation_angle_obj.rotation_angle,
                                self.qubit_1)
        return program
        
    def _braket_gate(self):
        
        raise NotImplementedError()
        
    def _vector_gate(self, input_obj):
        input_obj.apply_rx(self.qubit_1, self.rotation_angle_obj.rotation_angle)
        
        
class RZ(OneQubitGate):
        
    def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
        circuit.rz(self.rotation_angle_obj.rotation_angle, self.qubit_1)
        return circuit
    
    def _pyquil_gate(self, program: quilProgram) -> quilProgram:

        program += quilgates.RZ(self.rotation_angle_obj.rotation_angle,
                                self.qubit_1)
        return program
        
    def _braket_gate(self):
        
        raise NotImplementedError()
        
    def _vector_gate(self, input_obj):
        input_obj.apply_rz(self.qubit_1, self.rotation_angle_obj.rotation_angle)
        
        
class TwoQubitGate(LowLevelGate):
    
    def __init__(self, qubit_indices: List[int]):
        
        self.qubit_1 = qubit_indices[0]
        self.qubit_2 = qubit_indices[1]


class CZ(TwoQubitGate):
    
    def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
        circuit.cz(self.qubit_1, self.qubit_2)
        return circuit
    
    def _pyquil_gate(self, program: quilProgram) -> quilProgram:
        
        program += quilgates.CZ(self.qubit_1, self.qubit_2)
        return program
        
    def _braket_gate(self):
        
        raise NotImplementedError()
        
    def _vector_gate(self):
        
        raise NotImplementedError()

        
class CX(TwoQubitGate):
        
    @property
    def _in_XY(self) -> List[Tuple[LowLevelGate, List]]:
    
        return [(RX, [self.qubit_2, np.pi/2]), 
                (RZ, [self.qubit_1, -np.pi/2]), 
                (RZ, [self.qubit_1, np.pi/2]), 
                (RiSWAP, [[self.qubit_1, self.qubit_2,np.pi]]), 
                (RX, [self.qubit_1, np.pi/2]), 
                (RiSWAP, [[self.qubit_1, self.qubit_2,np.pi]]), 
                (RZ, [self.qubit_2, np.pi/2])]
    
    @property
    def _in_CZ(self) -> List[Tuple[LowLevelGate, List]]:
        
        return [(RY, [self.qubit_2, np.pi/2]), 
                (RX, [self.qubit_2, np.pi]), 
                (CZ, [[self.qubit_1, self.qubit_2]]), 
                (RY, [self.qubit_2, np.pi/2]), 
                (RX, [self.qubit_2, np.pi])]
    
    def apply_gate(self, circuit, circuit_library: str, mode: Optional[str] = 'CX'):
        
        if circuit_library == 'ibm':
            return self._ibm_gate(circuit, mode)
        elif circuit_library == 'pyquil':
            return self._pyquil_gate(circuit, mode)
        elif circuit_library == 'braket':
            return self._braket_gate(circuit, mode)
        elif circuit_library == 'vector':
            return self._vector_gate(circuit)
    
    def _ibm_gate(self, circuit: qkQuantumCircuit, mode: str) -> qkQuantumCircuit:
        
        if mode == 'CX':
            circuit.cx(self.qubit_1, self.qubit_2)
        elif mode == 'CZ':
            for each_object, init_params in self.in_CZ:
                circuit = each_object(*init_params).apply_gate(circuit, 'ibm')
        
        return circuit
    
    def _pyquil_gate(self, program: quilProgram, mode: str) -> quilProgram:
        
        program += quilgates.CNOT(self.qubit_1,self.qubit_2)
        return program
        
    def _braket_gate(self):
        
        raise NotImplementedError()
        
    def _vector_gate(self, input_obj):
        
        raise NotImplementedError()
    

class TwoQubitGateWithAngle(TwoQubitGate):
    
    def __init__(self, qubit_indices: List[int], rotation_angle_obj: RotationAngle):
        
        super().__init__(qubit_indices)
        self.gate_label = rotation_angle_obj.pauli_label
        self.rotation_angle_obj = rotation_angle_obj
        
        
class RXX(TwoQubitGateWithAngle):
    
    def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
        circuit.rxx(self.rotation_angle_obj.rotation_angle,
                                self.qubit_1,
                                self.qubit_2)
        return circuit
        
    def _pyquil_gate(self):
        
        raise NotImplementedError()
        
    def _braket_gate(self):
        
        raise NotImplementedError()
        
    def _vector_gate(self, input_obj):
        input_obj.apply_rxx(self.qubit_1, self.qubit_2, self.rotation_angle_obj.rotation_angle)
        
        
class RYY(TwoQubitGateWithAngle):
    
    def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
        circuit.ryy(self.rotation_angle_obj.rotation_angle,
                                self.qubit_1,
                                self.qubit_2)
        return circuit
        
    def _pyquil_gate(self):
        
        raise NotImplementedError()
        
    def _braket_gate(self):
        
        raise NotImplementedError()
        
    def _vector_gate(self, input_obj):
        input_obj.apply_ryy(self.qubit_1, self.qubit_2, self.rotation_angle_obj.rotation_angle)
    
    
class RZZ(TwoQubitGateWithAngle):
        
    def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
        circuit.rzz(self.rotation_angle_obj.rotation_angle,
                                self.qubit_1,
                                self.qubit_2)
        return circuit
    
    def _pyquil_gate(self) -> quilProgram:
        
        raise NotImplementedError()


    def _braket_gate(self):
        
        raise NotImplementedError()
        
    def _vector_gate(self, input_obj):
        input_obj.apply_rzz(self.qubit_1, self.qubit_2, self.rotation_angle_obj.rotation_angle)
    
    
class RXY(TwoQubitGateWithAngle):
        
    def _ibm_gate(self):
        
        raise NotImplementedError()
    
    def _pyquil_gate(self):
        
        raise NotImplementedError()
        
    def _braket_gate(self):
        
        raise NotImplementedError()
        
    def _vector_gate(self, input_obj):
        input_obj.apply_rxy(self.qubit_1, self.qubit_2, self.rotation_angle_obj.rotation_angle)
    
    
class RXZ(TwoQubitGateWithAngle):
        
    def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
        circuit.rzx(self.rotation_angle_obj.rotation_angle,
                                self.qubit_2,
                                self.qubit_1)
        return circuit
    
    def _pyquil_gate(self):
        
        raise NotImplementedError()
        
    def _braket_gate(self):
        
        raise NotImplementedError()
        
    def _vector_gate(self, input_obj):
        input_obj.apply_rxz(self.qubit_1, self.qubit_2, self.rotation_angle_obj.rotation_angle)
    
    
class RYZ(TwoQubitGateWithAngle):

    def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
        raise NotImplementedError()
    
    def _pyquil_gate(self):
        
        raise NotImplementedError()
        
    def _braket_gate(self):
        
        raise NotImplementedError()
        
    def _vector_gate(self, input_obj):
        input_obj.apply_ryz(self.qubit_1, self.qubit_2, self.rotation_angle_obj.rotation_angle)

class CPHASE(TwoQubitGateWithAngle):
    
    def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
        circuit.crz(self.rotation_angle_obj.rotation_angle,
                                self.qubit_1,
                                self.qubit_2)
        return circuit
    
    def _pyquil_gate(self, program: quilProgram) -> quilProgram:
        
        program += quilgates.CPHASE(self.rotation_angle_obj.rotation_angle,
                                    self.qubit_1, self.qubit_2)
        return program

    def _braket_gate(self):
        
        raise NotImplementedError()
            
    def _vector_gate(self):
        
        raise NotImplementedError()

        
class RiSWAP(TwoQubitGateWithAngle):
    
    def _ibm_gate(self, circuit: qkQuantumCircuit) -> qkQuantumCircuit:
        
        raise NotImplementedError()
        
    def _pyquil_gate(self, program: quilProgram) -> quilProgram:
        
        program += quilgates.XY(self.rotation_angle_obj.rotation_angle,
                                self.qubit_1, self.qubit_2)
        return program
        
    def _braket_gate(self):
        
        raise NotImplementedError()
        
    def _vector_gate(self):
        
        raise NotImplementedError()