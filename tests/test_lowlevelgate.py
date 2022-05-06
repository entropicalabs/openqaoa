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

import unittest
import numpy as np
from qiskit import QuantumCircuit
from pyquil import Program, quilbase
from pyquil.gates import RX as p_RX
from pyquil.gates import RY as p_RY
from pyquil.gates import RZ as p_RZ
from pyquil.gates import CZ as p_CZ
from pyquil.gates import CNOT as p_CX
from pyquil.gates import XY as p_XY
from pyquil.gates import CPHASE as p_CPHASE

from openqaoa.qaoa_parameters.pauligate import RY, RX, RZ, CZ, CX, RXX, RYY, RZZ, RXZ, CPHASE, RiSWAP
from openqaoa.qaoa_parameters.rotationangle import RotationAngle

class TestingLowLevelGate(unittest.TestCase):
    
    def test_ibm_gates_1q(self):
        
        # One Qubit Gate Tests
        rotation_angle_obj = RotationAngle(lambda x: x, [], np.pi)
        
        empty_circuit = QuantumCircuit(1)
        llgate = RY(0, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit, 'ibm')

        test_circuit = QuantumCircuit(1)
        test_circuit.ry(np.pi, 0)
        
        self.assertEqual(test_circuit.to_instruction().definition, output_circuit.to_instruction().definition)
        
        empty_circuit = QuantumCircuit(1)
        llgate = RX(0, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit, 'ibm')

        test_circuit = QuantumCircuit(1)
        test_circuit.rx(np.pi, 0)
        
        self.assertEqual(test_circuit.to_instruction().definition, output_circuit.to_instruction().definition)
        
        empty_circuit = QuantumCircuit(1)
        llgate = RZ(0, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit, 'ibm')

        test_circuit = QuantumCircuit(1)
        test_circuit.rz(np.pi, 0)
        
        self.assertEqual(test_circuit.to_instruction().definition, output_circuit.to_instruction().definition)
        
    def test_ibm_gates_2q(self):
            
        # Two Qubit Gate Tests
        empty_circuit = QuantumCircuit(2)
        llgate = CZ([0, 1])
        output_circuit = llgate.apply_gate(empty_circuit, 'ibm')

        test_circuit = QuantumCircuit(2)
        test_circuit.cz(0, 1)
        
        self.assertEqual(test_circuit.to_instruction().definition, output_circuit.to_instruction().definition)
        
        empty_circuit = QuantumCircuit(2)
        llgate = CX([0, 1])
        output_circuit = llgate.apply_gate(empty_circuit, 'ibm', 'CX')

        test_circuit = QuantumCircuit(2)
        test_circuit.cx(0, 1)
        
        self.assertEqual(test_circuit.to_instruction().definition, output_circuit.to_instruction().definition)
        
        empty_circuit = QuantumCircuit(2)
        llgate = CX([0, 1])
        output_circuit = llgate.apply_gate(empty_circuit, 'ibm', 'CZ')

        test_circuit = QuantumCircuit(2)
        test_circuit.ry(np.pi/2, 1)
        test_circuit.rx(np.pi, 1)
        test_circuit.cz(0, 1)
        test_circuit.ry(np.pi/2, 1)
        test_circuit.rx(np.pi, 1)
        
        self.assertEqual(test_circuit.to_instruction().definition, output_circuit.to_instruction().definition)
        
    def test_ibm_gates_2q_w_gates(self):
        
        # Two Qubit Gate with Angles Tests
        rotation_angle_obj = RotationAngle(lambda x: x, [], np.pi)
        
        empty_circuit = QuantumCircuit(2)
        llgate = RXX([0, 1], rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit, 'ibm')

        test_circuit = QuantumCircuit(2)
        test_circuit.rxx(np.pi, 0, 1)
        
        self.assertEqual(test_circuit.to_instruction().definition, output_circuit.to_instruction().definition)
        
        empty_circuit = QuantumCircuit(2)
        llgate = RYY([0, 1], rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit, 'ibm')

        test_circuit = QuantumCircuit(2)
        test_circuit.ryy(np.pi, 0, 1)
        
        self.assertEqual(test_circuit.to_instruction().definition, output_circuit.to_instruction().definition)
        
        empty_circuit = QuantumCircuit(2)
        llgate = RZZ([0, 1], rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit, 'ibm')

        test_circuit = QuantumCircuit(2)
        test_circuit.rzz(np.pi, 0, 1)
        
        self.assertEqual(test_circuit.to_instruction().definition, output_circuit.to_instruction().definition)
        
        empty_circuit = QuantumCircuit(2)
        llgate = RXZ([0, 1], rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit, 'ibm')

        test_circuit = QuantumCircuit(2)
        test_circuit.rzx(np.pi, 1, 0)
        
        self.assertEqual(test_circuit.to_instruction().definition, output_circuit.to_instruction().definition)
        
        empty_circuit = QuantumCircuit(2)
        llgate = CPHASE([0, 1], rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit, 'ibm')

        test_circuit = QuantumCircuit(2)
        test_circuit.crz(np.pi, 0, 1)
        
        self.assertEqual(test_circuit.to_instruction().definition, output_circuit.to_instruction().definition)
        
    def test_pyquil_gates_1q(self):
        
        # One Qubit Gate Tests
        rotation_angle_obj = RotationAngle(lambda x: x, [], np.pi)
        
        empty_program = Program()
        llgate = RY(0, rotation_angle_obj)
        output_program = llgate.apply_gate(empty_program, 'pyquil')
        
        test_program = Program().inst(p_RY(np.pi, 0))
        
        output_gate_names = [instr.name for instr in output_program if type(instr) == quilbase.Gate]
        test_gate_names = [instr.name for instr in test_program if type(instr) == quilbase.Gate]
        self.assertEqual(output_gate_names, test_gate_names)
        
        empty_program = Program()
        llgate = RX(0, rotation_angle_obj)
        output_program = llgate.apply_gate(empty_program, 'pyquil')
        
        test_program = Program().inst(p_RX(np.pi, 0))
        
        output_gate_names = [instr.name for instr in output_program if type(instr) == quilbase.Gate]
        test_gate_names = [instr.name for instr in test_program if type(instr) == quilbase.Gate]
        self.assertEqual(output_gate_names, test_gate_names)
        
        empty_program = Program()
        llgate = RZ(0, rotation_angle_obj)
        output_program = llgate.apply_gate(empty_program, 'pyquil')
        
        test_program = Program().inst(p_RZ(np.pi, 0))
        
        output_gate_names = [instr.name for instr in output_program if type(instr) == quilbase.Gate]
        test_gate_names = [instr.name for instr in test_program if type(instr) == quilbase.Gate]
        self.assertEqual(output_gate_names, test_gate_names)
        
    def test_pyquil_gates_2q(self):
        
        # Two Qubit Gate Tests
        empty_program = Program()
        llgate = CZ([0, 1])
        output_program = llgate.apply_gate(empty_program, 'pyquil')

        test_program = Program().inst(p_CZ(0, 1))
        
        output_gate_names = [instr.name for instr in output_program if type(instr) == quilbase.Gate]
        test_gate_names = [instr.name for instr in test_program if type(instr) == quilbase.Gate]
        self.assertEqual(output_gate_names, test_gate_names)
        
        empty_program = Program()
        llgate = CX([0, 1])
        output_program = llgate.apply_gate(empty_program, 'pyquil')

        test_program = Program().inst(p_CX(0, 1))
        
        output_gate_names = [instr.name for instr in output_program if type(instr) == quilbase.Gate]
        test_gate_names = [instr.name for instr in test_program if type(instr) == quilbase.Gate]
        self.assertEqual(output_gate_names, test_gate_names)
        
    def test_pyquil_gates_2q_w_gates(self):
        
        # Two Qubit Gate with Angles Tests
        rotation_angle_obj = RotationAngle(lambda x: x, [], np.pi)
        
        empty_program = Program()
        llgate = CPHASE([0, 1], rotation_angle_obj)
        output_program = llgate.apply_gate(empty_program, 'pyquil')

        test_program = Program().inst(p_CPHASE(np.pi, 0, 1))
        
        output_gate_names = [instr.name for instr in output_program if type(instr) == quilbase.Gate]
        test_gate_names = [instr.name for instr in test_program if type(instr) == quilbase.Gate]
        self.assertEqual(output_gate_names, test_gate_names)
        
        empty_program = Program()
        llgate = RiSWAP([0, 1], rotation_angle_obj)
        output_program = llgate.apply_gate(empty_program, 'pyquil')

        test_program = Program().inst(p_XY(np.pi, 0, 1))
        
        output_gate_names = [instr.name for instr in output_program if type(instr) == quilbase.Gate]
        test_gate_names = [instr.name for instr in test_program if type(instr) == quilbase.Gate]
        self.assertEqual(output_gate_names, test_gate_names)
    

if __name__ == '__main__':
    unittest.main()