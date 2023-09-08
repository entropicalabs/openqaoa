import unittest
import numpy as np

from pyquil import Program, quilbase
from pyquil.gates import RX as p_RX
from pyquil.gates import RY as p_RY
from pyquil.gates import RZ as p_RZ
from pyquil.gates import CZ as p_CZ
from pyquil.gates import CNOT as p_CX
from pyquil.gates import XY as p_XY
from pyquil.gates import CPHASE as p_CPHASE

from openqaoa.qaoa_components.ansatz_constructor.gates import (
    RY,
    RX,
    RZ,
    CZ,
    CX,
    RXX,
    RYY,
    RZZ,
    RZX,
    CPHASE,
    RiSWAP,
)
from openqaoa.qaoa_components.ansatz_constructor.rotationangle import RotationAngle
from openqaoa_pyquil.backends.gates_pyquil import PyquilGateApplicator


class TestingGate(unittest.TestCase):
    def setUp(self):
        self.pyquil_gate_applicator = PyquilGateApplicator()

    def test_pyquil_gates_1q(self):
        # Pyquil Gate Applicator
        gate_applicator = self.pyquil_gate_applicator

        # One Qubit Gate Tests
        rotation_angle_obj = RotationAngle(lambda x: x, [], np.pi)

        empty_program = Program()
        llgate = RY(gate_applicator, 0, rotation_angle_obj)
        output_program = llgate.apply_gate(empty_program)

        test_program = Program().inst(p_RY(np.pi, 0))

        output_gate_names = [
            instr.name for instr in output_program if type(instr) == quilbase.Gate
        ]
        test_gate_names = [
            instr.name for instr in test_program if type(instr) == quilbase.Gate
        ]
        self.assertEqual(output_gate_names, test_gate_names)

        empty_program = Program()
        llgate = RX(gate_applicator, 0, rotation_angle_obj)
        output_program = llgate.apply_gate(empty_program)

        test_program = Program().inst(p_RX(np.pi, 0))

        output_gate_names = [
            instr.name for instr in output_program if type(instr) == quilbase.Gate
        ]
        test_gate_names = [
            instr.name for instr in test_program if type(instr) == quilbase.Gate
        ]
        self.assertEqual(output_gate_names, test_gate_names)

        empty_program = Program()
        llgate = RZ(gate_applicator, 0, rotation_angle_obj)
        output_program = llgate.apply_gate(empty_program)

        test_program = Program().inst(p_RZ(np.pi, 0))

        output_gate_names = [
            instr.name for instr in output_program if type(instr) == quilbase.Gate
        ]
        test_gate_names = [
            instr.name for instr in test_program if type(instr) == quilbase.Gate
        ]
        self.assertEqual(output_gate_names, test_gate_names)

    def test_pyquil_gates_2q(self):
        # Pyquil Gate Applicator
        gate_applicator = self.pyquil_gate_applicator

        # Two Qubit Gate Tests
        empty_program = Program()
        llgate = CZ(gate_applicator, 0, 1)
        output_program = llgate.apply_gate(empty_program)

        test_program = Program().inst(p_CZ(0, 1))

        output_gate_names = [
            instr.name for instr in output_program if type(instr) == quilbase.Gate
        ]
        test_gate_names = [
            instr.name for instr in test_program if type(instr) == quilbase.Gate
        ]
        self.assertEqual(output_gate_names, test_gate_names)

        empty_program = Program()
        llgate = CX(gate_applicator, 0, 1)
        output_program = llgate.apply_gate(empty_program)

        test_program = Program().inst(p_CX(0, 1))

        output_gate_names = [
            instr.name for instr in output_program if type(instr) == quilbase.Gate
        ]
        test_gate_names = [
            instr.name for instr in test_program if type(instr) == quilbase.Gate
        ]
        self.assertEqual(output_gate_names, test_gate_names)

    def test_pyquil_gates_2q_w_gates(self):
        # Pyquil Gate Applicator
        gate_applicator = self.pyquil_gate_applicator

        # Two Qubit Gate with Angles Tests
        rotation_angle_obj = RotationAngle(lambda x: x, [], np.pi)

        empty_program = Program()
        llgate = CPHASE(gate_applicator, 0, 1, rotation_angle_obj)
        output_program = llgate.apply_gate(empty_program)

        test_program = Program().inst(p_CPHASE(np.pi, 0, 1))

        output_gate_names = [
            instr.name for instr in output_program if type(instr) == quilbase.Gate
        ]
        test_gate_names = [
            instr.name for instr in test_program if type(instr) == quilbase.Gate
        ]
        self.assertEqual(output_gate_names, test_gate_names)

        empty_program = Program()
        llgate = RiSWAP(gate_applicator, 0, 1, rotation_angle_obj)
        output_program = llgate.apply_gate(empty_program)

        test_program = Program().inst(p_XY(np.pi, 0, 1))

        output_gate_names = [
            instr.name for instr in output_program if type(instr) == quilbase.Gate
        ]
        test_gate_names = [
            instr.name for instr in test_program if type(instr) == quilbase.Gate
        ]
        self.assertEqual(output_gate_names, test_gate_names)


if __name__ == "__main__":
    unittest.main()
