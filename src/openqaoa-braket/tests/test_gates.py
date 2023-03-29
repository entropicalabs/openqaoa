import unittest
import numpy as np

from braket.circuits import gates as braketgates
from braket.circuits import Circuit
from braket.circuits.free_parameter import FreeParameter

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
from openqaoa_braket.backends.gates_braket import BraketGateApplicator


class TestingGate(unittest.TestCase):
    
    def setUp(self):
        
        self.braket_gate_applicator = BraketGateApplicator()
    
    def test_braket_gates_1q(self):
        
        # Braket Gate Applicator
        gate_applicator = self.braket_gate_applicator

        # One Qubit Gate Tests
        rotation_angle_obj = RotationAngle(lambda x: x, [], FreeParameter("test_angle"))
        
        empty_circuit = Circuit()
        llgate = RY(gate_applicator, 0, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)
        output_circuit = output_circuit.make_bound_circuit({"test_angle": np.pi})

        test_circuit = Circuit()
        test_circuit += braketgates.Ry.ry(0, np.pi)

        self.assertEqual(test_circuit, output_circuit)

        empty_circuit = Circuit()
        llgate = RX(gate_applicator, 0, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)
        output_circuit = output_circuit.make_bound_circuit({"test_angle": np.pi})

        test_circuit = Circuit()
        test_circuit += braketgates.Rx.rx(0, np.pi)

        self.assertEqual(test_circuit, output_circuit)

        empty_circuit = Circuit()
        llgate = RZ(gate_applicator, 0, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)
        output_circuit = output_circuit.make_bound_circuit({"test_angle": np.pi})

        test_circuit = Circuit()
        test_circuit += braketgates.Rz.rz(0, np.pi)

        self.assertEqual(test_circuit, output_circuit)

    def test_braket_gates_2q(self):
        
        # Braket Gate Applicator
        gate_applicator = self.braket_gate_applicator

        # Two Qubit Gate Tests
        empty_circuit = Circuit()
        llgate = CZ(gate_applicator, 0, 1)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = Circuit()
        test_circuit += braketgates.CZ.cz(0, 1)

        self.assertEqual(test_circuit, output_circuit)

        empty_circuit = Circuit()
        llgate = CX(gate_applicator, 0, 1)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = Circuit()
        test_circuit += braketgates.CNot.cnot(0, 1)

        self.assertEqual(test_circuit, output_circuit)

    def test_braket_gates_2q_w_gates(self):
        
        # Braket Gate Applicator
        gate_applicator = self.braket_gate_applicator

        # Two Qubit Gate with Angles Tests
        rotation_angle_obj = RotationAngle(lambda x: x, [], FreeParameter("test_angle"))

        empty_circuit = Circuit()
        llgate = RXX(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)
        output_circuit = output_circuit.make_bound_circuit({"test_angle": np.pi})

        test_circuit = Circuit()
        test_circuit += braketgates.XX.xx(0, 1, np.pi)

        self.assertEqual(test_circuit, output_circuit)

        empty_circuit = Circuit()
        llgate = RYY(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)
        output_circuit = output_circuit.make_bound_circuit({"test_angle": np.pi})

        test_circuit = Circuit()
        test_circuit += braketgates.YY.yy(0, 1, np.pi)

        self.assertEqual(test_circuit, output_circuit)

        empty_circuit = Circuit()
        llgate = RZZ(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)
        output_circuit = output_circuit.make_bound_circuit({"test_angle": np.pi})

        test_circuit = Circuit()
        test_circuit += braketgates.ZZ.zz(0, 1, np.pi)

        self.assertEqual(test_circuit, output_circuit)

        empty_circuit = Circuit()
        llgate = CPHASE(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)
        output_circuit = output_circuit.make_bound_circuit({"test_angle": np.pi})

        test_circuit = Circuit()
        test_circuit += braketgates.CPhaseShift.cphaseshift(0, 1, np.pi)

        self.assertEqual(test_circuit, output_circuit)

        empty_circuit = Circuit()
        llgate = RiSWAP(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)
        output_circuit = output_circuit.make_bound_circuit({"test_angle": np.pi})

        test_circuit = Circuit()
        test_circuit += braketgates.XY.xy(0, 1, np.pi)

        self.assertEqual(test_circuit, output_circuit)


if __name__ == "__main__":
    unittest.main()
