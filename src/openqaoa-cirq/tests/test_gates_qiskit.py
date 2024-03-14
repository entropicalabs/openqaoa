import unittest
import numpy as np

from qiskit import QuantumCircuit

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
from openqaoa_qiskit.backends.gates_qiskit import QiskitGateApplicator


class TestingGate(unittest.TestCase):
    def setUp(self):
        self.qiskit_gate_applicator = QiskitGateApplicator()

    def test_ibm_gates_1q(self):
        # Qiskit Gate Applicator
        gate_applicator = self.qiskit_gate_applicator

        # One Qubit Gate Tests
        rotation_angle_obj = RotationAngle(lambda x: x, [], np.pi)

        empty_circuit = QuantumCircuit(1)
        llgate = RY(gate_applicator, 0, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(1)
        test_circuit.ry(np.pi, 0)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

        empty_circuit = QuantumCircuit(1)
        llgate = RX(gate_applicator, 0, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(1)
        test_circuit.rx(np.pi, 0)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

        empty_circuit = QuantumCircuit(1)
        llgate = RZ(gate_applicator, 0, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(1)
        test_circuit.rz(np.pi, 0)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

    def test_ibm_gates_2q(self):
        # Qiskit Gate Applicator
        gate_applicator = self.qiskit_gate_applicator

        # Two Qubit Gate Tests
        empty_circuit = QuantumCircuit(2)
        llgate = CZ(gate_applicator, 0, 1)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(2)
        test_circuit.cz(0, 1)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

        empty_circuit = QuantumCircuit(2)
        llgate = CX(gate_applicator, 0, 1)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(2)
        test_circuit.cx(0, 1)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

    #         empty_circuit = QuantumCircuit(2)
    #         llgate = CX(gate_applicator)
    #         output_circuit = llgate.apply_ibm_gate([0, 1], empty_circuit)

    #         test_circuit = QuantumCircuit(2)
    #         test_circuit.ry(np.pi / 2, 1)
    #         test_circuit.rx(np.pi, 1)
    #         test_circuit.cz(0, 1)
    #         test_circuit.ry(np.pi / 2, 1)
    #         test_circuit.rx(np.pi, 1)

    #         self.assertEqual(
    #             test_circuit.to_instruction().definition,
    #             output_circuit.to_instruction().definition,
    #         )

    def test_ibm_gates_2q_w_gates(self):
        # Qiskit Gate Applicator
        gate_applicator = self.qiskit_gate_applicator

        # Two Qubit Gate with Angles Tests
        rotation_angle_obj = RotationAngle(lambda x: x, [], np.pi)

        empty_circuit = QuantumCircuit(2)
        llgate = RXX(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(2)
        test_circuit.rxx(np.pi, 0, 1)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

        empty_circuit = QuantumCircuit(2)
        llgate = RYY(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(2)
        test_circuit.ryy(np.pi, 0, 1)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

        empty_circuit = QuantumCircuit(2)
        llgate = RZZ(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(2)
        test_circuit.rzz(np.pi, 0, 1)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

        empty_circuit = QuantumCircuit(2)
        llgate = RZX(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(2)
        test_circuit.rzx(np.pi, 0, 1)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

        empty_circuit = QuantumCircuit(2)
        llgate = CPHASE(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(2)
        test_circuit.crz(np.pi, 0, 1)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )


if __name__ == "__main__":
    unittest.main()
