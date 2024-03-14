import unittest

from qiskit.circuit import gate as qsk_gate
from qiskit.circuit import controlledgate as qsk_c_gate
from qiskit.circuit.library import standard_gates as qsk_s_gate

# from qiskit.circuit import singleton as qsk_sgt
from qiskit import QuantumCircuit

import openqaoa
import openqaoa.qaoa_components.ansatz_constructor.gates as oq_gate_mod
from openqaoa_qiskit.backends.gates_qiskit import QiskitGateApplicator
from openqaoa.qaoa_components.ansatz_constructor.rotationangle import RotationAngle


class TestQiskitGateApplicator(unittest.TestCase):
    def setUp(self):
        available_gates = [
            each_subclass_gate for each_subclass_gate in qsk_gate.Gate.__subclasses__()
        ]
        available_gates.extend(
            [
                each_subclass_gate
                for each_subclass_gate in qsk_c_gate.ControlledGate.__subclasses__()
            ]
        )
        # available_gates.extend(
        #     [
        #         each_subclass_gate
        #         for each_subclass_gate in qsk_sgt.SingletonGate.__subclasses__()
        #     ]
        # )
        # available_gates.extend(
        #     [
        #         each_subclass_gate
        #         for each_subclass_gate in qsk_sgt.SingletonControlledGate.__subclasses__()
        #     ]
        # )

        available_qiskit_gates_name = [
            each_gate.__name__.lower() for each_gate in available_gates
        ]
        self.available_qiskit_gates = dict(
            zip(available_qiskit_gates_name, available_gates)
        )

        self.oq_available_gates = (
            tuple(
                [each_gate for each_gate in oq_gate_mod.OneQubitGate.__subclasses__()]
            )
            + tuple(
                [
                    each_gate
                    for each_gate in oq_gate_mod.OneQubitRotationGate.__subclasses__()
                ]
            )
            + tuple(
                [each_gate for each_gate in oq_gate_mod.TwoQubitGate.__subclasses__()]
            )
            + tuple(
                [
                    each_gate
                    for each_gate in oq_gate_mod.TwoQubitRotationGate.__subclasses__()
                ]
            )
        )

        # OneQubitGate, OneQubitRotationGate, TwoQubitGate, TwoQubitRotationGate are not acceptable inputs into the function even though they are of the Gate class
        # RXY, RiSWAP and RYZ are not supported by Qiskit library
        self.qiskit_excluded_gates = [
            oq_gate_mod.OneQubitGate,
            oq_gate_mod.OneQubitRotationGate,
            oq_gate_mod.TwoQubitGate,
            oq_gate_mod.TwoQubitRotationGate,
            oq_gate_mod.RXY,
            oq_gate_mod.RiSWAP,
            oq_gate_mod.RYZ,
        ]

    def test_gate_applicator_mapper(self):
        """
        The mapper to the gate applicator should only contain gates that
        are trivially support by the library.
        """

        for each_gate in QiskitGateApplicator.QISKIT_OQ_GATE_MAPPER.values():
            self.assertTrue(
                each_gate.__name__.lower() in self.available_qiskit_gates.keys(),
                "{}, {}".format(each_gate.__name__, self.available_qiskit_gates.keys()),
            )

    def test_gate_selector(self):
        """
        This method should return the Qiskit Gate object based on the input OQ
        Gate object.
        """

        gate_applicator = QiskitGateApplicator()

        oq_gate_list = tuple(
            [each_gate for each_gate in oq_gate_mod.Gate.__subclasses__()]
        )

        oq_gate_list_1q = oq_gate_list + tuple(
            [each_gate for each_gate in oq_gate_mod.OneQubitGate.__subclasses__()]
        )

        for each_gate in oq_gate_list_1q:
            if each_gate not in self.qiskit_excluded_gates:
                returned_gate = gate_applicator.gate_selector(
                    each_gate(applicator=None, qubit_1=None)
                )
                self.assertEqual(
                    self.available_qiskit_gates[returned_gate.__name__.lower()],
                    returned_gate,
                )

        oq_gate_list_1qr = oq_gate_list + tuple(
            [
                each_gate
                for each_gate in oq_gate_mod.OneQubitRotationGate.__subclasses__()
            ]
        )

        for each_gate in oq_gate_list_1qr:
            if each_gate not in self.qiskit_excluded_gates:
                returned_gate = gate_applicator.gate_selector(
                    each_gate(applicator=None, qubit_1=None, rotation_object=None)
                )
                self.assertEqual(
                    self.available_qiskit_gates[returned_gate.__name__.lower()],
                    returned_gate,
                )

        oq_gate_list_2q = oq_gate_list + tuple(
            [each_gate for each_gate in oq_gate_mod.TwoQubitGate.__subclasses__()]
        )

        for each_gate in oq_gate_list_2q:
            if each_gate not in self.qiskit_excluded_gates:
                returned_gate = gate_applicator.gate_selector(
                    each_gate(applicator=None, qubit_1=None, qubit_2=None)
                )
                self.assertEqual(
                    self.available_qiskit_gates[returned_gate.__name__.lower()],
                    returned_gate,
                )

        oq_gate_list_2qr = oq_gate_list + tuple(
            [
                each_gate
                for each_gate in oq_gate_mod.TwoQubitRotationGate.__subclasses__()
            ]
        )

        for each_gate in oq_gate_list_2qr:
            if each_gate not in self.qiskit_excluded_gates:
                returned_gate = gate_applicator.gate_selector(
                    each_gate(
                        applicator=None,
                        qubit_1=None,
                        qubit_2=None,
                        rotation_object=None,
                    )
                )
                self.assertEqual(
                    self.available_qiskit_gates[returned_gate.__name__.lower()],
                    returned_gate,
                )

    def test_not_supported_gates(self):
        """
        If an unsupported Gate object is passed into the apply_gate method,
        a KeyError should be raised.
        The unsupported Gate object does not exist on the mapper.
        """

        unsupported_list = [oq_gate_mod.RXY, oq_gate_mod.RiSWAP, oq_gate_mod.RYZ]
        qubit_1 = 0
        qubit_2 = 1
        rotation_object = None
        circuit = QuantumCircuit(2)

        gate_applicator = QiskitGateApplicator()

        for each_gate in unsupported_list:
            with self.assertRaises(KeyError):
                gate_applicator.apply_gate(
                    each_gate, qubit_1, qubit_2, rotation_object, circuit
                )

    def test_wrong_argument(self):
        """
        If a supported Gate object is passed into the apply_gate method with the
        incorrect set of arguments, a TypeError should be raised.
        """

        input_gates = {
            tuple(oq_gate_mod.OneQubitRotationGate.__subclasses__()): {
                "wrong_args": [0, 1, None],
                "n_qubits": 1,
            },
            tuple(oq_gate_mod.TwoQubitGate.__subclasses__()): {
                "wrong_args": [0, 1, None],
                "n_qubits": 2,
            },
            tuple(oq_gate_mod.TwoQubitRotationGate.__subclasses__()): {
                "wrong_args": [0, 1, None],
                "n_qubits": 2,
            },
        }
        circuit = QuantumCircuit(2)
        gate_applicator = QiskitGateApplicator()

        for each_gate_set, input_arguments in input_gates.items():
            for each_gate in each_gate_set:
                each_gate.n_qubits = input_arguments["n_qubits"]
                if each_gate not in self.qiskit_excluded_gates:
                    with self.assertRaises(TypeError):
                        gate_applicator.apply_gate(
                            each_gate, *input_arguments["wrong_args"], circuit
                        )

    def test_wrong_n_qubits(self):
        """
        If a supported Gate object is passed into the apply_gate method with the
        n_qubits attribute that is not 1 or 2, a ValueError should be raised.
        """

        input_gates = {
            tuple(oq_gate_mod.OneQubitRotationGate.__subclasses__()): {
                "args": [0, None],
                "n_qubits": 3,
            }
        }
        circuit = QuantumCircuit(2)
        gate_applicator = QiskitGateApplicator()

        for each_gate_set, input_arguments in input_gates.items():
            for each_gate in each_gate_set:
                each_gate.n_qubits = input_arguments["n_qubits"]
                if each_gate not in self.qiskit_excluded_gates:
                    with self.assertRaises(ValueError):
                        gate_applicator.apply_gate(
                            each_gate, *input_arguments["args"], circuit
                        )
                    break

    def test_static_methods_1q(self):
        """
        Checks that the static method, apply_1q_rotation_gate, apply the correct
        gate to the circuit object.
        This method is check directly as there are currently no OQ Gate objects
        that call this method through apply_gate.
        """

        gate_applicator = QiskitGateApplicator()
        circuit = QuantumCircuit(1)
        output_circuit = gate_applicator.apply_1q_fixed_gate(
            qsk_s_gate.XGate, 0, circuit
        )

        self.assertEqual(
            [
                each_qubit_ci.index
                for circuit_instruction in output_circuit.data
                for each_qubit_ci in circuit_instruction.qubits
            ],
            [0],
        )
        self.assertEqual(
            [
                circuit_instruction.operation.name
                for circuit_instruction in output_circuit.data
            ],
            ["x"],
        )

    def test_static_methods_1qr(self):
        """
        Checks that the static method, apply_1q_rotation_gate, apply the correct
        gate to the circuit object.
        """

        gate_applicator = QiskitGateApplicator()
        input_angle = 1
        rot_obj = RotationAngle(lambda x: x, None, input_angle)

        each_sub_gate = [
            each_gate for each_gate in oq_gate_mod.OneQubitRotationGate.__subclasses__()
        ]

        for each_gate in each_sub_gate:
            circuit = QuantumCircuit(1)

            output_circuit = gate_applicator.apply_gate(
                each_gate(gate_applicator, 0, rot_obj), 0, rot_obj, circuit
            )

            self.assertEqual(
                [
                    circuit_instruction.operation.params
                    for circuit_instruction in output_circuit.data
                ],
                [[input_angle]],
            )
            self.assertEqual(
                [
                    each_qubit_ci.index
                    for circuit_instruction in output_circuit.data
                    for each_qubit_ci in circuit_instruction.qubits
                ],
                [0],
            )
            self.assertEqual(
                [
                    circuit_instruction.operation.name
                    for circuit_instruction in output_circuit.data
                ],
                [
                    gate_applicator.QISKIT_OQ_GATE_MAPPER[
                        each_gate.__name__
                    ].__name__.lower()[:-4]
                ],
            )

    def test_static_methods_2q(self):
        """
        Checks that the static method, apply_2q_fixed_gate, apply the correct
        gate to the circuit object.
        """

        gate_applicator = QiskitGateApplicator()

        each_sub_gate = [
            each_gate for each_gate in oq_gate_mod.TwoQubitGate.__subclasses__()
        ]

        for each_gate in each_sub_gate:
            if each_gate not in self.qiskit_excluded_gates:
                circuit = QuantumCircuit(2)
                output_circuit = gate_applicator.apply_gate(
                    each_gate(gate_applicator, 0, 1), 0, 1, circuit
                )

                self.assertEqual(
                    [
                        each_qubit_ci.index
                        for circuit_instruction in output_circuit.data
                        for each_qubit_ci in circuit_instruction.qubits
                    ],
                    [0, 1],
                )
                self.assertEqual(
                    [
                        circuit_instruction.operation.name
                        for circuit_instruction in output_circuit.data
                    ],
                    [
                        gate_applicator.QISKIT_OQ_GATE_MAPPER[
                            each_gate.__name__
                        ].__name__.lower()[:-4]
                    ],
                )

    def test_static_methods_2qr(self):
        """
        Checks that the static method, apply_2q_rotation_gate, apply the correct
        gate to the circuit object.
        """

        gate_applicator = QiskitGateApplicator()
        input_angle = 1
        rot_obj = RotationAngle(lambda x: x, None, input_angle)

        each_sub_gate = [
            each_gate for each_gate in oq_gate_mod.TwoQubitRotationGate.__subclasses__()
        ]

        for each_gate in each_sub_gate:
            if each_gate not in self.qiskit_excluded_gates:
                circuit = QuantumCircuit(2)
                output_circuit = gate_applicator.apply_gate(
                    each_gate(gate_applicator, 0, 1, rot_obj), 0, 1, rot_obj, circuit
                )

                self.assertEqual(
                    [
                        circuit_instruction.operation.params
                        for circuit_instruction in output_circuit.data
                    ],
                    [[input_angle]],
                )
                self.assertEqual(
                    [
                        each_qubit_ci.index
                        for circuit_instruction in output_circuit.data
                        for each_qubit_ci in circuit_instruction.qubits
                    ],
                    [0, 1],
                )
                self.assertEqual(
                    [
                        circuit_instruction.operation.name
                        for circuit_instruction in output_circuit.data
                    ],
                    [
                        gate_applicator.QISKIT_OQ_GATE_MAPPER[
                            each_gate.__name__
                        ].__name__.lower()[:-4]
                    ],
                )


if __name__ == "__main__":
    unittest.main()
