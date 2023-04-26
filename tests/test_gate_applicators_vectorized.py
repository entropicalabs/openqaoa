import unittest
from unittest.mock import Mock

import openqaoa
from openqaoa.backends.qaoa_vectorized import QAOAvectorizedBackendSimulator
import openqaoa.qaoa_components.ansatz_constructor.gates as oq_gate_mod
from openqaoa.backends.gates_vectorized import VectorizedGateApplicator
from openqaoa.qaoa_components.ansatz_constructor.rotationangle import RotationAngle


class TestVectorizedGateApplicator(unittest.TestCase):
    def setUp(self):
        available_vectorized_gates_name = [
            each_name
            for each_name in dir(QAOAvectorizedBackendSimulator)
            if "apply" in each_name
        ]
        available_gates = [
            getattr(QAOAvectorizedBackendSimulator, each_name)
            for each_name in available_vectorized_gates_name
        ]

        self.available_vectorized_gates = dict(
            zip(available_vectorized_gates_name, available_gates)
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
        # CZ, CX, RXY and CPhase are not supported by Vectorized library
        self.vectorized_excluded_gates = [
            oq_gate_mod.OneQubitGate,
            oq_gate_mod.OneQubitRotationGate,
            oq_gate_mod.TwoQubitGate,
            oq_gate_mod.TwoQubitRotationGate,
            oq_gate_mod.X,
            oq_gate_mod.CZ,
            oq_gate_mod.CX,
            oq_gate_mod.RXY,
            oq_gate_mod.CPHASE,
        ]

    def test_gate_applicator_mapper(self):
        """
        The mapper to the gate applicator should only contain gates that
        are trivially support by the library.
        """

        for each_gate in VectorizedGateApplicator.VECTORIZED_OQ_GATE_MAPPER(
            QAOAvectorizedBackendSimulator
        ).values():
            self.assertTrue(
                each_gate.__name__.lower() in self.available_vectorized_gates.keys(),
                "{}, {}".format(
                    each_gate.__name__, self.available_vectorized_gates.keys()
                ),
            )

    def test_gate_selector(self):
        """
        This method should apply the Vectorized Gate Function based on the input OQ
        Gate object.
        """

        gate_applicator = VectorizedGateApplicator()

        oq_gate_list = tuple(
            [each_gate for each_gate in oq_gate_mod.Gate.__subclasses__()]
        )

        # Vectorized Gate Function for 1 Qubit Gates without Rotation Angle
        oq_gate_list_1q = oq_gate_list + tuple(
            [each_gate for each_gate in oq_gate_mod.OneQubitGate.__subclasses__()]
        )

        for each_gate in oq_gate_list_1q:
            if each_gate not in self.vectorized_excluded_gates:
                returned_gate = gate_applicator.gate_selector(
                    each_gate(applicator=None, qubit_1=None, vectorized_backend=None),
                    QAOAvectorizedBackendSimulator,
                )
                self.assertEqual(
                    self.available_vectorized_gates[
                        returned_gate._extract_mock_name.lower()
                    ],
                    returned_gate,
                )

        # Vectorized Gate Function for 1 Qubit Gates with Rotation Angle
        oq_gate_list_1qr = oq_gate_list + tuple(
            [
                each_gate
                for each_gate in oq_gate_mod.OneQubitRotationGate.__subclasses__()
            ]
        )

        for each_gate in oq_gate_list_1qr:
            if each_gate not in self.vectorized_excluded_gates:
                returned_gate = gate_applicator.gate_selector(
                    each_gate(applicator=None, qubit_1=None, rotation_object=None),
                    QAOAvectorizedBackendSimulator,
                )
                self.assertEqual(
                    self.available_vectorized_gates[returned_gate.__name__.lower()],
                    returned_gate,
                )

        # Vectorized Gate Function for 2 Qubit Gate without Rotation Angle
        oq_gate_list_2q = oq_gate_list + tuple(
            [each_gate for each_gate in oq_gate_mod.TwoQubitGate.__subclasses__()]
        )

        for each_gate in oq_gate_list_2q:
            if each_gate not in self.vectorized_excluded_gates:
                returned_gate = gate_applicator.gate_selector(
                    each_gate(applicator=None, qubit_1=None, qubit_2=None),
                    QAOAvectorizedBackendSimulator,
                )
                self.assertEqual(
                    self.available_vectorized_gates[returned_gate.__name__.lower()],
                    returned_gate,
                )

        # Vectorized Gate Function for 2 Qubit Gate with Rotation Angle
        oq_gate_list_2qr = oq_gate_list + tuple(
            [
                each_gate
                for each_gate in oq_gate_mod.TwoQubitRotationGate.__subclasses__()
            ]
        )

        for each_gate in oq_gate_list_2qr:
            if each_gate not in self.vectorized_excluded_gates:
                returned_gate = gate_applicator.gate_selector(
                    each_gate(
                        applicator=None,
                        qubit_1=None,
                        qubit_2=None,
                        rotation_object=None,
                    ),
                    QAOAvectorizedBackendSimulator,
                )
                self.assertEqual(
                    self.available_vectorized_gates[returned_gate.__name__.lower()],
                    returned_gate,
                )

    def test_not_supported_gates(self):
        """
        If an unsupported Gate object is passed into the apply_gate method,
        a KeyError should be raised.
        The unsupported Gate object does not exist on the mapper.
        """

        unsupported_list = [
            oq_gate_mod.CZ,
            oq_gate_mod.CX,
            oq_gate_mod.RXY,
            oq_gate_mod.CPHASE,
        ]
        qubit_1 = 0
        qubit_2 = 1
        rotation_object = None

        gate_applicator = VectorizedGateApplicator()

        for each_gate in unsupported_list:
            with self.assertRaises(KeyError):
                gate_applicator.apply_gate(
                    each_gate,
                    qubit_1,
                    qubit_2,
                    rotation_object,
                    QAOAvectorizedBackendSimulator,
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
        gate_applicator = VectorizedGateApplicator()

        for each_gate_set, input_arguments in input_gates.items():
            for each_gate in each_gate_set:
                each_gate.n_qubits = input_arguments["n_qubits"]
                if each_gate not in self.vectorized_excluded_gates:
                    with self.assertRaises(TypeError):
                        gate_applicator.apply_gate(
                            each_gate,
                            *input_arguments["wrong_args"],
                            QAOAvectorizedBackendSimulator
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
        gate_applicator = VectorizedGateApplicator()

        for each_gate_set, input_arguments in input_gates.items():
            for each_gate in each_gate_set:
                each_gate.n_qubits = input_arguments["n_qubits"]
                if each_gate not in self.vectorized_excluded_gates:
                    with self.assertRaises(ValueError):
                        gate_applicator.apply_gate(
                            each_gate,
                            *input_arguments["args"],
                            QAOAvectorizedBackendSimulator
                        )
                    break

    def test_static_methods_1q(self):
        """
        Checks that the static method, apply_1q_rotation_gate, apply the correct
        gate function.
        This method is check directly as there are currently no OQ Gate objects
        that call this method through apply_gate.
        """

        gate_applicator = VectorizedGateApplicator()
        QAOAvectorizedBackendSimulator.apply_x = Mock()
        gate_applicator.apply_1q_fixed_gate(QAOAvectorizedBackendSimulator.apply_x, 0)

        QAOAvectorizedBackendSimulator.apply_x.assert_called_with(0)

    def test_static_methods_1qr(self):
        """
        Checks that the static method, apply_1q_rotation_gate, apply the correct
        gate function.
        """

        gate_applicator = VectorizedGateApplicator()
        input_angle = 1
        rot_obj = RotationAngle(lambda x: x, None, input_angle)

        each_sub_gate = [
            each_gate for each_gate in oq_gate_mod.OneQubitRotationGate.__subclasses__()
        ]

        for each_gate in each_sub_gate:
            mock_vectorized_backend = Mock()
            gate_applicator.apply_gate(
                each_gate(gate_applicator, 0, rot_obj),
                0,
                rot_obj,
                mock_vectorized_backend,
            )

            method_name = VectorizedGateApplicator.VECTORIZED_OQ_GATE_MAPPER(
                mock_vectorized_backend
            )[each_gate.__name__]._extract_mock_name()[5:]
            # Check that the method is called with the right arguments
            getattr(mock_vectorized_backend, method_name).assert_called_with(
                0, rot_obj.rotation_angle
            )

    def test_static_methods_2q(self):
        """
        Checks that the static method, apply_2q_fixed_gate, apply the correct
        gate function.
        """

        gate_applicator = VectorizedGateApplicator()

        each_sub_gate = [
            each_gate for each_gate in oq_gate_mod.TwoQubitGate.__subclasses__()
        ]

        for each_gate in each_sub_gate:
            if each_gate not in self.vectorized_excluded_gates:
                mock_vectorized_backend = Mock()
                gate_applicator.apply_gate(
                    each_gate(gate_applicator, 0, 1), 0, 1, mock_vectorized_backend
                )

                method_name = VectorizedGateApplicator.VECTORIZED_OQ_GATE_MAPPER(
                    mock_vectorized_backend
                )[each_gate.__name__]._extract_mock_name()[5:]
                # Check that the method is called with the right arguments
                getattr(mock_vectorized_backend, method_name).assert_called_with(0, 1)

    def test_static_methods_2qr(self):
        """
        Checks that the static method, apply_2q_rotation_gate, apply the correct
        gate function.
        """

        gate_applicator = VectorizedGateApplicator()
        input_angle = 1
        rot_obj = RotationAngle(lambda x: x, None, input_angle)

        each_sub_gate = [
            each_gate for each_gate in oq_gate_mod.TwoQubitRotationGate.__subclasses__()
        ]

        for each_gate in each_sub_gate:
            if each_gate not in self.vectorized_excluded_gates:
                mock_vectorized_backend = Mock()
                gate_applicator.apply_gate(
                    each_gate(gate_applicator, 0, 1, rot_obj),
                    0,
                    1,
                    rot_obj,
                    mock_vectorized_backend,
                )

                method_name = VectorizedGateApplicator.VECTORIZED_OQ_GATE_MAPPER(
                    mock_vectorized_backend
                )[each_gate.__name__]._extract_mock_name()[5:]
                # Check that the method is called with the right arguments
                getattr(mock_vectorized_backend, method_name).assert_called_with(
                    0, 1, rot_obj.rotation_angle
                )


if __name__ == "__main__":
    unittest.main()
