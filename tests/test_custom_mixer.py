import unittest

import networkx as nx

from openqaoa.qaoa_components.ansatz_constructor import (
    GateMap,
    SWAPGateMap,
    RotationGateMap,
    RXGateMap,
    RZXGateMap,
    RXXGateMap,
    RYYGateMap,
)
from openqaoa.qaoa_components import QAOADescriptor, create_qaoa_variational_params
from openqaoa.backends import create_device
from openqaoa.optimizers import get_optimizer
from openqaoa.backends.qaoa_backend import get_qaoa_backend
from openqaoa.utilities import (
    quick_create_mixer_for_topology,
    X_mixer_hamiltonian,
    XY_mixer_hamiltonian,
)
from openqaoa.problems import MinimumVertexCover
from openqaoa.qaoa_components.ansatz_constructor.gatemaplabel import GateMapType


class TestingCustomMixer(unittest.TestCase):
    def setUp(self):

        nodes = 6
        edge_probability = 0.7
        g = nx.generators.fast_gnp_random_graph(n=nodes, p=edge_probability, seed=34)
        mini_cov = MinimumVertexCover(g, field=1.0, penalty=1.0)
        self.PROBLEM_QUBO = mini_cov.qubo

        # Case with Mixer with 1 AND 2-qubit terms
        self.MANUAL_GATEMAP_LIST, self.MANUAL_COEFFS = [
            RXGateMap(0),
            RZXGateMap(0, 1),
            RZXGateMap(0, 2),
            RZXGateMap(0, 3),
            RZXGateMap(0, 4),
            RXGateMap(4),
            RZXGateMap(0, 5),
            RXXGateMap(1, 2),
        ], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.MANUAL_SEQUENCE = [0, 0, 1, 2, 3, 1, 4, 5]

        zx_gatemap_list, zx_gatemap_coeffs = quick_create_mixer_for_topology(
            RZXGateMap, 6, qubit_connectivity="star"
        )
        xx_gatemap_list, xx_gatemap_coeffs = quick_create_mixer_for_topology(
            RXXGateMap, 6, qubit_connectivity="full"
        )

        zx_gatemap_list.extend(xx_gatemap_list)
        zx_gatemap_coeffs.extend(xx_gatemap_coeffs)

        # Case with Multiple types of 2-qubit gates
        self.COMPLICATED_GATEMAP_LIST, self.COMPLICATED_COEFFS = (
            zx_gatemap_list,
            zx_gatemap_coeffs,
        )
        self.COMPLICATED_SEQUENCE = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
        ]

        self.TESTING_GATEMAPS = [
            [self.MANUAL_GATEMAP_LIST, self.MANUAL_COEFFS, self.MANUAL_SEQUENCE],
            [
                self.COMPLICATED_GATEMAP_LIST,
                self.COMPLICATED_COEFFS,
                self.COMPLICATED_SEQUENCE,
            ],
        ]

    def test_custom_mixer_basic_workflow(self):

        """
        Check that using custom mixers works.
        Custom Mixers are only available in Manual mode.
        """

        for each_gatemap_list, each_gatemap_coeffs, _ in self.TESTING_GATEMAPS:

            custom_mixer_block_gatemap = each_gatemap_list
            custom_mixer_block_coeffs = each_gatemap_coeffs

            qaoa_descriptor = QAOADescriptor(
                self.PROBLEM_QUBO.hamiltonian,
                custom_mixer_block_gatemap,
                p=1,
                mixer_coeffs=custom_mixer_block_coeffs,
            )
            device_local = create_device(location="local", name="qiskit.shot_simulator")
            variate_params = create_qaoa_variational_params(
                qaoa_descriptor, "standard", "rand"
            )
            backend_local = get_qaoa_backend(qaoa_descriptor, device_local, n_shots=500)
            optimizer = get_optimizer(
                backend_local, variate_params, {"method": "cobyla", "maxiter": 10}
            )
            optimizer.optimize()

    def test_mixer_block_properties_sequence(self):

        """
        The custom mixers should have sequences that are correct.
        The sequence values are based on the position of the gate in the block
        relative to other gates of the same qubit count.
        """

        for (
            each_gatemap_list,
            each_gatemap_coeffs,
            correct_seq,
        ) in self.TESTING_GATEMAPS:

            gatemap_list_sequence = []
            one_qubit_count = 0
            two_qubit_count = 0

            for each_gatemap in each_gatemap_list:

                if each_gatemap.gate_label.n_qubits == 1:

                    gatemap_list_sequence.append(one_qubit_count)
                    one_qubit_count += 1

                elif each_gatemap.gate_label.n_qubits == 2:

                    gatemap_list_sequence.append(two_qubit_count)
                    two_qubit_count += 1

            # Test Equality between hand-written and programmatic assignment
            self.assertEqual(gatemap_list_sequence, correct_seq)

            qaoa_descriptor = QAOADescriptor(
                self.PROBLEM_QUBO.hamiltonian,
                each_gatemap_list,
                p=1,
                mixer_coeffs=each_gatemap_coeffs,
            )

            descriptor_mixer_seq = [
                each_mixer_gatemap.gate_label.sequence
                for each_mixer_gatemap in qaoa_descriptor.mixer_block
            ]

            # Test Equality between OQ and hand-written sequence
            self.assertEqual(descriptor_mixer_seq, correct_seq)

    def test_set_block_sequence(self):

        """
        Check that the set block sequence method is correct.
        """

        for (
            each_gatemap_list,
            each_gatemap_coeffs,
            correct_seq,
        ) in self.TESTING_GATEMAPS:

            output_gatemap_list = QAOADescriptor.set_block_sequence(each_gatemap_list)
            output_seq = [
                each_gatemap.gate_label.sequence for each_gatemap in output_gatemap_list
            ]

            self.assertEqual(output_seq, correct_seq)

    def test_set_block_sequence_error_raises(self):

        """
        Check that the set block sequence method raises the right error when the
        wrong type is passed. A TypeError should be raised if the input_gatemap_list
        if not a List of RotationGateMap Objects.
        """

        incorrect_input_iterable = [""], [1], [SWAPGateMap(0, 1)], [RXGateMap]

        for each_iterable in incorrect_input_iterable:
            with self.assertRaises(TypeError):
                QAOADescriptor.set_block_sequence(each_iterable)

    def test_block_setter(self):

        """
        Check that block_setter method correctly maps the sequence and the type
        of the RotationGateMap Objects returned.
        """

        input_enum_type = GateMapType.MIXER

        for (
            each_gatemap_list,
            each_gatemap_coeffs,
            correct_seq,
        ) in self.TESTING_GATEMAPS:

            output_gatemap_list = QAOADescriptor.block_setter(
                each_gatemap_list, input_enum_type
            )

            output_type = [
                each_gatemap.gate_label.type for each_gatemap in output_gatemap_list
            ]
            output_seq = [
                each_gatemap.gate_label.sequence for each_gatemap in output_gatemap_list
            ]

            # sequence and type should be labelled correctly.
            self.assertEqual(output_seq, correct_seq)
            self.assertEqual(
                output_type, [input_enum_type for _ in range(len(correct_seq))]
            )

    def test_block_setter_error_raises(self):

        """
        The block_setter method should raise a ValueError if the input_object is
        not of the type Hamiltonian or List. It should also raise a TypeError if
        the List contains elements that are not of the type RotationGateMap.
        """

        incorrect_input_object = [(), "", 1]

        for each_input in incorrect_input_object:
            with self.assertRaises(ValueError):
                QAOADescriptor.block_setter(
                    input_object=each_input, block_type=GateMapType.MIXER
                )

        incorrect_input_iterable = [
            [SWAPGateMap(0, 1)],
            [RXGateMap],
            [RXGateMap(0), SWAPGateMap(0, 1)],
        ]

        for each_input_iterable in incorrect_input_iterable:
            with self.assertRaises(TypeError):
                QAOADescriptor.block_setter(
                    input_object=each_input_iterable, block_type=GateMapType.MIXER
                )

    def test_block_setter_equivalence_simple(self):

        """
        A Hamiltonian Object and a list of RotationGateMap should have both their
        sequence and type assigned the same if they represent the same gate sequence.
        """

        # 1-Qubit
        test_hamiltonian = X_mixer_hamiltonian(3)
        test_gatemap_list = [RXGateMap(0), RXGateMap(1), RXGateMap(2)]

        return_gatemap_list_h = QAOADescriptor.block_setter(
            input_object=test_hamiltonian, block_type=GateMapType.MIXER
        )
        return_gatemap_list_gl = QAOADescriptor.block_setter(
            input_object=test_gatemap_list, block_type=GateMapType.MIXER
        )

        # Both gatemap list should be equivalent
        self.assertEqual(
            [
                each_gatemap.gate_label.sequence
                for each_gatemap in return_gatemap_list_h
            ],
            [
                each_gatemap.gate_label.sequence
                for each_gatemap in return_gatemap_list_gl
            ],
        )
        self.assertEqual(
            [each_gatemap.gate_label.type for each_gatemap in return_gatemap_list_h],
            [each_gatemap.gate_label.type for each_gatemap in return_gatemap_list_gl],
        )

        # 2-Qubit
        test_hamiltonian = XY_mixer_hamiltonian(3)
        test_gatemap_list = [
            RXXGateMap(0, 1),
            RXXGateMap(0, 2),
            RXXGateMap(1, 2),
            RYYGateMap(0, 1),
            RYYGateMap(0, 2),
            RYYGateMap(1, 2),
        ]

        return_gatemap_list_h = QAOADescriptor.block_setter(
            input_object=test_hamiltonian, block_type=GateMapType.MIXER
        )
        return_gatemap_list_gl = QAOADescriptor.block_setter(
            input_object=test_gatemap_list, block_type=GateMapType.MIXER
        )

        # Both gatemap list should be equivalent
        self.assertEqual(
            [
                each_gatemap.gate_label.sequence
                for each_gatemap in return_gatemap_list_h
            ],
            [
                each_gatemap.gate_label.sequence
                for each_gatemap in return_gatemap_list_gl
            ],
        )
        self.assertEqual(
            [each_gatemap.gate_label.type for each_gatemap in return_gatemap_list_h],
            [each_gatemap.gate_label.type for each_gatemap in return_gatemap_list_gl],
        )


if __name__ == "__main__":
    unittest.main()
