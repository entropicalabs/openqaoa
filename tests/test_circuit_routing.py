# unit testing for circuit routing functionality in OQ
import unittest
import numpy as np
from typing import List, Optional
import pytest

from openqaoa import QAOA, create_device, QUBO
from openqaoa.qaoa_components import (
    create_qaoa_variational_params,
    QAOADescriptor,
    PauliOp,
    Hamiltonian,
)
from openqaoa.utilities import X_mixer_hamiltonian
from openqaoa.problems import NumberPartition, Knapsack, MaximumCut, ShortestPath
from openqaoa_pyquil.backends import DevicePyquil, QAOAPyQuilQPUBackend
from openqaoa.backends.devices_core import DeviceBase
from openqaoa.qaoa_components.ansatz_constructor.gatemap import SWAPGateMap


class TestingQAOAPyquilQVM_QR(unittest.TestCase):

    """Tests pyquil backend compatibility with routing_function.

    For all of these tests, qvm and quilc must be running.
    """

    def test_no_swap(self):
        """
        Tests that QAOADescriptor with a trivial `routing_function` input (with no swaps) returns identical
        results as QAOADescriptor with no `routing_function` input, by comparing output of seeded QVM run.
        Different values of p, arguments, and cost hamiltonian coefficients are tested.

        """

        def routing_function_test1(device, problem_to_solve):
            # tuples ordered from 0,n, both SWAP and ising gates
            gate_list_indices = [[0, 1]]

            # True for SWAP
            swap_mask = [False]

            # {QPU: (0 to n index)}
            initial_physical_to_logical_mapping = {0: 0, 1: 1}

            # 0 to n, permuted
            final_mapping = [0, 1]

            return (
                gate_list_indices,
                swap_mask,
                initial_physical_to_logical_mapping,
                final_mapping,
            )

        args_lst = [
            [np.pi / 8, np.pi / 4],
            [np.pi / 3.5, np.pi / 3],
            [np.pi / 8, np.pi / 4],
            [np.pi / 3.5, np.pi / 3],
            [1, 2, 3, 4],
            [np.pi / 8, np.pi / 4, np.pi / 8, np.pi / 4],
            [1, 2, 3, 4],
            [np.pi / 8, np.pi / 4, np.pi / 8, np.pi / 4],
        ]
        p_lst = [1, 1, 1, 1, 2, 2, 2, 2]
        cost_hamil_lst = [
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 1, 1],
                1,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 1, 1],
                1,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [0.5, 0, 2],
                0.7,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 1.2, 1],
                0,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 1, 1],
                1,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 1, 1],
                1,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [0.5, 0, 2],
                0.7,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 1.2, 1],
                0,
            ),
        ]
        shots = 2
        seed = 1

        device_pyquil = DevicePyquil(
            device_name="2q-qvm", as_qvm=True, execution_timeout=5, compiler_timeout=5
        )
        device_pyquil.quantum_computer.qam.random_seed = seed

        for i in range(len(p_lst)):
            p = p_lst[i]
            args = args_lst[i]
            cost_hamil = cost_hamil_lst[i]

            # With routing
            mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
            qaoa_descriptor = QAOADescriptor(
                cost_hamil, mixer_hamil, routing_function=routing_function_test1, p=p
            )
            variate_params = create_qaoa_variational_params(
                qaoa_descriptor, "standard", "ramp"
            )

            variate_params.update_from_raw(args)
            backend_obj_pyquil = QAOAPyQuilQPUBackend(
                qaoa_descriptor=qaoa_descriptor,
                device=device_pyquil,
                prepend_state=None,
                append_state=None,
                init_hadamard=True,
                cvar_alpha=1,
                n_shots=shots,
            )
            expt_pyquil_w_qr = backend_obj_pyquil.expectation(variate_params)

            # No routing
            mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
            variate_params = create_qaoa_variational_params(
                qaoa_descriptor, "standard", "ramp"
            )

            variate_params.update_from_raw(args)
            backend_obj_pyquil = QAOAPyQuilQPUBackend(
                qaoa_descriptor=qaoa_descriptor,
                device=device_pyquil,
                prepend_state=None,
                append_state=None,
                init_hadamard=True,
                cvar_alpha=1,
                n_shots=shots,
            )
            expt_pyquil_no_qr = backend_obj_pyquil.expectation(variate_params)

            self.assertAlmostEqual(expt_pyquil_w_qr, expt_pyquil_no_qr)

    def test_cancelled_swap(self):
        """
        Tests that QAOADescriptor with a trivial `routing_function` input (with two swaps that cancel each other)
        returns identical results as QAOADescriptor with no `routing_function` input,
        by comparing output of seeded QVM run. Different values of p, arguments,
        and cost hamiltonian coefficients are tested.
        """

        def routing_function_test1(device, problem_to_solve):
            # tuples ordered from 0,n, both SWAP and ising gates
            gate_list_indices = [[0, 1], [0, 1], [0, 1]]

            # True for SWAP
            swap_mask = [True, True, False]

            # {QPU: (0 to n index)}
            initial_physical_to_logical_mapping = {0: 0, 1: 1}

            # 0 to n, permuted
            final_mapping = [0, 1]

            return (
                gate_list_indices,
                swap_mask,
                initial_physical_to_logical_mapping,
                final_mapping,
            )

        args_lst = [
            [np.pi / 8, np.pi / 4],
            [np.pi / 3.5, np.pi / 3],
            [np.pi / 8, np.pi / 4],
            [np.pi / 3.5, np.pi / 3],
            [1, 2, 3, 4],
            [np.pi / 8, np.pi / 4, np.pi / 8, np.pi / 4],
            [1, 2, 3, 4],
            [np.pi / 8, np.pi / 4, np.pi / 8, np.pi / 4],
        ]
        p_lst = [1, 1, 1, 1, 2, 2, 2, 2]
        cost_hamil_lst = [
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 1, 1],
                1,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 1, 1],
                1,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [0.5, 0, 2],
                0.7,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1.5, 1.2, 1],
                0,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 1, 1],
                1,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 1, 1],
                1,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [0.5, 0, 2],
                0.7,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 5.2, 1],
                0,
            ),
        ]
        shots = 3
        seed = 4

        device_pyquil = DevicePyquil(
            device_name="2q-qvm", as_qvm=True, execution_timeout=5, compiler_timeout=5
        )
        device_pyquil.quantum_computer.qam.random_seed = seed

        for i in range(len(p_lst)):
            p = p_lst[i]
            args = args_lst[i]
            cost_hamil = cost_hamil_lst[i]

            # With routing
            mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
            qaoa_descriptor = QAOADescriptor(
                cost_hamil, mixer_hamil, routing_function=routing_function_test1, p=p
            )
            variate_params = create_qaoa_variational_params(
                qaoa_descriptor, "standard", "ramp"
            )

            variate_params.update_from_raw(args)
            backend_obj_pyquil = QAOAPyQuilQPUBackend(
                qaoa_descriptor=qaoa_descriptor,
                device=device_pyquil,
                prepend_state=None,
                append_state=None,
                init_hadamard=True,
                cvar_alpha=1,
                n_shots=shots,
            )
            expt_pyquil_w_qr = backend_obj_pyquil.expectation(variate_params)

            # No routing
            mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
            variate_params = create_qaoa_variational_params(
                qaoa_descriptor, "standard", "ramp"
            )

            variate_params.update_from_raw(args)
            backend_obj_pyquil = QAOAPyQuilQPUBackend(
                qaoa_descriptor=qaoa_descriptor,
                device=device_pyquil,
                prepend_state=None,
                append_state=None,
                init_hadamard=True,
                cvar_alpha=1,
                n_shots=shots,
            )
            expt_pyquil_no_qr = backend_obj_pyquil.expectation(variate_params)

            self.assertAlmostEqual(expt_pyquil_w_qr, expt_pyquil_no_qr)

    def test_simplest_swap(self):
        """
        Tests that QAOADescriptor with a trivial `routing_function` input (with no swaps) returns identical
        results as QAOADescriptor with no `routing_function` input, by comparing output of seeded QVM run.
        Different values of p, arguments, and cost hamiltonian coefficients are tested.

        Note : Even with a fixed seed, insertion of swaps changes measurement statistics.
        Final assertion is therefore only up to a tolerance, chosen by eyeballing results for a chosen seed.

        """

        def routing_function_test1(device, problem_to_solve):
            # tuples ordered from 0,n, both SWAP and ising gates
            gate_list_indices = [[0, 1], [0, 1]]

            # True for SWAP
            swap_mask = [True, False]

            # {QPU: (0 to n index)}
            initial_physical_to_logical_mapping = {0: 0, 1: 1}

            # 0 to n, permuted
            final_mapping = [1, 0]

            return (
                gate_list_indices,
                swap_mask,
                initial_physical_to_logical_mapping,
                final_mapping,
            )

        args_lst = [
            [np.pi / 8, np.pi / 4],
            [np.pi / 3.5, np.pi / 3],
            [np.pi / 8, np.pi / 4],
            [np.pi / 3.5, np.pi / 3],
            [1, 2, 3, 4],
            [np.pi / 8, np.pi / 4, np.pi / 8, np.pi / 4],
            [1, 2, 3, 4],
            [np.pi / 8, np.pi / 4, np.pi / 8, np.pi / 4],
        ]
        p_lst = [1, 1, 1, 1, 2, 2, 2, 2]
        cost_hamil_lst = [
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 1, 1],
                1,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 1, 1],
                1,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [0.5, 0, 2],
                0.7,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1.5, 1.2, 1],
                0,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 1, 1],
                1,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 1, 1],
                1,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [0.5, 0, 2],
                0.7,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 5.2, 1],
                0,
            ),
        ]
        shots = 10
        seed = 4

        device_pyquil = DevicePyquil(
            device_name="2q-qvm", as_qvm=True, execution_timeout=10, compiler_timeout=10
        )
        device_pyquil.quantum_computer.qam.random_seed = seed

        for i in range(len(p_lst)):
            p = p_lst[i]
            args = args_lst[i]
            cost_hamil = cost_hamil_lst[i]

            # With routing
            mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
            qaoa_descriptor = QAOADescriptor(
                cost_hamil, mixer_hamil, routing_function=routing_function_test1, p=p
            )
            variate_params = create_qaoa_variational_params(
                qaoa_descriptor, "standard", "ramp"
            )

            variate_params.update_from_raw(args)
            backend_obj_pyquil = QAOAPyQuilQPUBackend(
                qaoa_descriptor=qaoa_descriptor,
                device=device_pyquil,
                prepend_state=None,
                append_state=None,
                init_hadamard=True,
                cvar_alpha=1,
                n_shots=shots,
            )
            expt_pyquil_w_qr = backend_obj_pyquil.expectation(variate_params)

            # No routing
            mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
            variate_params = create_qaoa_variational_params(
                qaoa_descriptor, "standard", "ramp"
            )

            variate_params.update_from_raw(args)
            backend_obj_pyquil = QAOAPyQuilQPUBackend(
                qaoa_descriptor=qaoa_descriptor,
                device=device_pyquil,
                prepend_state=None,
                append_state=None,
                init_hadamard=True,
                cvar_alpha=1,
                n_shots=shots,
            )
            expt_pyquil_no_qr = backend_obj_pyquil.expectation(variate_params)

            # Note : Even with a fixed seed, insertion of swaps changes measurement statistics.
            # Final assertion is therefore only up to a tolerance, chosen by eyeballing results for a chosen seed.
            self.assertAlmostEqual(expt_pyquil_w_qr, expt_pyquil_no_qr, delta=1)

    def test_different_topologies(self):
        """
        Tests QAOADescriptor with different devices.
        results as QAOADescriptor with no `routing_function` input, by comparing output of seeded QVM run.
        Different values of p, arguments, and cost hamiltonian coefficients are tested.

        """

        def routing_function_test1(device, problem_to_solve):
            # tuples ordered from 0,n, both SWAP and ising gates
            gate_list_indices = [[0, 1], [1, 0], [0, 1]]

            # True for SWAP
            swap_mask = [True, True, False]

            # {QPU: (0 to n index)}
            initial_physical_to_logical_mapping = {0: 0, 1: 1}

            # 0 to n, permuted
            final_mapping = [0, 1]

            return (
                gate_list_indices,
                swap_mask,
                initial_physical_to_logical_mapping,
                final_mapping,
            )

        args_lst = [[np.pi / 8, np.pi / 4], [np.pi / 8, np.pi / 4, 1, 2], [1, 2, 3, 4]]
        p_lst = [1, 2, 2]
        cost_hamil_lst = [
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 1.5, 2],
                0.5,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 1.5, 2],
                0.5,
            ),
            Hamiltonian(
                [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
                [1, 1.5, 2],
                0.5,
            ),
        ]

        device_name_lst = ["2q-qvm", "3q-qvm", "Aspen-M-3"]

        shots = 2
        seed = 1

        for i in range(len(p_lst)):
            p = p_lst[i]
            args = args_lst[i]
            cost_hamil = cost_hamil_lst[i]

            device_pyquil = DevicePyquil(
                device_name=device_name_lst[i],
                as_qvm=True,
                execution_timeout=5,
                compiler_timeout=5,
            )
            device_pyquil.quantum_computer.qam.random_seed = seed

            # With routing
            mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
            qaoa_descriptor = QAOADescriptor(
                cost_hamil, mixer_hamil, routing_function=routing_function_test1, p=p
            )
            variate_params = create_qaoa_variational_params(
                qaoa_descriptor, "standard", "ramp"
            )

            variate_params.update_from_raw(args)
            backend_obj_pyquil = QAOAPyQuilQPUBackend(
                qaoa_descriptor=qaoa_descriptor,
                device=device_pyquil,
                prepend_state=None,
                append_state=None,
                init_hadamard=True,
                cvar_alpha=1,
                n_shots=shots,
            )
            expt_pyquil_w_qr = backend_obj_pyquil.expectation(variate_params)

            # No routing
            mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
            variate_params = create_qaoa_variational_params(
                qaoa_descriptor, "standard", "ramp"
            )

            variate_params.update_from_raw(args)
            backend_obj_pyquil = QAOAPyQuilQPUBackend(
                qaoa_descriptor=qaoa_descriptor,
                device=device_pyquil,
                prepend_state=None,
                append_state=None,
                init_hadamard=True,
                cvar_alpha=1,
                n_shots=shots,
            )
            expt_pyquil_no_qr = backend_obj_pyquil.expectation(variate_params)

            self.assertAlmostEqual(expt_pyquil_w_qr, expt_pyquil_no_qr)


class ExpectedRouting:
    def __init__(
        self,
        qubo,
        device_name,
        device_location,
        qpu_credentials,
        problem_to_solve,
        initial_mapping,
        gate_indices_list,
        swap_mask,
        initial_physical_to_logical_mapping,
        final_logical_qubit_order,
    ):
        self.qubo = qubo
        self.device = create_device(
            name=device_name, location=device_location, **qpu_credentials
        )

        self.device_name = device_name
        self.device_location = device_location
        self.problem_to_solve = problem_to_solve
        self.initial_mapping = initial_mapping

        self.gate_indices_list = gate_indices_list
        self.swap_mask = swap_mask
        self.initial_physical_to_logical_mapping = initial_physical_to_logical_mapping
        self.final_logical_qubit_order = final_logical_qubit_order

    def values_input(self):
        return (
            self.device_name,
            self.device_location,
            self.problem_to_solve,
            self.initial_mapping,
        )

    def values_return(self):
        return (
            self.gate_indices_list,
            self.swap_mask,
            self.initial_physical_to_logical_mapping,
            self.final_logical_qubit_order,
        )


class TestingQubitRouting(unittest.TestCase):
    def setUp(self):
        """
        Test edge cases
        """
        # case qubits device > qubits problem (IBM NAIROBI)
        self.IBM_NAIROBI_KNAPSACK = ExpectedRouting(
            qubo=Knapsack.random_instance(n_items=3, seed=20).qubo,
            device_location="ibmq",
            device_name="ibm_nairobi",
            qpu_credentials={
                "hub": "ibm-q",
                "group": "open",
                "project": "main",
                "as_emulator": True,
            },
            problem_to_solve=[
                (0, 1),
                (2, 3),
                (2, 4),
                (3, 4),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 3),
                (1, 4),
            ],
            initial_mapping=None,
            gate_indices_list=[
                [2, 4],
                [1, 2],
                [3, 5],
                [0, 5],
                [4, 5],
                [4, 5],
                [2, 4],
                [0, 5],
                [2, 4],
                [1, 2],
                [4, 5],
                [0, 5],
                [1, 2],
                [2, 4],
                [2, 4],
                [0, 5],
                [4, 5],
            ],
            swap_mask=[
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                True,
                False,
            ],
            initial_physical_to_logical_mapping={6: 0, 2: 1, 1: 2, 4: 3, 3: 4, 5: 5},
            final_logical_qubit_order=[5, 4, 0, 1, 2, 3],
        )

        # case qubits problem == 2 (IBM OSLO)
        self.IBM_OSLO_QUBO2 = ExpectedRouting(
            qubo=QUBO.from_dict(
                {
                    "terms": [[0, 1], [1]],
                    "weights": [9.800730090617392, 26.220558065741773],
                    "n": 2,
                }
            ),
            device_location="ibmq",
            device_name="ibm_oslo",
            qpu_credentials={
                "hub": "ibm-q",
                "group": "open",
                "project": "main",
                "as_emulator": True,
            },
            problem_to_solve=[(0, 1)],
            initial_mapping=None,
            gate_indices_list=[[0, 2], [1, 2]],
            swap_mask=[True, False],
            initial_physical_to_logical_mapping={2: 0, 3: 1, 1: 2},
            final_logical_qubit_order=[2, 1, 0],
        )

        # case qubits device == qubits problem (IBM OSLO)
        self.IBM_OSLO_NPARTITION = ExpectedRouting(
            qubo=NumberPartition.random_instance(n_numbers=7, seed=2).qubo,
            device_location="ibmq",
            device_name="ibm_oslo",
            qpu_credentials={
                "hub": "ibm-q",
                "group": "open",
                "project": "main",
                "as_emulator": True,
            },
            problem_to_solve=[
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (0, 6),
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (1, 6),
                (2, 3),
                (2, 4),
                (2, 5),
                (2, 6),
                (3, 4),
                (3, 5),
                (3, 6),
                (4, 5),
                (4, 6),
                (5, 6),
            ],
            initial_mapping=None,
            gate_indices_list=[
                [0, 5],
                [1, 5],
                [3, 4],
                [4, 5],
                [2, 3],
                [3, 6],
                [4, 5],
                [0, 5],
                [1, 5],
                [3, 4],
                [3, 4],
                [2, 3],
                [3, 6],
                [4, 5],
                [0, 5],
                [1, 5],
                [2, 3],
                [3, 6],
                [3, 4],
                [3, 4],
                [3, 6],
                [0, 5],
                [1, 5],
                [4, 5],
                [4, 5],
                [1, 5],
                [3, 6],
                [3, 4],
                [3, 4],
                [1, 5],
                [4, 5],
            ],
            swap_mask=[
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                True,
                False,
                True,
                False,
                True,
                True,
                False,
            ],
            initial_physical_to_logical_mapping={
                0: 0,
                2: 1,
                4: 2,
                5: 3,
                3: 4,
                1: 5,
                6: 6,
            },
            final_logical_qubit_order=[3, 5, 1, 0, 6, 2, 4],
        )

        # case qubits device > qubits problem (RIGETTI)
        self.RIGETTI_SHORTESTPATH = ExpectedRouting(
            qubo=ShortestPath.random_instance(
                n_nodes=4, edge_probability=0.9, seed=20
            ).qubo,
            device_location="qcs",
            device_name="9q-square-qvm",
            qpu_credentials={
                "as_qvm": True,
                "execution_timeout": 10,
                "compiler_timeout": 100,
            },
            problem_to_solve=[
                (2, 3),
                (0, 2),
                (2, 4),
                (2, 5),
                (0, 4),
                (4, 5),
                (0, 5),
                (1, 3),
                (3, 5),
                (1, 5),
            ],
            initial_mapping=None,
            gate_indices_list=[
                [4, 5],
                [2, 4],
                [1, 3],
                [3, 5],
                [4, 5],
                [1, 4],
                [2, 4],
                [0, 6],
                [2, 6],
                [2, 6],
                [2, 4],
                [4, 5],
                [2, 4],
                [1, 3],
                [1, 6],
            ],
            swap_mask=[
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
            ],
            initial_physical_to_logical_mapping={
                8: 0,
                4: 1,
                6: 2,
                1: 3,
                3: 4,
                0: 5,
                7: 6,
            },
            final_logical_qubit_order=[2, 3, 6, 1, 4, 5, 0],
        )

        # case qubits device == qubits problem (RIGETTI)
        self.RIGETTI_MAXCUT = ExpectedRouting(
            qubo=MaximumCut.random_instance(
                n_nodes=9, edge_probability=0.9, seed=20
            ).qubo,
            device_location="qcs",
            device_name="9q-square-qvm",
            qpu_credentials={
                "as_qvm": True,
                "execution_timeout": 10,
                "compiler_timeout": 100,
            },
            problem_to_solve=[
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (0, 7),
                (0, 8),
                (1, 2),
                (1, 4),
                (1, 5),
                (1, 6),
                (1, 7),
                (1, 8),
                (2, 3),
                (2, 5),
                (2, 6),
                (2, 7),
                (2, 8),
                (3, 4),
                (3, 5),
                (3, 6),
                (3, 8),
                (4, 7),
                (4, 8),
                (5, 6),
                (5, 7),
                (5, 8),
                (6, 7),
                (6, 8),
                (7, 8),
            ],
            initial_mapping=None,
            gate_indices_list=[
                [5, 6],
                [3, 5],
                [0, 3],
                [1, 7],
                [0, 4],
                [1, 8],
                [4, 7],
                [6, 8],
                [6, 7],
                [1, 2],
                [2, 4],
                [4, 7],
                [0, 4],
                [1, 2],
                [6, 7],
                [5, 6],
                [1, 7],
                [3, 7],
                [4, 7],
                [6, 8],
                [0, 3],
                [0, 4],
                [3, 5],
                [5, 6],
                [6, 8],
                [3, 5],
                [1, 8],
                [1, 2],
                [6, 8],
                [1, 7],
                [3, 7],
                [4, 7],
                [4, 7],
                [6, 7],
                [0, 4],
                [0, 4],
                [2, 4],
            ],
            swap_mask=[
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
            ],
            initial_physical_to_logical_mapping={
                2: 0,
                7: 1,
                8: 2,
                1: 3,
                5: 4,
                0: 5,
                3: 6,
                4: 7,
                6: 8,
            },
            final_logical_qubit_order=[3, 8, 7, 4, 2, 6, 1, 5, 0],
        )

        # create a list of all the cases
        self.list_of_cases = []
        for value in self.__dict__.values():
            if isinstance(value, ExpectedRouting):
                self.list_of_cases.append(value)

    def __routing_function_mock(
        self,
        device: DeviceBase,
        problem_to_solve: List[List[int]],
        initial_mapping: Optional[List[int]] = None,
    ):
        """
        function that imitates the routing function for testing purposes.
        """

        for case in self.list_of_cases:
            if case.values_input() == (
                device.device_name,
                device.device_location,
                problem_to_solve,
                initial_mapping,
            ):
                return case.values_return()

        raise ValueError(
            """The input values are not in the list of expected values, 
                            check the expected cases and the input values. 
                            The input values are: device_name: {}, device_location: {}, problem_to_solve: {}, 
                            initial_mapping: {}""".format(
                device.device_name,
                device.device_location,
                problem_to_solve,
                initial_mapping,
            )
        )

    def __compare_results(self, expected: ExpectedRouting, p: int):
        """
        function that runs qaoa with the routing function and the problem to solve and
        compares the expected and actual results of the routing function.

        :param expected: ExpectedRouting object that contains the input and the expected results of the routing function
        :param p: number of layers of the qaoa circuit
        """
        print(f"\t Testing p={p}")

        device = expected.device
        qubo = expected.qubo

        qaoa = QAOA()
        qaoa.set_device(device)
        qaoa.set_circuit_properties(
            p=p, param_type="standard", init_type="rand", mixer_hamiltonian="x"
        )
        qaoa.set_backend_properties(prepend_state=None, append_state=None)
        qaoa.set_classical_optimizer(
            method="nelder-mead",
            maxiter=1,
            cost_progress=True,
            parameter_log=True,
            optimization_progress=True,
        )
        qaoa.compile(qubo, routing_function=self.__routing_function_mock)
        qaoa.optimize()

        backend, result = qaoa.backend, qaoa.result

        # do the checks

        assert backend.n_qubits == len(
            expected.final_logical_qubit_order
        ), """Number of qubits in the circuit is not equal to the number of qubits given by routing"""

        assert (
            backend.problem_qubits == qubo.n
        ), f"""Number of nodes in problem is not equal to backend.problem_qubits, 
        is '{backend.problem_qubits }' but should be '{qubo.n}'"""

        assert (
            len(list(result.optimized["measurement_outcomes"].keys())[0]) == qubo.n
        ), "The number of qubits in the optimized circuit is not equal to the number of qubits in the problem."

        # check that swap gates are applied in the correct position
        swap_mask_new = []
        for gate in backend.abstract_circuit:
            if not gate.gate_label.n_qubits == 1:
                swap_mask_new.append(isinstance(gate, SWAPGateMap))

        # create the expected swap mask (for p==2 the second swap mask is reversed)
        expected_swap_mask = []
        for i in range(p):
            expected_swap_mask += expected.swap_mask[:: (-1) ** (i % 2)]

        assert (
            swap_mask_new == expected_swap_mask
        ), "Swap gates are not in the correct position"

        # check that the correct qubits are used in the gates
        gate_indices_list_new = []
        for gate in backend.abstract_circuit:
            if not gate.gate_label.n_qubits == 1:
                gate_indices_list_new.append([gate.qubit_1, gate.qubit_2])

        # create the expected swap mask (for p==2 the second swap mask is reversed)
        expected_gate_indices_list = []
        for i in range(p):
            expected_gate_indices_list += expected.gate_indices_list[:: (-1) ** (i % 2)]

        assert (
            gate_indices_list_new == expected_gate_indices_list
        ), "The qubits used in the gates are not correct"

    @pytest.mark.qpu
    def test_qubit_routing(self):
        for i, case in enumerate(self.list_of_cases):
            print("Test case {} out of {}:".format(i + 1, len(self.list_of_cases)))
            self.__compare_results(case, p=i % 4 + 1)
            print("Test passed for case: {}".format(case.values_input()))


if __name__ == "__main__":
    unittest.main()
