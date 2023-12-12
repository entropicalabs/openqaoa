# unit testing for circuit routing functionality in OQ
import unittest
from typing import List, Optional
import pytest

from openqaoa import QAOA
from openqaoa.backends import create_device
from openqaoa.problems import NumberPartition, QUBO, Knapsack
from openqaoa.backends.devices_core import DeviceBase
from openqaoa.qaoa_components.ansatz_constructor.gatemap import SWAPGateMap


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
    @pytest.mark.qpu
    def setUp(self):
        # case qubits device > qubits problem (IBM KYOTO)
        self.IBM_KYOTO_KNAPSACK = ExpectedRouting(
            qubo=Knapsack.random_instance(n_items=3, seed=20).qubo,
            # qubo = NumberPartition(list(range(1,4))).qubo,
            device_location="ibmq",
            device_name="ibm_kyoto",
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
            # problem_to_solve = [[0,1],[1,2],[2,3]],
            initial_mapping=None,
            gate_indices_list=[
                [0, 1],
                [2, 3],
                [3, 4],
                [1, 2],
                [2, 3],
                [3, 4],
                [1, 2],
                [0, 1],
                [1, 2],
                [1, 2],
                [2, 3],
                [3, 4],
                [2, 3],
                [0, 1],
                [1, 2],
                [2, 3],
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
                True,
                False,
            ],
            initial_physical_to_logical_mapping={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
            final_logical_qubit_order=[1, 2, 4, 0, 3],
        )

        # case qubits problem == 2 (IBM kyoto)
        self.IBM_KYOTO_QUBO2 = ExpectedRouting(
            qubo=QUBO.from_dict(
                {
                    "terms": [[0, 1], [0, 2], [1]],
                    "weights": [1, 9.800730090617392, 26.220558065741773],
                    "n": 3,
                }
            ),
            device_location="ibmq",
            device_name="ibm_kyoto",
            qpu_credentials={
                "hub": "ibm-q",
                "group": "open",
                "project": "main",
                "as_emulator": True,
            },
            problem_to_solve=[(0, 1), (0, 2)],
            initial_mapping=None,
            gate_indices_list=[[0, 1], [1, 2], [0, 1]],
            swap_mask=[False, True, False],
            initial_physical_to_logical_mapping={0: 0, 1: 1, 2: 2},
            final_logical_qubit_order=[0, 2, 1],
        )

        # case qubits device == qubits problem (IBM perth)
        self.IBM_KYOTO_NPARTITION = ExpectedRouting(
            qubo=NumberPartition.random_instance(n_numbers=7, seed=2).qubo,
            device_location="ibmq",
            device_name="ibm_kyoto",
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
                [4, 5],
                [1, 2],
                [0, 1],
                [2, 3],
                [3, 4],
                [5, 6],
                [1, 2],
                [0, 1],
                [2, 3],
                [3, 4],
                [2, 3],
                [4, 5],
                [2, 3],
                [1, 2],
                [4, 5],
                [5, 6],
                [3, 4],
                [0, 1],
                [1, 2],
                [3, 4],
                [5, 6],
                [4, 5],
                [2, 3],
                [1, 2],
                [4, 5],
                [3, 4],
                [1, 2],
                [0, 1],
                [2, 3],
                [3, 4],
                [3, 4],
                [4, 5],
                [5, 6],
                [2, 3],
                [0, 1],
                [1, 2],
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
                True,
                False,
                True,
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
                True,
                True,
                False,
            ],
            initial_physical_to_logical_mapping={
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                4: 4,
                5: 5,
                6: 6,
            },
            final_logical_qubit_order=[5, 4, 1, 6, 3, 0, 2],
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
