import json, os, gzip
import unittest
import networkx as nw
import numpy as np
import datetime

from qiskit.providers.fake_provider import FakeVigo
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import QasmSimulator

from openqaoa import QAOA, RQAOA
from openqaoa.algorithms import QAOAResult, RQAOAResult
from openqaoa.algorithms.baseworkflow import Workflow
from openqaoa.utilities import X_mixer_hamiltonian, XY_mixer_hamiltonian, is_valid_uuid
from openqaoa.algorithms.workflow_properties import (
    BackendProperties,
    ClassicalOptimizer,
    CircuitProperties,
)
from openqaoa.algorithms.rqaoa.rqaoa_workflow_properties import RqaoaParameters
from openqaoa.backends import create_device, DeviceLocal
from openqaoa.backends.devices_core import SUPPORTED_LOCAL_SIMULATORS
from openqaoa.qaoa_components import (
    Hamiltonian,
    QAOADescriptor,
    QAOAVariationalStandardParams,
    QAOAVariationalStandardWithBiasParams,
    QAOAVariationalExtendedParams,
    QAOAVariationalFourierParams,
    QAOAVariationalFourierExtendedParams,
    QAOAVariationalFourierWithBiasParams,
)
from openqaoa.backends import QAOAvectorizedBackendSimulator
from openqaoa.backends.basebackend import QAOABaseBackendStatevector
from openqaoa.problems import MinimumVertexCover, QUBO, MaximumCut
from openqaoa.optimizers.qaoa_optimizer import available_optimizers
from openqaoa.optimizers.training_vqa import (
    ScipyOptimizer,
    CustomScipyGradientOptimizer,
    PennyLaneOptimizer,
)
from openqaoa_qiskit.backends import DeviceQiskit
from openqaoa_qiskit.backends import (
    QAOAQiskitBackendShotBasedSimulator,
    QAOAQiskitBackendStatevecSimulator,
)


ALLOWED_LOCAL_SIMUALTORS = SUPPORTED_LOCAL_SIMULATORS
LOCAL_DEVICES = ALLOWED_LOCAL_SIMUALTORS + ["6q-qvm", "Aspen-11"]


def _compare_qaoa_results(dict_old, dict_new):
    for key in dict_old.keys():
        if key == "cost_hamiltonian":  ## CHECK WHAT DO WITH THIS
            pass
        elif key == "_QAOAResult__type_backend":
            if issubclass(dict_old[key], QAOABaseBackendStatevector):
                assert (
                    dict_new[key] == QAOABaseBackendStatevector
                ), "Type of backend is not correct."
            else:
                assert dict_new[key] == "", "Type of backend should be empty string."
        elif key == "optimized":
            for key2 in dict_old[key].keys():
                if key2 == "measurement_outcomes":
                    assert np.all(
                        dict_old[key][key2] == dict_new[key][key2]
                    ), "Optimized params are not the same."
                else:
                    assert (
                        dict_old[key][key2] == dict_new[key][key2]
                    ), "Optimized params are not the same."
        elif key == "intermediate":
            for key2 in dict_old[key].keys():
                if key2 == "measurement_outcomes":
                    for step in range(len(dict_old[key][key2])):
                        assert np.all(
                            dict_old[key][key2][step] == dict_new[key][key2][step]
                        ), "Intermediate params are not the same."
                else:
                    assert (
                        dict_old[key][key2] == dict_new[key][key2]
                    ), "Intermediate params are not the same."
        else:
            assert dict_old[key] == dict_new[key], f"'{key}' is not the same"


def _test_keys_in_dict(obj, expected_keys):
    """
    private function to test the keys. It recursively tests the keys of the nested dictionaries, or lists of dictionaries
    """

    if isinstance(obj, dict):
        for key in obj:
            if key in expected_keys.keys():
                expected_keys[key] = True

            if isinstance(obj[key], dict):
                _test_keys_in_dict(obj[key], expected_keys)
            elif isinstance(obj[key], list):
                for item in obj[key]:
                    _test_keys_in_dict(item, expected_keys)
    elif isinstance(obj, list):
        for item in obj:
            _test_keys_in_dict(item, expected_keys)


class TestingVanillaQAOA(unittest.TestCase):

    """
    Unit test based testing of the QAOA workflow class
    """

    def test_set_device_cloud(self):
        """ "
        Check that all QPU-provider related devices are correctly initialised
        """
        q = QAOA()
        q.set_device(
            create_device(
                "ibmq",
                name="place_holder",
                **{"hub": "***", "group": "***", "project": "***"},
            )
        )
        assert type(q.device) == DeviceQiskit
        assert q.device.device_name == "place_holder"
        assert q.device.device_location == "ibmq"

    def test_compile_before_optimise(self):
        """
        Assert that compilation has to be called before optimisation
        """
        g = nw.circulant_graph(6, [1])
        # vc = MinimumVertexCover(g, field =1.0, penalty=10).qubo

        q = QAOA()
        q.set_classical_optimizer(optimization_progress=True)

        self.assertRaises(ValueError, lambda: q.optimize())

    def test_set_backend_properties_check_backend_qiskit_qasm(self):
        """
        Check if the backend returned by set_backend_properties is correct
        Based on the input device. For qiskit qasm simulator.
        Also Checks if defaults from workflows are used in the backend.
        """

        nodes = 6
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes, p=edge_probability)
        problem = MinimumVertexCover(g, field=1.0, penalty=10)
        qubo_problem = problem.qubo

        q = QAOA()
        q.set_device(create_device(location="local", name="qiskit.qasm_simulator"))
        q.compile(problem=qubo_problem)

        self.assertEqual(type(q.backend), QAOAQiskitBackendShotBasedSimulator)

        self.assertEqual(q.backend.init_hadamard, True)
        self.assertEqual(q.backend.prepend_state, None)
        self.assertEqual(q.backend.append_state, None)
        self.assertEqual(q.backend.cvar_alpha, 1)
        self.assertEqual(q.backend.n_shots, 100)

    def test_set_backend_properties_check_backend_qiskit_statevector(self):
        """
        Check if the backend returned by set_backend_properties is correct
        Based on the input device. For qiskit statevector simulator.
        Also Checks if defaults from workflows are used in the backend.
        """

        nodes = 6
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes, p=edge_probability)
        problem = MinimumVertexCover(g, field=1.0, penalty=10)
        qubo_problem = problem.qubo

        q = QAOA()
        q.set_device(
            create_device(location="local", name="qiskit.statevector_simulator")
        )
        q.compile(problem=qubo_problem)

        self.assertEqual(type(q.backend), QAOAQiskitBackendStatevecSimulator)

        self.assertEqual(q.backend.init_hadamard, True)
        self.assertEqual(q.backend.prepend_state, None)
        self.assertEqual(q.backend.append_state, None)
        self.assertEqual(q.backend.cvar_alpha, 1)

        self.assertRaises(AttributeError, lambda: q.backend.n_shots)

    def test_qaoa_asdict_with_noise(self):
        """test to check that we can serialize a QAOA object with noise"""
        device_backend = FakeVigo()
        device = QasmSimulator.from_backend(device_backend)
        noise_model = NoiseModel.from_backend(device)
        q_noisy_shot = QAOA()

        # device
        qiskit_noisy_shot = create_device(
            location="local", name="qiskit.qasm_simulator"
        )
        q_noisy_shot.set_device(qiskit_noisy_shot)
        # circuit properties
        q_noisy_shot.set_circuit_properties(
            p=2, param_type="standard", init_type="rand", mixer_hamiltonian="x"
        )
        # backend properties
        q_noisy_shot.set_backend_properties(n_shots=200, noise_model=noise_model)
        # classical optimizer properties
        q_noisy_shot.set_classical_optimizer(
            method="COBYLA", maxiter=200, cost_progress=True, parameter_log=True
        )
        q_noisy_shot.compile(QUBO.random_instance(n=8))
        q_noisy_shot.optimize()
        q_noisy_shot.asdict()

    def test_qaoa_from_dict_and_load(self):
        """
        test loading the QAOA object from a dictionary
        methods: from_dict, load, loads
        """

        # problem
        maxcut_qubo = MaximumCut(
            nw.generators.fast_gnp_random_graph(n=6, p=0.6, seed=42)
        ).qubo

        # run rqaoa with different devices, and save the objcets in a list
        qaoas = []
        for device in [create_device(location="local", name="qiskit.shot_simulator")]:
            q = QAOA()
            q.set_device(device)
            q.set_circuit_properties(
                p=1, param_type="extended", init_type="rand", mixer_hamiltonian="x"
            )
            q.set_backend_properties(n_shots=50)
            q.set_classical_optimizer(maxiter=10, optimization_progress=True)
            q.set_exp_tags({"add_tag": "test"})
            q.set_header(
                project_id="8353185c-b175-4eda-9628-b4e58cb0e41b",
                description="test",
                run_by="raul",
                provider="-",
                target="-",
                cloud="local",
                client="-",
            )

            # test that you can convert the rqaoa object to a dictionary and then load it before optimization
            _ = QAOA.from_dict(q.asdict())
            _.compile(maxcut_qubo)
            _.optimize()
            assert isinstance(
                _, QAOA
            ), "The object loaded from a dictionary is not an RQAOA object."

            # compile and optimize the original rqaoa object
            q.compile(maxcut_qubo)
            q.optimize()

            qaoas.append(q)

        # for each rqaoa object, create a new rqaoa object from dict, json string, json file, and compressed json file and compare them with the original object
        for q in qaoas:
            new_q_list = []

            # get new qaoa from dict
            new_q_list.append(QAOA.from_dict(q.asdict()))
            # get new qaoa from json string
            new_q_list.append(QAOA.loads(q.dumps()))
            # get new qaoa from json file
            q.dump("test.json", prepend_id=False)
            new_q_list.append(QAOA.load("test.json"))
            os.remove("test.json")  # delete file test.json
            # get new qaoa from compressed json file
            q.dump("test.json", prepend_id=False, compresslevel=3)
            new_q_list.append(QAOA.load("test.json.gz"))
            os.remove("test.json.gz")  # delete file test.json

            for new_q in new_q_list:
                # check that the new object is an QAOA object
                assert isinstance(new_q, QAOA), "new_r is not an RQAOA object"

                # check that the attributes of the new object are of the correct type
                attributes_types = [
                    ("header", dict),
                    ("exp_tags", dict),
                    ("problem", QUBO),
                    ("result", QAOAResult),
                    ("backend_properties", BackendProperties),
                    ("classical_optimizer", ClassicalOptimizer),
                    ("circuit_properties", CircuitProperties),
                ]
                for attribute, type_ in attributes_types:
                    assert isinstance(
                        getattr(new_q, attribute), type_
                    ), f"attribute {attribute} is not type {type_}"

                # get the two objects (old and new) as dictionaries
                q_asdict = q.asdict()
                new_q_asdict = new_q.asdict()

                # compare the two dictionaries
                for key, value in q_asdict.items():
                    if key == "header":
                        assert value == new_q_asdict[key], "Header is not the same"

                    elif key == "data":
                        for key2, value2 in value.items():
                            if key2 == "input_parameters":
                                # pop key device since it is not returned completely when using asdict/dump(s)
                                value2.pop("device")
                                new_q_asdict[key][key2].pop("device")
                            if key2 == "result":
                                _compare_qaoa_results(value2, new_q_asdict[key][key2])
                            else:
                                assert (
                                    value2 == new_q_asdict[key][key2]
                                ), "{} not the same".format(key2)

                # compile and optimize the new qaoa, to check if everything is working
                new_q.compile(maxcut_qubo)
                new_q.optimize()

        # check that the RQAOA.from_dict method raises an error when using a QAOA dictionary
        error = False
        try:
            RQAOA.from_dict(q.asdict())
        except Exception:
            error = True
        assert (
            error
        ), "RQAOA.from_dict should raise an error when using a QAOA dictionary"


class TestingRQAOA(unittest.TestCase):
    """
    Unit test based testing of the RQAOA workflow class
    """

    def _test_default_values(self, x):
        """
        General function to check default values of rqaoa and qaoa
        """

        # circuit_properties
        cp = x.circuit_properties
        assert cp.param_type == "standard"
        assert cp.init_type == "ramp"
        assert cp.p == 1
        assert cp.q == None
        assert cp.mixer_hamiltonian == "x"

        # device
        d = x.device
        assert d.device_location == "local"
        assert d.device_name == "vectorized"

    def test_rqaoa_default_values(self):
        """
        Tests all default values are correct
        """
        r = RQAOA()

        assert r.rqaoa_parameters.rqaoa_type == "custom"
        assert r.rqaoa_parameters.n_cutoff == 5
        assert r.rqaoa_parameters.n_max == 1
        assert r.rqaoa_parameters.steps == 1
        assert r.rqaoa_parameters.original_hamiltonian == None
        assert r.rqaoa_parameters.counter == 0

        self._test_default_values(r)

    def test_rqaoa_compile_and_qoao_default_values(self):
        """
        Test creation of the qaoa object and its default values
        """
        r = RQAOA()
        r.compile(QUBO.random_instance(n=7))

        self._test_default_values(r._RQAOA__q)

    def __run_rqaoa(
        self,
        type,
        problem=None,
        n_cutoff=5,
        eliminations=1,
        p=1,
        param_type="standard",
        mixer="x",
        method="cobyla",
        maxiter=15,
        name_device="qiskit.statevector_simulator",
        return_object=False,
    ):
        if problem == None:
            problem = MaximumCut.random_instance(
                n_nodes=8, edge_probability=0.5, seed=2
            ).qubo

        r = RQAOA()
        qiskit_device = create_device(location="local", name=name_device)
        r.set_device(qiskit_device)
        if type == "adaptive":
            r.set_rqaoa_parameters(
                n_cutoff=n_cutoff, n_max=eliminations, rqaoa_type=type
            )
        else:
            r.set_rqaoa_parameters(
                n_cutoff=n_cutoff, steps=eliminations, rqaoa_type=type
            )
        r.set_circuit_properties(p=p, param_type=param_type, mixer_hamiltonian=mixer)
        r.set_backend_properties(prepend_state=None, append_state=None)
        r.set_classical_optimizer(
            method=method,
            maxiter=maxiter,
            optimization_progress=True,
            cost_progress=True,
            parameter_log=True,
        )
        r.set_header(
            project_id="8353185c-b175-4eda-9628-b4e58cb0e41b",
            description="header",
            run_by="raul",
            provider="-",
            target="-",
            cloud="local",
            client="-",
        )
        r.set_exp_tags(tags={"tag1": "value1", "tag2": "value2"})
        r.compile(problem)
        r.optimize()

        if return_object:
            return r
        return r.result.get_solution()

    def test_rqaoa_optimize_multiple_times(self):
        """
        Test that the rqaoa can not be optimized multiple times
        """
        graph = nw.circulant_graph(10, [1])
        problem = MinimumVertexCover(graph, field=1, penalty=10).qubo

        r = RQAOA()
        exception = False
        try:
            r.optimize()
        except:
            exception = True

        assert exception, "RQAOA should not be able to optimize without compilation"

        r.compile(problem)
        r.optimize()
        exception = False
        try:
            r.optimize()
        except:
            exception = True

        assert (
            exception
        ), "RQAOA should not be able to optimize twice without compilation"

    def test_example_1_adaptive_custom(self):
        # Number of qubits
        n_qubits = 12

        # Elimination schemes
        Nmax = [1, 2, 3, 4]
        schedules = [1, [1, 2, 1, 2, 7]]
        n_cutoff = 5

        # Edges and weights of the graph
        pair_edges = [(i, i + 1) for i in range(n_qubits - 1)] + [(0, n_qubits - 1)]
        self_edges = [(i,) for i in range(n_qubits)]
        pair_weights = [1 for _ in range(len(pair_edges))]  # All weights equal to 1
        self_weights = [10 ** (-4) for _ in range(len(self_edges))]

        edges = pair_edges + self_edges
        weights = pair_weights + self_weights
        problem = QUBO(n_qubits, edges, weights)

        # list of solutions of rqaoa
        solutions = []

        # run RQAOA and append solution
        for nmax in Nmax:
            solutions.append(self.__run_rqaoa("adaptive", problem, n_cutoff, nmax))

        for schedule in schedules:
            solutions.append(self.__run_rqaoa("custom", problem, n_cutoff, schedule))

        # Correct solution
        exact_soutions = {"101010101010": -12, "010101010101": -12}

        # Check computed solutions are among the correct ones
        for solution in solutions:
            for key in solution:
                assert solution[key] == exact_soutions[key]

    def test_example_2_adaptive_custom(self):
        # Elimination scheme
        n_cutoff = 3

        # Define problem instance (Ring graph 10 qubits)
        graph = nw.circulant_graph(10, [1])
        problem = MinimumVertexCover(graph, field=1, penalty=10).qubo

        # run RQAOA and append solution in list
        solutions = []
        solutions.append(self.__run_rqaoa("adaptive", problem, n_cutoff))
        solutions.append(self.__run_rqaoa("custom", problem, n_cutoff))

        # Correct solution
        exact_soutions = {"1010101010": 5, "0101010101": 5}

        # Check computed solutions are among the correct ones
        for solution in solutions:
            for key in solution:
                assert solution[key] == exact_soutions[key]

    def test_example_3_adaptive_custom(self):
        # Elimination scheme
        step = 2
        nmax = 4
        n_cutoff = 3

        # Define problem instance (Ring graph 10 qubits)
        graph = nw.complete_graph(10)
        problem = MinimumVertexCover(graph, field=1, penalty=10).qubo

        # run RQAOA and append solution in list
        solutions = []
        solutions.append(self.__run_rqaoa("adaptive", problem, n_cutoff, nmax))
        solutions.append(self.__run_rqaoa("custom", problem, n_cutoff, step))

        # Correct solution
        exact_soutions = {
            "0111111111": 9,
            "1011111111": 9,
            "1101111111": 9,
            "1110111111": 9,
            "1111011111": 9,
            "1111101111": 9,
            "1111110111": 9,
            "1111111011": 9,
            "1111111101": 9,
            "1111111110": 9,
        }

        # Check computed solutions are among the correct ones
        for solution in solutions:
            for key in solution:
                assert solution[key] == exact_soutions[key]

    def test_example_4_adaptive_custom(self):
        # Number of qubits
        n_qubits = 10

        # Elimination schemes
        Nmax = [1, 2, 3, 4]
        schedules = [1, 2, 3]
        n_cutoff = 3

        # Edges and weights of the graph
        edges = [(i, j) for j in range(n_qubits) for i in range(j)]
        weights = [1 for _ in range(len(edges))]

        problem = QUBO(n_qubits, edges, weights)

        # list of solutions of rqaoa
        solutions = []

        # run RQAOA and append solution
        for nmax in Nmax:
            solutions.append(self.__run_rqaoa("adaptive", problem, n_cutoff, nmax))

        for schedule in schedules:
            solutions.append(self.__run_rqaoa("custom", problem, n_cutoff, schedule))

        # Correct solution
        exact_states = [
            "1111100000",
            "1111010000",
            "1110110000",
            "1101110000",
            "1011110000",
            "0111110000",
            "1111001000",
            "1110101000",
            "1101101000",
            "1011101000",
            "0111101000",
            "1110011000",
            "1101011000",
            "1011011000",
            "0111011000",
            "1100111000",
            "1010111000",
            "0110111000",
            "1001111000",
            "0101111000",
            "0011111000",
            "1111000100",
            "1110100100",
            "1101100100",
            "1011100100",
            "0111100100",
            "1110010100",
            "1101010100",
            "1011010100",
            "0111010100",
            "1100110100",
            "1010110100",
            "0110110100",
            "1001110100",
            "0101110100",
            "0011110100",
            "1110001100",
            "1101001100",
            "1011001100",
            "0111001100",
            "1100101100",
            "1010101100",
            "0110101100",
            "1001101100",
            "0101101100",
            "0011101100",
            "1100011100",
            "1010011100",
            "0110011100",
            "1001011100",
            "0101011100",
            "0011011100",
            "1000111100",
            "0100111100",
            "0010111100",
            "0001111100",
            "1111000010",
            "1110100010",
            "1101100010",
            "1011100010",
            "0111100010",
            "1110010010",
            "1101010010",
            "1011010010",
            "0111010010",
            "1100110010",
            "1010110010",
            "0110110010",
            "1001110010",
            "0101110010",
            "0011110010",
            "1110001010",
            "1101001010",
            "1011001010",
            "0111001010",
            "1100101010",
            "1010101010",
            "0110101010",
            "1001101010",
            "0101101010",
            "0011101010",
            "1100011010",
            "1010011010",
            "0110011010",
            "1001011010",
            "0101011010",
            "0011011010",
            "1000111010",
            "0100111010",
            "0010111010",
            "0001111010",
            "1110000110",
            "1101000110",
            "1011000110",
            "0111000110",
            "1100100110",
            "1010100110",
            "0110100110",
            "1001100110",
            "0101100110",
            "0011100110",
            "1100010110",
            "1010010110",
            "0110010110",
            "1001010110",
            "0101010110",
            "0011010110",
            "1000110110",
            "0100110110",
            "0010110110",
            "0001110110",
            "1100001110",
            "1010001110",
            "0110001110",
            "1001001110",
            "0101001110",
            "0011001110",
            "1000101110",
            "0100101110",
            "0010101110",
            "0001101110",
            "1000011110",
            "0100011110",
            "0010011110",
            "0001011110",
            "0000111110",
            "1111000001",
            "1110100001",
            "1101100001",
            "1011100001",
            "0111100001",
            "1110010001",
            "1101010001",
            "1011010001",
            "0111010001",
            "1100110001",
            "1010110001",
            "0110110001",
            "1001110001",
            "0101110001",
            "0011110001",
            "1110001001",
            "1101001001",
            "1011001001",
            "0111001001",
            "1100101001",
            "1010101001",
            "0110101001",
            "1001101001",
            "0101101001",
            "0011101001",
            "1100011001",
            "1010011001",
            "0110011001",
            "1001011001",
            "0101011001",
            "0011011001",
            "1000111001",
            "0100111001",
            "0010111001",
            "0001111001",
            "1110000101",
            "1101000101",
            "1011000101",
            "0111000101",
            "1100100101",
            "1010100101",
            "0110100101",
            "1001100101",
            "0101100101",
            "0011100101",
            "1100010101",
            "1010010101",
            "0110010101",
            "1001010101",
            "0101010101",
            "0011010101",
            "1000110101",
            "0100110101",
            "0010110101",
            "0001110101",
            "1100001101",
            "1010001101",
            "0110001101",
            "1001001101",
            "0101001101",
            "0011001101",
            "1000101101",
            "0100101101",
            "0010101101",
            "0001101101",
            "1000011101",
            "0100011101",
            "0010011101",
            "0001011101",
            "0000111101",
            "1110000011",
            "1101000011",
            "1011000011",
            "0111000011",
            "1100100011",
            "1010100011",
            "0110100011",
            "1001100011",
            "0101100011",
            "0011100011",
            "1100010011",
            "1010010011",
            "0110010011",
            "1001010011",
            "0101010011",
            "0011010011",
            "1000110011",
            "0100110011",
            "0010110011",
            "0001110011",
            "1100001011",
            "1010001011",
            "0110001011",
            "1001001011",
            "0101001011",
            "0011001011",
            "1000101011",
            "0100101011",
            "0010101011",
            "0001101011",
            "1000011011",
            "0100011011",
            "0010011011",
            "0001011011",
            "0000111011",
            "1100000111",
            "1010000111",
            "0110000111",
            "1001000111",
            "0101000111",
            "0011000111",
            "1000100111",
            "0100100111",
            "0010100111",
            "0001100111",
            "1000010111",
            "0100010111",
            "0010010111",
            "0001010111",
            "0000110111",
            "1000001111",
            "0100001111",
            "0010001111",
            "0001001111",
            "0000101111",
            "0000011111",
        ]
        exact_soutions = {state: -5 for state in exact_states}

        # Check computed solutions are among the correct ones
        for solution in solutions:
            for key in solution:
                assert solution[key] == exact_soutions[key]

    def test_rqaoa_asdict_dumps(self):
        """Test the asdict method of the RQAOA class."""

        # rqaoa
        rqaoa = self.__run_rqaoa("custom", return_object=True)

        # check RQAOA asdict
        self.__test_expected_keys(rqaoa.asdict(), method="asdict")

        # check RQAOA asdict deleting some keys
        exclude_keys = ["corr_matrix", "number_steps"]
        self.__test_expected_keys(
            rqaoa.asdict(exclude_keys=exclude_keys), exclude_keys, method="asdict"
        )

        # check RQAOA dumps
        self.__test_expected_keys(json.loads(rqaoa.dumps()), method="dumps")

        # check RQAOA dumps deleting some keys
        exclude_keys = ["project_id", "counter"]
        self.__test_expected_keys(
            json.loads(rqaoa.dumps(exclude_keys=exclude_keys)),
            exclude_keys,
            method="dumps",
        )

        # check RQAOA dump
        file_name = "test_dump_rqaoa.json"
        project_id, experiment_id, atomic_id = (
            rqaoa.header["project_id"],
            rqaoa.header["experiment_id"],
            rqaoa.header["atomic_id"],
        )
        full_name = f"{project_id}--{experiment_id}--{atomic_id}--{file_name}"

        rqaoa.dump(file_name, prepend_id=True, indent=None)
        assert os.path.isfile(full_name), "Dump file does not exist"
        with open(full_name, "r") as file:
            assert file.read() == rqaoa.dumps(
                indent=None
            ), "Dump file does not contain the correct data"
        os.remove(full_name)

        # check RQAOA dump whitout prepending the experiment_id and atomic_id
        rqaoa.dump(file_name, indent=None, prepend_id=False)
        assert os.path.isfile(
            file_name
        ), "Dump file does not exist, when not prepending the experiment_id and atomic_id"

        # check RQAOA dump fails when the file already exists
        error = False
        try:
            rqaoa.dump(file_name, indent=None, prepend_id=False)
        except FileExistsError:
            error = True
        assert error, "Dump file does not fail when the file already exists"

        # check that we can overwrite the file
        rqaoa.dump(file_name, indent=None, prepend_id=False, overwrite=True)
        assert os.path.isfile(file_name), "Dump file does not exist, when overwriting"
        os.remove(file_name)

        # check RQAOA dump fails when prepend_id is True and file_name is not given
        error = False
        try:
            rqaoa.dump(prepend_id=False)
        except ValueError:
            error = True
        assert (
            error
        ), "Dump file does not fail when prepend_id is True and file_name is not given"

        # check RQAOA dump with no arguments
        error = False
        try:
            rqaoa.dump()
        except ValueError:
            error = True
        assert (
            error
        ), "Dump file does not fail when no arguments are given, should be the same as dump(prepend_id=False)"

        # check you can dump to a file with no arguments, just prepend_id=True
        rqaoa.dump(prepend_id=True)
        assert os.path.isfile(
            f"{project_id}--{experiment_id}--{atomic_id}.json"
        ), "Dump file does not exist, when no name is given"
        os.remove(f"{project_id}--{experiment_id}--{atomic_id}.json")

        # check RQAOA dump deleting some keys
        exclude_keys = ["schedule", "singlet"]
        rqaoa.dump(file_name, exclude_keys=exclude_keys, indent=None, prepend_id=True)
        assert os.path.isfile(
            full_name
        ), "Dump file does not exist, when deleting some keys"
        with open(full_name, "r") as file:
            assert file.read() == rqaoa.dumps(
                exclude_keys=exclude_keys, indent=None
            ), "Dump file does not contain the correct data, when deleting some keys"
        os.remove(full_name)

        # check RQAOA dump with compression
        rqaoa.dump(file_name, compresslevel=2, indent=None, prepend_id=True)
        assert os.path.isfile(
            full_name + ".gz"
        ), "Dump file does not exist, when compressing"
        with gzip.open(full_name + ".gz", "rb") as file:
            assert (
                file.read() == rqaoa.dumps(indent=None).encode()
            ), "Dump file does not contain the correct data, when compressing"
        os.remove(full_name + ".gz")

    def __test_expected_keys(self, obj, exclude_keys=[], method="asdict"):
        """
        method to test if the dictionary has all the expected keys
        """

        # create a dictionary with all the expected keys and set them to False
        expected_keys = [
            "header",
            "atomic_id",
            "experiment_id",
            "project_id",
            "algorithm",
            "description",
            "run_by",
            "provider",
            "target",
            "cloud",
            "client",
            "qubit_number",
            "execution_time_start",
            "execution_time_end",
            "metadata",
            "tag1",
            "tag2",
            "problem_type",
            "n_shots",
            "optimizer_method",
            "param_type",
            "init_type",
            "p",
            "rqaoa_type",
            "rqaoa_n_max",
            "rqaoa_n_cutoff",
            "data",
            "exp_tags",
            "input_problem",
            "terms",
            "weights",
            "constant",
            "n",
            "problem_instance",
            "input_parameters",
            "device",
            "device_location",
            "device_name",
            "backend_properties",
            "init_hadamard",
            "prepend_state",
            "append_state",
            "cvar_alpha",
            "noise_model",
            "initial_qubit_mapping",
            "seed_simulator",
            "qiskit_simulation_method",
            "active_reset",
            "rewiring",
            "disable_qubit_rewiring",
            "classical_optimizer",
            "optimize",
            "method",
            "maxiter",
            "maxfev",
            "jac",
            "hess",
            "constraints",
            "bounds",
            "tol",
            "optimizer_options",
            "jac_options",
            "hess_options",
            "parameter_log",
            "optimization_progress",
            "cost_progress",
            "save_intermediate",
            "circuit_properties",
            "qubit_register",
            "q",
            "variational_params_dict",
            "total_annealing_time",
            "annealing_time",
            "linear_ramp_time",
            "mixer_hamiltonian",
            "mixer_qubit_connectivity",
            "mixer_coeffs",
            "seed",
            "rqaoa_parameters",
            "n_max",
            "steps",
            "n_cutoff",
            "original_hamiltonian",
            "counter",
            "result",
            "solution",
            "classical_output",
            "minimum_energy",
            "optimal_states",
            "elimination_rules",
            "pair",
            "correlation",
            "schedule",
            "number_steps",
            "intermediate_steps",
            "problem",
            "qaoa_results",
            "evals",
            "number_of_evals",
            "jac_evals",
            "qfim_evals",
            "most_probable_states",
            "solutions_bitstrings",
            "bitstring_energy",
            "intermediate",
            "angles",
            "cost",
            "measurement_outcomes",
            "job_id",
            "optimized",
            "eval_number",
            "exp_vals_z",
            "corr_matrix",
            "atomic_ids",
        ]
        expected_keys = {item: False for item in expected_keys}

        # test the keys, it will set the keys to True if they are found
        _test_keys_in_dict(obj, expected_keys)

        # Check if the dictionary has all the expected keys except the ones that were not included
        for key, value in expected_keys.items():
            if key not in exclude_keys:
                assert (
                    value == True
                ), f'Key "{key}" not found in the dictionary, when using "{method}" method.'
            else:
                assert (
                    value == False
                ), f'Key "{key}" was found in the dictionary, but it should not be there, when using "{method}" method.'

        """
        to get the list of expected keys, run the following code:

            def get_keys(obj, list_keys):
                if isinstance(obj, dict):
                    for key in obj:
                        if not key in list_keys: list_keys.append(key)

                        if isinstance(obj[key], dict):
                            get_keys(obj[key], list_keys)
                        elif isinstance(obj[key], list):
                            for item in obj[key]:
                                get_keys(item, list_keys)
                elif isinstance(obj, list):
                    for item in obj:
                        get_keys(item, list_keys)

            expected_keys = []
            get_keys(rqaoa.asdict(), expected_keys)
            print(expected_keys)
        """

    def test_rqaoa_dumping_step_by_step(self):
        """
        test dumping the RQAOA object step by step
        """

        # define the problem
        problem = QUBO.random_instance(n=8)
        problem.set_metadata(
            {"metadata_key1": "metadata_value1", "metadata_key2": "metadata_value2"}
        )

        # define the RQAOA object
        r = RQAOA()

        # set experimental tags
        r.set_exp_tags({"tag1": "value1", "tag2": "value2"})

        # set the classical optimizer
        r.set_classical_optimizer(optimization_progress=True)

        # compile the problem
        r.compile(problem)

        # optimize the problem while dumping the data at each step
        r.optimize(
            dump=True,
            dump_options={
                "file_name": "test_dumping_step_by_step",
                "compresslevel": 2,
                "indent": None,
                "prepend_id": True,
            },
        )

        # create list of expected file names
        project_id = "None"
        experiment_id, atomic_id = r.header["experiment_id"], r.header["atomic_id"]
        file_names = {
            id: project_id
            + "--"
            + experiment_id
            + "--"
            + id
            + "--"
            + "test_dumping_step_by_step.json.gz"
            for id in r.result["atomic_ids"].values()
        }
        file_names[atomic_id] = (
            project_id
            + "--"
            + experiment_id
            + "--"
            + atomic_id
            + "--"
            + "test_dumping_step_by_step.json.gz"
        )

        # check if the files exist
        for file_name in file_names.values():
            assert os.path.isfile(file_name), f"File {file_name} does not exist."

        # put each file in a dictionary
        files = {}
        for atomic_id, file_name in file_names.items():
            with gzip.open(file_name, "rb") as file:
                files[atomic_id] = json.loads(file.read().decode())

        rqaoa_files, qaoa_files = 0, 0

        # check if the files have the expected keys
        for atomic_id, dictionary in files.items():
            file_name = file_names[atomic_id]

            if r.header["atomic_id"] == atomic_id:  # rqaoa files
                rqaoa_files += 1

                assert (
                    dictionary["header"]["experiment_id"] == r.header["experiment_id"]
                ), f"File {file_name} has a different experiment_id than the RQAOA object."
                assert (
                    dictionary["header"]["atomic_id"] == r.header["atomic_id"]
                ), f"File {file_name} has a different atomic_id than the RQAOA object."
                assert (
                    dictionary["header"]["algorithm"] == "rqaoa"
                ), f"File {file_name} has a different algorithm than rqaoa, which is the expected algorithm."

                # check that the intermediate measurements are empty
                for step in dictionary["data"]["result"]["intermediate_steps"]:
                    assert (
                        step["qaoa_results"]["intermediate"]["measurement_outcomes"]
                        == []
                    ), f"File {file_name} has intermediate measurements, but it should not have them."

            else:  # qaoa files
                qaoa_files += 1

                assert (
                    dictionary["header"]["atomic_id"] == atomic_id
                ), f"File {file_name} has a different atomic_id than expected."
                assert (
                    dictionary["header"]["algorithm"] == "qaoa"
                ), f"File {file_name} has a different algorithm than qaoa, which is the expected algorithm."

                # check that the intermediate mesuraments are not empty
                assert (
                    len(
                        dictionary["data"]["result"]["intermediate"][
                            "measurement_outcomes"
                        ]
                    )
                    > 0
                ), f"File {file_name} does not have intermediate measurements, but it should have them."

        assert rqaoa_files == 1, f"Expected 1 rqaoa file, but {rqaoa_files} were found."
        assert qaoa_files == len(
            r.result["atomic_ids"]
        ), f'Expected {len(r.result["atomic_ids"])} qaoa files, but {qaoa_files} were found.'

        # erase the files
        for file_name in file_names.values():
            os.remove(file_name)

    def test_rqaoa_from_dict_and_load(self):
        """
        test loading the RQAOA object from a dictionary
        methods: from_dict, load, loads
        """

        # problem
        maxcut_qubo = MaximumCut(
            nw.generators.fast_gnp_random_graph(n=6, p=0.6, seed=42)
        ).qubo

        # run rqaoa with different devices, and save the objcets in a list
        rqaoas = []
        for device in [
            create_device(location="local", name="qiskit.shot_simulator"),
        ]:
            r = RQAOA()
            r.set_device(device)
            r.set_circuit_properties(
                p=1, param_type="extended", init_type="rand", mixer_hamiltonian="x"
            )
            r.set_backend_properties(n_shots=50)
            r.set_classical_optimizer(maxiter=10, optimization_progress=True)
            r.set_rqaoa_parameters(rqaoa_type="adaptive", n_cutoff=3)
            r.set_exp_tags({"tag1": "value1", "tag2": "value2"})
            r.set_header(
                project_id="8353185c-b175-4eda-9628-b4e58cb0e41b",
                description="test",
                run_by="raul",
                provider="-",
                target="-",
                cloud="local",
                client="-",
            )

            # test that you can convert the rqaoa object to a dictionary and then load it before optimization
            _ = RQAOA.from_dict(r.asdict())
            _.compile(maxcut_qubo)
            _.optimize()
            assert isinstance(
                _, RQAOA
            ), "The object loaded from a dictionary is not an RQAOA object."

            # compile and optimize the original rqaoa object
            r.compile(maxcut_qubo)
            r.optimize()

            rqaoas.append(r)

        # for each rqaoa object, create a new rqaoa object from dict, json string, json file, and compressed json file and compare them with the original object
        for r in rqaoas:
            new_r_list = []

            # get new qaoa from dict
            new_r_list.append(RQAOA.from_dict(r.asdict()))
            # get new qaoa from json string
            new_r_list.append(RQAOA.loads(r.dumps()))
            # get new qaoa from json file
            r.dump("test.json", prepend_id=False)
            new_r_list.append(RQAOA.load("test.json"))
            os.remove("test.json")  # delete file test.json
            # get new qaoa from compressed json file
            r.dump("test.json", prepend_id=False, compresslevel=3)
            new_r_list.append(RQAOA.load("test.json.gz"))
            os.remove("test.json.gz")  # delete file test.json

            for new_r in new_r_list:
                # check that the new object is an RQAOA object
                assert isinstance(new_r, RQAOA), "new_r is not an RQAOA object"

                # check that the attributes of the new object are of the correct type
                attributes_types = [
                    ("header", dict),
                    ("exp_tags", dict),
                    ("problem", QUBO),
                    ("result", RQAOAResult),
                    ("backend_properties", BackendProperties),
                    ("classical_optimizer", ClassicalOptimizer),
                    ("circuit_properties", CircuitProperties),
                    ("rqaoa_parameters", RqaoaParameters),
                ]
                for attribute, type_ in attributes_types:
                    assert isinstance(
                        getattr(new_r, attribute), type_
                    ), f"attribute {attribute} is not type {type_}"

                # get the two objects (old and new) as dictionaries
                r_asdict = r.asdict()
                new_r_asdict = new_r.asdict()

                # compare the two dictionaries
                for key, value in r_asdict.items():
                    if key == "header":
                        assert value == new_r_asdict[key], "Header is not the same"

                    elif key == "data":
                        for key2, value2 in value.items():
                            if key2 == "input_parameters":
                                # pop key device
                                value2.pop("device")
                                new_r_asdict[key][key2].pop("device")
                            if key2 == "result":
                                for step in range(len(value2["intermediate_steps"])):
                                    for key3 in value2["intermediate_steps"][
                                        step
                                    ].keys():
                                        if key3 == "qaoa_results":
                                            _compare_qaoa_results(
                                                value2["intermediate_steps"][step][
                                                    "qaoa_results"
                                                ],
                                                new_r_asdict[key][key2][
                                                    "intermediate_steps"
                                                ][step]["qaoa_results"],
                                            )
                                        else:
                                            assert (
                                                value2["intermediate_steps"][step][key3]
                                                == new_r_asdict[key][key2][
                                                    "intermediate_steps"
                                                ][step][key3]
                                            ), f"{key3} is not the same"
                            else:
                                assert (
                                    value2 == new_r_asdict[key][key2]
                                ), "{} not the same".format(key2)

                # compile and optimize the new rqaoa, to check if everything is working
                new_r.compile(maxcut_qubo)
                new_r.optimize()

        # check that the Optimizer.from_dict method raises an error when using a RQAOA dictionary
        error = False
        try:
            Workflow.from_dict(r.asdict())
        except Exception:
            error = True
        assert (
            error
        ), "Optimizer.from_dict should raise an error when using a RQAOA dictionary"


if __name__ == "__main__":
    unittest.main()
