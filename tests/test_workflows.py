import json
import os
import gzip
import unittest
import networkx as nw
import numpy as np
import datetime
from copy import deepcopy

from qiskit.providers.fake_provider import FakeVigo
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import QasmSimulator

from openqaoa import QAOA, RQAOA
from openqaoa.problems import NumberPartition
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
from openqaoa.backends.cost_function import cost_function

# from openqaoa.backends.devices_core import SUPPORTED_LOCAL_SIMULATORS
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
from openqaoa_pyquil.backends import DevicePyquil
from openqaoa_pyquil.backends import QAOAPyQuilWavefunctionSimulatorBackend
from openqaoa_qiskit.backends import DeviceQiskit
from openqaoa_qiskit.backends import (
    QAOAQiskitBackendShotBasedSimulator,
    QAOAQiskitBackendStatevecSimulator,
)
from openqaoa_azure.backends import DeviceAzure

from openqaoa.qaoa_components.variational_parameters.variational_params_factory import (
    PARAMS_CLASSES_MAPPER,
)


def _compare_qaoa_results(dict_old, dict_new):
    for key in dict_old.keys():
        if key == "cost_hamiltonian":  # CHECK WHAT DO WITH THIS
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
                        ), f"Intermediate params are not the same. Expected {dict_old[key][key2][step]} but \
                            received {dict_new[key][key2][step]}"
                else:
                    assert (
                        dict_old[key][key2] == dict_new[key][key2]
                    ), f"Intermediate params are not the same. Expected {dict_old[key][key2]}, but \
                        received {dict_new[key][key2]}"
        else:
            assert dict_old[key] == dict_new[key], f"'{key}' is not the same"


def _test_keys_in_dict(obj, expected_keys):
    """
    private function to test the keys.
    It recursively tests the keys of the nested dictionaries, or lists of dictionaries
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

    def test_vanilla_qaoa_default_values(self):
        q = QAOA()
        assert q.circuit_properties.p == 1
        assert q.circuit_properties.param_type == "standard"
        assert q.circuit_properties.init_type == "ramp"
        assert q.device.device_location == "local"
        assert q.device.device_name == "vectorized"

    def test_end_to_end_vectorized(self):
        g = nw.circulant_graph(6, [1])
        vc = MinimumVertexCover(g, field=1.0, penalty=10).qubo

        q = QAOA()
        q.set_classical_optimizer(optimization_progress=True)
        q.compile(vc)
        q.optimize()

        result = q.result.most_probable_states["solutions_bitstrings"][0]
        assert "010101" == result or "101010" == result

    def test_set_device_local(self):
        """ "
        Check that all local devices are correctly initialised
        """
        q = QAOA()
        for d in q.local_simulators:
            q.set_device(create_device(location="local", name=d))
            assert type(q.device) == DeviceLocal
            assert q.device.device_name == d
            assert q.device.device_location == "local"

    def test_set_device_cloud(self):
        """ "
        Check that all QPU-provider related devices are correctly initialised
        """
        q = QAOA()
        q.set_device(
            create_device(
                "qcs",
                name="6q-qvm",
                **{"as_qvm": True, "execution_timeout": 10, "compiler_timeout": 10},
            )
        )
        assert type(q.device) == DevicePyquil
        assert q.device.device_name == "6q-qvm"
        assert q.device.device_location == "qcs"

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

        q.set_device(
            create_device(
                "azure", name="place_holder", resource_id="***", az_location="***"
            )
        )
        assert type(q.device) == DeviceAzure
        assert q.device.device_name == "place_holder"
        assert q.device.device_location == "azure"

    def test_compile_before_optimise(self):
        """
        Assert that compilation has to be called before optimisation
        """

        q = QAOA()
        q.set_classical_optimizer(optimization_progress=True)

        self.assertRaises(ValueError, lambda: q.optimize())

    def test_cost_hamil(self):
        g = nw.circulant_graph(6, [1])
        problem = MinimumVertexCover(g, field=1.0, penalty=10)
        qubo_problem = problem.qubo

        test_hamil = Hamiltonian.classical_hamiltonian(
            terms=qubo_problem.terms,
            coeffs=qubo_problem.weights,
            constant=qubo_problem.constant,
        )

        q = QAOA()

        q.compile(problem=qubo_problem)

        self.assertEqual(q.cost_hamil.expression, test_hamil.expression)
        self.assertEqual(
            q.qaoa_descriptor.cost_hamiltonian.expression, test_hamil.expression
        )

    def test_set_circuit_properties_fourier_q(self):
        """
        The value of q should be None if the param_type used is not fourier.
        Else if param_type is fourier, fourier_extended or fourier_w_bias, it
        should be the value of q, if it is provided.
        """

        fourier_param_types = ["fourier", "fourier_extended", "fourier_w_bias"]

        q = QAOA()

        for each_param_type in fourier_param_types:
            q.set_circuit_properties(param_type=each_param_type, q=1)
            self.assertEqual(q.circuit_properties.q, 1)

        q.set_circuit_properties(param_type="standard", q=1)

        self.assertEqual(q.circuit_properties.q, None)

    def test_set_circuit_properties_annealing_time_linear_ramp_time(self):
        """
        Check that linear_ramp_time and annealing_time are updated appropriately
        as the value of p is changed.
        """

        q = QAOA()

        q.set_circuit_properties(p=3)

        self.assertEqual(q.circuit_properties.annealing_time, 0.7 * 3)
        self.assertEqual(q.circuit_properties.linear_ramp_time, 0.7 * 3)

        q.set_circuit_properties(p=2)

        self.assertEqual(q.circuit_properties.annealing_time, 0.7 * 2)
        self.assertEqual(q.circuit_properties.linear_ramp_time, 0.7 * 2)

    def test_set_circuit_properties_qaoa_descriptor_mixer_x(self):
        """
        Checks if the X mixer created by the X_mixer_hamiltonian method
        and the automated methods in workflows do the same thing.

        For each qubit, there should be 1 RXGateMap per layer of p.
        """

        nodes = 6
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes, p=edge_probability)
        problem = MinimumVertexCover(g, field=1.0, penalty=10)

        q = QAOA()
        q.set_circuit_properties(mixer_hamiltonian="x", p=2)

        q.compile(problem=problem.qubo)

        self.assertEqual(type(q.qaoa_descriptor), QAOADescriptor)
        self.assertEqual(q.qaoa_descriptor.p, 2)

        mixer_hamil = X_mixer_hamiltonian(n_qubits=nodes)

        self.assertEqual(q.mixer_hamil.expression, mixer_hamil.expression)

        self.assertEqual(len(q.qaoa_descriptor.mixer_qubits_singles), 6)
        self.assertEqual(len(q.qaoa_descriptor.mixer_qubits_pairs), 0)
        for each_gatemap_name in q.qaoa_descriptor.mixer_qubits_singles:
            self.assertEqual(each_gatemap_name, "RXGateMap")

        for j in range(2):
            for i in range(6):
                self.assertEqual(q.qaoa_descriptor.mixer_blocks[j][i].qubit_1, i)

    def test_set_circuit_properties_qaoa_descriptor_mixer_xy(self):
        """
        Checks if the XY mixer created by the XY_mixer_hamiltonian method
        and the automated methods in workflows do the same thing.

        Depending on the qubit connectivity selected. (chain, full or star)
        For each pair of connected qubits, there should be 1 RXXGateMap and RYYGateMap per layer of p.
        """

        g_c = nw.circulant_graph(6, [1])
        g_f = nw.complete_graph(6)
        # A 5-sided star graoh requires 6 qubit. (Center Qubit of the pattern)
        g_s = nw.star_graph(5)
        problems = [
            MinimumVertexCover(g_c, field=1.0, penalty=10),
            MinimumVertexCover(g_f, field=1.0, penalty=10),
            MinimumVertexCover(g_s, field=1.0, penalty=10),
        ]
        qubit_connectivity_name = ["chain", "full", "star"]

        for i in range(3):
            q = QAOA()
            q.set_circuit_properties(
                mixer_hamiltonian="xy",
                mixer_qubit_connectivity=qubit_connectivity_name[i],
                p=2,
            )

            q.compile(problem=problems[i].qubo)

            self.assertEqual(type(q.qaoa_descriptor), QAOADescriptor)
            self.assertEqual(q.qaoa_descriptor.p, 2)

            mixer_hamil = XY_mixer_hamiltonian(
                n_qubits=6, qubit_connectivity=qubit_connectivity_name[i]
            )

            self.assertEqual(q.mixer_hamil.expression, mixer_hamil.expression)

            self.assertEqual(len(q.qaoa_descriptor.mixer_qubits_singles), 0)
            for i in range(0, len(q.qaoa_descriptor.mixer_qubits_pairs), 2):
                self.assertEqual(q.qaoa_descriptor.mixer_qubits_pairs[i], "RXXGateMap")
                self.assertEqual(
                    q.qaoa_descriptor.mixer_qubits_pairs[i + 1], "RYYGateMap"
                )

    def test_set_circuit_properties_variate_params(self):
        """
        Ensure that the Varitional Parameter Object created based on the input string , param_type, is correct.

        TODO: Check if q=None is the appropriate default.
        """

        param_type_names = [
            "standard",
            "standard_w_bias",
            "extended",
            "fourier",
            "fourier_extended",
            "fourier_w_bias",
        ]
        object_types = [
            QAOAVariationalStandardParams,
            QAOAVariationalStandardWithBiasParams,
            QAOAVariationalExtendedParams,
            QAOAVariationalFourierParams,
            QAOAVariationalFourierExtendedParams,
            QAOAVariationalFourierWithBiasParams,
        ]

        nodes = 6
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes, p=edge_probability)
        problem = MinimumVertexCover(g, field=1.0, penalty=10)

        for i in range(len(object_types)):
            q = QAOA()
            q.set_circuit_properties(param_type=param_type_names[i], q=1)

            q.compile(problem=problem.qubo)

            self.assertEqual(type(q.variate_params), object_types[i])

    def test_set_circuit_properties_change(self):
        """
        Ensure that once a property has beefn changed via set_circuit_properties.
        The attribute has been appropriately updated.
        Updating all attributes at the same time.
        """

        #         default_pairings = {'param_type': 'standard',
        #                             'init_type': 'ramp',
        #                             'qubit_register': [],
        #                             'p': 1,
        #                             'q': None,
        #                             'annealing_time': 0.7,
        #                             'linear_ramp_time': 0.7,
        #                             'variational_params_dict': {},
        #                             'mixer_hamiltonian': 'x',
        #                             'mixer_qubit_connectivity': None,
        #                             'mixer_coeffs': None,
        #                             'seed': None}

        q = QAOA()

        # TODO: Some weird error related to the initialisation of QAOA here
        #         for each_key, each_value in default_pairings.items():
        #             print(each_key, getattr(q.circuit_properties, each_key), each_value)
        #             self.assertEqual(getattr(q.circuit_properties, each_key), each_value)

        update_pairings = {
            "param_type": "fourier",
            "init_type": "rand",
            "qubit_register": [0, 1],
            "p": 2,
            "q": 2,
            "annealing_time": 1.0,
            "linear_ramp_time": 1.0,
            "variational_params_dict": {"key": "value"},
            "mixer_hamiltonian": "xy",
            "mixer_qubit_connectivity": "chain",
            "mixer_coeffs": [0.1, 0.2],
            "seed": 45,
        }

        q.set_circuit_properties(**update_pairings)

        for each_key, each_value in update_pairings.items():
            self.assertEqual(getattr(q.circuit_properties, each_key), each_value)

    def test_set_circuit_properties_rejected_values(self):
        """
        Some properties of CircuitProperties Object return a ValueError
        if the specified property has not been whitelisted in the code.
        This checks that the ValueError is raised if the argument is not whitelisted.
        """

        q = QAOA()

        self.assertRaises(
            ValueError, lambda: q.set_circuit_properties(param_type="wrong name")
        )
        self.assertRaises(
            ValueError, lambda: q.set_circuit_properties(init_type="wrong name")
        )
        self.assertRaises(
            ValueError, lambda: q.set_circuit_properties(mixer_hamiltonian="wrong name")
        )
        self.assertRaises(ValueError, lambda: q.set_circuit_properties(p=-1))

    def test_set_backend_properties_change(self):
        """
        Ensure that once a property has been changed via set_backend_properties.
        The attribute has been appropriately updated.
        Updating all attributes at the same time.
        """

        default_pairings = {
            "prepend_state": None,
            "append_state": None,
            "init_hadamard": True,
            "n_shots": 100,
            "cvar_alpha": 1.0,
        }

        q = QAOA()

        for each_key, each_value in default_pairings.items():
            self.assertEqual(getattr(q.backend_properties, each_key), each_value)

        update_pairings = {
            "prepend_state": [[0, 0]],
            "append_state": [[0, 0]],
            "init_hadamard": False,
            "n_shots": 10,
            "cvar_alpha": 0.5,
        }

        q.set_backend_properties(**update_pairings)

        for each_key, each_value in update_pairings.items():
            self.assertEqual(getattr(q.backend_properties, each_key), each_value)

    def test_set_backend_properties_check_backend_vectorized(self):
        """
        Check if the backend returned by set_backend_properties is correct
        Based on the input device.
        Also Checks if defaults from workflows are used in the backend.
        """

        nodes = 6
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes, p=edge_probability)
        problem = MinimumVertexCover(g, field=1.0, penalty=10)
        qubo_problem = problem.qubo

        q = QAOA()
        q.set_device(create_device(location="local", name="vectorized"))
        q.compile(problem=qubo_problem)

        self.assertEqual(type(q.backend), QAOAvectorizedBackendSimulator)

        self.assertEqual(q.backend.init_hadamard, True)
        self.assertEqual(q.backend.prepend_state, None)
        self.assertEqual(q.backend.append_state, None)
        self.assertEqual(q.backend.cvar_alpha, 1)

        self.assertRaises(AttributeError, lambda: q.backend.n_shots)

    def test_set_backend_properties_check_backend_vectorized_w_custom(self):
        """
        Check if the backend returned by set_backend_properties is correct
        Based on the input device.
        Uses custom values for attributes in backend_properties and checks if the
        backend object responds appropriately.
        """

        nodes = 3
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes, p=edge_probability)
        problem = MinimumVertexCover(g, field=1.0, penalty=10)
        qubo_problem = problem.qubo

        q = QAOA()
        q.set_device(create_device(location="local", name="vectorized"))

        prepend_state_rand = np.random.rand(2**3)
        append_state_rand = np.eye(2**3)

        update_pairings = {
            "prepend_state": prepend_state_rand,
            "append_state": append_state_rand,
            "init_hadamard": False,
            "n_shots": 10,
            "cvar_alpha": 1,
        }

        q.set_backend_properties(**update_pairings)

        q.compile(problem=qubo_problem)

        self.assertEqual(type(q.backend), QAOAvectorizedBackendSimulator)

        self.assertEqual(q.backend.init_hadamard, False)
        self.assertEqual((q.backend.prepend_state == prepend_state_rand).all(), True)
        self.assertEqual((q.backend.append_state == append_state_rand).all(), True)
        self.assertEqual(q.backend.cvar_alpha, 1)

        self.assertRaises(AttributeError, lambda: q.backend.n_shots)

    def test_set_backend_properties_check_backend_vectorized_error_values(self):
        """
        If the values provided from the workflows are incorrect, we should
        receive the appropriate error messages from the vectorized backend.

        Checks:
        Incorrect size of prepend state and append state.
        """

        nodes = 3
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes, p=edge_probability)
        problem = MinimumVertexCover(g, field=1.0, penalty=10)
        qubo_problem = problem.qubo

        q = QAOA()
        q.set_device(create_device(location="local", name="vectorized"))

        prepend_state_rand = np.random.rand(2**2)

        update_pairings = {"prepend_state": prepend_state_rand, "append_state": None}

        q.set_backend_properties(**update_pairings)

        self.assertRaises(ValueError, lambda: q.compile(problem=qubo_problem))

        q = QAOA()
        q.set_device(create_device(location="local", name="vectorized"))

        append_state_rand = np.random.rand(2**2, 2**2)

        update_pairings = {"prepend_state": None, "append_state": append_state_rand}

        q.set_backend_properties(**update_pairings)

        self.assertRaises(ValueError, lambda: q.compile(problem=qubo_problem))

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

    def test_set_backend_properties_check_backend_pyquil_statevector(self):
        """
        Check if the backend returned by set_backend_properties is correct
        Based on the input device. For pyquil statevector simulator.
        Also Checks if defaults from workflows are used in the backend.
        """

        nodes = 6
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes, p=edge_probability)
        problem = MinimumVertexCover(g, field=1.0, penalty=10)
        qubo_problem = problem.qubo

        q = QAOA()
        q.set_device(
            create_device(location="local", name="pyquil.statevector_simulator")
        )
        q.compile(problem=qubo_problem)

        self.assertEqual(type(q.backend), QAOAPyQuilWavefunctionSimulatorBackend)

        self.assertEqual(q.backend.init_hadamard, True)
        self.assertEqual(q.backend.prepend_state, None)
        self.assertEqual(q.backend.append_state, None)
        self.assertEqual(q.backend.cvar_alpha, 1)

        self.assertRaises(AttributeError, lambda: q.backend.n_shots)

    def test_set_classical_optimizer_defaults(self):
        """
        Check if the fields in the default classical_optimizer dict are correct
        """

        default_pairings = {
            "optimize": True,
            "method": "cobyla",
            "maxiter": 100,
            "jac": None,
            "hess": None,
            "constraints": None,
            "bounds": None,
            "tol": None,
            "optimizer_options": None,
            "jac_options": None,
            "hess_options": None,
            "optimization_progress": False,
            "cost_progress": True,
            "parameter_log": True,
        }

        q = QAOA()

        for each_key, each_value in default_pairings.items():
            self.assertEqual(getattr(q.classical_optimizer, each_key), each_value)

            # if each_value is None: LD --> I don't think we really need this test, since asdict()
            #     assert isinstance(q.classical_optimizer.each_key, None)

    def test_set_classical_optimizer_jac_hess_casing(self):
        """
        jac and hess should be in lower case if it is a string.
        """

        q = QAOA()
        q.set_classical_optimizer(jac="JaC", hess="HeSS")

        self.assertEqual(q.classical_optimizer.jac, "jac")
        self.assertEqual(q.classical_optimizer.hess, "hess")

    def test_set_classical_optimizer_method_selectors(self):
        """
        Different methods would return different Optimizer classes.
        Check that the correct class is returned.
        """

        nodes = 6
        edge_probability = 0.6
        g = nw.generators.fast_gnp_random_graph(n=nodes, p=edge_probability)
        problem = MinimumVertexCover(g, field=1.0, penalty=10)
        qubo_problem = problem.qubo

        for each_method in available_optimizers()["scipy"]:
            q = QAOA()
            q.set_classical_optimizer(method=each_method, jac="grad_spsa")
            q.compile(problem=qubo_problem)

            self.assertEqual(isinstance(q.optimizer, ScipyOptimizer), True)
            self.assertEqual(
                isinstance(q.optimizer, CustomScipyGradientOptimizer), False
            )
            self.assertEqual(isinstance(q.optimizer, PennyLaneOptimizer), False)

        for each_method in available_optimizers()["custom_scipy_gradient"]:
            q = QAOA()
            q.set_classical_optimizer(
                method=each_method, jac="grad_spsa", hess="finite_difference"
            )
            q.compile(problem=qubo_problem)

            self.assertEqual(isinstance(q.optimizer, ScipyOptimizer), False)
            self.assertEqual(
                isinstance(q.optimizer, CustomScipyGradientOptimizer), True
            )
            self.assertEqual(isinstance(q.optimizer, PennyLaneOptimizer), False)

        for each_method in available_optimizers()["custom_scipy_pennylane"]:
            q = QAOA()
            q.set_classical_optimizer(method=each_method, jac="grad_spsa")
            q.compile(problem=qubo_problem)

            self.assertEqual(isinstance(q.optimizer, ScipyOptimizer), False)
            self.assertEqual(
                isinstance(q.optimizer, CustomScipyGradientOptimizer), False
            )
            self.assertEqual(isinstance(q.optimizer, PennyLaneOptimizer), True)

    def test_set_header(self):
        """
        Test the test_set_header method of the QAOA class. Step by step it is checked that the header is set correctly.
        """

        # create a QAOA object
        qaoa: QAOA = QAOA()

        # check if the header values are set to None, except for the experiment_id and algorithm
        for key, value in qaoa.header.items():
            if key == "experiment_id":
                assert is_valid_uuid(
                    qaoa.header["experiment_id"]
                ), "The experiment_id is not a valid uuid."
            elif key == "algorithm":
                assert qaoa.header["algorithm"] == "qaoa"
            else:
                assert (
                    value is None
                ), "The value of the key {} (of the dictionary qaoa.header) is not None, when it should be.".format(
                    key
                )

        # save the experiment_id
        experiment_id = qaoa.header["experiment_id"]

        # set the header
        qaoa.set_header(
            project_id="8353185c-b175-4eda-9628-b4e58cb0e41b",
            description="test",
            run_by="OpenQAOA",
            provider="-",
            target="vectorized",
            cloud="local",
            client="-",
        )

        # check that the experiment_id has not changed, since it is not set in the set_header method
        assert (
            qaoa.header["experiment_id"] == experiment_id
        ), "The experiment_id has changed when it should not have."

        # now set the experiment_id
        experiment_id = experiment_id[:-2] + "00"

        # set the header
        qaoa.set_header(
            project_id="8353185c-b175-4eda-9628-b4e58cb0e41b",
            description="test",
            run_by="OpenQAOA",
            provider="-",
            target="vectorized",
            cloud="local",
            client="-",
            experiment_id=experiment_id,
        )

        # check if the header values are set to the correct values, except for the
        # qubit_number, atomic_id, execution_time_start, and execution_time_end (which are set to None)
        dict_values = {
            "experiment_id": experiment_id,
            "project_id": "8353185c-b175-4eda-9628-b4e58cb0e41b",
            "algorithm": "qaoa",
            "description": "test",
            "run_by": "OpenQAOA",
            "provider": "-",
            "target": "vectorized",
            "cloud": "local",
            "client": "-",
            "qubit_number": None,
            "atomic_id": None,
            "execution_time_start": None,
            "execution_time_end": None,
        }
        for key, value in qaoa.header.items():
            assert (
                dict_values[key] == value
            ), "The value of the key {} (of the dictionary qaoa.header) is not correct.".format(
                key
            )

        # compile the QAOA object
        qaoa.compile(problem=QUBO.random_instance(n=8))

        # check if the header values are still set to the correct values, except for execution_time_start, and
        # execution_time_end (which are set to None).
        # Now atomic_id should be set to a valid uuid.
        # And qubit_number should be set to 8 (number of qubits of the problem)
        dict_values["qubit_number"] = 8
        for key, value in qaoa.header.items():
            if key not in ["atomic_id"]:
                assert (
                    dict_values[key] == value
                ), "The value of the key {} (of the dictionary qaoa.header) is not correct.".format(
                    key
                )
        assert is_valid_uuid(
            qaoa.header["atomic_id"]
        ), "The atomic_id is not a valid uuid."

        # save the atomic_id
        atomic_id = qaoa.header["atomic_id"]

        # optimize the QAOA object
        qaoa.optimize()

        # check if the header values are still set to the correct values, now everything should be set to a valid value
        # (execution_time_start and execution_time_end should be integers>1672933928)
        dict_values["atomic_id"] = atomic_id
        for key, value in qaoa.header.items():
            if key not in ["execution_time_start", "execution_time_end"]:
                assert (
                    dict_values[key] == value
                ), "The value of the key {} (of the dictionary qaoa.header) is not correct.".format(
                    key
                )
        assert datetime.datetime.strptime(
            qaoa.header["execution_time_start"], "%Y-%m-%dT%H:%M:%S"
        ), "The execution_time_start is not valid."
        assert datetime.datetime.strptime(
            qaoa.header["execution_time_end"], "%Y-%m-%dT%H:%M:%S"
        ), "The execution_time_end is not valid."

        # test if an error is raised when the project_id is not a valid string
        error = False
        try:
            qaoa.set_header(
                project_id="test",
                description="test",
                run_by="OpenQAOA",
                provider="-",
                target="vectorized",
                cloud="local",
                client="-",
            )
        except:
            error = True
        assert error, "The project_id is not valid string, but no error was raised."

        # test if an error is raised when the experiment_id is not a valid string
        error = False
        try:
            qaoa.set_header(
                project_id="8353185c-b175-4eda-9628-b4e58cb0e41b",
                experiment_id="test",
                description="test",
                run_by="OpenQAOA",
                provider="-",
                target="vectorized",
                cloud="local",
                client="-",
            )
        except:
            error = True
        assert error, "The experiment_id is not valid string, but no error was raised."

    def test_set_exp_tags(self):
        """
        Test the set_exp_tags method of the QAOA class.
        """

        qaoa = QAOA()
        qaoa.set_exp_tags(tags={"tag1": "value1", "tag2": "value2"})
        qaoa.set_exp_tags(tags={"tag1": "value9"})
        qaoa.compile(problem=QUBO.random_instance(n=8))
        qaoa.optimize()

        tags = {
            "tag1": "value9",
            "tag2": "value2",
            "init_type": "ramp",
            "optimizer_method": "cobyla",
            "p": 1,
            "param_type": "standard",
            "qubit_number": 8,
        }

        assert qaoa.exp_tags == tags, "Experiment tags are not set correctly."

        error = False
        try:
            qaoa.set_exp_tags(tags={"tag1": complex(1, 2)})
        except:
            error = True
        assert error, "Experiment tag values should be primitives."

        error = False
        try:
            qaoa.set_exp_tags(tags={(1, 2): "test"})
        except:
            error = True
        assert error, "Experiment tag keys should be strings."

    def test_qaoa_asdict_with_noise(self):
        "test to check that we can serialize a QAOA object with noise"
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

    def test_qaoa_asdict_dumps(self):
        """Test the asdict method of the QAOA class."""

        # qaoa
        qaoa = QAOA()
        qaoa.compile(problem=QUBO.random_instance(n=8))
        # set the header
        qaoa.set_header(
            project_id="8353185c-b175-4eda-9628-b4e58cb0e41b",
            description="test",
            run_by="OpenQAOA",
            provider="-",
            target="vectorized",
            cloud="local",
            client="-",
        )
        qaoa.optimize()

        # check QAOA asdict
        self.__test_expected_keys(qaoa.asdict(), method="asdict")

        # check QAOA asdict deleting some keys
        exclude_keys = ["corr_matrix", "number_steps"]
        self.__test_expected_keys(
            qaoa.asdict(exclude_keys=exclude_keys), exclude_keys, method="asdict"
        )

        # check QAOA dumps
        self.__test_expected_keys(json.loads(qaoa.dumps()), method="dumps")

        # check QAOA dumps deleting some keys
        exclude_keys = ["parent_id", "counter"]
        self.__test_expected_keys(
            json.loads(qaoa.dumps(exclude_keys=exclude_keys)),
            exclude_keys,
            method="dumps",
        )

        # check QAOA dump
        file_name = "test_dump_qaoa.json"
        project_id, experiment_id, atomic_id = (
            qaoa.header["project_id"],
            qaoa.header["experiment_id"],
            qaoa.header["atomic_id"],
        )
        full_name = f"{project_id}--{experiment_id}--{atomic_id}--{file_name}"

        qaoa.dump(file_name, indent=None, prepend_id=True)
        assert os.path.isfile(full_name), "Dump file does not exist"
        with open(full_name, "r") as file:
            assert file.read() == qaoa.dumps(
                indent=None
            ), "Dump file does not contain the correct data"
        os.remove(full_name)

        # check QAOA dump whitout prepending the experiment_id and atomic_id
        qaoa.dump(file_name, indent=None, prepend_id=False)
        assert os.path.isfile(
            file_name
        ), "Dump file does not exist, when not prepending the experiment_id and atomic_id"

        # check QAOA dump fails when the file already exists
        error = False
        try:
            qaoa.dump(file_name, indent=None, prepend_id=False)
        except FileExistsError:
            error = True
        assert error, "Dump file does not fail when the file already exists"

        # check that we can overwrite the file
        qaoa.dump(file_name, indent=None, prepend_id=False, overwrite=True)
        assert os.path.isfile(file_name), "Dump file does not exist, when overwriting"
        os.remove(file_name)

        # check QAOA dump fails when prepend_id is True and file_name is not given
        error = False
        try:
            qaoa.dump(prepend_id=False)
        except ValueError:
            error = True
        assert (
            error
        ), "Dump file does not fail when prepend_id is True and file_name is not given"

        # check QAOA dump with no arguments
        error = False
        try:
            qaoa.dump()
        except ValueError:
            error = True
        assert (
            error
        ), "Dump file does not fail when no arguments are given, should be the same as dump(prepend_id=False)"

        # check you can dump to a file with no arguments, just prepend_id=True
        qaoa.dump(prepend_id=True)
        assert os.path.isfile(
            f"{project_id}--{experiment_id}--{atomic_id}.json"
        ), "Dump file does not exist, when no name is given"
        os.remove(f"{project_id}--{experiment_id}--{atomic_id}.json")

        # check QAOA dump deleting some keys
        exclude_keys = ["schedule", "singlet"]
        qaoa.dump(file_name, exclude_keys=exclude_keys, indent=None, prepend_id=True)
        assert os.path.isfile(
            full_name
        ), "Dump file does not exist, when deleting some keys"
        with open(full_name, "r") as file:
            assert file.read() == qaoa.dumps(
                exclude_keys=exclude_keys, indent=None
            ), "Dump file does not contain the correct data, when deleting some keys"
        os.remove(full_name)

        # check QAOA dump with compression
        qaoa.dump(file_name, compresslevel=2, indent=None, prepend_id=True)
        assert os.path.isfile(
            full_name + ".gz"
        ), "Dump file does not exist, when compressing"
        with gzip.open(full_name + ".gz", "rb") as file:
            assert (
                file.read() == qaoa.dumps(indent=None).encode()
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
            "problem_type",
            "n_shots",
            "optimizer_method",
            "param_type",
            "init_type",
            "p",
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
            "result",
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
        ]
        expected_keys = {item: False for item in expected_keys}

        # test the keys, it will set the keys to True if they are found
        _test_keys_in_dict(obj, expected_keys)

        # Check if the dictionary has all the expected keys except the ones that were not included
        for key, value in expected_keys.items():
            if key not in exclude_keys:
                assert (
                    value is True
                ), f'Key "{key}" not found in the dictionary, when using "{method}" method.'
            else:
                assert (
                    value is False
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
            get_keys(qaoa.asdict(), expected_keys)
            print(expected_keys)
        """

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
        for device in [
            create_device(location="local", name="qiskit.shot_simulator"),
            create_device(location="local", name="vectorized"),
        ]:
            q = QAOA()
            q.set_device(device)
            q.set_circuit_properties(
                p=1, param_type="extended", init_type="rand", mixer_hamiltonian="x"
            )
            q.set_backend_properties(n_shots=50)
            q.set_classical_optimizer(maxiter=2, optimization_progress=True)
            q.set_exp_tags({"add_tag": "test"})
            q.set_header(
                project_id="8353185c-b175-4eda-9628-b4e58cb0e41b",
                description="test",
                run_by="oq",
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

        # for each rqaoa object, create a new rqaoa object from dict, json string, json file,
        # and compressed json file and compare them with the original object
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

    def test_qaoa_evaluate_circuit(self):
        """
        test the evaluate_circuit method
        """

        # problem
        problem = MinimumVertexCover.random_instance(
            n_nodes=6, edge_probability=0.8
        ).qubo

        # run qaoa with different param_type, and save the objcets in a list
        qaoas = []
        for param_type in PARAMS_CLASSES_MAPPER.keys():
            q = QAOA()
            q.set_circuit_properties(p=3, param_type=param_type, init_type="rand")
            q.compile(problem)
            qaoas.append(q)

        # for each qaoa object, test the evaluate_circuit method
        for q in qaoas:
            # evaluate the circuit with random dict of params
            params = {
                k: np.random.rand(*v.shape)
                for k, v in q.variate_params.asdict().items()
            }
            result = q.evaluate_circuit(params)
            assert (
                abs(result["cost"]) >= 0
            ), f"param_type={q.circuit_properties.param_type}. `evaluate_circuit` \
                should return a cost, here cost is {result['cost']}"
            assert (
                abs(result["uncertainty"]) > 0
            ), f"param_type={q.circuit_properties.param_type}. `evaluate_circuit` should return an uncertanty, \
                here uncertainty is {result['uncertainty']}"
            assert (
                len(result["measurement_results"]) > 0
            ), f"param_type={q.circuit_properties.param_type}. `evaluate_circuit` should return \
                a wavefunction when using a state-based simulator"

            # evaluate the circuit with a list of params, taking the params from the dict,
            # so we should get the same result
            params2 = []
            for value in params.values():
                params2 += value.flatten().tolist()
            result2 = q.evaluate_circuit(params2)
            assert (
                result == result2
            ), f"param_type={q.circuit_properties.param_type}. `evaluate_circuit` should return the same result \
                when passing a dict or a list of params"

            # evaluate the circuit with np.ndarray of params, taking the params from the dict,
            # so we should get the same result
            result2 = q.evaluate_circuit(np.array(params2))
            assert (
                result == result2
            ), f"param_type={q.circuit_properties.param_type}. `evaluate_circuit` should return the same result \
                when passing a dict or a list of params"

            # evaluate the circuit with the params as a QAOAVariationalBaseParams object,
            # so we should get the same result
            params_obj = deepcopy(q.variate_params)
            params_obj.update_from_raw(params2)
            result3 = q.evaluate_circuit(params_obj)
            assert (
                result == result3
            ), f"param_type={q.circuit_properties.param_type}. `evaluate_circuit` should return the same result \
                when passing a dict or a list of params"

            # run the circuit with the params manually, we should get the same result
            result4 = {}
            (
                result4["cost"],
                result4["uncertainty"],
            ) = q.backend.expectation_w_uncertainty(params_obj)
            result4["measurement_results"] = q.backend.wavefunction(params_obj)
            assert (
                result == result4
            ), f"param_type={q.circuit_properties.param_type}. `evaluate_circuit` should return the same result when \
                  passing the optimized params manually"

            # evaluate the circuit with a wrong input, it should raise an error
            error = False
            try:
                q.evaluate_circuit(1)
            except Exception:
                error = True
            assert (
                error
            ), f"param_type={q.circuit_properties.param_type}. `evaluate_circuit` should raise an error when \
                  passing a wrong input"

            # evaluate the circuit with a list longer than it should, it should raise an error
            error = False
            try:
                q.evaluate_circuit(params2 + [1])
            except Exception:
                error = True
            assert (
                error
            ), f"param_type={q.circuit_properties.param_type}. `evaluate_circuit` should raise an error when \
                passing a list longer than it should"

            # evaluate the circuit with a list shorter than it should, it should raise an error
            error = False
            try:
                q.evaluate_circuit(params2[:-1])
            except Exception:
                error = True
            assert (
                error
            ), f"param_type={q.circuit_properties.param_type}. `evaluate_circuit` should raise an error when \
                passing a list shorter than it should"

            # evaluate the circuit with a dict with a wrong key, it should raise an error
            error = False
            try:
                q.evaluate_circuit({**params, "wrong_key": 1})
            except Exception:
                error = True
            assert (
                error
            ), f"param_type={q.circuit_properties.param_type}. `evaluate_circuit` should raise an error \
                when passing a dict with a wrong key"

            # evaluate the circuit with a dict with a value longer than it should, it should raise an error
            error = False
            try:
                q.evaluate_circuit(
                    {**params, list(params.keys())[0]: np.random.rand(40)}
                )
            except Exception:
                error = True
            assert (
                error
            ), f"param_type={q.circuit_properties.param_type}. `evaluate_circuit` should raise an error when \
                passing a dict with a value longer than it should"

            # evaluate the circuit without passing any param, it should raise an error
            error = False
            try:
                q.evaluate_circuit()
            except Exception:
                error = True
            assert (
                error
            ), f"param_type={q.circuit_properties.param_type}. `evaluate_circuit` should raise an error when \
                  not passing any param"

        # check that it works with shots
        q = QAOA()
        device = create_device(location="local", name="qiskit.qasm_simulator")
        q.set_device(device)
        q.set_circuit_properties(p=3)

        # try to evaluate the circuit before compiling
        error = False
        try:
            q.evaluate_circuit()
        except Exception:
            error = True
        assert (
            error
        ), f"param_type={param_type}. `evaluate_circuit` should raise an error if the circuit is not compiled"

        # compile and evaluate the circuit, and check that the result is correct
        q.compile(problem)
        result = q.evaluate_circuit([1, 2, 1, 2, 1, 2])
        assert isinstance(
            result["measurement_results"], dict
        ), "When using a shot-based simulator, `evaluate_circuit` should return a dict of counts"
        assert (
            abs(result["cost"]) >= 0
        ), "When using a shot-based simulator, `evaluate_circuit` should return a cost"
        assert (
            abs(result["uncertainty"]) > 0
        ), "When using a shot-based simulator, `evaluate_circuit` should return an uncertanty"

        cost = cost_function(
            result["measurement_results"],
            q.backend.qaoa_descriptor.cost_hamiltonian,
            q.backend.cvar_alpha,
        )
        cost_sq = cost_function(
            result["measurement_results"],
            q.backend.qaoa_descriptor.cost_hamiltonian.hamiltonian_squared,
            q.backend.cvar_alpha,
        )
        uncertainty = np.sqrt(cost_sq - cost**2)
        assert (
            np.round(cost, 12) == result["cost"]
        ), "When using a shot-based simulator, `evaluate_circuit` not returning the correct cost"
        assert (
            np.round(uncertainty, 12) == result["uncertainty"]
        ), "When using a shot-based simulator, `evaluate_circuit` not returning the correct uncertainty"

        # check that it works with analytical simulator
        q = QAOA()
        device = create_device(location="local", name="analytical_simulator")
        q.set_device(device)
        q.set_circuit_properties(p=1, param_type="standard")
        q.compile(problem)
        result = q.evaluate_circuit([1, 2])
        assert (
            abs(result["cost"]) >= 0
        ), "When using an analytical simulator, `evaluate_circuit` should return a cost"
        assert (
            result["uncertainty"] is None
        ), "When using an analytical simulator, `evaluate_circuit` should return uncertainty None"
        assert (
            result["measurement_results"] is None
        ), "When using an analytical simulator, `evaluate_circuit` should return no measurement results"

    def test_change_properties_after_compilation(self):
        device = create_device(location="local", name="qiskit.shot_simulator")
        q = QAOA()
        q.compile(QUBO.random_instance(4))

        with self.assertRaises(ValueError):
            q.set_device(device)
        with self.assertRaises(ValueError):
            q.set_circuit_properties(
                p=1, param_type="standard", init_type="rand", mixer_hamiltonian="x"
            )
        with self.assertRaises(ValueError):
            q.set_backend_properties(prepend_state=None, append_state=None)
        with self.assertRaises(ValueError):
            q.set_classical_optimizer(
                maxiter=100, method="vgd", jac="finite_difference"
            )

    def test_numpy_serialize(self):
        np_qubo = NumberPartition([1, 2, 3]).qubo

        q = QAOA()
        q.compile(np_qubo)
        q.optimize()

        # add numpy results
        numpy_dict = {
            "000": np.int64(85),
            "100": np.int64(85),
            "010": np.int64(85),
            "111": 12,
        }
        numpy_cost = np.float64(85.123)

        q.result.optimized["intermediate"] = [
            numpy_cost,
            numpy_cost,
            numpy_cost,
            0.123,
            -123.123,
        ]
        q.result.intermediate["measurement_outcomes"] = [
            numpy_dict,
            numpy_dict,
            numpy_dict,
        ]
        q.result.optimized["measurement_outcomes"] = numpy_dict
        q.result.optimized["cost"] = numpy_cost

        q.dumps()


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
        assert cp.q is None
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
        assert r.rqaoa_parameters.original_hamiltonian is None
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
        if problem is None:
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
            run_by="OpenQAOA",
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
                    value is True
                ), f'Key "{key}" not found in the dictionary, when using "{method}" method.'
            else:
                assert (
                    value is False
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

        project_id = "d3a6f03b-1484-423a-8432-38e57c4e9ec7"

        # define the problem
        problem = QUBO.random_instance(n=8)
        problem.set_metadata(
            {"metadata_key1": "metadata_value1", "metadata_key2": "metadata_value2"}
        )

        r = RQAOA()
        r.set_header(project_id=project_id)
        r.set_exp_tags({"tag1": "value1", "tag2": "value2"})
        r.set_classical_optimizer(optimization_progress=True)
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

                # check that the intermediate mesuraments are empty
                for step in dictionary["data"]["result"]["intermediate_steps"]:
                    assert (
                        step["qaoa_results"]["intermediate"]["measurement_outcomes"]
                        == []
                    ), f"File {file_name} has intermediate mesuraments, but it should not have them."

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
                ), f"File {file_name} does not have intermediate mesuraments, but it should have them."

        assert rqaoa_files == 1, f"Expected 1 rqaoa file, but {rqaoa_files} were found."
        assert qaoa_files == len(
            r.result["atomic_ids"]
        ), f'Expected {len(r.result["atomic_ids"])} qaoa files, but {qaoa_files} were found.'

        # erease the files
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
            create_device(location="local", name="vectorized"),
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
                run_by="OpenQAOA",
                provider="-",
                target="vectorized",
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

        # for each rqaoa object, create a new rqaoa object from dict, json string, json file,
        # and compressed json file and compare them with the original object
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

    def test_change_properties_after_compilation(self):
        device = create_device(location="local", name="qiskit.shot_simulator")
        r = RQAOA()
        r.compile(QUBO.random_instance(4))

        with self.assertRaises(ValueError):
            r.set_device(device)
        with self.assertRaises(ValueError):
            r.set_circuit_properties(
                p=1, param_type="standard", init_type="rand", mixer_hamiltonian="x"
            )
        with self.assertRaises(ValueError):
            r.set_backend_properties(prepend_state=None, append_state=None)
        with self.assertRaises(ValueError):
            r.set_classical_optimizer(
                maxiter=100, method="vgd", jac="finite_difference"
            )
        with self.assertRaises(ValueError):
            r.set_rqaoa_parameters(rqaoa_type="adaptive", n_cutoff=3, n_steps=3)


if __name__ == "__main__":
    unittest.main()
