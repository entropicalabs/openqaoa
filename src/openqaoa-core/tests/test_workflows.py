import json
import os
import gzip
import unittest
import networkx as nw
import numpy as np
import datetime
from copy import deepcopy

from openqaoa import QAOA, RQAOA
from openqaoa.problems import NumberPartition
from openqaoa.algorithms import QAOAResult, RQAOAResult
from openqaoa.algorithms.baseworkflow import Workflow
from openqaoa.utilities import X_mixer_hamiltonian, XY_mixer_hamiltonian, is_valid_uuid, ground_state_hamiltonian
from openqaoa.algorithms.workflow_properties import (
    BackendProperties,
    ClassicalOptimizer,
    CircuitProperties,
)
from openqaoa.algorithms.rqaoa.rqaoa_workflow_properties import RqaoaParameters
from openqaoa.backends import create_device, DeviceLocal
from openqaoa.backends.cost_function import cost_function

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

    def test_qaoa_brute_force(self):
        """
        test the brute_force method
        """
        # problem
        problem = MinimumVertexCover.random_instance(
            n_nodes=6, edge_probability=0.8
        ).qubo

        # check if the brute force method gets the same results as the ground state hamiltonian method
        qaoa = QAOA()
        qaoa.compile(problem)
        qaoa.solve_brute_force()
        min_energy_bf, config_strings_bf = qaoa.brute_force_results["energy"], qaoa.brute_force_results["configuration"]
        (min_energy_gsh, config_strings_ghs) = ground_state_hamiltonian(qaoa.cost_hamil)
        assert (
            min_energy_bf == min_energy_gsh and config_strings_bf == config_strings_ghs
        ), f"The energy and config strings of brute forcing should match the ground state hamiltonian method results"

        # Check if an uncompiled QAOA raises an error when calling its brute_force method
        qaoa_uncompiled = QAOA()
        error = False
        try:
            qaoa_uncompiled.brute_force()
        except Exception:
            error = True
        assert(error), f"An uncompiled QAOA should raise an error when brute-forcing it"

        # Check if bounded=True disallows computation for more than 25 qubits
        qaoa = QAOA()
        large_problem = MaximumCut.random_instance(
            n_nodes=26, edge_probability=1
        ).qubo
        qaoa.compile(large_problem)
        error = False
        try:
            qaoa.solve_brute_force()
        except:
            error = True
        assert(error), f"Brute forcing should not compute for large problems (> 25 qubits) when bounded=True"

    

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

    def test_qaoa_evaluate_circuit_shot(self):

        # problem
        problem = MinimumVertexCover.random_instance(
            n_nodes=6, edge_probability=0.8
        ).qubo

        if "qiskit.qasm_simulator" not in SUPPORTED_LOCAL_SIMULATORS:
            self.skipTest(reason="Qiskit QASM Simulator is not available. Please install the qiskit plugin: openqaoa-qiskit.")
        else:
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

    def test_qaoa_evaluate_circuit_analytical_sim(self):

        # problem
        problem = MinimumVertexCover.random_instance(
            n_nodes=6, edge_probability=0.8
        ).qubo

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

        device = create_device(location="local", name="vectorized")
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
