import unittest
import itertools
import networkx as nw
import numpy as np

from openqaoa import QAOA, RQAOA
from openqaoa.algorithms import QAOAResult, RQAOAResult
from openqaoa.backends.qaoa_backend import (
    DEVICE_NAME_TO_OBJECT_MAPPER,
    DEVICE_ACCESS_OBJECT_MAPPER,
)
from openqaoa.backends import create_device
from openqaoa.backends.basebackend import QAOABaseBackendStatevector
from openqaoa.backends.devices_core import SUPPORTED_LOCAL_SIMULATORS
from openqaoa.problems import MinimumVertexCover, QUBO, MaximumCut
from openqaoa.qaoa_components import Hamiltonian


def _compare_qaoa_results(dict_old, dict_new, bool_cmplx_str):
    for key in dict_old.keys():
        if key == "cost_hamiltonian":  ## CHECK WHAT DO WITH THIS
            pass
        elif key == "_QAOAResult__type_backend":
            if issubclass(dict_old[key], QAOABaseBackendStatevector):
                assert (
                    dict_new[key] == QAOABaseBackendStatevector
                ), "Type of backend is not correct, complex_to_string = {}".format(
                    bool_cmplx_str
                )
            else:
                assert (
                    dict_new[key] == ""
                ), "Type of backend should be empty string, complex_to_string = {}".format(
                    bool_cmplx_str
                )
        elif key == "optimized":
            for key2 in dict_old[key].keys():
                if key2 == "measurement_outcomes":
                    assert np.all(
                        dict_old[key][key2] == dict_new[key][key2]
                    ), "Optimized params are not the same, complex_to_string = {}".format(
                        bool_cmplx_str
                    )
                else:
                    assert (
                        dict_old[key][key2] == dict_new[key][key2]
                    ), "Optimized params are not the same, complex_to_string = {}".format(
                        bool_cmplx_str
                    )
        elif key == "intermediate":
            for key2 in dict_old[key].keys():
                if key2 == "measurement_outcomes":
                    for step in range(len(dict_old[key][key2])):
                        assert np.all(
                            dict_old[key][key2][step] == dict_new[key][key2][step]
                        ), "Intermediate params are not the same, complex_to_string = {}".format(
                            bool_cmplx_str
                        )
                else:
                    assert (
                        dict_old[key][key2] == dict_new[key][key2]
                    ), "Intermediate params are not the same, complex_to_string = {}".format(
                        bool_cmplx_str
                    )
        else:
            assert (
                dict_old[key] == dict_new[key]
            ), f"{key} is not the same, complex_to_string = {bool_cmplx_str}"


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


class TestingResultOutputs(unittest.TestCase):

    """
    Test the Results Output after an optimization loop
    """

    def test_flags_result_outputs_workflow(self):
        """
        Run an optimization problem for 5 iterations.
        Should expect certain fields of the results output to be filled based
        on some of the users inputs. (Default settings)
        Can be checked for cobyla.

        Check for all available supported local backends.
        """

        g = nw.circulant_graph(3, [1])
        vc = MinimumVertexCover(g, field=1.0, penalty=10).qubo

        choice_combination = list(
            itertools.product([True, False], [True, False], [True, False])
        )
        recorded_evals = [0, 5]

        for device_name in SUPPORTED_LOCAL_SIMULATORS:
            for each_choice in choice_combination:
                q = QAOA()
                q.set_classical_optimizer(
                    method="cobyla",
                    parameter_log=each_choice[0],
                    cost_progress=each_choice[1],
                    optimization_progress=each_choice[2],
                    maxiter=5,
                )
                device = create_device("local", device_name)
                print(device.device_name)
                q.set_device(device)
                q.compile(vc)
                q.optimize()
                self.assertEqual(
                    recorded_evals[each_choice[0]], len(q.result.intermediate["angles"])
                )
                self.assertEqual(
                    recorded_evals[each_choice[1]], len(q.result.intermediate["cost"])
                )
                self.assertEqual(
                    recorded_evals[each_choice[2]],
                    len(q.result.intermediate["measurement_outcomes"]),
                )

    def test_qaoa_result_asdict(self):
        """
        Test the qaoa result.asdict method
        """

        # run the QAOA
        qaoa = QAOA()
        qaoa.compile(problem=QUBO.random_instance(n=8))
        qaoa.optimize()

        # get dict
        results_dict = qaoa.result.asdict()

        # list of expected keys
        expected_keys = [
            "method",
            "cost_hamiltonian",
            "n_qubits",
            "terms",
            "qubit_indices",
            "pauli_str",
            "phase",
            "coeffs",
            "constant",
            "qubits_pairs",
            "qubits_singles",
            "single_qubit_coeffs",
            "pair_qubit_coeffs",
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
            "angles",
            "cost",
            "measurement_outcomes",
            "job_id",
        ]

        # we append all the keys that we find in rqaoa.results, so if we introduce a new key, we will know that we need to update the result.asdict method
        for key in vars(qaoa.result).keys():
            if not key in expected_keys and not "_QAOAResult__" in key:
                expected_keys.append(key)

        # create a dictionary with all the expected keys and set them to False
        expected_keys_dict = {item: False for item in expected_keys}

        # test the keys, it will set the keys to True if they are found
        _test_keys_in_dict(results_dict, expected_keys_dict)

        # Check if the dictionary has all the expected keys
        for key, value in expected_keys_dict.items():
            assert (
                value == True
            ), f"Key {key} was not found in the dictionary of the QAOA Result class."

        ## now we repeat the same test but we do not include the cost hamiltonian

        # get dict without cost hamiltonian
        results_dict = qaoa.result.asdict(keep_cost_hamiltonian=False)

        # expected keys
        expected_keys_dict = {item: False for item in expected_keys}
        expected_keys_not_in_dict = [
            "cost_hamiltonian",
            "n_qubits",
            "terms",
            "qubit_indices",
            "pauli_str",
            "phase",
            "coeffs",
            "constant",
            "qubits_pairs",
            "qubits_singles",
            "single_qubit_coeffs",
            "pair_qubit_coeffs",
        ]

        # test the keys, it will set the keys to True if they are found, except the ones that were not included which should be those in expected_keys_not_in_dict
        _test_keys_in_dict(results_dict, expected_keys_dict)

        # Check if the dictionary has all the expected keys except the ones that were not included
        for key, value in expected_keys_dict.items():
            if not key in expected_keys_not_in_dict:
                assert (
                    value == True
                ), f"Key {key} was not found in the dictionary of the RQAOAResult class."
            else:
                assert (
                    value == False
                ), f"Key {key} was found in the dictionary of the RQAOAResult class, but it should not have been."

        ## now we repeat the same test but we do not include some keys

        # get dict without some values
        results_dict = qaoa.result.asdict(
            exclude_keys=["solutions_bitstrings", "method"]
        )

        # expected keys
        expected_keys_dict = {item: False for item in expected_keys}
        expected_keys_not_in_dict = ["solutions_bitstrings", "method"]

        # test the keys, it will set the keys to True if they are found, except the ones that were not included which should be those in expected_keys_not_in_dict
        _test_keys_in_dict(results_dict, expected_keys_dict)

        # Check if the dictionary has all the expected keys except the ones that were not included
        for key, value in expected_keys_dict.items():
            if not key in expected_keys_not_in_dict:
                assert (
                    value == True
                ), f"Key {key} was not found in the dictionary of the RQAOAResult class."
            else:
                assert (
                    value == False
                ), f"Key {key} was found in the dictionary of the RQAOAResult class, but it should not have been."

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
            get_keys(rqaoa.results.asdict(), expected_keys)
            print(expected_keys)
        """

    # test eval_number
    def test_qaoa_result_eval_number(self):
        """
        Test the eval_number method for the QAOA result class
        """

        for method in ["cobyla", "spsa", "vgd", "newton", "natural_grad_descent"]:
            # run the QAOA and get the results
            q = QAOA()
            q.set_classical_optimizer(
                maxiter=15,
                method=method,
                jac="finite_difference",
                hess="finite_difference",
            )
            q.compile(problem=QUBO.random_instance(n=8))
            q.optimize()

            # test the eval_number method
            assert (
                q.result.intermediate["cost"].index(min(q.result.intermediate["cost"]))
                + 1
                == q.result.optimized["eval_number"]
            ), "optimized eval_number does not return the correct number of the optimized evaluation, when using {} method".format(
                method
            )

    def test_qaoa_results_from_dict(self):
        """
        test loading the QAOA Result object from a dictionary
        methods: from_dict
        """

        # problem
        maxcut_qubo = MaximumCut(
            nw.generators.fast_gnp_random_graph(n=6, p=0.6, seed=42)
        ).qubo

        # run qaoa with different devices, and save the objects in a list
        qaoas = []
        for device in [
            create_device(location="local", name=each_device_name)
            for each_device_name in SUPPORTED_LOCAL_SIMULATORS
            if each_device_name != "analytical_simulator"
        ]:
            q = QAOA()
            q.set_device(device)
            q.set_circuit_properties(
                p=1, param_type="extended", init_type="rand", mixer_hamiltonian="x"
            )
            q.set_backend_properties(prepend_state=None, append_state=None)
            q.set_classical_optimizer(maxiter=10, optimization_progress=True)

            q.compile(maxcut_qubo)

            q.optimize()

            qaoas.append(q)

        for q in qaoas:
            results = q.result
            for bool_cmplx_str in [True, False]:
                results_from_dict = QAOAResult.from_dict(
                    results.asdict(complex_to_string=bool_cmplx_str)
                )

                # assert that results_from_dict is intance of Result
                assert isinstance(
                    results_from_dict, QAOAResult
                ), "results_from_dict is not an instance of Result"

                _compare_qaoa_results(
                    results.__dict__, results_from_dict.__dict__, bool_cmplx_str
                )

    def test_qaoa_results_calculate_statistics_full(self):

        maxcut_qubo = MaximumCut(
            nw.generators.fast_gnp_random_graph(n=6, p=0.6, seed=42)
        ).qubo

        qaoas = []
        for device_name in SUPPORTED_LOCAL_SIMULATORS:
            if device_name == "analytical_simulator":
                continue

            qaoa = QAOA()

            qaoa.set_device(create_device("local", device_name))
            qaoa.set_classical_optimizer(optimization_progress=True)

            qaoa.compile(maxcut_qubo)
            qaoa.optimize()

            qaoas.append(qaoa)

        for qaoa in qaoas:
            result = qaoa.result.calculate_statistics(include_intermediate=True)
            result_optimized = result['optimized']
            result_intermediate = result['intermediate']
            optimized_sorted, optimized_mean, optimized_std_deviation = list(result_optimized['sorted'].values()), \
                result_optimized['mean'], result_optimized['std_deviation']

            for ri in result_intermediate:
                intermediate_sorted, intermediate_mean, intermediate_std_deviation = list(ri['sorted'].values()), ri['mean'], ri['std_deviation']

                assert all(intermediate_sorted[i] >= intermediate_sorted[i + 1] for i in range(len(intermediate_sorted) - 1)), "Counts in descending order."
                assert intermediate_mean > 0, "Mean is greater then zero."
                assert intermediate_std_deviation > 0, "Standard deviation is greater than zero."

            assert all(optimized_sorted[i] >= optimized_sorted[i + 1] for i in range(len(optimized_sorted) - 1)), "Counts in descending order."
            assert optimized_mean > 0, "Mean is greater then zero."
            assert optimized_std_deviation > 0, "Standard deviation is greater than zero."

    def test_qaoa_results_calculate_statistics_without_intermediate(self):

        maxcut_qubo = MaximumCut(
            nw.generators.fast_gnp_random_graph(n=6, p=0.6, seed=42)
        ).qubo

        qaoas = []
        for device_name in SUPPORTED_LOCAL_SIMULATORS:
            if device_name == "analytical_simulator":
                continue

            qaoa = QAOA()
            qaoa.set_device(create_device("local", device_name))
            qaoa.set_classical_optimizer(optimization_progress=False)

            qaoa.compile(maxcut_qubo)
            qaoa.optimize()

            qaoas.append(qaoa)

        for qaoa in qaoas:
            result = qaoa.result.calculate_statistics(include_intermediate=False)
            result_optimized = result['optimized']
            optimized_sorted, optimized_mean, optimized_std_deviation = list(result_optimized['sorted'].values()), \
                result_optimized['mean'], result_optimized['std_deviation']

            assert result['intermediate'] == [], "Statistics for intermediate measurements are empty."
            assert all(optimized_sorted[i] >= optimized_sorted[i + 1] for i in range(len(optimized_sorted) - 1)), "Counts in descending order."
            assert optimized_mean > 0, "Mean is greater then zero."
            assert optimized_std_deviation > 0, "Standard deviation is greater than zero."

    def test_qaoa_results_calculate_statistics_raise_value_error(self):
        """
        The method raise an error if the user requires statistics on all 
        intermediate measurement outcomes but hasn't specified saving them during optimization.
        """
        maxcut_qubo = MaximumCut(
            nw.generators.fast_gnp_random_graph(n=6, p=0.6, seed=42)
        ).qubo

        qaoa = QAOA()
        qaoa.set_device(create_device(location="local", name="vectorized"))
        qaoa.set_classical_optimizer(optimization_progress=False)

        qaoa.compile(maxcut_qubo)
        qaoa.optimize()

        try:
            qaoa.result.calculate_statistics(include_intermediate=True)
        except ValueError as e:
            self.assertEqual(
                str(e),
                "The underlying QAOA object does not seem to have any intermediate measurement result. Please, consider saving intermediate measurements during optimization by setting `optimization_progress=True` in your workflow.",
            )


class TestingRQAOAResultOutputs(unittest.TestCase):
    """
    Test the  Results Output after a full RQAOA loop
    """

    def __run_rqaoa(
        self,
        type="custom",
        eliminations=1,
        p=1,
        param_type="standard",
        mixer="x",
        method="cobyla",
        maxiter=15,
        name_device="vectorized",
    ):
        """
        private function to run the RQAOA
        """

        n_qubits = 6
        n_cutoff = 3
        g = nw.circulant_graph(n_qubits, [1])
        problem = MinimumVertexCover(g, field=1.0, penalty=10).qubo

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
        r.compile(problem)
        r.optimize()

        return r.result

    def test_rqaoa_result_outputs(self):
        """
        Test the result outputs for the RQAOA class
        """

        n_qubits = 6
        n_cutoff = 3

        # Test for the standard RQAOA
        results = self.__run_rqaoa()
        assert isinstance(
            results, RQAOAResult
        ), "Results of RQAOA are not of type RQAOAResult"
        for key in results["solution"].keys():
            assert len(key) == n_qubits, "Number of qubits solution is not correct"
        assert isinstance(results["classical_output"]["minimum_energy"], float)
        assert isinstance(results["classical_output"]["optimal_states"], list)
        for rule_list in results["elimination_rules"]:
            for rule in rule_list:
                assert isinstance(
                    rule, dict
                ), "Elimination rule item is not a dictionary"
        assert isinstance(results["schedule"], list), "Schedule is not a list"
        assert (
            sum(results["schedule"]) + n_cutoff == n_qubits
        ), "Schedule is not correct"
        for step in results["intermediate_steps"]:
            assert isinstance(step["problem"], QUBO), "problem is not of type QUBO"
            assert isinstance(
                step["qaoa_results"], QAOAResult
            ), "QAOA_results is not of type QAOA Results"
            assert isinstance(
                step["exp_vals_z"], np.ndarray
            ), "exp_vals_z is not of type numpy array"
            assert isinstance(
                step["corr_matrix"], np.ndarray
            ), "corr_matrix is not of type numpy array"
        assert isinstance(
            results["number_steps"], int
        ), "Number of steps is not an integer"

        # Test for the adaptive RQAOA
        results = self.__run_rqaoa(type="adaptive")
        assert isinstance(
            results, RQAOAResult
        ), "Results of RQAOA are not of type RQAOAResult"
        for key in results["solution"].keys():
            assert len(key) == n_qubits, "Number of qubits solution is not correct"
        assert isinstance(results["classical_output"]["minimum_energy"], float)
        assert isinstance(results["classical_output"]["optimal_states"], list)
        for rule_list in results["elimination_rules"]:
            for rule in rule_list:
                assert isinstance(
                    rule, dict
                ), "Elimination rule item is not a dictionary"
        assert isinstance(results["schedule"], list), "Schedule is not a list"
        assert (
            sum(results["schedule"]) + n_cutoff == n_qubits
        ), "Schedule is not correct"
        for step in results["intermediate_steps"]:
            assert isinstance(step["problem"], QUBO), "QUBO is not of type QUBO"
            assert isinstance(
                step["qaoa_results"], QAOAResult
            ), "QAOA_results is not of type QAOA Results"
            assert isinstance(
                step["exp_vals_z"], np.ndarray
            ), "exp_vals_z is not of type numpy array"
            assert isinstance(
                step["corr_matrix"], np.ndarray
            ), "corr_matrix is not of type numpy array"
        assert isinstance(
            results["number_steps"], int
        ), "Number of steps is not an integer"

    def test_rqaoa_result_methods_steps(self):
        """
        Test the methods for the RQAOAResult class for the steps
        """

        # run the RQAOA
        results = self.__run_rqaoa()

        # test the solution method
        assert (
            results.get_solution() == results["solution"]
        ), "get_solution method is not correct"

        # test the methods for the intermediate steps
        for i in range(results["number_steps"]):
            # methods for intermediate qaao results
            assert (
                results.get_qaoa_results(i)
                == results["intermediate_steps"][i]["qaoa_results"]
            ), "get_qaoa_results method is not correct"
            assert (
                results.get_qaoa_optimized_angles(i)
                == results.get_qaoa_results(i).optimized["angles"]
            ), "get_qaoa_optimized_angles method is not correct"

            # methods for intermediate qubo
            assert (
                results.get_problem(i) == results["intermediate_steps"][i]["problem"]
            ), "get_qubo method is not correct"
            assert isinstance(
                results.get_hamiltonian(i), Hamiltonian
            ), "get_hamiltonian method is not correct"

            # methods for intermediate exp_vals_z and corr_matrix
            assert (
                results.get_exp_vals_z(i)
                is results["intermediate_steps"][i]["exp_vals_z"]
            ), "get_exp_vals_z method is not correct"
            assert (
                results.get_corr_matrix(i)
                is results["intermediate_steps"][i]["corr_matrix"]
            ), "get_corr_matrix method is not correct"

    def test_rqaoa_result_plot_corr_matrix(self):
        """
        Test the plot_corr_matrix method for the RQAOAResult class
        """

        # run the RQAOA
        results = self.__run_rqaoa()

        # test the plot_corr_matrix method
        for i in range(results["number_steps"]):
            results.plot_corr_matrix(step=i)

    def test_rqaoa_result_asdict(self):
        """
        Test the plot_exp_vals_z method for the RQAOAResult class
        """

        # run the RQAOA
        results = self.__run_rqaoa()

        # get dict
        results_dict = results.asdict()

        # create a list of expected keys
        expected_keys = [
            "solution",
            "classical_output",
            "minimum_energy",
            "optimal_states",
            "elimination_rules",
            "singlet",
            "bias",
            "pair",
            "correlation",
            "schedule",
            "intermediate_steps",
            "problem",
            "terms",
            "weights",
            "constant",
            "n",
            "qaoa_results",
            "method",
            "cost_hamiltonian",
            "n_qubits",
            "qubit_indices",
            "pauli_str",
            "phase",
            "coeffs",
            "qubits_pairs",
            "qubits_singles",
            "single_qubit_coeffs",
            "pair_qubit_coeffs",
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
            "angles",
            "cost",
            "measurement_outcomes",
            "job_id",
            "exp_vals_z",
            "corr_matrix",
            "number_steps",
        ]

        # we append all the keys that we find in rqaoa.results, so if we introduce a new key, we will know that we need to update the result.asdict method
        for key in results.keys():
            if not key in expected_keys:
                expected_keys.append(key)
        for key in results["intermediate_steps"][0].keys():
            if not key in expected_keys:
                expected_keys.append(key)

        # dictionary with all the expected keys and set them to False
        expected_keys = {item: False for item in expected_keys}

        # test the keys, it will set the keys to True if they are found
        _test_keys_in_dict(results_dict, expected_keys)

        # Check if the dictionary has all the expected keys except the ones that were not included
        for key, value in expected_keys.items():
            assert (
                value == True
            ), f"Key {key} was not found in the dictionary of the RQAOAResult class."

        ## now we repeat the same test but we do not include some keys

        # get dict without some values
        results_dict = results.asdict(exclude_keys=["solutions_bitstrings", "method"])

        # expected keys
        expected_keys_dict = {item: False for item in expected_keys}
        expected_keys_not_in_dict = ["solutions_bitstrings", "method"]

        # test the keys, it will set the keys to True if they are found, except the ones that were not included which should be those in expected_keys_not_in_dict
        _test_keys_in_dict(results_dict, expected_keys_dict)

        # Check if the dictionary has all the expected keys except the ones that were not included
        for key, value in expected_keys_dict.items():
            if not key in expected_keys_not_in_dict:
                assert (
                    value == True
                ), f"Key {key} was not found in the dictionary of the RQAOAResult class."
            else:
                assert (
                    value == False
                ), f"Key {key} was found in the dictionary of the RQAOAResult class, but it should not have been."

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
            get_keys(rqaoa.results.asdict(), expected_keys)
            print(expected_keys)
        """

    def test_rqaoa_results_from_dict(self):
        """
        test loading the QAOA Result object from a dictionary
        methods: from_dict
        """

        # problem
        maxcut_qubo = MaximumCut(
            nw.generators.fast_gnp_random_graph(n=6, p=0.6, seed=42)
        ).qubo

        # run rqaoa with different devices, and save the objcets in a list
        rqaoas = []

        for device in [
            create_device(location="local", name=each_device_name)
            for each_device_name in SUPPORTED_LOCAL_SIMULATORS
            if each_device_name != "analytical_simulator"
        ]:
            r = RQAOA()
            r.set_device(device)
            r.set_circuit_properties(
                p=1, param_type="extended", init_type="rand", mixer_hamiltonian="x"
            )
            r.set_backend_properties(prepend_state=None, append_state=None)
            r.set_classical_optimizer(maxiter=10, optimization_progress=True)

            r.compile(maxcut_qubo)

            r.optimize()

            rqaoas.append(r)

        # for each rqaoa object, we check that we can create a new results object from the dictionary of the old one
        for r in rqaoas:
            new_results = RQAOAResult.from_dict(r.result.asdict())
            old_results = r.result

            # assert that new_results is an instance of RQAOAResult
            assert isinstance(
                new_results, RQAOAResult
            ), "new_results is not an instance of RQAOAResult"

            for key in old_results:
                if key == "intermediate_steps":
                    for i in range(len(old_results[key])):
                        for key2 in old_results[key][i]:
                            if key2 == "problem":
                                assert (
                                    old_results[key][i][key2].asdict()
                                    == new_results[key][i][key2].asdict()
                                ), f"{key2} is not the same"
                            elif key2 == "qaoa_results":
                                _compare_qaoa_results(
                                    old_results[key][i][key2].asdict(),
                                    new_results[key][i][key2].asdict(),
                                    None,
                                )
                            else:
                                assert np.all(
                                    old_results[key][i][key2]
                                    == new_results[key][i][key2]
                                ), f"{key2} is not the same"
                else:
                    assert (
                        old_results[key] == new_results[key]
                    ), f"{key} is not the same"


if __name__ == "__main__":
    unittest.main()
