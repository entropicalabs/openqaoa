import unittest
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from docplex.mp.model import Model

from openqaoa import QAOA
from openqaoa.problems import MinimumVertexCover
from openqaoa.qaoa_components import PauliOp, Hamiltonian
from openqaoa.algorithms.qaoa.qaoa_result import QAOAResult, most_probable_bitstring
from openqaoa.utilities import qaoa_probabilities
from openqaoa.problems.converters import FromDocplex2IsingModel
from openqaoa.backends import create_device


class TestingLoggerClass(unittest.TestCase):
    def test_attribute_existence(self):
        """
        This test checks that 'most_probable_states', 'evals', 'intermediate', 'optimized' are correctly
        created by the default workflow
        """

        # Create the problem
        g = nx.circulant_graph(6, [1])
        vc = MinimumVertexCover(g, field=1.0, penalty=10).qubo

        q = QAOA()
        q.compile(vc, verbose=False)
        q.optimize()

        assert set(
            ["most_probable_states", "evals", "intermediate", "optimized"]
        ) <= set([a for a in dir(q.result) if not a.startswith("__")])

    def test_most_probable_bitstring(self):

        cost_hamil = Hamiltonian(
            [
                PauliOp("ZZ", (0, 3)),
                PauliOp("ZZ", (0, 1)),
                PauliOp("ZZ", (1, 2)),
                PauliOp("ZZ", (2, 3)),
                PauliOp("Z", (0,)),
                PauliOp("Z", (1,)),
                PauliOp("Z", (2,)),
                PauliOp("Z", (3,)),
            ],
            [2.5, 2.5, 2.5, 2.5, 4.5, 4.5, 4.5, 4.5],
            12,
        )

        # From a statevector backend
        optimized_measurement_outcomes_sv = np.array(
            [
                0.18256422 - 0.1296918j,
                -0.03786405 - 0.13158363j,
                -0.03786405 - 0.13158363j,
                -0.01910622 + 0.3151214j,
                -0.03786405 - 0.13158363j,
                0.42313283 + 0.15529604j,
                -0.01910622 + 0.3151214j,
                0.016895 + 0.05117122j,
                -0.03786405 - 0.13158363j,
                -0.01910622 + 0.3151214j,
                0.42313283 + 0.15529604j,
                0.016895 + 0.05117122j,
                -0.01910622 + 0.3151214j,
                0.016895 + 0.05117122j,
                0.016895 + 0.05117122j,
                0.23799586 + 0.04022615j,
            ]
        )

        # from a shot simulator
        optimized_measurement_outcomes_shot = {
            "0010": 1,
            "0110": 2,
            "1000": 1,
            "1001": 13,
            "0000": 1,
            "1100": 10,
            "1110": 201,
            "1111": 37,
            "0101": 225,
            "1101": 57,
            "0011": 8,
            "1010": 228,
            "0111": 74,
            "1011": 115,
        }

        mps_sv = most_probable_bitstring(
            cost_hamil, qaoa_probabilities(optimized_measurement_outcomes_sv)
        )
        mps_shot = most_probable_bitstring(
            cost_hamil, optimized_measurement_outcomes_shot
        )

        assert set(["0101", "1010"]) == set(
            mps_sv["solutions_bitstrings"]
        )  # Equality for sv
        assert set(["0101", "1010"]) >= set(
            mps_shot["solutions_bitstrings"]
        )  # Subset for shot/qpu

    def test_plot_cost(self):

        # Create the problem
        g = nx.circulant_graph(6, [1])
        vc = MinimumVertexCover(g, field=1.0, penalty=10).qubo

        q = QAOA()
        q.compile(vc, verbose=False)
        q.optimize()

        q.result.plot_cost()

    def test_plot_probabilities(self):

        # Create the problem
        g = nx.circulant_graph(6, [1])
        vc = MinimumVertexCover(g, field=1.0, penalty=10).qubo

        # first for state-vector based simulator:
        q_sv = QAOA()
        q_sv.set_circuit_properties(p=3, init_type="ramp")
        q_sv.compile(vc)

        q_sv.optimize()

        q_sv.result.plot_probabilities()

        # then for shot based simulator:
        q_shot = QAOA()
        q_shot_dev = create_device(location="local", name="qiskit.shot_simulator")
        q_shot.set_device(q_shot_dev)

        q_shot.compile(vc)
        q_shot.optimize()

        q_shot.result.plot_probabilities()

    def test_plot_n_shots(self):

        # Create the problem
        g = nx.circulant_graph(6, [1])
        vc = MinimumVertexCover(g, field=1.0, penalty=10).qubo

        for method in ["cans", "icans"]:
            q = QAOA()
            q.set_classical_optimizer(
                method=method,
                maxiter=50,
                jac="finite_difference",
                optimizer_options={
                    "stepsize": 0.01,
                    "n_shots_min": 5,
                    "n_shots_max": 50,
                    "n_shots_budget": 1000,
                },
            )
            q.compile(vc, verbose=False)
            q.optimize()

            q.result.plot_n_shots()

            # all combinations to check
            test_dict = {
                "none": (
                    None,
                    {
                        "label": [None, ["t1", "t2"]],
                        "linestyle": ["-", ["--", "-"]],
                        "color": [None, ["red", "green"]],
                    },
                ),
                "int": (
                    0,
                    {
                        "label": [None, "t1", ["t2"]],
                        "linestyle": ["-", ["--"]],
                        "color": [None, ["red"], "green"],
                    },
                ),
                "list_one": (
                    1,
                    {
                        "label": [None, "t1", ["t2"]],
                        "linestyle": ["-", ["--"]],
                        "color": [None, ["red"], "green"],
                    },
                ),
                "list_two": (
                    [0, 1],
                    {
                        "label": [None, ["t1", "t2"]],
                        "linestyle": ["-", ["--", "-"]],
                        "color": [None, ["red", "green"]],
                    },
                ),
            }
            # using the test_dict we plot with different options
            for value in test_dict.values():
                for label, line, color in zip(
                    value[1]["label"], value[1]["linestyle"], value[1]["color"]
                ):
                    q.result.plot_n_shots(
                        param_to_plot=value[0],
                        label=label,
                        linestyle=line,
                        color=color,
                        title=f"method: {method}, param_to_plot: {value[0]}, label: {label}, linestyle: {line}, color: {color}",
                    )
                    plt.close()

            # function to test that errors are raised, when trying to plot with incorrect inputs
            def test_incorrect_arguments(argument: str, inputs_to_try: list):
                for x in inputs_to_try:
                    error = False
                    try:
                        q.result.plot_n_shots(**{argument: x})
                    except Exception as e:
                        assert len(str(e)) > 0, "No error message was raised"
                        error = True
                    assert error, "Error not raised, when it should have been"
                plt.close()

            # check that errors are raised, when trying to plot with incorrect inputs
            test_incorrect_arguments(
                argument="param_to_plot", inputs_to_try=["0", 2, [0, 1, 2]]
            )
            test_incorrect_arguments(
                argument="linestyle",
                inputs_to_try=[0, ["one", "two", "three"], [1, "two"]],
            )
            test_incorrect_arguments(
                argument="label", inputs_to_try=[0, ["one", "two", "three"], [1, "two"]]
            )
            test_incorrect_arguments(
                argument="color", inputs_to_try=[0, ["b", "c", "g"], [1, "g"]]
            )

    def test_get_counts(self):

        # measurement outcome from a statevector_simulator backend
        optimized_measurement_outcomes_sv = np.array(
            [
                0.18256422 - 0.1296918j,
                -0.03786405 - 0.13158363j,
                -0.03786405 - 0.13158363j,
                -0.01910622 + 0.3151214j,
                -0.03786405 - 0.13158363j,
                0.42313283 + 0.15529604j,
                -0.01910622 + 0.3151214j,
                0.016895 + 0.05117122j,
                -0.03786405 - 0.13158363j,
                -0.01910622 + 0.3151214j,
                0.42313283 + 0.15529604j,
                0.016895 + 0.05117122j,
                -0.01910622 + 0.3151214j,
                0.016895 + 0.05117122j,
                0.016895 + 0.05117122j,
                0.23799586 + 0.04022615j,
            ]
        )

        counts_from_sv = qaoa_probabilities(optimized_measurement_outcomes_sv)

        # measurement outcome from a shot simulator or QPU
        optimized_measurement_outcomes_shot = {
            "0010": 1,
            "0110": 2,
            "1000": 1,
            "1001": 13,
            "0000": 1,
            "1100": 10,
            "1110": 201,
            "1111": 37,
            "0101": 225,
            "1101": 57,
            "0011": 8,
            "1010": 228,
            "0111": 74,
            "1011": 115,
        }

        assert optimized_measurement_outcomes_shot == QAOAResult.get_counts(
            optimized_measurement_outcomes_shot
        )
        assert counts_from_sv == QAOAResult.get_counts(
            optimized_measurement_outcomes_sv
        )

    def test_best_result(self):
        """Test lowest_cost_bitstring attribute and FromDocplex2IsingModel model generation"""

        mdl = Model()  # Create a docplex model
        x = mdl.binary_var_list(5, name="x")
        mdl.maximize(np.sum(x))  # Add an objective function
        mdl.add_constraint(np.sum(x[:3]) == 0)
        q = QAOA()
        qubo = FromDocplex2IsingModel(mdl)  # translate the docplex model to qubo
        q.compile(qubo.ising_model)  # complining the ising representation of the qubo
        q.optimize()
        result = q.result
        lowest_energy = result.lowest_cost_bitstrings()
        assert (
            lowest_energy["solutions_bitstrings"][0] == "00011"
        )  # bitstring optimal solution
        assert np.isclose(lowest_energy["bitstrings_energies"][0], -2.0)  # solution


if __name__ == "__main__":
    unittest.main()
