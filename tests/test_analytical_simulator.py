import unittest
import numpy as np

from openqaoa.backends.qaoa_analytical_sim import QAOABackendAnalyticalSimulator
from openqaoa.algorithms import QAOA, RQAOA
from openqaoa.problems import MaximumCut
from openqaoa.backends.qaoa_device import create_device
from openqaoa.utilities import (
    X_mixer_hamiltonian,
    XY_mixer_hamiltonian,
    ring_of_disagrees,
    random_k_regular_graph,
)
from openqaoa.qaoa_components import QAOADescriptor, QAOAVariationalStandardParams

"""
A set of tests for the analytical simulator backend which computes the energy of a given quantum circuit as a function of beta and gamma.
"""


def Disagrees_SetUp(n_qubits, p, mixer_hamil, betas, gammas):
    """
    Helper function for the tests below
    """

    register = range(n_qubits)
    cost_hamil = ring_of_disagrees(register)

    qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p)
    variational_params_std = QAOAVariationalStandardParams(
        qaoa_descriptor, betas, gammas
    )
    # Get the part of the Hamiltonian proportional to the identity

    return register, cost_hamil, qaoa_descriptor, variational_params_std


class TestingQAOABackendAnalyticalSimulator(unittest.TestCase):
    """
    Unittest based testing of QAOABackendAnalyticalSimulator
    """

    def test_expectation(self):
        """
        Testing if the analytical formula returns the energy expectation value for a given beta and gamma on an easy problem (Ring of disagree).
        """
        n_qubits = 8
        p = 1
        mixer_hamil = X_mixer_hamiltonian(n_qubits)
        betas = [np.pi / 8]
        gammas = [np.pi / 4]
        register, cost_hamil, qaoa_descriptor, variate_params = Disagrees_SetUp(
            n_qubits, p, mixer_hamil, betas, gammas
        )

        backend_analytical = QAOABackendAnalyticalSimulator(qaoa_descriptor)
        exp_val = backend_analytical.expectation(variate_params)

        # Check correct expecation value
        assert np.isclose(exp_val, -6)

    def test_p_not_1_fails(self):
        """
        Testing if the analytical backend fails if the number of layers, p, is different than 1.
        """
        n_qubits = 8
        p = 2
        mixer_hamil = X_mixer_hamiltonian(n_qubits)
        betas = [np.pi / 8, np.pi / 8]
        gammas = [np.pi / 4, np.pi / 4]
        register, cost_hamil, qaoa_descriptor, variate_params = Disagrees_SetUp(
            n_qubits, p, mixer_hamil, betas, gammas
        )
        exception = False
        try:
            backend_analytical = QAOABackendAnalyticalSimulator(qaoa_descriptor)
        except:
            exception = True

        assert exception, "p != 1 didn't fail."

    def test_different_mixer_fails(self):
        """
        Testing if the analytical backend fails if the mixer hamiltonian is different than X.
        """
        exception = False
        n_qubits = 8
        p = 1
        mixer_hamil = XY_mixer_hamiltonian(n_qubits)
        betas = [np.pi / 8]
        gammas = [np.pi / 4]
        register, cost_hamil, qaoa_descriptor, variate_params = Disagrees_SetUp(
            n_qubits, p, mixer_hamil, betas, gammas
        )
        try:
            backend_analytical = QAOABackendAnalyticalSimulator(qaoa_descriptor)
        except:
            exception = True

        assert exception, "XY mixer Hamiltonian didn't fail."

    def test_not_standard_params_fails(self):
        """
        Testing if the analytical backend fails if the parametrization of the circuit is extended or fourier.
        """
        # Create a 3-regular weighted graph and a qubo problem
        g = random_k_regular_graph(
            degree=3, nodes=range(8), seed=2642, weighted=True, biases=False
        )
        maxcut_qubo = MaximumCut(g).qubo

        # Testing for Extended params
        exception = False
        q = QAOA()
        q.set_circuit_properties(
            p=1, param_type="extended", init_type="rand", mixer_hamiltonian="x"
        )
        q.set_classical_optimizer(method="nelder-mead", maxiter=10)
        analytical_device = create_device(location="local", name="analytical_simulator")
        q.set_device(analytical_device)
        q.compile(maxcut_qubo)
        try:
            q.optimize()
        except:
            exception = True
        assert exception, "Extended params didn't fail"

        # Testing for Fourier params
        q = QAOA()
        q.set_device(analytical_device)
        q.set_circuit_properties(
            p=1, param_type="fourier", init_type="rand", mixer_hamiltonian="x"
        )
        exception = False
        try:
            q.optimize()
        except:
            exception = True
        assert exception, "Fourier params didn't fail"

    def test_end_to_end_rqaoa(self):
        """
        Testing the whole rqaoa workflow if the device is set to 'analytical_simulator'.
        """
        # Create a 3-regular weighted graph and a qubo problem
        g = random_k_regular_graph(
            degree=3, nodes=range(8), seed=2642, weighted=True, biases=False
        )
        maxcut_qubo = MaximumCut(g).qubo

        # Define the RQAOA object and set its params
        r = RQAOA()
        r.set_rqaoa_parameters(steps=1, n_cutoff=3)
        r.set_circuit_properties(
            p=1,
            init_type="custom",
            variational_params_dict={"betas": [0.26], "gammas": [0.42]},
            mixer_hamiltonian="x",
        )

        # Define the device to be the analytical simulator
        device = create_device(location="local", name="analytical_simulator")
        r.set_device(device)

        # Set the classical method used to optimize over QAOA angles and its properties
        r.set_classical_optimizer(
            method="rmsprop",
            optimizer_options={"stepsize": 10 ** (-10)},
            tol=10 ** (-1),
            maxfev=1,
            maxiter=1,
            jac="finite_difference",
            optimization_progress=False,
            cost_progress=False,
            parameter_log=False,
        )

        # Compile and optimize the problem instance on RQAOA
        r.compile(maxcut_qubo)
        r.optimize()

        opt_results = r.result
        opt_solution = opt_results["solution"]
        opt_solution_string = list(opt_solution.keys())[0]

        assert opt_solution_string == "01011010"

    def test_exact_solution(self):
        """
        Testing the exact solution which is a property of every backend.

        """

        n_qubits = 8
        register = range(n_qubits)
        p = 1

        correct_energy = -8
        correct_config = [0, 1, 0, 1, 0, 1, 0, 1]

        # The tests pass regardless of the value of betas and gammas is this correct?
        betas = [np.pi / 8]
        gammas = [np.pi / 4]

        cost_hamiltonian = ring_of_disagrees(register)
        mixer_hamiltonian = X_mixer_hamiltonian(n_qubits)
        qaoa_descriptor = QAOADescriptor(cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = QAOAVariationalStandardParams(
            qaoa_descriptor, betas, gammas
        )

        backend_analytical = QAOABackendAnalyticalSimulator(qaoa_descriptor)
        # exact solution is defined as the property of the cost function
        energy_vec, config_vec = backend_analytical.exact_solution

        assert np.isclose(energy_vec, correct_energy)

        config_vec = [config.tolist() for config in config_vec]

        assert correct_config in config_vec


if __name__ == "__main__":
    unittest.main()
