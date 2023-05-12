import warnings
import numpy as np
import unittest

# OpenQAOA imports
from openqaoa.backends import QAOAvectorizedBackendSimulator
from openqaoa.qaoa_components import (
    QAOADescriptor,
    Hamiltonian,
    create_qaoa_variational_params,
)
from openqaoa.utilities import X_mixer_hamiltonian
from openqaoa.optimizers.logger_vqa import Logger
from openqaoa.derivatives.derivative_functions import derivative

"""
Unittest based testing of derivative computations.
"""


class TestQAOACostBaseClass(unittest.TestCase):

    """
    def test_gradient_agreement(self):
        "Test agreement between gradients computed from finite difference, parameter shift and SPS (all gates sampled) for weighted and unweighted graphs at several parameters."

        # unweighted graph
        terms = [[0,1], [0,2], [1,3], [2]]
        weights = [1, 1, 1, 1]
        register = [0, 1, 2, 3]
        p = 2
        nqubits = 4

        cost_hamiltonian = Hamiltonian.classical_hamiltonian(
            terms, weights, constant=0)
        mixer_hamiltonian = X_mixer_hamiltonian(nqubits)
        qaoa_qaoa_descriptor = QAOADescriptor(
            cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = create_qaoa_variational_params(
            qaoa_qaoa_descriptor, 'standard', 'ramp')
        backend_vectorized = QAOAvectorizedBackendSimulator(
            qaoa_qaoa_descriptor, prepend_state=None, append_state=None, init_hadamard=True)

        grad_stepsize = 0.00000001
        gradient_ps = backend_vectorized.derivative_function(variational_params_std, 'gradient', 'param_shift')
        gradient_fd = backend_vectorized.derivative_function(variational_params_std, 'gradient', 'finite_difference', {'stepsize': grad_stepsize})
        gradient_sps = backend_vectorized.derivative_function(variational_params_std, 'gradient', 'stoch_param_shift', {'stepsize':grad_stepsize, 'n_beta_single':-1, 'n_beta_pair':-1, 'n_gamma_pair':-1, 'n_gamma_single':-1})

        params = [[0,0,0,0], [1,1,1,1], [np.pi/2, np.pi/2, np.pi/2, np.pi/2]]

        for param in params:
            grad_fd = gradient_fd(param)
            grad_ps = gradient_ps(param)
            grad_sps = gradient_sps(param)

            for i, grad in enumerate(grad_fd):
                assert np.isclose(grad, grad_ps[i], rtol=1e-05, atol=1e-05)
                assert np.isclose(grad, grad_sps[i], rtol=1e-05, atol=1e-05)

        # weighted graph with bias
        terms = [[0,1], [1,2], [0,3], [2], [1]]
        weights = [1, 1.1, 1.5, 2, -0.8]
        register = [0, 1, 2, 3]
        p = 2
        nqubits = 4

        cost_hamiltonian = Hamiltonian.classical_hamiltonian(
            terms, weights, constant=0.8)
        mixer_hamiltonian = X_mixer_hamiltonian(nqubits)
        qaoa_qaoa_descriptor = QAOADescriptor(
            cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = create_qaoa_variational_params(
            qaoa_qaoa_descriptor, 'standard', 'ramp')
        backend_vectorized = QAOAvectorizedBackendSimulator(
            qaoa_qaoa_descriptor, prepend_state=None, append_state=None, init_hadamard=True)

        grad_stepsize = 0.00000001
        gradient_ps = backend_vectorized.derivative_function(variational_params_std, 'gradient', 'param_shift')
        gradient_fd = backend_vectorized.derivative_function(variational_params_std, 'gradient', 'finite_difference', {'stepsize': grad_stepsize})
        gradient_sps = backend_vectorized.derivative_function(variational_params_std, 'gradient', 'stoch_param_shift', {'stepsize':grad_stepsize, 'n_beta_single':-1, 'n_beta_pair':-1, 'n_gamma_pair':-1, 'n_gamma_single':-1})

        params = [[0,0,0,0], [1,1,1,1], [np.pi/2, np.pi/2, np.pi/2, np.pi/2]]

        for param in params:
            grad_fd = gradient_fd(param)
            grad_ps = gradient_ps(param)
            grad_sps = gradient_sps(param)

            for i, grad in enumerate(grad_fd):
                assert np.isclose(grad, grad_ps[i], rtol=1e-05, atol=1e-05)
                assert np.isclose(grad, grad_sps[i], rtol=1e-05, atol=1e-05)
    """

    def setUp(self):
        self.log = Logger(
            {
                "func_evals": {
                    "history_update_bool": False,
                    "best_update_string": "HighestOnly",
                },
                "jac_func_evals": {
                    "history_update_bool": False,
                    "best_update_string": "HighestOnly",
                },
                "n_shots": {
                    "history_update_bool": True,
                    "best_update_string": "Replace",
                },
            },
            {
                "root_nodes": ["func_evals", "jac_func_evals", "n_shots"],
                "best_update_structure": [],
            },
        )

        self.log.log_variables({"func_evals": 0})
        self.log.log_variables({"jac_func_evals": 0})

    def __backend_params(self, terms: list, weights: list, p: int, nqubits: int):
        cost_hamiltonian = Hamiltonian.classical_hamiltonian(terms, weights, constant=0)
        mixer_hamiltonian = X_mixer_hamiltonian(nqubits)
        qaoa_qaoa_descriptor = QAOADescriptor(cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = create_qaoa_variational_params(
            qaoa_qaoa_descriptor, "standard", "ramp"
        )
        backend_vectorized = QAOAvectorizedBackendSimulator(
            qaoa_qaoa_descriptor,
            prepend_state=None,
            append_state=None,
            init_hadamard=True,
        )
        return backend_vectorized, variational_params_std

    def test_gradient_computation(self):
        "Test gradient computation by param. shift, finite difference, and SPS (all gates sampled) on barbell graph."

        # Analytical cost expression : C(b,g) = -sin(4b)*sin(2g)
        backend, params = self.__backend_params(
            terms=[[0, 1]], weights=[1], p=1, nqubits=2
        )

        gradients_types_list = ["finite_difference", "param_shift", "stoch_param_shift"]
        gradients_fun_list = [
            derivative(
                backend, params, self.log, "gradient", type_, {"stepsize": 0.0000001}
            )
            for type_ in gradients_types_list
        ]

        test_points = [[0, 0], [np.pi / 2, np.pi / 3], [1, 2]]

        for point in test_points:
            beta, gamma = point

            dCdb, dCdg = -4 * np.cos(4 * beta) * np.sin(2 * gamma), -2 * np.sin(
                4 * beta
            ) * np.cos(2 * gamma)

            for gradient_fun, gradient_name in zip(
                gradients_fun_list, gradients_types_list
            ):
                dCdb_, dCdg_ = gradient_fun(point)
                assert np.isclose(
                    dCdb, dCdb_, rtol=1e-05, atol=1e-05
                ), f"Gradient computation failed for {gradient_name} on barbell graph. dCdb: {dCdb}, dCdb_: {dCdb_}"
                assert np.isclose(
                    dCdg, dCdg_, rtol=1e-05, atol=1e-05
                ), f"Gradient computation failed for {gradient_name} on barbell graph. dCdg: {dCdg}, dCdg_: {dCdg_}"

    def test_gradient_w_variance_computation(self):
        "Test gradient computation by param. shift, finite difference, and SPS (all gates sampled) on barbell graph."

        # Analytical cost expression : C(b,g) = -sin(4b)*sin(2g)
        backend, params = self.__backend_params(
            terms=[[0, 1]], weights=[1], p=1, nqubits=2
        )

        gradients_types_list = ["finite_difference", "param_shift", "stoch_param_shift"]
        gradients_fun_list = [
            derivative(
                backend,
                params,
                self.log,
                "gradient_w_variance",
                type_,
                {"stepsize": 0.0000001},
            )
            for type_ in gradients_types_list
        ]

        test_points = [[0, 0], [np.pi / 2, np.pi / 3], [1, 2]]

        for point in test_points:
            for gradient_fun, gradient_name in zip(
                gradients_fun_list, gradients_types_list
            ):
                # compute gradient for each point with number of shots 1000 for ech function evaluation
                grad, var, n_shots = gradient_fun(point, n_shots=1000)

                # check if there is a gradient and variance (we can't check the value of the gradient and variance because it is randomly computed, since n_shots is 1000)
                for g in grad:
                    assert (
                        np.abs(g) >= 0
                    ), f"Gradient computation failed for {gradient_name} on barbell graph. grad: {grad}"
                for v in var:
                    assert (
                        v > 0
                    ), f"Error computing the variance of the gradient for {gradient_name} on barbell graph."

                # check if the number of shots is correct
                x = (
                    4 if gradient_name == "finite_difference" else 6
                )  # that is also checking that  SPS samples all gates when (n_beta, n_gamma_pair, n_gamma_single) is (-1, -1, -1)
                assert (
                    n_shots == x * 1000
                ), f"The number of shots should be {x*1000} but is {n_shots}."

    def test_SPS_sampling(self):
        "Test that SPS samples the number of gates specified by the user."
        backend, params = self.__backend_params(
            terms=[[0, 1]], weights=[1], p=1, nqubits=2
        )
        gradient_fun = derivative(
            backend,
            params,
            self.log,
            "gradient_w_variance",
            "stoch_param_shift",
            {
                "stepsize": 0.1,
                "n_beta_single": 1,
                "n_beta_pair": 0,
                "n_gamma_pair": 1,
                "n_gamma_single": 0,
            },
        )

        n_shots = gradient_fun([0, 0], n_shots=1000)[2]
        # check if the number of shots is correct, we know that we are sampling 2 gates (1 beta and 1 gamma) -> 2*2*1000 (the second 2* comes from two evaluations for each gradient)
        assert (
            n_shots == 4000
        ), f"The number of shots should be {4000} but is {n_shots}."

    def test_hessian_computation(self):
        "Test Hessian computation by finite difference on barbell graph"

        # Analytical cost expression : C(b,g) = -sin(4b)*sin(2g)
        backend, params = self.__backend_params(
            terms=[[0, 1]], weights=[1], p=1, nqubits=2
        )

        hessian_fd = derivative(
            backend,
            params,
            self.log,
            "hessian",
            "finite_difference",
            {"stepsize": 0.001},
        )

        test_points = np.round([[0, 0], [np.pi / 2, np.pi / 3], [1, 2]], 12)

        for point in test_points:
            beta, gamma = point[0], point[1]

            dCdbb = 16 * np.sin(4 * beta) * np.sin(2 * gamma)
            dCdbg = -8 * np.cos(4 * beta) * np.cos(2 * gamma)
            dCdgb = dCdbg
            dCdgg = 4 * np.sin(4 * beta) * np.sin(2 * gamma)

            assert np.isclose(dCdbb, hessian_fd(point)[0][0], rtol=1e-05, atol=1e-05)
            assert np.isclose(dCdbg, hessian_fd(point)[0][1], rtol=1e-05, atol=1e-05)
            assert np.isclose(dCdgb, hessian_fd(point)[1][0], rtol=1e-05, atol=1e-05)
            assert np.isclose(dCdgg, hessian_fd(point)[1][1], rtol=1e-05, atol=1e-05)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PendingDeprecationWarning)
        unittest.main()
