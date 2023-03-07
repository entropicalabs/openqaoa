import warnings
import numpy as np
import networkx as nw
import unittest
from unittest.mock import Mock
import warnings
import os

import numpy as np
from scipy.optimize._minimize import MINIMIZE_METHODS

from openqaoa.qaoa_components import (
    create_qaoa_variational_params,
    QAOADescriptor,
    PauliOp,
    Hamiltonian,
)
from openqaoa.utilities import X_mixer_hamiltonian
from openqaoa.backends.qaoa_backend import get_qaoa_backend
from openqaoa.backends import create_device
from openqaoa.optimizers import get_optimizer
from openqaoa.algorithms.qaoa.qaoa_result import QAOAResult
from openqaoa.derivatives.derivative_functions import derivative
from openqaoa.optimizers.logger_vqa import Logger
from openqaoa.derivatives.qfim import qfim
from openqaoa.problems import MinimumVertexCover

"""
Unittest based testing of custom optimizers.
"""

cost_hamiltonian_1 = Hamiltonian(
    [
        PauliOp("ZZ", (0, 1)),
        PauliOp("ZZ", (1, 2)),
        PauliOp("ZZ", (0, 3)),
        PauliOp("Z", (2,)),
        PauliOp("Z", (1,)),
    ],
    [1, 1.1, 1.5, 2, -0.8],
    0.8,
)


class TestQAOACostBaseClass(unittest.TestCase):
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
                "qfim_func_evals": {
                    "history_update_bool": False,
                    "best_update_string": "HighestOnly",
                },
            },
            {
                "root_nodes": ["func_evals", "jac_func_evals", "qfim_func_evals"],
                "best_update_structure": [],
            },
        )

        self.log.log_variables(
            {"func_evals": 0, "jac_func_evals": 0, "qfim_func_evals": 0}
        )

    def __backend_params(self, cost_hamil, n_qubits):
        """Helper function to create a backend and a parameters onjects for testing."""
        mixer_hamil = X_mixer_hamiltonian(n_qubits=n_qubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=1)
        device = create_device("local", "vectorized")
        backend_obj_vectorized = get_qaoa_backend(qaoa_descriptor, device)
        variate_params = create_qaoa_variational_params(
            qaoa_descriptor, "standard", "ramp"
        )
        return backend_obj_vectorized, variate_params

    def test_saving_feature(self):
        """Test save_intermediate in OptimizeVQA"""
        backend_obj_vectorized, variate_params = self.__backend_params(
            cost_hamiltonian_1, 4
        )
        niter = 5
        grad_stepsize = 0.0001
        stepsize = 0.001

        params_array = variate_params.raw().copy()
        jac = derivative(
            backend_obj_vectorized,
            variate_params,
            self.log,
            "gradient",
            "finite_difference",
            {"stepsize": grad_stepsize},
        )

        # Optimize
        vector_optimizer = get_optimizer(
            backend_obj_vectorized,
            variate_params,
            optimizer_dict={
                "method": "vgd",
                "tol": 10 ** (-9),
                "jac": jac,
                "maxiter": niter,
                "optimizer_options": {"stepsize": stepsize},
                "save_intermediate": True,
            },
        )
        vector_optimizer()

    def test_scipy_optimizers_global(self):
        "Check that final value of all scipy MINIMIZE_METHODS optimizers agrees with pre-computed optimized value."

        # Create problem instance, cost function, and gradient functions
        backend_obj_vectorized, variate_params = self.__backend_params(
            cost_hamiltonian_1, 4
        )

        niter = 5
        stepsize = 0.001
        y_precomp = [
            -2.4345058914425626,
            -2.5889608823632795,
            -2.588960865651421,
            -2.5889608823632786,
            -2.5889608823632795,
            -2.588960882363273,
            -2.5889608823632786,
            0.7484726235465329,
            -2.588960882363272,
            -2.588960882363281,
            -2.5889608823632786,
            -2.5889608823632795,
            -2.5889608823632786,
            -2.5889608823632786,
        ]
        optimizer_dicts = []
        for method in MINIMIZE_METHODS:
            optimizer_dicts.append(
                {"method": method, "maxiter": niter, "tol": 10 ** (-9)}
            )

        for i, optimizer_dict in enumerate(optimizer_dicts):

            optimizer_dict["jac"] = derivative(
                backend_obj_vectorized,
                variate_params,
                self.log,
                "gradient",
                "finite_difference",
            )
            optimizer_dict["hess"] = derivative(
                backend_obj_vectorized,
                variate_params,
                self.log,
                "hessian",
                "finite_difference",
            )

            # Optimize
            vector_optimizer = get_optimizer(
                backend_obj_vectorized, variate_params, optimizer_dict=optimizer_dict
            )
            vector_optimizer()

            y_opt = vector_optimizer.qaoa_result.intermediate["cost"]

            assert np.isclose(
                y_precomp[i], y_opt[-1], rtol=1e-04, atol=1e-04
            ), f"{optimizer_dict['method']} failed the test."

    def test_gradient_optimizers_global(self):
        "Check that final value of all implemented gradient optimizers agrees with pre-computed optimized value."

        backend_obj_vectorized, variate_params = self.__backend_params(
            cost_hamiltonian_1, 4
        )
        niter = 10
        stepsize = 0.001

        # pre-computed final optimized costs
        y_precomp = [
            -2.4212581335011456,
            -2.4246393953483825,
            -2.47312715451289,
            -2.5031221706241906,
        ]

        optimizer_dicts = []
        optimizer_dicts.append(
            {
                "method": "vgd",
                "tol": 10 ** (-9),
                "jac": "finite_difference",
                "maxiter": niter,
                "optimizer_options": {"stepsize": stepsize},
            }
        )
        optimizer_dicts.append(
            {
                "method": "newton",
                "tol": 10 ** (-9),
                "jac": "finite_difference",
                "hess": "finite_difference",
                "maxiter": niter,
                "optimizer_options": {"stepsize": stepsize},
            }
        )
        optimizer_dicts.append(
            {
                "method": "natural_grad_descent",
                "tol": 10 ** (-9),
                "jac": "finite_difference",
                "maxiter": niter,
                "optimizer_options": {"stepsize": 0.01},
            }
        )
        optimizer_dicts.append(
            {
                "method": "rmsprop",
                "tol": 10 ** (-9),
                "jac": "finite_difference",
                "maxiter": niter,
                "optimizer_options": {"stepsize": stepsize, "decay": 0.9, "eps": 1e-07},
            }
        )

        for i, optimizer_dict in enumerate(optimizer_dicts):

            # Optimize
            vector_optimizer = get_optimizer(
                backend_obj_vectorized, variate_params, optimizer_dict=optimizer_dict
            )
            vector_optimizer()

            y_opt = vector_optimizer.qaoa_result.intermediate["cost"]

            assert np.isclose(
                y_precomp[i], y_opt[-1], rtol=1e-04, atol=1e-04
            ), f"{optimizer_dict['method']} method failed the test."

    def test_gradient_optimizers_cans(self):
        n_qubits = 10
        graph = nw.circulant_graph(n_qubits, [1])
        cost_hamil = MinimumVertexCover(graph, field=1, penalty=10).qubo.hamiltonian
        backend_obj_vectorized, variate_params = self.__backend_params(
            cost_hamil, n_qubits
        )

        optimizer_list = [
            {
                "method": method,
                "jac": "param_shift",
                "maxiter": 1000,
                "optimizer_options": {
                    "stepsize": 0.0001,
                    "n_shots_min": 5,
                    "n_shots_max": 200,
                    "n_shots_budget": 1000,
                },
            }
            for method in ["cans", "icans"]
        ]

        for optimizer_dict in optimizer_list:
            # optimize qaoa
            vector_optimizer = get_optimizer(
                backend_obj_vectorized, variate_params, optimizer_dict=optimizer_dict
            )
            vector_optimizer()

            # check that the final cost is the optimal value
            assert (
                int(
                    vector_optimizer.qaoa_result.most_probable_states[
                        "bitstring_energy"
                    ]
                )
                == 5
            ), "{} failed the test.".format(optimizer_dict["method"])
            # check that the number of shots are correct
            n_shots_used = np.array(vector_optimizer.qaoa_result.n_shots)
            assert (
                np.sum(np.int0(n_shots_used < 5) + np.int8(n_shots_used > 200)) == 0
            ), "Optimizer {} did not use the correct number of shots.".format(
                optimizer_dict["method"]
            )
            # check n_shots_budget worked
            assert (
                np.sum(20 * n_shots_used[:-1].T[0] + 40 * n_shots_used[:-1].T[1]) < 1000
                and np.sum(20 * n_shots_used.T[0] + 40 * n_shots_used.T[1]) >= 1000
            ), "Optimizer {} did not use the correct number of shots.".format(
                optimizer_dict["method"]
            )

            assert (
                np.sum(n_shots_used < 5) == 0
            ), "Optimizer {} did not use the correct number of shots.".format(
                optimizer_dict["method"]
            )

    def test_gradient_descent_step(self):
        """
        Check that implemented gradient descent takes the first two steps correctly.
        """

        backend_obj_vectorized, variate_params = self.__backend_params(
            cost_hamiltonian_1, 4
        )
        niter = 5
        grad_stepsize = 0.0001
        stepsize = 0.001

        params_array = variate_params.raw().copy()
        jac = derivative(
            backend_obj_vectorized,
            variate_params,
            self.log,
            "gradient",
            "finite_difference",
            {"stepsize": grad_stepsize},
        )

        # Optimize
        vector_optimizer = get_optimizer(
            backend_obj_vectorized,
            variate_params,
            optimizer_dict={
                "method": "vgd",
                "tol": 10 ** (-9),
                "jac": jac,
                "maxiter": niter,
                "optimizer_options": {"stepsize": stepsize},
            },
        )
        vector_optimizer()
        y_opt = vector_optimizer.qaoa_result.intermediate["cost"][1:4]

        # Stepwise optimize
        def step(x0):
            x1 = x0 - stepsize * jac(x0)

            variate_params.update_from_raw(x1)

            return [x1, np.real(backend_obj_vectorized.expectation(variate_params))]

        x0 = params_array
        variate_params.update_from_raw(x0)
        y0 = backend_obj_vectorized.expectation(variate_params)
        [x1, y1] = step(x0)
        [x2, y2] = step(x1)

        y = [y0, y1, y2]

        for i, yi in enumerate(y):
            assert np.isclose(yi, y_opt[i], rtol=1e-05, atol=1e-05)

    def test_newton_step(self):
        """
        Check that implemented Newton descent takes the first two steps correctly.
        """

        backend_obj_vectorized, variate_params = self.__backend_params(
            cost_hamiltonian_1, 4
        )
        niter = 5
        grad_stepsize = 0.0001
        stepsize = 0.001

        params_array = variate_params.raw().copy()
        jac = derivative(
            backend_obj_vectorized,
            variate_params,
            self.log,
            "gradient",
            "finite_difference",
            {"stepsize": grad_stepsize},
        )
        hess = derivative(
            backend_obj_vectorized,
            variate_params,
            self.log,
            "hessian",
            "finite_difference",
            {"stepsize": grad_stepsize},
        )

        # Optimize
        vector_optimizer = get_optimizer(
            backend_obj_vectorized,
            variate_params,
            optimizer_dict={
                "method": "newton",
                "tol": 10 ** (-9),
                "jac": jac,
                "hess": hess,
                "maxiter": niter,
                "optimizer_options": {"stepsize": stepsize},
            },
        )
        vector_optimizer()
        y_opt = vector_optimizer.qaoa_result.intermediate["cost"][1:4]

        # Stepwise optimize
        def step(x0):
            scaled_gradient = np.linalg.solve(hess(x0), jac(x0))
            x1 = x0 - stepsize * scaled_gradient
            variate_params.update_from_raw(x1)
            return [x1, np.real(backend_obj_vectorized.expectation(variate_params))]

        x0 = params_array
        variate_params.update_from_raw(x0)
        y0 = backend_obj_vectorized.expectation(variate_params)
        [x1, y1] = step(x0)
        [x2, y2] = step(x1)

        y = [y0, y1, y2]

        for i, yi in enumerate(y):
            assert np.isclose(yi, y_opt[i], rtol=1e-05, atol=1e-05)

    def test_natural_gradient_descent_step(self):
        """
        Check that implemented natural gradient descent takes the first two steps correctly.
        """

        backend_obj_vectorized, variate_params = self.__backend_params(
            cost_hamiltonian_1, 4
        )
        niter = 5
        grad_stepsize = 0.0001
        stepsize = 0.001

        params_array = variate_params.raw().copy()
        jac = derivative(
            backend_obj_vectorized,
            variate_params,
            self.log,
            "gradient",
            "finite_difference",
            {"stepsize": grad_stepsize},
        )

        # Optimize
        vector_optimizer = get_optimizer(
            backend_obj_vectorized,
            variate_params,
            optimizer_dict={
                "method": "natural_grad_descent",
                "tol": 10 ** (-9),
                "jac": jac,
                "maxiter": niter,
                "optimizer_options": {"stepsize": stepsize},
            },
        )
        vector_optimizer()
        y_opt = vector_optimizer.qaoa_result.intermediate["cost"][1:4]

        # Stepwise optimize
        def step(x0):
            qfim_ = qfim(backend_obj_vectorized, variate_params, self.log)
            scaled_gradient = np.linalg.solve(qfim_(x0), jac(x0))
            x1 = x0 - stepsize * scaled_gradient
            variate_params.update_from_raw(x1)
            return [x1, np.real(backend_obj_vectorized.expectation(variate_params))]

        x0 = params_array
        variate_params.update_from_raw(x0)
        y0 = backend_obj_vectorized.expectation(variate_params)
        [x1, y1] = step(x0)
        [x2, y2] = step(x1)

        y = [y0, y1, y2]

        for i, yi in enumerate(y):
            assert np.isclose(yi, y_opt[i], rtol=1e-05, atol=1e-05)

    def test_rmsprop_step(self):
        """
        Check that implemented RMSProp takes the first two steps correctly.
        """

        backend_obj_vectorized, variate_params = self.__backend_params(
            cost_hamiltonian_1, 4
        )
        niter = 5
        grad_stepsize = 0.0001
        stepsize = 0.001

        params_array = variate_params.raw().copy()
        jac = derivative(
            backend_obj_vectorized,
            variate_params,
            self.log,
            "gradient",
            "finite_difference",
            {"stepsize": grad_stepsize},
        )

        decay = 0.9
        eps = 1e-07

        # Optimize
        vector_optimizer = get_optimizer(
            backend_obj_vectorized,
            variate_params,
            optimizer_dict={
                "method": "rmsprop",
                "tol": 10 ** (-9),
                "jac": jac,
                "maxiter": niter,
                "optimizer_options": {"stepsize": stepsize, "decay": decay, "eps": eps},
            },
        )

        vector_optimizer()
        y_opt = vector_optimizer.qaoa_result.intermediate["cost"][1:4]

        # Stepwise optimize
        def step(x0, sqgrad0):
            sqgrad = decay * sqgrad0 + (1 - decay) * jac(x0) ** 2
            x1 = x0 - stepsize * jac(x0) / (np.sqrt(sqgrad) + eps)
            variate_params.update_from_raw(x1)
            return [
                x1,
                np.real(backend_obj_vectorized.expectation(variate_params)),
                sqgrad0,
            ]

        x0 = params_array
        variate_params.update_from_raw(x0)
        y0 = backend_obj_vectorized.expectation(variate_params)
        sqgrad0 = jac(x0) ** 2
        [x1, y1, sqgrad1] = step(x0, sqgrad0)
        [x2, y2, sqgrad2] = step(x1, sqgrad1)

        y = [y0, y1, y2]

        for i, yi in enumerate(y):
            assert np.isclose(yi, y_opt[i], rtol=1e-05, atol=1e-05)

    def test_optimize_loop_crash(self):

        """
        This tests that the optimization loop doesnt crash silently.
        An Exception gets raised.
        """

        cost_hamil = Hamiltonian(
            [
                PauliOp("ZZ", (0, 1)),
                PauliOp("ZZ", (1, 2)),
                PauliOp("ZZ", (0, 3)),
                PauliOp("Z", (2,)),
                PauliOp("Z", (1,)),
            ],
            [1, 1.1, 1.5, 2, -0.8],
            0.8,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=4)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=2)
        device = create_device("local", "vectorized")
        backend_obj_vectorized = get_qaoa_backend(qaoa_descriptor, device)
        variate_params = create_qaoa_variational_params(
            qaoa_descriptor, "standard", "ramp"
        )
        niter = 5

        # Optimize
        vector_optimizer = get_optimizer(
            backend_obj_vectorized,
            variate_params,
            optimizer_dict={
                "method": "nelder-mead",
                "maxiter": niter,
            },
        )
        vector_optimizer.vqa.expectation = Mock(side_effect=Exception("Error!"))
        self.assertRaises(Exception, lambda: vector_optimizer.optimize())

        # Check that QAOA Result exists
        self.assertEqual(type(vector_optimizer.qaoa_result), QAOAResult)

    @classmethod
    def tearDownClass(cls):

        output_csv = ["oq_saved_info_job_ids.csv", "oq_saved_info_param_log.csv"]
        for each_csv in output_csv:
            if os.path.exists(each_csv):
                os.remove(each_csv)
            else:
                raise FileNotFoundError(
                    "Unable to remove the generated csv file: {}".format(each_csv)
                )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PendingDeprecationWarning)
        unittest.main()
