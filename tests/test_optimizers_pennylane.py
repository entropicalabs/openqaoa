import warnings
import unittest

import numpy as np
import networkx as nx
import openqaoa.optimizers.pennylane as pl
import copy
import inspect

from openqaoa.workflows.optimizer import QAOA
from openqaoa.devices import create_device
from openqaoa.problems.problem import MinimumVertexCover
from openqaoa.optimizers.training_vqa import PennyLaneOptimizer
from openqaoa.optimizers.pennylane.optimization_methods_pennylane import AVAILABLE_OPTIMIZERS
from openqaoa.derivative_functions import derivative
from openqaoa.optimizers.logger_vqa import Logger
from openqaoa.qaoa_parameters import create_qaoa_variational_params, QAOACircuitParams, PauliOp, Hamiltonian
from openqaoa.utilities import X_mixer_hamiltonian
from openqaoa.backends.qaoa_backend import get_qaoa_backend
from openqaoa.optimizers import get_optimizer
from openqaoa.qfim import qfim as Qfim
from openqaoa.problems.problem import QUBO


#list of optimizers to test, pennylane optimizers
list_optimizers = PennyLaneOptimizer.PENNYLANE_OPTIMIZERS

#create a problem
g = nx.circulant_graph(4, [1])
problem = MinimumVertexCover(g, field =1.0, penalty=10)
qubo_problem_1 = problem.get_qubo_problem()
qubo_problem_2 = QUBO.random_instance(5)
qubo_problem_3 = QUBO.random_instance(6)


class TestPennylaneOptimizers(unittest.TestCase):

    def setUp(self):
        
        self.log = Logger({'func_evals': 
                           {
                               'history_update_bool': False, 
                               'best_update_string': 'HighestOnly'
                           },
                           'jac_func_evals':
                           {
                               'history_update_bool': False, 
                               'best_update_string': 'HighestOnly'
                           },
                           'qfim_func_evals': 
                           {
                               'history_update_bool': False, 
                               'best_update_string': 'HighestOnly'
                           }
                          }, 
                          {
                              'root_nodes': ['func_evals', 'jac_func_evals', 
                                             'qfim_func_evals'], 
                              'best_update_structure': []
                          })
        
        self.log.log_variables({'func_evals': 0, 'jac_func_evals': 0, 'qfim_func_evals': 0})

    def _run_method_workflows(self, method, problem):
        " helper function to run the test for any method using workflows"
        q = QAOA()
        q.set_classical_optimizer(method=method, maxiter=3, jac='finite_difference')
        q.compile(problem) 
        q.optimize()

        assert len(q.results.most_probable_states['solutions_bitstrings'][0]) > 0

    def _run_method_manual(self, method, problem):
        " helper function tu run the test for any method using manual mode"

        cost_hamil = problem.hamiltonian
        mixer_hamil = X_mixer_hamiltonian(n_qubits=problem.n)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=2)
        device = create_device('local','vectorized')
        backend_obj_vectorized = get_qaoa_backend(circuit_params,device)
        variate_params = create_qaoa_variational_params(circuit_params, 'standard', 'ramp')
        niter = 5
        grad_stepsize = 0.0001
        stepsize = 0.001

        # declare needed functions
        jac = derivative(backend_obj_vectorized, variate_params, self.log, 
                    'gradient', 'finite_difference', 
                    {'stepsize': grad_stepsize})
        qfim = Qfim(backend_obj_vectorized, variate_params, self.log)   


        # Optimize
        vector_optimizer = get_optimizer(backend_obj_vectorized, variate_params, optimizer_dict={
                                        'method': method, 'jac': jac, 'maxiter': niter, 'qfim': qfim,
                                        'optimizer_options' : {'stepsize': stepsize}})
        vector_optimizer()

        # saving the results
        results = vector_optimizer.qaoa_result

        assert len(results.most_probable_states['solutions_bitstrings'][0]) == problem.n

    def test_pennylane_optimizers_workflows(self):
        " function to run the tests for pennylane optimizers, workflows"

        i = 0
        for problem in [qubo_problem_3, qubo_problem_2, qubo_problem_1]:
            for opt in list_optimizers:
                self._run_method_workflows(opt, problem)
                i += 1

        assert i == 3*len(list_optimizers)

    def test_pennylane_optimizers_manual(self):
        " function to run the tests for pennylane optimizers, manual mode"

        i = 0
        for problem in [qubo_problem_3, qubo_problem_2, qubo_problem_1]:
            for opt in list_optimizers:
                self._run_method_manual(opt, problem)
                i += 1

        assert i == 3*len(list_optimizers)

    def _pennylane_step(self, params_array, cost, optimizer, method, jac, qfim):
        " helper function to run a setp of the pennylane optimizer"
        params_array = pl.numpy.array(params_array, requires_grad=True)
        if method in ['natural_grad_descent']: 
            x, y = optimizer.step_and_cost(cost, params_array, grad_fn=jac, metric_tensor_fn=qfim) 
        if method in ['adagrad', 'adam', 'vgd', 'momentum', 'nesterov_momentum', 'rmsprop']:
            x, y = optimizer.step_and_cost(cost, params_array, grad_fn=jac)
        if method in ['rotosolve']: 
            x, y = optimizer.step_and_cost(
                                                    cost, params_array,
                                                    nums_frequency={'params': {(i,):1 for i in range(params_array.size)}},
                                                    # spectra=spectra,
                                                    # shifts=shifts,
                                                    # full_output=False,
                                                )
        if method in ['spsa']:       
            x, y = optimizer.step_and_cost(cost, params_array)

        return x, y

    def test_step_and_cost(self):
        " function to run the tests for steps of pennylane optimizers "

        # define some problem
        cost_hamil = Hamiltonian([PauliOp('ZZ', (0, 1)), PauliOp('ZZ', (1, 2)), PauliOp(
            'ZZ', (0, 3)), PauliOp('Z', (2,)), PauliOp('Z', (1,))], [1, 1.1, 1.5, 2, -0.8], 0.8)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=4)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=2)
        device = create_device('local','vectorized')
        backend_obj_vectorized = get_qaoa_backend(circuit_params,device)
        variate_params = create_qaoa_variational_params(circuit_params, 'standard', 'ramp')
        niter = 5
        grad_stepsize = 0.0001
        stepsize = 0.001

        # declare needed functions
        jac = derivative(backend_obj_vectorized, variate_params, self.log, 
                    'gradient', 'finite_difference', 
                    {'stepsize': grad_stepsize})
        qfim = Qfim(backend_obj_vectorized, variate_params, self.log)   
        def cost(params):
            variate_params.update_from_raw(params)
            return np.real(backend_obj_vectorized.expectation(variate_params))

        i = 0
        for method in list_optimizers:
        
            pennylane_method = method.replace('pennylane_', '')

            # copy the parameters
            x0 = copy.deepcopy(variate_params.raw().copy())

            # Optimize with the implemented optimizer in OpenQAOA
            vector_optimizer = get_optimizer(backend_obj_vectorized, variate_params, optimizer_dict={
                                            'method': method, 'jac': jac, 'maxiter': niter, 'qfim': qfim,
                                            'optimizer_options' : {'stepsize': stepsize}})
            vector_optimizer()

            # formatting the data
            y_opt = vector_optimizer.qaoa_result.intermediate['intermediate cost'][1:4]
            if pennylane_method in ['rotosolve']: y_opt = vector_optimizer.qaoa_result.intermediate['intermediate cost'][4:40:12]

            # get optimizer to try
            optimizer = AVAILABLE_OPTIMIZERS[pennylane_method]
            #get optimizer arguments
            arguments = inspect.signature(optimizer).parameters.keys()

            #check if stepsize is in the optimizer arguments
            options = {}
            if 'stepsize' in arguments: options['stepsize'] = stepsize
            if 'maxiter'  in arguments: options['maxiter'] = niter

            #pass the argument to the optimizer
            optimizer = optimizer(**options) 

            # reinitialize variables
            variate_params.update_from_raw(x0)
            x0 = variate_params.raw().copy()
            y0 = cost(x0)

            # compute steps (depends on the optimizer)
            x1, y1 = self._pennylane_step(x0, cost, optimizer, pennylane_method, jac, qfim)
            x2, y2 = self._pennylane_step(x1, cost, optimizer, pennylane_method, jac, qfim)
            x3, y3 = self._pennylane_step(x2, cost, optimizer, pennylane_method, jac, qfim)

            # list of results
            y = [y1, y2, y3]

            # check that the results are ok
            if pennylane_method in ['spsa']: 
                assert np.sum(np.abs(np.array(y)) >= 0) == 3
            else:
                for yi, y_opt_i in zip(y, y_opt):
                    assert np.isclose(yi, y_opt_i, rtol=0.001, atol=0.001)

            i += 1                

        assert i == len(list_optimizers)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=PendingDeprecationWarning)
        unittest.main()
