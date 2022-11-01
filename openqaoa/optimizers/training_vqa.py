#   Copyright 2022 Entropica Labs
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os
import numpy as np
import pickle
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Type, Callable, List
from datetime import datetime

from scipy.optimize._minimize import minimize, MINIMIZE_METHODS
from scipy.optimize import LinearConstraint, NonlinearConstraint, Bounds

from ..basebackend import VQABaseBackend
from ..qaoa_parameters.baseparams import QAOAVariationalBaseParams
from . import optimization_methods as om

from .logger_vqa import Logger
from .result import Result

from ..derivative_functions import derivative
from ..qfim import qfim

###
# TODO: Find better place for this

import pandas as pd
from typing import Any

def save_parameter(parameter_name: str, parameter_value: Any):
    
    filename = 'oq_saved_info_' + parameter_name
    
    try:
        opened_csv = pd.read_csv(filename + '.csv')
    except Exception:
        opened_csv = pd.DataFrame(columns = [parameter_name])
    
    if type(parameter_value) not in [str, float, int]:
        parameter_value = str(parameter_value)
        
    update_df = pd.DataFrame(data = {parameter_name: parameter_value}, index = [0])
    new_df = pd.concat([opened_csv, update_df], ignore_index = True)
    
    new_df.to_csv(filename + '.csv', index = False)
    
    print('Parameter Saving Successful')
    
###


class OptimizeVQA(ABC):
    '''    
    Training Class for optimizing VQA algorithm that wraps around VQABaseBackend and QAOAVariationalBaseParams objects.
    This function utilizes the `update_from_raw` of the QAOAVariationalBaseParams class and `expectation` method of 
    the VQABaseBackend class to create a wrapper callable which is passed into scipy.optimize.minimize for minimization.
    Only the trainable parameters should be passed instead of the complete
    AbstractParams object. The construction is completely backend and type 
    of VQA agnostic. 

    This class is an AbstractBaseClass on top of which other specific Optimizer
    classes are built.
    
    .. Tip:: 
        Optimizer that usually work the best for quantum optimization problems
        
        * Gradient free optimizer - Cobyla

    Parameters
    ----------
    vqa_object:
        Backend object of class VQABaseBackend which contains information on the backend used to perform computations, and the VQA circuit.
    
    variational_params:
        Object of class QAOAVariationalBaseParams, which contains information on the circuit to be executed,  the type of parametrisation, and the angles of the VQA circuit.
    
    method: 
        which method to use for optimization. Choose a method from the list
        of supported methods by scipy optimize, or from the list of custom gradient optimisers.

    optimizer_dict:
        All extra parameters needed for customising the optimising, as a dictionary.

    #Optimizers that usually work the best for quantum optimization problems:
        1) Gradient free optimizer: BOBYQA, ImFil, Cobyla
        2) Gradient based optimizer: L-BFGS, ADAM (With parameter shift gradients)
        Note: Adam is not a part of scipy, it will added in a future version
    '''

    def __init__(self,
                 vqa_object: Type[VQABaseBackend],
                 variational_params: Type[QAOAVariationalBaseParams],
                 optimizer_dict: dict):

        if not isinstance(vqa_object, VQABaseBackend):
            raise TypeError(
                f'The specified cost object must be of type VQABaseBackend')

        self.vqa = vqa_object
        # extract initial parameters from the params of vqa_object
        self.variational_params = variational_params
        self.initial_params = variational_params.raw()
        self.method = optimizer_dict['method'].lower()
        self.save_to_csv = optimizer_dict.get('save_intermediate', False)
        
        self.log = Logger({'cost': 
                           {
                               'history_update_bool': optimizer_dict.get('cost_progress',True), 
                               'best_update_string': 'LowestOnly'
                           }, 
                           'measurement_outcomes': 
                           {
                               'history_update_bool': optimizer_dict.get('optimization_progress',False), 
                               'best_update_string': 'Replace'
                           },
                           'param_log': 
                           {
                               'history_update_bool': optimizer_dict.get('parameter_log',True), 
                               'best_update_string': 'Replace'
                           }, 
                           'func_evals': 
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
                           }, 
                           'job_ids':
                           {
                               'history_update_bool': True,
                               'best_update_string': 'Replace'
                           }
                          }, 
                          {
                              'root_nodes': ['cost', 'func_evals', 'jac_func_evals', 
                                             'qfim_func_evals'],
                              'best_update_structure': (['cost', 'param_log'], 
                                                        ['cost', 'measurement_outcomes'], 
                                                        ['cost', 'job_ids'])
                          })
        
        self.log.log_variables({'func_evals': 0, 'jac_func_evals': 0, 'qfim_func_evals': 0})

    @abstractmethod
    def __repr__(self):
        """
        Overview of the instantiated optimier/trainer.
        """
        string = f"Optimizer for VQA of type: {type(self.vqa).__base__.__name__} \n"
        string += f"Backend: {type(self.vqa).__name__} \n"
        string += f"Method: {str(self.method).upper()}\n"

        return string

    def __call__(self):
        """
        Call the class instance to initiate the training process.
        """
        self.optimize()
        return self

    # def evaluate_jac(self, x):

    def optimize_this(self, x):
        '''
        A function wrapper to execute the circuit in the backend. This function 
        will be passed as argument to be optimized by scipy optimize.

        Parameters
        ----------
        x: 
            Parameters (a list of floats) over which optimization is performed.

        Returns 
        -------
        cost value: 
            Cost value which is evaluated on the declared backend.

        Returns
        -------
        :
            Cost Value evaluated on the declared backed or on the Wavefunction Simulator if specified so
        '''
        
        log_dict = {}
        log_dict.update({'param_log': deepcopy(x)})
        
        self.variational_params.update_from_raw(deepcopy(x))
        
        if hasattr(self.vqa, 'log_with_backend') and callable(getattr(self.vqa, 'log_with_backend')):
            self.vqa.log_with_backend(metric_name="variational_params",
                                      value=self.variational_params,
                                      iteration_number=self.log.func_evals.best[0])
        
        if self.save_to_csv:
            save_parameter('param_log', deepcopy(x))
        
        callback_cost = self.vqa.expectation(self.variational_params)
        
        log_dict.update({'cost': callback_cost})
        current_eval = self.log.func_evals.best[0]
        current_eval += 1
        log_dict.update({'func_evals': current_eval})
        
        log_dict.update({'measurement_outcomes': self.vqa.measurement_outcomes})
        
        if hasattr(self.vqa, 'log_with_backend') and callable(getattr(self.vqa, 'log_with_backend')):
            self.vqa.log_with_backend(metric_name="measurement_outcomes",
                                      value=self.vqa.measurement_outcomes,
                                      iteration_number=self.log.func_evals.best[0])
        
        if hasattr(self.vqa, 'job_id'):
            log_dict.update({'job_ids': self.vqa.job_id})
            
            if self.save_to_csv:
                save_parameter('job_ids', self.vqa.job_id)
            
        self.log.log_variables(log_dict)

        return callback_cost

    @abstractmethod
    def optimize(self):
        '''
        Main method which implements the optimization process.
        Child classes must implement this method according to their respective
        optimization process.

        Returns
        -------
        :
            The optimized return object from the ``scipy.optimize`` package the result is assigned to the attribute ``opt_result``
        '''
        pass

    def results_dictionary(self,
                           file_path: str = None,
                           file_name: str = None):
        '''
        This method formats a dictionary that consists of all the results from 
        the optimization process. The dictionary is returned by this method.
        The results can also be saved by providing the path to save the pickled file

        .. Important:: 
            Child classes must implement this method so that the returned object,
            a ``Dictionary`` is consistent across all Optimizers.

        TODO: 
            Decide results datatype: dictionary or namedtuple?

        Parameters
        ----------
        file_path: 
            To save the results locally on the machine in pickle format, specify 
            the entire file path to save the result_dictionary.

        file_name:
            Custom name for to save the data; a generic name with the time of 
            optimization is used if not specified

        Returns
        -------
        :
            Dictionary with the following keys
                
                #. "solution"
                    #. "bitstring"
                    #. "degeneracy"
                #. "number of evals"
                #. "jac evals"
                #. "qfim evals"
                #. "parameter log"
                #. "optimized param"
                #. "intermediate cost"
                #. "optimized cost"
                #. "intermediate measurement outcomes"
                #. "optimized measurement outcomes"
                #. "optimization method"
        '''
        date_time = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
        file_name = f'opt_results_{date_time}' if file_name is None else file_name
        
        self.qaoa_result = Result(self.log, self.method, self.vqa.cost_hamiltonian)
        
        if(file_path and os.path.isdir(file_path)):
            print('Saving results locally')
            pickled_file = open(f'{file_path}/{file_name}.pcl', 'wb')
            pickle.dump(self.qaoa_result, pickled_file)
            pickled_file.close()

        return  # result_dict


class ScipyOptimizer(OptimizeVQA):
    """
    Python vanilla scipy based optimizer for the VQA class.
    
    .. Tip::
        Using bounds may result in lower optimization performance

    Parameters
    ----------
    vqa_object:
        Backend object of class VQABaseBackend which contains information on the backend used to perform computations, and the VQA circuit.
    
    variational_params:
        Object of class QAOAVariationalBaseParams, which contains information on the circuit to be executed,  the type of parametrisation, and the angles of the VQA circuit.

    optimizer_dict:
        jac: 
            gradient as `Callable` if defined. else None

        hess: 
            hessian as `Callable` if defined. else None

        * bounds
        
            * parameter bounds while training, defaults to ``None``

        * constraints
        
            * Linear/Non-Linear constraints (only for COBYLA, SLSQP and trust-constr)

        * tol
        
            * Tolerance for termination

        * maxiters
        
            * sets ``maxiters = 100`` by default if not specified.
            
        * maxfev
        
            * sets ``maxfev = 100`` by default if not specified.
            
        * optimizer_options
        
            * Dictionary of optimiser-specific arguments, defaults to ``None``

    """
    GRADIENT_FREE = ['cobyla', 'nelder-mead', 'powell', 'slsqp']
    SCIPY_METHODS = MINIMIZE_METHODS

    def __init__(self,
                 vqa_object: Type[VQABaseBackend],
                 variational_params: Type[QAOAVariationalBaseParams],
                 optimizer_dict: dict):

        super().__init__(vqa_object, variational_params, optimizer_dict)

        self.vqa_object = vqa_object
        self._validate_and_set_params(optimizer_dict)

    def _validate_and_set_params(self, optimizer_dict):
        """
        Verify that the specified arguments are valid for the particular optimizer.
        """

        if self.method not in ScipyOptimizer.SCIPY_METHODS:
            raise ValueError(
                "Specified method not supported by Scipy Minimize")

        jac = optimizer_dict.get('jac', None)
        hess = optimizer_dict.get('hess', None)
        jac_options = optimizer_dict.get('jac_options', None)
        hess_options = optimizer_dict.get('hess_options', None)

        if self.method not in ScipyOptimizer.GRADIENT_FREE and (jac is None or not isinstance(jac, (Callable, str))):
            raise ValueError(
                "Please specify either a string or provide callable gradient in order to use gradient based methods")
        else:
            if isinstance(jac, str):
                self.jac = derivative(
                    self.vqa_object, self.variational_params, self.log, 'gradient', 
                    jac, jac_options)
            else:
                self.jac = jac

        hess = optimizer_dict.get('hess', None)
        if hess is not None and not isinstance(hess, (Callable, str)):
            raise ValueError("Hessian needs to be of type Callable or str")
        else:
            if isinstance(hess, str):
                self.hess = derivative(
                    self.vqa_object, self.variational_params, self.log, 'hessian', 
                    hess, hess_options)
            else:
                self.hess = hess

        constraints = optimizer_dict.get('constraints', ())
        if constraints == () or isinstance(constraints, LinearConstraint) or isinstance(constraints, NonlinearConstraint):
            self.constraints = constraints
        else:
            raise ValueError(
                f"Constraints for Scipy optimization should be of type {LinearConstraint} or {NonlinearConstraint}")

        bounds = optimizer_dict.get('bounds', None)

        if bounds is None or isinstance(bounds, Bounds):
            self.bounds = bounds
        elif isinstance(bounds, List):
            lb = np.array(bounds).T[0]
            ub = np.array(bounds).T[1]
            self.bounds = Bounds(lb, ub)
        else:
            raise ValueError(
                f"Bounds for Scipy optimization should be of type {Bounds}, or a list in the form [[ub1, lb1], [ub2, lb2], ...]")
        
        self.options = optimizer_dict.get('optimizer_options', {})
        self.options["maxiter"] = optimizer_dict.get('maxiter', None)
        if optimizer_dict.get('maxfev') is not None:
            self.options["maxfev"] = optimizer_dict.get('maxfev', None)

        self.tol = optimizer_dict.get('tol', None)

        return self

    def __repr__(self):
        """
        Overview of the instantiated optimier/trainer.
        """
        maxiter = self.options["maxiter"]
        string = f"Optimizer for VQA of type: {type(self.vqa).__base__.__name__} \n"
        string += f"Backend: {type(self.vqa).__name__} \n"
        string += f"Method: {str(self.method).upper()} with Max Iterations: {maxiter}\n"

        return string

    def optimize(self):
        '''
        Main method which implements the optimization process using ``scipy.minimize``.

        Returns
        -------
        : 
            Returns self after the optimization process is completed.
        '''
        
        try:
            if self.method not in ScipyOptimizer.GRADIENT_FREE:
                if self.hess == None:
                    result = minimize(self.optimize_this, x0=self.initial_params, method=self.method,
                                      jac=self.jac, tol=self.tol, constraints=self.constraints,
                                      options=self.options, bounds=self.bounds)
                else:
                    result = minimize(self.optimize_this, x0=self.initial_params, method=self.method,
                                      jac=self.jac, hess=self.hess, tol=self.tol,
                                      constraints=self.constraints, options=self.options, bounds=self.bounds)
            else:
                result = minimize(self.optimize_this, x0=self.initial_params, method=self.method,
                                  tol=self.tol, constraints=self.constraints, options=self.options, bounds=self.bounds)
        except ConnectionError as e:
            print(e, '\n')
            print("The optimization has been terminated early. You can retrieve results from the optimization runs that were completed through the .results_information method.")
        finally:
            self.results_dictionary()
            return self


class CustomScipyGradientOptimizer(OptimizeVQA):
    """
    Python custom scipy gradient based optimizer for the VQA class.

    .. Tip::
        Using bounds may result in lower optimization performance

    Parameters
    ----------
    vqa_object:
        Backend object of class VQABaseBackend which contains information on the backend used to perform computations, and the VQA circuit.
    
    variational_params:
        Object of class QAOAVariationalBaseParams, which contains information on the circuit to be executed,  the type of parametrisation, and the angles of the VQA circuit.

    optimizer_dict:
        * jac
        
            * gradient as ``Callable``, if defined else ``None``

        * hess
        
            * hessian as ``Callable``, if defined else ``None``

        * bounds
        
            * parameter bounds while training, defaults to ``None``

        * constraints
        
            * Linear/Non-Linear constraints (only for COBYLA, SLSQP and trust-constr)

        * tol
        
            * Tolerance for termination

        * maxiters
        
            * sets ``maxiters = 100`` by default if not specified.
            
        * maxfev
        
            * sets ``maxfev = 100`` by default if not specified.
            
        * optimizer_options
        
            * Dictionary of optimiser-specific arguments, defaults to ``None``

    """
    CUSTOM_GRADIENT_OPTIMIZERS = ['vgd', 'newton',
                                  'rmsprop', 'natural_grad_descent', 'spsa']

    def __init__(self,
                 vqa_object: Type[VQABaseBackend],
                 variational_params: Type[QAOAVariationalBaseParams],
                 optimizer_dict: dict):

        super().__init__(vqa_object, variational_params, optimizer_dict)

        self.vqa_object = vqa_object
        self._validate_and_set_params(optimizer_dict)

    def _validate_and_set_params(self, optimizer_dict):
        """
        Verify that the specified arguments are valid for the particular optimizer.
        """

        if self.method not in CustomScipyGradientOptimizer.CUSTOM_GRADIENT_OPTIMIZERS:
            raise ValueError(
                f"Please choose from the supported methods: {CustomScipyGradientOptimizer.CUSTOM_GRADIENT_OPTIMIZERS}")

        jac = optimizer_dict.get('jac', None)
        hess = optimizer_dict.get('hess', None)
        jac_options = optimizer_dict.get('jac_options', None)
        hess_options = optimizer_dict.get('hess_options', None)

        if jac is None or not isinstance(jac, (Callable, str)):
            raise ValueError(
                "Please specify either a string or provide callable gradient in order to use gradient based methods")
        else:
            if isinstance(jac, str):
                self.jac = derivative(
                    self.vqa_object, self.variational_params, self.log, 
                    'gradient', jac, jac_options)
            else:
                self.jac = jac

        if hess is not None and not isinstance(hess, (Callable, str)):
            raise ValueError("Hessian needs to be of type Callable or str")
        else:
            if isinstance(hess, str):
                self.hess = derivative(
                    self.vqa_object, self.variational_params, self.log, 
                    'hessian', hess, hess_options)
            else:
                self.hess = hess

        constraints = optimizer_dict.get('constraints', ())
        if constraints == () or isinstance(constraints, LinearConstraint) or isinstance(constraints, NonlinearConstraint):
            self.constraints = constraints
        else:
            raise ValueError(
                f"Constraints for Scipy optimization should be of type {LinearConstraint} or {NonlinearConstraint}")

        bounds = optimizer_dict.get('bounds', None)
        if bounds is None or isinstance(bounds, Bounds):
            self.bounds = bounds
        else:
            raise ValueError(
                f"Bounds for Scipy optimization should be of type {Bounds}")

        self.options = optimizer_dict.get('optimizer_options', {})
        self.options["maxiter"] = optimizer_dict.get('maxiter', None)
        if optimizer_dict.get('maxfev') is not None:
            self.options["maxfev"] = optimizer_dict.get('maxfev', None)

        self.tol = optimizer_dict.get('tol', None)

        return self

    def __repr__(self):
        """
        Overview of the instantiated optimier/trainer.
        """
        maxiter = self.options["maxiter"]
        string = f"Optimizer for VQA of type: {type(self.vqa).__base__.__name__} \n"
        string += f"Backend: {type(self.vqa).__name__} \n"
        string += f"Method: {str(self.method).upper()} with Max Iterations: {maxiter}\n"

        return string

    def optimize(self):
        '''
        Main method which implements the optimization process using ``scipy.minimize``.

        Returns
        -------
        : 
            The optimized return object from the ``scipy.optimize`` package the result is assigned to the attribute ``opt_result``
        '''
        if self.method == 'vgd':
            method = om.grad_descent
        elif self.method == 'newton':
            method = om.newton_descent
        elif self.method == 'rmsprop':
            method = om.rmsprop
        elif self.method == 'natural_grad_descent':
            method = om.natural_grad_descent
            self.options['qfim'] = qfim(self.vqa_object,
                self.variational_params, self.log)
        elif self.method == 'spsa':
            print("Warning : SPSA is an experimental feature.")
            method = om.SPSA
        
        try:
            if self.hess == None:
                result = minimize(self.optimize_this, x0=self.initial_params, method=method,
                                  jac=self.jac, tol=self.tol, constraints=self.constraints,
                                  options=self.options, bounds=self.bounds)
            else:
                result = minimize(self.optimize_this, x0=self.initial_params, method=method,
                                  jac=self.jac, hess=self.hess, tol=self.tol, constraints=self.constraints,
                                  options=self.options, bounds=self.bounds)
        except ConnectionError as e:
            print("The optimization has been terminated early. Most likely due to a connection error. You can retrieve results from the optimization runs that were completed through the .results_information method.")
        finally:
            self.results_dictionary()
            return self
