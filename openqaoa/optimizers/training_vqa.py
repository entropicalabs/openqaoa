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


class OptimizeVQA(ABC):
    '''    
    Training Class for optimizing VQA algorithm based on VQABaseBackend.
    This function utilizes the __call__ method of the cost class.
    Only the trainable parameters should be passed instead of the complete
    AbstractParams object. The construction is completely backend and type 
    of VQA agnostic. 

    This class is an AbstractBaseClass on top of which other specific Optimizer
    classes are built.

    PARAMETERS
    ----------
    vqa_object:
        object of class qaoa which is contains methods and attributes 
        of class qaoa which will be useful for optimization in this
        module

    method: 
        which method to use for optimization. Choose a method from the list
        of supported methods by scipy optimize

    optimizer_dict:
        All extra parameters needed for customising the optimising, as a dictionary

    #Optimizers that usually work the best for quantum optimization problems:
        1) Gradient free optimizer: BOBYQA, ImFil, Cobyla
        2) Gradient based optimizer: L-BFGS, ADAM
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

        self.cost_progress = []
        self.count_progress = []
        self.probability_progress = []
        self.param_log = []
        self.opt_result = None

        self.optimization_progress_bool = optimizer_dict.get(
            'optimization_progress', False)
        self.cost_progress_bool = optimizer_dict.get('cost_progress', False)
        self.parameter_log_bool = optimizer_dict.get('parameter_log', False)

    @abstractmethod
    def __repr__(self):
        """
        Overview of the instantiated optimier/trainer
        """
        string = f"Optimizer for VQA of type: {type(self.vqa).__base__.__name__} \n"
        string += f"Backend: {type(self.vqa).__name__} \n"
        string += f"Method: {str(self.method).upper()}\n"

        return string

    def __call__(self):
        """
        Call the class instance to initiate the training process
        """
        self.optimize()
        return self

    # def evaluate_jac(self, x):

    def optimize_this(self, x):
        '''
        A function wrapper for the qaoa cost method of the class
        qaoa_pulse. This function will be passed as argument to be 
        optimized by scipy optimize

        Parameters
        ----------
        x: parameters over which optimization is performed

        Returns 
        -------
        1) cost value: evaluated either on the declared backed or on the Wavefunction
        Simulator if specified so

        Actions
        -------
        1) Appends the cost value after each iteration in the optimization
            process to self.cost_progress list
        2) Appends all intermediate parameters in self.param_log list
        3) Checks if self.vqa has the self.counts attribute. If it exists, appends 
            the counts of each state for that circuit evaluation to 
            self.count_progress list.
        4) Checks if self.vqa has the self.probability attribute. If it exists, 
            appends the probability of each state for that circuit evaluation to 
            self.count_progress list.

        '''
        # First we append the new parameters and its cost.
        self.param_log.append(deepcopy(x))
        self.variational_params.update_from_raw(deepcopy(x))
        callback_cost = self.vqa.expectation(self.variational_params)
        self.cost_progress.append(callback_cost)

        if hasattr(self.vqa, 'counts'):
            self.count_progress.append(self.vqa.counts)
        elif hasattr(self.vqa, 'probability'):
            self.probability_progress.append(self.vqa.probability)

        # If optimization progress boolean is false, we do not record the
        # counts and probablity progress. We only keep the best results.
        if self.optimization_progress_bool == False:
            if np.argmin(self.cost_progress) == len(self.cost_progress):
                if hasattr(self.vqa, 'counts'):
                    self.count_progress = [self.count_progress[-1]]
                elif hasattr(self.vqa, 'probability'):
                    self.probability_progress = [self.probability_progress[-1]]
            else:
                if hasattr(self.vqa, 'counts'):
                    self.count_progress = [self.count_progress[0]]
                elif hasattr(self.vqa, 'probability'):
                    self.probability_progress = [self.probability_progress[0]]

        return callback_cost

    @abstractmethod
    def optimize(self):
        '''
        Main method which implements the optimization process
        Child classes must implement this method according to their respective
        optimization process.

        RETURNS:
        result: the optimized return object from the scipy.optimize package
                the result is assigned to the attribute 'opt_result'
        '''
        pass

    def results_dictionary(self,
                           final_params,
                           opt_cost,
                           nfev,
                           file_path: str = None,
                           file_name: str = None):
        '''
        This method returns a dictionary of all results of optimization
        The results can also be saved by providing the path to save the pickled file

        NOTE: Child classes must implement this method so that the returned object,
              a ``Dictionary`` is consistent across all Optimizers.

        TODO: 
            Decide results datatype: dictionary or namedtuple?


        PARAMS:
        file_path: to save the results locally on the machine in pickle format,
                    specify the entire file path to save the result_dictionary.

        file_name: custom name for to save the data; a generic name with the time of 
                    optimization is used if not specified

        RETURNS:
        Dictionary with the following keys:
            1) "opt result"
            2) "parameter log"
            3) "final params"
            4) "cost progress list"
            5) "optimization method"
            6) "cost function calls"
            7) "optimal cost"
        '''
        date_time = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
        file_name = f'opt_results_{date_time}' if file_name is None else file_name

        result_dict = {
            'opt result': self.opt_result,
            'parameter log': np.array(self.param_log).tolist(),
            'final params': np.array(final_params).tolist(),
            'cost progress list': np.array(self.cost_progress).tolist(),
            'count progress list': np.array(self.count_progress).tolist(),
            'probability progress list': np.array(self.probability_progress).tolist(),
            'optimization method': self.method,
            'cost function calls': nfev,
            'optimal cost': opt_cost
        }

        if(file_path and os.path.isdir(file_path)):
            print('Saving results locally')
            pickled_file = open(f'{file_path}/{file_name}.pcl', 'wb')
            pickle.dump(result_dict, pickled_file)
            pickled_file.close()

        return result_dict


class ScipyOptimizer(OptimizeVQA):
    """
    Python vanilla scipy based optimizer for the VQA class

    Parameters
    ----------
    vqa_object: 
        object of class qaoa which is contains methods and attributes 
        of class qaoa which will be useful for optimization in this
        module

    method: 
        which method to use for optimization. Choose a method from the list
        of supported methods by scipy optimize

    optimizer_dict:
        jac: 
            gradient as `Callable`, if defined else None

        hess: 
            hessian as `Callable`, if defined else None

        bounds: 
            parameter bounds while training, defaults to None
            NOTE: "using bounds may result in lower optimization performance"

        constraints: 
            Linear/Non-Linear constraints
            (only for COBYLA, SLSQP and trust-constr)

        tol: 
            Tolerance for termination

        maxiters:
            sets maxiters = 100 by default if not specified.

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
        Verify that the specified arguments are valid for the particular optimizer
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
                self.jac = self.vqa_object.derivative_function(
                    self.variational_params, 'gradient', jac, jac_options)
            else:
                self.jac = jac

        hess = optimizer_dict.get('hess', None)
        if hess is not None and not isinstance(hess, (Callable, str)):
            raise ValueError("Hessian needs to be of type Callable or str")
        else:
            if isinstance(hess, str):
                self.hess = self.vqa_object.derivative_function(
                    self.variational_params, 'hessian', hess, hess_options)
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

        maxiter = optimizer_dict.get('maxiter', 100)
        self.options = {'maxiter': maxiter}

        self.tol = optimizer_dict.get('tol', None)

        return self

    def __repr__(self):
        """
        Overview of the instantiated optimier/trainer
        """
        maxiter = self.options["maxiter"]
        string = f"Optimizer for VQA of type: {type(self.vqa).__base__.__name__} \n"
        string += f"Backend: {type(self.vqa).__name__} \n"
        string += f"Method: {str(self.method).upper()} with Max Iterations: {maxiter}\n"

        return string

    def optimize(self):
        '''
        Main method which implements the optimization process using scipy.minimize

        RETURNS
        -------
        result: the optimized return object from the scipy.optimize package
                the result is assigned to the attribute 'opt_result'
        '''
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

        self.opt_result = result
        return self

    def results_information(self, file_path: str = None, file_name: str = None):
        '''
        This method returns a dictionary of all results of optimization
        The results can also be saved by providing the path to save the pickled file

        Parameters
        ----------
        file_path: 
            to save the results locally on the machine in pickle format,
            specify the entire file path to save the result_dictionary.

        file_name: 
            File name to save the data. 
            [DEFAULT] a generic name with the time of optimization is used if not specified

        RETURNS
        -------
        Dictionary with the following keys:
            1) "opt result"
            2) "parameter log"
            3) "final params"
            4) "cost progress list"
            5) "optimization method"
            6) "cost function calls"
            7) "optimal cost"
        '''
        final_params = self.opt_result.x
        opt_cost = self.opt_result.fun
        nfev = self.opt_result.nfev

        results = self.results_dictionary(
            final_params, opt_cost, nfev, file_path, file_name)
        return results


class CustomScipyGradientOptimizer(OptimizeVQA):
    """
    Python vanilla scipy based optimizer for the VQA class

    Parameters
    ----------
    vqa_object: 
        object of class qaoa which is contains methods and attributes 
        of class qaoa which will be useful for optimization in this
        module

    method: 
        which method to use for optimization. Choose a method from the list
        of supported methods by scipy optimize

    optimizer_dict:
        jac: 
            gradient as `Callable`, if defined else None

        hess: 
            hessian as `Callable`, if defined else None

        bounds: 
            parameter bounds while training, defaults to None
            NOTE: "using bounds may result in lower optimization performance"

        constraints: 
            Linear/Non-Linear constraints
            (only for COBYLA, SLSQP and trust-constr)

        tol: 
            Tolerance for termination

        maxiters:
            sets maxiters = 100 by default if not specified.
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
        Verify that the specified arguments are valid for the particular optimizer
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
                self.jac = self.vqa_object.derivative_function(
                    self.variational_params, 'gradient', jac, jac_options)
            else:
                self.jac = jac

        if hess is not None and not isinstance(hess, (Callable, str)):
            raise ValueError("Hessian needs to be of type Callable or str")
        else:
            if isinstance(hess, str):
                self.hess = self.vqa_object.derivative_function(
                    self.variational_params, 'hessian', hess, hess_options)
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

        self.options = optimizer_dict

        # Remove redundant keys (because self.jac and self.hess already exist)
        optimizer_dict.pop('jac', None)
        optimizer_dict.pop('hess', None)

        self.tol = optimizer_dict.get('tol', None)

        return self

    def __repr__(self):
        """
        Overview of the instantiated optimier/trainer
        """
        maxiter = self.options["maxiter"]
        string = f"Optimizer for VQA of type: {type(self.vqa).__base__.__name__} \n"
        string += f"Backend: {type(self.vqa).__name__} \n"
        string += f"Method: {str(self.method).upper()} with Max Iterations: {maxiter}\n"

        return string

    def optimize(self):
        '''
        Main method which implements the optimization process using scipy.minimize

        RETURNS
        -------
        result: the optimized return object from the scipy.optimize package
                the result is assigned to the attribute 'opt_result'
        '''
        if self.method == 'vgd':
            method = om.grad_descent
        elif self.method == 'newton':
            method = om.newton_descent
        elif self.method == 'rmsprop':
            method = om.rmsprop
        elif self.method == 'natural_grad_descent':
            method = om.natural_grad_descent
            self.options['qfim'] = self.vqa_object.qfim(
                self.variational_params)
        elif self.method == 'spsa':
            print("Warning : SPSA is an experimental feature.")
            method = om.SPSA

        if self.hess == None:
            result = minimize(self.optimize_this, x0=self.initial_params, method=method,
                              jac=self.jac, tol=self.tol, constraints=self.constraints,
                              options=self.options, bounds=self.bounds)
        else:
            result = minimize(self.optimize_this, x0=self.initial_params, method=method,
                              jac=self.jac, hess=self.hess, tol=self.tol, constraints=self.constraints,
                              options=self.options, bounds=self.bounds)

        self.opt_result = result
        return self

    def results_information(self, file_path: str = None, file_name: str = None):
        '''
        This method returns a dictionary of all results of optimization
        The results can also be saved by providing the path to save the pickled file

        Parameters
        ----------
        file_path: 
            to save the results locally on the machine in pickle format,
            specify the entire file path to save the result_dictionary.

        file_name: 
            File name to save the data. 
            [DEFAULT] a generic name with the time of optimization is used if not specified

        RETURNS
        -------
        Dictionary with the following keys:
            1) "opt result"
            2) "parameter log"
            3) "final params"
            4) "cost progress list"
            5) "optimization method"
            6) "cost function calls"
            7) "optimal cost"
        '''
        final_params = self.opt_result.x
        opt_cost = self.opt_result.fun
        nfev = self.opt_result.nfev

        results = self.results_dictionary(
            final_params, opt_cost, nfev, file_path, file_name)
        return results
