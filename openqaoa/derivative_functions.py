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

from __future__ import annotations

import numpy as np
import random

from copy import deepcopy
from .qaoa_parameters.extendedparams import QAOAVariationalExtendedParams 
from .qaoa_parameters.baseparams import QAOAVariationalBaseParams
from .basebackend import QAOABaseBackend
from .optimizers.logger_vqa import Logger
from .cost_function import cost_function


def update_and_compute_expectation(backend_obj: QAOABaseBackend, 
                                   params: QAOAVariationalBaseParams, 
                                   logger: Logger):
    """
    Helper function that returns a callable that takes in a list/nparray of raw parameters.
    This function will handle:
    
        #. Updating logger object with `logger.log_variables`
        #. Updating variational parameters with `update_from_raw` 
        #. Computing expectation with `backend_obj.expectation`
    
    Parameters
    ----------
    backend_obj: QAOABaseBackend
        `QAOABaseBackend` object that contains information about the backend that is being used to perform the QAOA circuit
    params : QAOAVariationalBaseParams
        `QAOAVariationalBaseParams` object containing variational angles. 
    logger: Logger
        Logger Class required to log information from the evaluations required for the jacobian/hessian computation.
    Returns
    -------
    out:
        A callable that accepts a list/array of parameters, and returns the computed expectation value. 
    """
    
    def fun(args, n_shots=None):
        current_total_eval = logger.func_evals.best[0]
        current_total_eval += 1
        current_jac_eval = logger.jac_func_evals.best[0]
        current_jac_eval += 1
        logger.log_variables({'func_evals': current_total_eval, 
                              'jac_func_evals': current_jac_eval})
        params.update_from_raw(args)
        
        n_shots_dict = {'n_shots':n_shots} if n_shots else {}
        return backend_obj.expectation(params, **n_shots_dict)

    return fun

def update_and_get_counts(  backend_obj: QAOABaseBackend, 
                            params: QAOAVariationalBaseParams, 
                            logger: Logger):
    
    """
    Helper function that returns a callable that takes in a list/nparray of raw parameters.
    This function will handle:
        (1) Updating logger object with `logger.log_variables`
        (2) Updating variational parameters with `update_from_raw` 
        (3) Getting the counts dictonary with `backend_obj.get_counts`
    
    PARAMETERS
    ----------
    backend_obj: QAOABaseBackend
        `QAOABaseBackend` object that contains information about the backend that is being used to perform the QAOA circuit
        
    params : QAOAVariationalBaseParams
        `QAOAVariationalBaseParams` object containing variational angles.
        
    logger: Logger
        Logger Class required to log information from the evaluations required for the jacobian/hessian computation.
    
    Returns
    -------
    out:
        A callable that accepts a list/array of parameters, and returns the counts dictonary. 
    """
    
    def fun(args, n_shots=None):
        current_total_eval = logger.func_evals.best[0]
        current_total_eval += 1
        current_jac_eval = logger.jac_func_evals.best[0]
        current_jac_eval += 1
        logger.log_variables({'func_evals': current_total_eval, 
                              'jac_func_evals': current_jac_eval,
                              'n_shots': n_shots})
        params.update_from_raw(args)
        
        n_shots_dict = {'n_shots':n_shots} if n_shots else {}
        return backend_obj.get_counts(params, **n_shots_dict)

    return fun

def derivative(backend_obj: QAOABaseBackend, 
               params: QAOAVariationalBaseParams, 
               logger: Logger, 
               derivative_type: str = None, 
               derivative_method: str = None, 
               derivative_options: dict = None):
    """
    Returns a callable function that calculates the gradient according to the specified `gradient_method`.

    Parameters
    ----------
    backend_obj: QAOABaseBackend
        `QAOABaseBackend` object that contains information about the backend that is being used to perform the QAOA circuit
    params : QAOAVariationalBaseParams
        `QAOAVariationalBaseParams` object containing variational angles.
    logger: Logger
        Logger Class required to log information from the evaluations required for the jacobian/hessian computation.
    derivative_type : str
        Type of derivative to compute. Either `gradient` or `hessian`.
    derivative_method : str
        Computational method of the derivative. Either `finite_difference`, `param_shift`, `stoch_param_shift`, or `grad_spsa`.
    derivative_options : dict
        Dictionary containing options specific to each `derivative_method`.
    cost_std : QAOACost
        `QAOACost` object that computes expectation values when executed. Standard parametrisation.
    cost_ext : QAOACost
        `QAOACost` object that computes expectation values when executed. Extended parametrisation. Mainly used to compute parameter shifts at each individual gate, which is summed to recover the parameter shift for a parametrised layer.
    Returns
    -------
    out:
        The callable derivative function of the cost function, generated based on the `derivative_type`, `derivative_method`, and `derivative_options` specified.
    """
    # Default derivative_options used if none are specified.
    default_derivative_options = {"stepsize": 0.00001,
                                  "n_beta_single": -1,
                                  "n_beta_pair": -1,
                                  "n_gamma_single": -1,
                                  "n_gamma_pair": -1}

    derivative_options = {**default_derivative_options, **derivative_options
                          } if derivative_options is not None else default_derivative_options

    # cost_std = derivative_dict['cost_std']
    # cost_ext = derivative_dict['cost_ext']
    params_ext = QAOAVariationalExtendedParams.empty(backend_obj.circuit_params)

    derivative_types = ['gradient', 'gradient_w_variance', 'hessian']
    assert derivative_type in derivative_types,\
        "Unknown derivative type specified - please choose between " + \
        str(derivative_types)

    derivative_methods = ['finite_difference',
                          'param_shift', 'stoch_param_shift', 'grad_spsa']
    assert derivative_method in derivative_methods,\
        "Unknown derivative computation method specified - please choose between " + \
        str(derivative_methods)
    
    params = deepcopy(params)

    if derivative_type == 'gradient':

        if derivative_method == 'finite_difference':
            out = grad_fd(backend_obj, params, derivative_options, logger)
        elif derivative_method == 'param_shift':
            assert params.__class__.__name__ == 'QAOAVariationalStandardParams', f"{params.__class__.__name__} not supported - only Standard Parametrisation is supported for parameter shift/stochastic parameter shift for now."
            out = grad_ps(backend_obj, params, params_ext, logger)
        elif derivative_method == 'stoch_param_shift':
            assert params.__class__.__name__ == 'QAOAVariationalStandardParams', f"{params.__class__.__name__} not supported - only Standard Parametrisation is supported for parameter shift/stochastic parameter shift for now."
            out = grad_sps(backend_obj, params, params_ext, derivative_options, logger)
        elif derivative_method == 'grad_spsa':
            out = grad_spsa(backend_obj, params, derivative_options, logger)

    elif derivative_type == 'gradient_w_variance':

        #TODO: complete this
        out = grad_w_variance(backend_obj, params, derivative_options, logger, derivative_method)

    elif derivative_type == 'hessian':

        if derivative_method == 'finite_difference':
            out = hessian_fd(backend_obj, params, derivative_options, logger)
        else:
            raise ValueError('Only support hessian derivative method is finite_difference. Your choice: {}'.format(derivative_method))

    return out

def __grad_variance_n_shots(args, backend_obj, params, gradient_options, logger, method='finite_difference'):
    """
    returns a callable function that Computes the gradient and its variance for the i-th parameter using finite difference.
    Or the gradient and its variance using spsa.
    """    

    def fun(i=0, n_shots=None):
        """
        Computes the gradient and its variance for the i-th parameter using finite difference.
        Or the gradient and its variance using spsa.
        """

        # get value of eta, the function to get counts, hamiltonian, and alpha
        eta = gradient_options['stepsize']
        fun = update_and_get_counts(backend_obj, params, logger)
        hamiltonian = backend_obj.circuit_params.cost_hamiltonian
        alpha = backend_obj.cvar_alpha
        
        if method in ['finite_difference']:
            # vector of zeros with length equal to the number of parameters, and 1 at the i-th position
            vect_eta = np.zeros(len(args))
            vect_eta[i] = 1
        elif method in ['grad_spsa']:
            # vector of random numbers (1 or -1) with length equal to the number of parameters
            vect_eta = np.random.choice([-1, 1], size=len(args))

        # get counts f(x+eta/2) and f(x-eta/2)
        counts_i_dict = fun(args - (eta/2)*vect_eta, n_shots=n_shots)
        counts_f_dict = fun(args + (eta/2)*vect_eta, n_shots=n_shots)

        #compute cost for each state in the counts dictionary
        costs_dict = {key: cost_function({key: 1}, hamiltonian, alpha) for key in counts_i_dict.keys() | counts_f_dict.keys()}

        # for each count get the cost and create a list of shot costs
        eval_i_list = [costs_dict[key] for key, value in counts_i_dict.items() for _ in range(value)]
        eval_f_list = [costs_dict[key] for key, value in counts_f_dict.items() for _ in range(value)]

        # compute a list of gradients of one shot cost
        grad_list =  np.real((np.array(eval_f_list) - np.array(eval_i_list))/eta)

        if method in ['grad_spsa']:
            # return average and variance for the gradient for the arguments
            return np.array([np.mean(grad_list)/vect_eta, np.var(grad_list)/np.abs(vect_eta)])
        
        # return average and variance for the gradient for this argument
        return np.array([np.mean(grad_list), np.var(grad_list)])

    return fun
        

def grad_w_variance(backend_obj, params, gradient_options, logger, method='finite_difference'):
    """
    Returns a callable function that calculates the gradient and its variance with the method specified.

    PARAMETERS
    ----------
    backend_obj : `QAOABaseBackend`
        backend object that computes expectation values when executed. 
    params : `QAOAVariationalBaseParams`
        parameters of the variational circuit.
    gradient_options : `dict`
        stepsize : 
            Stepsize of finite difference.
    logger : `Logger`
        logger object to log the number of function evaluations.
    method : `str`
        Method to compute the gradient. Currently only 'finite_difference' and 'grad_spsa' are supported.

    RETURNS
    -------
    grad_func: `Callable`
        Callable derivative function.
    """

    def grad_func(args, n_shots=None):
        """
        Computes the gradient and its variance for the given parameters.

        PARAMETERS
        ----------
        args : `list`
            List of parameters.
        n_shots : `int` or `List[int]`
            Number of shots to use for the finite difference method. If None, the number of shots specified in the backend object is used.
            If a list is given, each item of the list is used for the corresponding parameter.

        RETURNS
        -------
        grad : `np.array`
            Gradient of the cost function.
        grad_var : `np.array`
            Variance of the gradient of the cost function.
        """

        # function to compute gradient and its variance
        fun_grad = __grad_variance_n_shots(args, backend_obj, params, gradient_options, logger, method) 

        if method in ['finite_difference']:
            # if n_shots is int or None create a list with len of args 
            if n_shots == None or isinstance(n_shots, int):
                n_shots_list = [n_shots for _ in range(len(args))]
            else:
                n_shots_list = n_shots

            # check if n_shots_list has the same length as args
            if len(n_shots_list) != len(args):
                raise ValueError("When computing gradient, 'n_shots' and 'args' do not have the same length.")

            # lists of gradients and variances for each argument
            grad, variance = np.array([fun_grad(i_args, n_shots) for i_args, n_shots in enumerate(n_shots_list)]).T

        elif method in ['grad_spsa']:
            # compute gradient and variance for all arguments
            grad, variance = fun_grad(n_shots=n_shots)

        return grad, variance

    return grad_func




def grad_fd(backend_obj, params, gradient_options, logger):
    """
    Returns a callable function that calculates the gradient with the finite difference method.

    PARAMETERS
    ----------
    backend_obj : `QAOABaseBackend`
        backend object that computes expectation values when executed. 
    params : `QAOAVariationalBaseParams`
        parameters of the variational circuit.
    gradient_options : `dict`
        stepsize : 
            Stepsize of finite difference.
    logger : `Logger`
        logger object to log the number of function evaluations.

    RETURNS
    -------
    grad_fd_func: `Callable`
        Callable derivative function.
    """

    # Set default value of eta
    eta = gradient_options['stepsize']
    fun = update_and_compute_expectation(backend_obj, params, logger)

    def grad_fd_func(args):

        grad = np.zeros(len(args))

        for i in range(len(args)):
            vect_eta = np.zeros(len(args))
            vect_eta[i] = 1
            
            # Finite diff. calculation of gradient
            eval_i = fun(args - (eta/2)*vect_eta)
            eval_f = fun(args + (eta/2)*vect_eta)
            grad[i] = (eval_f-eval_i)/eta

        return grad

    return grad_fd_func


def grad_ps(backend_obj, params, params_ext, logger):
    """
    Returns a callable function that calculates the gradient with the parameter shift method.

    PARAMETERS
    ----------
    backend_obj : `QAOABaseBackend`
        backend object that computes expectation values when executed. 
    params : `QAOAVariationalStandardParams`
        variational parameters object, standard parametrisation.
    params_ext : `QAOAVariationalExtendedParams`
        variational parameters object, extended parametrisation.
    logger : `Logger`
        logger object to log the number of function evaluations.

    RETURNS
    -------
    grad_ps_func:
        Callable derivative function.
    """    
    # TODO : clean up conversion part + handle Fourier parametrisation
    
    fun = update_and_compute_expectation(backend_obj, params_ext, logger)
    
    coeffs_list = params.p*params.mixer_1q_coeffs + params.p*params.mixer_2q_coeffs + \
            params.p*params.cost_1q_coeffs + params.p*params.cost_2q_coeffs

    def grad_ps_func(args):

        # Convert standard to extended parameters before applying parameter shift
        args_ext = params.convert_to_ext(args)
        
        grad_ext = np.zeros(len(args_ext))

        # Apply parameter shifts
        for i in range(len(args_ext)):
            vect_eta = np.zeros(len(args_ext))
            vect_eta[i] = 1
            r = coeffs_list[i]
            grad_ext[i] = r*(fun(args_ext + (np.pi/(4*r))*vect_eta) -
                             fun(args_ext - (np.pi/(4*r))*vect_eta))

        # Convert extended param. gradient form back into std param. form
        
        m1q_entries = grad_ext[:params.p*len(params.mixer_1q_coeffs)]
        m2q_entries = grad_ext[params.p*len(params.mixer_1q_coeffs) : params.p*len(params.mixer_2q_coeffs)]
        c1q_entries = grad_ext[params.p*len(params.mixer_1q_coeffs) + params.p*len(params.mixer_2q_coeffs): params.p*len(params.mixer_1q_coeffs) + params.p*len(params.mixer_2q_coeffs) + params.p*len(params.cost_1q_coeffs)]
        c2q_entries = grad_ext[params.p*len(params.mixer_1q_coeffs) + params.p*len(params.mixer_2q_coeffs) + params.p*len(params.cost_1q_coeffs):]
        subdivided_ext_grad = [m1q_entries, m2q_entries, c1q_entries, c2q_entries]
        
        # Sum up gradients (due to the chain rule), and re-express in standard form.
        mat = np.zeros((4, params.p))
        for i in range(4):  # 4 types of terms
            for j in range(params.p):
                mat[i][j] = np.sum(subdivided_ext_grad[i][j*int(len(subdivided_ext_grad[i]) /
                                   params.p):(j+1)*int(len(subdivided_ext_grad[i])/params.p)])

        grad_std = list(np.sum(mat[:2], axis=0)) + \
            list(np.sum(mat[2:], axis=0))
        
        return np.array(grad_std)

    return grad_ps_func


def grad_sps(backend_obj, params_std, params_ext, gradient_options, logger):
    """
    Returns a callable function that approximates the gradient with the stochastic parameter shift method, which samples (n_beta_single, n_beta_pair, n_gamma_single, n_gamma_pair) gates at each layer instead of all gates. See "Algorithm 4" of https://arxiv.org/pdf/1910.01155.pdf. By convention, (n_beta_single, n_beta_pair, n_gamma_single, n_gamma_pair) = (-1, -1, -1, -1) will sample all gates (which is then equivalent to the full parameter shift rule).

    PARAMETERS
    ----------
    backend_obj : `QAOABaseBackend`
        backend object that computes expectation values when executed. 
    params_std : `QAOAVariationalStandardParams`
        variational parameters object, standard parametrisation.
    params_ext : `QAOAVariationalExtendedParams`
        variational parameters object, extended parametrisation.
    gradient_options :
        n_beta_single : 
            Number of single-qubit mixer gates to sample for the stochastic parameter shift.
        n_beta_pair : 
            Number of X two-qubit mixer gates to sample for the stochastic parameter shift.
        n_gamma_pair : 
            Number of two-qubit cost gates to sample for the stochastic parameter shift.
        n_gamma_single : 
            Number of single-qubit cost gates to sample for the stochastic parameter shift.
    logger : `Logger`
        logger object to log the number of function evaluations.

    RETURNS
    -------
    grad_sps_func:
        Callable derivative function.
    """
    
    n_beta_single = gradient_options['n_beta_single']
    n_beta_pair = gradient_options['n_beta_pair']
    n_gamma_single = gradient_options['n_gamma_single']
    n_gamma_pair = gradient_options['n_gamma_pair']
    
    beta_single_len = len(params_ext.betas_singles[0])
    beta_pair_len = len(params_ext.betas_pairs[0])
    gamma_single_len = len(params_ext.gammas_singles[0])
    gamma_pair_len = len(params_ext.gammas_pairs[0])
    
    coeffs_list = params_std.p*params_std.mixer_1q_coeffs + params_std.p*params_std.mixer_2q_coeffs + \
            params_std.p*params_std.cost_1q_coeffs + params_std.p*params_std.cost_2q_coeffs
    
    assert (-1 <= n_beta_single <= beta_single_len) and (-1 <= n_beta_pair <= beta_pair_len) and (-1 <= n_gamma_single <= gamma_single_len) and (-1 <= n_gamma_pair <= gamma_pair_len),\
        f"Invalid (n_beta_single, n_beta_pair, n_gamma_pair, n_gamma_single). Each n must be -1 or integers less than or equals to ({str(beta_single_len)}, {str(beta_pair_len)}, {str(gamma_single_len)}, {str(gamma_pair_len)})"

    if n_beta_single == -1: n_beta_single = beta_single_len
    if n_beta_pair == -1: n_beta_pair = beta_pair_len
    if n_gamma_single == -1: n_gamma_single = gamma_single_len
    if n_gamma_pair == -1: n_gamma_pair = gamma_pair_len
        
    fun = update_and_compute_expectation(backend_obj, params_ext, logger)
    
    def grad_sps_func(args):

        # Convert standard to extended parameters before applying parameter shift
        args_ext = params_std.convert_to_ext(args)
        
        grad_ext = np.zeros(len(args_ext))

        # Generate lists of random gates to sample. Note : Gates sampled in each layer is not necessarily the same, but the number of sampled gates in each layer is the same.
        sampled_indices = []
        for p in range(params_std.p):
            sampled_indices.append(random.sample(range(p*len(params_std.mixer_1q_coeffs) , (p+1)*len(params_std.mixer_1q_coeffs)), n_beta_single))

            sampled_indices.append(random.sample(range(params_std.p*len(params_std.mixer_1q_coeffs) + p*len(params_std.mixer_2q_coeffs) , params_std.p*len(params_std.mixer_1q_coeffs) + (p+1)*len(params_std.mixer_2q_coeffs)), n_beta_pair))

            sampled_indices.append(random.sample(range(params_std.p*(len(params_std.mixer_1q_coeffs) + len(params_std.mixer_2q_coeffs)) + p*len(params_std.cost_1q_coeffs) , params_std.p*(len(params_std.mixer_1q_coeffs) + len(params_std.mixer_2q_coeffs)) + (p+1)*len(params_std.cost_1q_coeffs)), n_gamma_single))

            sampled_indices.append(random.sample(range(params_std.p*(len(params_std.mixer_1q_coeffs) + len(params_std.mixer_2q_coeffs) + len(params_std.cost_1q_coeffs)) + p*len(params_std.cost_2q_coeffs) , params_std.p*(len(params_std.mixer_1q_coeffs) + len(params_std.mixer_2q_coeffs) + len(params_std.cost_1q_coeffs)) + (p+1)*len(params_std.cost_2q_coeffs)), n_gamma_pair))
    
        sampled_indices = [item for sublist in sampled_indices for item in sublist]

        # Apply parameter shifts
        for i in range(len(args_ext)):
            if (i) in sampled_indices:
                vect_eta = np.zeros(len(args_ext))
                vect_eta[i] = 1
                r = coeffs_list[i]
                grad_ext[i] = r*(fun(args_ext + (np.pi/(4*r))*vect_eta) -
                                 fun(args_ext - (np.pi/(4*r))*vect_eta))

        # Convert extended param. gradient form back into std param. form
        
        m1q_entries = grad_ext[:params_std.p*len(params_std.mixer_1q_coeffs)]
        m2q_entries = grad_ext[params_std.p*len(params_std.mixer_1q_coeffs) : params_std.p*len(params_std.mixer_2q_coeffs)]
        c1q_entries = grad_ext[params_std.p*len(params_std.mixer_1q_coeffs) + params_std.p*len(params_std.mixer_2q_coeffs): params_std.p*len(params_std.mixer_1q_coeffs) + params_std.p*len(params_std.mixer_2q_coeffs) + params_std.p*len(params_std.cost_1q_coeffs)]
        c2q_entries = grad_ext[params_std.p*len(params_std.mixer_1q_coeffs) + params_std.p*len(params_std.mixer_2q_coeffs) + params_std.p*len(params_std.cost_1q_coeffs):]
        subdivided_ext_grad = [m1q_entries, m2q_entries, c1q_entries, c2q_entries]
        
        # Sum up gradients (due to the chain rule), and re-express in standard form.
        mat = np.zeros((4, params_std.p))
        for i in range(4):  # 4 types of terms
            for j in range(params_std.p):
                mat[i][j] = np.sum(subdivided_ext_grad[i][j*int(len(subdivided_ext_grad[i]) /
                                   params_std.p):(j+1)*int(len(subdivided_ext_grad[i])/params_std.p)])

        grad_std = list(np.sum(mat[:2], axis=0)) + \
            list(np.sum(mat[2:], axis=0))
        
        return np.array(grad_std)

    return grad_sps_func


def hessian_fd(backend_obj, params, hessian_options, logger):
    """
    Returns a callable function that calculates the hessian with the finite difference method.

    PARAMETERS
    ----------
    backend_obj : `QAOABaseBackend`
        backend object that computes expectation values when executed.
    params : `QAOAVariationalBaseParams`
        variational parameters object.
    hessian_options :
        hessian_stepsize : 
            stepsize of finite difference.
    logger : `Logger`
        logger object to log the number of function evaluations.

    RETURNS
    -------
    hessian_fd_func:
        Callable derivative function.

    """

    eta = hessian_options['stepsize']
    fun = update_and_compute_expectation(backend_obj, params, logger)

    def hessian_fd_func(args):
        hess = np.zeros((len(args), len(args)))

        for i in range(len(args)):
            for j in range(len(args)):
                vect_eta1 = np.zeros(len(args))
                vect_eta2 = np.zeros(len(args))
                vect_eta1[i] = 1
                vect_eta2[j] = 1

                if i == j:
                    # Central diff. hessian diagonals (https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm)
                    hess[i][i] = (-fun(args+2*eta*vect_eta1) + 16*fun(args + eta*vect_eta1) - 30*fun(
                        args) + 16*fun(args-eta*vect_eta1)-fun(args-2*eta*vect_eta1))/(12*eta**2)
                #grad_diff[i] = (grad_fd_ext(params + (eta/2)*vect_eta)[i] - grad_fd_ext(params - (eta/2)*vect_eta)[i])/eta
                else:
                    hess[i][j] = (fun(args + eta*vect_eta1 + eta*vect_eta2)-fun(args + eta*vect_eta1 - eta*vect_eta2)-fun(
                        args - eta*vect_eta1 + eta*vect_eta2)+fun(args - eta*vect_eta1 - eta*vect_eta2))/(4*eta**2)

        return hess

    return hessian_fd_func


def grad_spsa(backend_obj, params, gradient_options, logger):
    """
    Returns a callable function that calculates the gradient approxmiation with the Simultaneous Perturbation Stochastic Approximation (SPSA) method.

    PARAMETERS
    ----------
    backend_obj : `QAOABaseBackend`
        backend object that computes expectation values when executed. 
    params : `QAOAVariationalBaseParams`
        variational parameters object.
    gradient_options : `dict`
        gradient_stepsize : 
            stepsize of stochastic shift.
    logger : `Logger`
        logger object to log the number of function evaluations.

    RETURNS
    -------
    grad_spsa_func: `Callable`
        Callable derivative function.

    """
    c = gradient_options['stepsize']
    fun = update_and_compute_expectation(backend_obj, params, logger)

    def grad_spsa_func(args):
        delta = (2*np.random.randint(0, 2, size=len(args))-1)
        return np.real((fun(args + c*delta) - fun(args - c*delta))*delta/(2*c)) 

    return grad_spsa_func
