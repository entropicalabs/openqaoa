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

"""
Collection of functions to return derivative computation functions. Called from QAOACosts' 'gradient_function' method.
New gradient/higher-order derivative computation methods can be added here. To add new computation methods:
    1. Write function in the format : new_function(cost_std, cost_ext, params, params_ext, gradient_options), or with less arguments.
    2. Give it a name, and add this under derivative_methods of DerivativeFunction, and as a possible 'out'.

"""
import numpy as np
import random


def update_and_compute_expectation(backend_obj, params, logger):
    # Helper function that returns a function that takes in a list of raw parameters
    # This function will handle (1) updating variational parameters with update_from_raw and (2) computing expectation.

    def fun(args):
        current_total_eval = logger.func_evals.best[0]
        current_total_eval += 1
        current_jac_eval = logger.jac_func_evals.best[0]
        current_jac_eval += 1
        logger.log_variables({'func_evals': current_total_eval, 
                              'jac_func_evals': current_jac_eval})
        params.update_from_raw(args)
        return backend_obj.expectation(params)

    return fun


def derivative(derivative_dict: dict):
    """
    Called from `cost_function.py`. Returns a callable derivative function, generated according to parameters in `derivative_dict`. 

    PARAMETERS
    ----------
    derivative_dict :
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

        params : QAOAVariationalBaseParams
            `QAOAVariationalBaseParams` object containing variational angles.

        params_ext : VariationalExtendedParams
            extended parametrisation object (for parameter shift and related methods)

    Returns
    -------
    out:
        Callable derivative function.
    """
    
    logger = derivative_dict['logger']
    
    derivative_type = derivative_dict['derivative_type']
    derivative_method = derivative_dict['derivative_method']

    # Default derivative_options used if none are specified.
    default_derivative_options = {"stepsize": 0.00001,
                                  "n_beta": -1,
                                  "n_gamma_pair": -1,
                                  "n_gamma_single": -1}

    derivative_options = {**default_derivative_options, **derivative_dict['derivative_options']
                          } if derivative_dict['derivative_options'] is not None else default_derivative_options

    backend_obj = derivative_dict['backend_obj']
    # cost_std = derivative_dict['cost_std']
    # cost_ext = derivative_dict['cost_ext']
    params = derivative_dict['params']
    params_ext = derivative_dict['params_ext']

    derivative_types = ['gradient', 'hessian']
    assert derivative_type in derivative_types,\
        "Unknown derivative type specified - please choose between " + \
        str(derivative_types)

    derivative_methods = ['finite_difference',
                          'param_shift', 'stoch_param_shift', 'grad_spsa']
    assert derivative_method in derivative_methods,\
        "Unknown derivative computation method specified - please choose between " + \
        str(derivative_methods)

    if derivative_type == 'gradient':

        if derivative_method == 'finite_difference':
            out = grad_fd(backend_obj, params, derivative_options, logger)
        elif derivative_method == 'param_shift':
            out = grad_ps(backend_obj, params, params_ext, logger)
        elif derivative_method == 'stoch_param_shift':
            out = grad_sps(backend_obj, params, params_ext, derivative_options, logger)
        elif derivative_method == 'grad_spsa':
            out = grad_spsa(backend_obj, params, derivative_options, logger)

    elif derivative_type == 'hessian':

        if derivative_method == 'finite_difference':
            out = hessian_fd(backend_obj, params, derivative_options, logger)

    return out


def grad_fd(backend_obj, params, gradient_options, logger):
    """
    Returns a callable function that calculates the gradient with the finite difference method.

    PARAMETERS
    ----------
    backend_obj : `QAOABaseBackend`
        backend object that computes expectation values when executed. 

    gradient_options : `dict`
        stepsize : 
            Stepsize of finite difference.

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

    params_std : `QAOAVariationalStandardParams`
        variational parameters object, standard parametrisation.

    params_ext : `QAOAVariationalExtendedParams`
        variational parameters object, extended parametrisation.

    RETURNS
    -------
    grad_ps_func:
        Callable derivative function.
    """

    fun = update_and_compute_expectation(backend_obj, params_ext, logger)

    def grad_ps_func(args):

        # Convert standard to extended parameters before applying parameter shift
        # TODO : clean this up
        terms_lst = [len(params.mixer_1q_coeffs), len(params.mixer_2q_coeffs), len(
            params.cost_1q_coeffs), len(params.cost_2q_coeffs)]
        terms_lst_p = np.repeat(terms_lst, [params.p]*len(terms_lst))
        args_ext = []
        for i in range(4):  # 4 types of terms
            for j in range(params.p):
                for k in range(terms_lst_p[i*params.p + j]):
                    if i < 2:
                        args_ext.append(params.raw()[j])
                    else:
                        args_ext.append(
                            params.raw()[j + int(len(params.raw())/2)])

        coeffs_list = params.mixer_1q_coeffs + params.mixer_2q_coeffs + \
            params.cost_1q_coeffs + params.cost_2q_coeffs
        grad_ext = np.zeros(len(args_ext))

        # Apply parameter shifts
        for i in range(len(args_ext)):
            vect_eta = np.zeros(len(args_ext))
            vect_eta[i] = 1
            r = coeffs_list[i]
            grad_ext[i] = r*(fun(args_ext + (np.pi/(4*r))*vect_eta) -
                             fun(args_ext - (np.pi/(4*r))*vect_eta))

        mat = np.zeros((4, params.p))

        # Convert extended param. gradient form back into std param. form
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
    Returns a callable function that approximates the gradient with the stochastic parameter shift method, which samples (n_beta, n_gamma_pair, n_gamma_single) gates at each layer instead of all gates. See "Algorithm 4" of https://arxiv.org/pdf/1910.01155.pdf. By convention, (n_beta, n_gamma_pair, n_gamma_single) = (-1, -1, -1) will sample all gates (which is then equivalent to the full parameter shift rule).

    PARAMETERS
    ----------
    backend_obj : `QAOABaseBackend`
        backend object that computes expectation values when executed. 

    params_std : `QAOAVariationalStandardParams`
        variational parameters object, standard parametrisation.

    params_ext : `QAOAVariationalExtendedParams`
        variational parameters object, extended parametrisation.

    gradient_options :
        n_beta : 
            Number of X gates to sample for the parameter shift.
        n_gamma_pair : 
            Number of ZZ gates to sample for the parameter shift.
        n_gamma_single : 
            Number of Z gates to sample for the parameter shift.

    RETURNS
    -------
    grad_sps_func:
        Callable derivative function.
    """

    n_beta = gradient_options['n_beta']
    n_gamma_pair = gradient_options['n_gamma_pair']
    n_gamma_single = gradient_options['n_gamma_single']

    beta_len = len(params_ext.betas[0])
    gamma_pair_len = len(params_ext.gammas_pairs[0])
    gamma_single_len = len(params_ext.gammas_singles[0])

    assert (-1 <= n_beta <= beta_len) and (-1 <= n_gamma_pair <= gamma_pair_len) and (-1 <= n_gamma_single <= gamma_single_len),\
        "Invalid (n_beta, n_gamma_pair, n_gamma_single). Each n must be -1 or integers less than or equals to (" + \
        str(beta_len) + ", " + str(gamma_pair_len) + \
        ", " + str(gamma_single_len) + ")."

    if n_beta == -1:
        n_beta = beta_len
    if n_gamma_pair == -1:
        n_gamma_pair = gamma_pair_len
    if n_gamma_single == -1:
        n_gamma_single = gamma_single_len

    timesteps = params_std.p
    p_ext = np.zeros(len(params_ext.raw()))
    grad_ext = np.zeros(len(params_ext.raw()))

    b = [len(x) for x in params_ext.betas]
    gs = [len(x) for x in params_ext.gammas_singles]
    gp = [len(x) for x in params_ext.gammas_pairs]

    pair_indices = [ind for ind, x in enumerate(
        params_std.hyperparameters.terms) if len(x) == 2]
    single_indices = [ind for ind, x in enumerate(
        params_std.hyperparameters.terms) if len(x) == 1]
    weights = params_std.hyperparameters.weights

    def grad_sps_func(params):
        cost_std(params)  # TODO : Temporary fix. See ent_10_13, part 3.
        grad = np.zeros(len(params))

        # Convert standard to extended parameters before applying parameter shift
        p_ext = np.repeat(list(params) + list(params_std.gammas), b + gs + gp)

        # Generate lists of random gates to sample. Note : Gates sampled in each layer is not necessarily the same, but the number of sampled gates in each layer is the same.
        sampled_beta, sampled_gamma_pair, sampled_gamma_single = [], [], []
        for i in range(timesteps):
            sampled_beta += random.sample(range(beta_len*i,
                                          beta_len*(i+1)), n_beta)
            sampled_gamma_pair += random.sample(
                range(gamma_pair_len*i, gamma_pair_len*(i+1)), n_gamma_pair)
            sampled_gamma_single += random.sample(
                range(gamma_single_len*i, gamma_single_len*(i+1)), n_gamma_single)

        # Apply parameter shift to ext parameters
        for i in range(len(params_ext.raw())):
            vect_eta = np.zeros(len(p_ext))
            vect_eta[i] = 1

            if i < len(params_ext.betas.flatten()):  # (beta/X terms)
                if (i) in sampled_beta:
                    r = 1
                    grad_ext[i] = (beta_len/n_beta)*r*(cost_ext(p_ext + (np.pi/(4*r))
                                                                * vect_eta) - cost_ext(p_ext - (np.pi/(4*r))*vect_eta))

            # (gamma/ZZ terms)
            elif i >= len(params_ext.betas.flatten()) + len(params_ext.gammas_singles.flatten()):
                if (i - len(params_ext.betas.flatten()) - len(params_ext.gammas_singles.flatten())) in sampled_gamma_pair:
                    r = weights[pair_indices[(i - len(params_ext.betas.flatten()) - len(
                        params_ext.gammas_singles.flatten())) % len(pair_indices)]]
                    grad_ext[i] = (gamma_pair_len/n_gamma_pair)*r*(cost_ext(
                        p_ext + (np.pi/(4*r))*vect_eta) - cost_ext(p_ext - (np.pi/(4*r))*vect_eta))

            else:  # (single Z terms)
                if (i - len(params_ext.betas.flatten())) in sampled_gamma_single:
                    r = weights[single_indices[(
                        i - len(params_ext.betas.flatten())) % len(single_indices)]]
                    grad_ext[i] = (gamma_single_len/n_gamma_single)*r*(cost_ext(
                        p_ext + (np.pi/(4*r))*vect_eta) - cost_ext(p_ext - (np.pi/(4*r))*vect_eta))

        # Convert gradient (ext back to std)
        s1 = np.split((np.tile([0] + list(np.cumsum(b + gs + gp)),
                      (2, 1)).flatten('F'))[1:-1], len(b + gs + gp))
        s2 = [grad_ext[x[0]:x[1]] for x in s1]
        a = [np.sum(x) for x in s2]
        grad = list(a[: len(params_ext.betas)]) + \
            list(
                np.sum(np.reshape(a[len(params_ext.betas):], (2, timesteps)), axis=0))

        return np.array(grad)

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


'''
def grad_sps(cost_std, cost_ext, params_std, params_ext, gradient_options):
    """
    Returns a callable function that calculates the gradient with the Stochastic Parameter Shift method. 
    See "Algorithm 4" of https://arxiv.org/pdf/1910.01155.pdf. (Deprecated)
    
    'gradient_options' parameters : 
    -------------------------------
        n_beta : Number of X gates to sample for the parameter shift.
        n_gamma : Number of ZZ gates to sample for the parameter shift.

    """
    
    # TODO : Update and integrate with grad_ps
    
    n_beta = gradient_options['n_beta']
    n_gamma = gradient_options['n_gamma']

    n_nodes = len(params_std.reg)
    timesteps = params_std.p
    beta_len = len(params_ext.betas[0])
    gamma_len = len(params_ext.gammas_pairs[0])

    assert n_beta <= beta_len and n_gamma <= gamma_len,\
    "Invalid (n_beta, n_gamma). Maximum is (" + str(beta_len) + ", " + str(gamma_len) + ")."

    if n_beta == 0:
        n_beta = beta_len
    if n_gamma == 0:
        n_gamma = gamma_len

    def grad_sps_func(params):

            p_ext = np.zeros(timesteps*(beta_len + gamma_len))
            grad = np.zeros(2*timesteps)

            # convert standard to extended parameters before applying parameter shift
            # note : built-in standard to extended converter not working?
            c = 0
            for j in range(len(p_ext)):
                if j < timesteps*beta_len: # (beta/X terms)

                    if ((j+1) % beta_len) == False:
                        p_ext[j] = params[c]
                        c +=1
                    else:
                        p_ext[j] = params[c]   
                else: # (gamma/Z terms)
                    if ((j+1 - timesteps*beta_len) % gamma_len)== False :
                        p_ext[j] = params[c]
                        c += 1
                    else:
                        p_ext[j] = params[c]


            for i in range(2*timesteps):

                if i < len(params_std.betas):

                    for k1 in random.sample(range(0, beta_len), n_beta):
                        vect_eta = np.zeros(len(p_ext))
                        c = i*beta_len + k1
                        vect_eta[c] = 1
                        r = 1
                        grad[i] += np.real((beta_len/n_beta)*r*(cost_ext(p_ext + (np.pi/(4*r))*vect_eta) - cost_ext(p_ext - (np.pi/(4*r))*vect_eta)))

                else:

                    for k2 in random.sample(range(0, gamma_len), n_gamma):
                        vect_eta = np.zeros(len(p_ext))
                        c = gamma_len*(i-len(params_std.betas)) + beta_len*len(params_ext.betas) + k2 
                        vect_eta[c] = 1
                        r = 1
                        grad[i] += np.real((gamma_len/n_gamma)*r*(cost_ext(p_ext + (np.pi/(4*r))*vect_eta) - cost_ext(p_ext - (np.pi/(4*r))*vect_eta)))

            return grad
    return grad_sps_func
'''
