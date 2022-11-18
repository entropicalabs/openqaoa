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
A set of functions to implement pennylane optimization algorithms.
Read https://docs.pennylane.ai/en/stable/introduction/interfaces.html#optimizers
Optimisers requiring a pennylane backend haven't been implemented yet.
Similarly as with the custom optimization methods Scipy `minimize` is used. Extends available scipy methods.
"""

from openqaoa.optimizers import pennylane as pl
import inspect
from scipy.optimize import OptimizeResult
import numpy as np

AVAILABLE_OPTIMIZERS = {  # optimizers implemented
                            'adagrad': pl.AdagradOptimizer, 
                            'adam': pl.AdamOptimizer, 
                            'vgd': pl.GradientDescentOptimizer, 
                            'momentum':  pl.MomentumOptimizer,
                            'nesterov_momentum': pl.NesterovMomentumOptimizer,
                            'rmsprop': pl.RMSPropOptimizer,
                            'rotosolve': pl.RotosolveOptimizer, 
                            'spsa': pl.SPSAOptimizer,
                        }



def pennylane_optimizer(fun, x0, args=(), maxfev=None, pennylane_method='vgd', 
                        maxiter=100, tol=10**(-6), jac=None, callback=None,                         
                        nums_frequency=None, spectra=None, shifts=None, **options):

    '''    
    Minimize a function `fun` using some pennylane method.
    To check available methods look at the available_methods_dict variable.
    Read https://docs.pennylane.ai/en/stable/introduction/interfaces.html#optimizers

    PARAMETERS
    ----------
    fun : callable
        Function to minimize
    x0 : ndarray
        Initial guess.
    args : sequence, optional
        Arguments to pass to `func`.
    maxfev : int, optional
        Maximum number of function evaluations.
    pennylane_method : string, optional
        Optimizer method to compute the steps.
    maxiter : int, optional
        Maximum number of iterations.
    tol : float
        Tolerance before the optimizer terminates; if `tol` is larger than the difference between two steps, terminate optimization.
    jac : callable, optinal
        Callable gradient function. Required for all methods but rotosolve and spsa.
    callback : callable, optional
        Called after each iteration, as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    options : dict, optional
        Dictionary where keys are the arguments for the optimizers object, and
        the values are the values to pass to these arguments.
        To know all the possible options see https://docs.pennylane.ai/en/stable/introduction/interfaces.html#optimizers.
    nums_frequency : dict[dict], optional
        It is required for rotosolve method
        The number of frequencies in the fun per parameter.
    spectra : dict[dict], optional
        It is required for rotosolve method
        Frequency spectra in the objective_fn per parameter.
    shifts : dict[dict], optional
        It is required for rotosolve method
        Shift angles for the reconstruction per parameter.
        Read https://docs.pennylane.ai/en/stable/code/api/pennylane.RotosolveOptimizer.html#pennylane.RotosolveOptimizer.step for more information.


    RETURNS
    -------
    OptimizeResult : OptimizeResult
        Scipy OptimizeResult object.
    '''

    def cost(params, **k): # define a function to convert the params list from pennylane to numpy
        return fun(np.array(params), *k)


    optimizer = AVAILABLE_OPTIMIZERS[pennylane_method] # define the optimizer

    #get optimizer arguments
    arguments = inspect.signature(optimizer).parameters.keys()
    options_keys = list(options.keys())

    #check which values of the options dict can be passed to the optimizer (pop the others)
    for key in options_keys:
        if key not in arguments: options.pop(key) 
        if 'maxiter' in arguments: options['maxiter'] = maxiter

    optimizer = optimizer(**options) #pass the arguments
    
    bestx = pl.numpy.array(x0, requires_grad=True)
    besty = cost(x0, *args)
    funcalls = 1  # tracks no. of function evals.
    niter = 0
    improved = True
    stop = False  

    testx = np.copy(bestx)
    testy = np.real(besty)
    while improved and not stop and niter < maxiter:
        improved = False

        # compute step (depends on the optimizer)
        if pennylane_method in ['adagrad', 'adam', 'vgd', 'momentum', 'nesterov_momentum', 'rmsprop']:
            testx, testy = optimizer.step_and_cost(cost, bestx, *args, grad_fn=jac)
        elif pennylane_method in ['rotosolve']: 
            testx, testy = optimizer.step_and_cost(
                                                    cost, bestx, *args,
                                                    nums_frequency={'params': {(i,):1 for i in range(bestx.size)}} if not nums_frequency else nums_frequency,
                                                    spectra=spectra,
                                                    shifts=shifts,
                                                    full_output=False,
                                                  )
        elif pennylane_method in ['spsa']:       
            testx, testy = optimizer.step_and_cost(cost, bestx, *args)

        # check if stable
        if np.abs(besty-testy) < tol and niter > 1:
            improved = False

        else:
            besty = testy
            bestx = testx
            improved = True

        if callback is not None:
            callback(bestx)
        if maxfev is not None and funcalls >= maxfev:
            stop = True
            break

        niter += 1
        
    return OptimizeResult(fun=besty, x=np.array(bestx), nit=niter,
                          nfev=funcalls, success=(niter > 1))


