

from importlib.metadata import requires
from operator import ne
import pennylane as pl
import inspect
from scipy.optimize import OptimizeResult
import numpy as np

import matplotlib.pyplot as plt

def pennylane_optimizer(fun, x0, args=(), maxfev=None, method='vgd', qfim=None,
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
    method : string, optional
        Optimizer method to compute the steps.
    qfim : callable, optional (required for natural_grad_descent)
        Callable Fubini-Study metric tensor
    maxiter : int, optional
        Maximum number of iterations.
    tol : float
        Tolerance before the optimizer terminates; if `tol` is larger than the difference between two steps, terminate optimization.
    jac : callable, optinal (required for all methods but rotosolve and spsa)
        Callable gradient function.
    callback : callable, optional
        Called after each iteration, as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    options : dict, optional
        Dictionary where keys are the arguments for the optimizers object, and
        the values are the values to pass to these arguments.
        To know all the possible argumets read
        https://docs.pennylane.ai/en/stable/introduction/interfaces.html#optimizers.


    (read https://docs.pennylane.ai/en/stable/code/api/pennylane.RotosolveOptimizer.html#pennylane.RotosolveOptimizer.step)
    nums_frequency : dict[dict], required for rotosolve
        The number of frequencies in the fun per parameter.
    spectra : dict[dict], required for rotosolve
        Frequency spectra in the objective_fn per parameter.
    shifts : dict[dict], required for rotosolve
        Shift angles for the reconstruction per parameter.


    RETURNS
    -------
    OptimizeResult : OptimizeResult
        Scipy OptimizeResult object.
    '''

    def cost(params, **k): # define a function to convert the params list from pennylane to numpy
        return fun(np.array(params), *k)

    available_methods_dict = {  # optimizers implemented
                                'adagrad': pl.AdagradOptimizer, 
                                'adam': pl.AdamOptimizer, 
                                'vgd': pl.GradientDescentOptimizer, 
                                'momentum':  pl.MomentumOptimizer,
                                'nesterov_momentum': pl.NesterovMomentumOptimizer,
                                'natural_grad_descent': pl.QNGOptimizer,
                                'rmsprop': pl.RMSPropOptimizer,
                                'rotosolve': pl.RotosolveOptimizer, 
                                'spsa': pl.SPSAOptimizer,
                             }

    optimizer = available_methods_dict[method] # define the optimizer

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
        niter += 1

        # compute step
        if qfim:    #natural_grad_descent
            testx, testy = optimizer.step_and_cost(cost, bestx, *args, grad_fn=jac, metric_tensor_fn=qfim) 
        elif jac:   #adagrad, adam, vgd, momentum, nesterov_momentum, rmsprop
            testx, testy = optimizer.step_and_cost(cost, bestx, *args, grad_fn=jac)
        elif method=='rotosolve': 
            testx, testy = optimizer.step_and_cost(
                                                    cost, bestx, *args,
                                                    nums_frequency={'params': {(i,):1 for i in range(bestx.size)}} if not nums_frequency else nums_frequency,
                                                    spectra=spectra,
                                                    shifts=shifts,
                                                    full_output=False,
                                                  )
        else:       #spsa  
            testx, testy = optimizer.step_and_cost(cost, bestx, *args)

        # check if stable
        if np.abs(besty-testy) < tol and niter > 2:
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
        
    return OptimizeResult(fun=besty, x=np.array(bestx), nit=niter,
                          nfev=funcalls, success=(niter > 1))


