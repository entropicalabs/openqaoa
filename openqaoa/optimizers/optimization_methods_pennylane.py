

from importlib.metadata import requires
from operator import ne
import pennylane as pl
import inspect
from scipy.optimize import OptimizeResult
import numpy as np

def pennylane_optimizer(fun, x0, args=(), maxfev=None, method = 'vgd', qfim=None,
                 nums_frequency=None, spectra=None, shifts=None, 
                 maxiter=100, tol=10**(-6), jac=None, callback=None, **options):

    def cost(params, **k):
        return fun(np.array(params), **k)

    available_methods_dict = {  
                                'adagrad': pl.AdagradOptimizer, 
                                'adam': pl.AdamOptimizer, 
                                'vgd': pl.GradientDescentOptimizer, 
                                'momentum':  pl.MomentumOptimizer,
                                'nesterov_momentum': pl.NesterovMomentumOptimizer,
                                'natural_grad_descent': pl.QNGOptimizer,
                                'rmsprop': pl.RMSPropOptimizer,
                                'rotosolve': pl.RotosolveOptimizer, 
                                'spsa': pl.QNSPSAOptimizer,
                             }

    optimizer = available_methods_dict[method]
    arguments = inspect.signature(optimizer).parameters.keys()
    options_keys = list(options.keys())

    print(arguments)

    for key in options_keys:
        if key not in arguments: options.pop(key) 

    optimizer = optimizer(**options)
    print(options, optimizer)

    
    bestx = pl.numpy.array(x0, requires_grad=True)
    besty = cost(x0)
    funcalls = 1  # tracks no. of function evals.
    niter = 0
    improved = True
    stop = False  

    testx = np.copy(bestx)
    testy = np.real(besty)
    while improved and not stop and niter < maxiter:
        improved = False
        niter += 1
        print(niter)

        # compute step
        if qfim:    #natural_grad_descent
            testx, testy = optimizer.step_and_cost(cost, bestx, grad_fn=jac, metric_tensor_fn=qfim) 
        elif jac:   #adagrad, adam, vgd, momentum, nesterov_momentum, rmsprop
            testx, testy = optimizer.step_and_cost(cost, bestx, grad_fn=jac)
        elif method=='rotosolve': 
            testx, testy = optimizer.step_and_cost(
                                                    cost, bestx, 
                                                    nums_frequency={'params': {(i,):1 for i in range(bestx.size)}} if not nums_frequency else nums_frequency,
                                                    spectra=spectra,
                                                    shifts=shifts,
                                                    full_output=False,
                                                  )
        else:      #spsa 
            testx, testy = optimizer.step_and_cost(cost, bestx)

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


