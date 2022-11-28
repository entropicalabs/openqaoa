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
Collection of custom optimization algorithms to be used by Scipy `minimize`. Extends available scipy methods.

"""

import numpy as np
from scipy.optimize import OptimizeResult

import time


def grad_descent(fun, x0, args=(), maxfev=None, stepsize=0.01,
                 maxiter=100, tol=10**(-6), jac=None, callback=None, **options):
    '''    
    Minimize a function `fun` using gradient descent. scipy.optimize.minimize compatible implementation of gradient descent for `method` == 'vgd'.

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
    stepsize : float
        Step size of each gradient descent step.
    maxiter : int, optional
        Maximum number of iterations.
    tol : float
        Tolerance before the optimizer terminates; if `tol` is larger than the difference between two steps, terminate optimization.
    jac : callable
        Callable gradient function.
    callback : callable, optional
        Called after each iteration, as ``callback(xk)``, where ``xk`` is the
        current parameter vector.

    RETURNS
    -------
    OptimizeResult : OptimizeResult
        Scipy OptimizeResult object.
    '''

    bestx = x0
    besty = fun(x0)
    funcalls = 1  # tracks no. of function evals.
    niter = 0
    improved = True
    stop = False

    testx = np.copy(bestx)
    testy = np.real(fun(testx, *args))
    while improved and not stop and niter < maxiter:
        improved = False
        niter += 1
        
        # compute gradient descent step
        testx = testx - stepsize*jac(testx)
        testy = np.real(fun(testx, *args))

        if np.abs(besty-testy) < tol:
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

    return OptimizeResult(fun=besty, x=bestx, nit=niter,
                          nfev=funcalls, success=(niter > 1))


def rmsprop(fun, x0, args=(), maxfev=None, stepsize=0.01,
            maxiter=100, tol=10**(-6), jac=None, callback=None, decay=0.9, eps=1e-07, **options):
    '''    
    Minimize a function `fun` using RMSProp. scipy.optimize.minimize compatible implementation of RMSProp for `method` == 'rmsprop'.

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
    stepsize : float
        Step size of each gradient descent step.
    maxiter : int, optional
        Maximum number of iterations.
    tol : float
        Tolerance before the optimizer terminates; if `tol` is larger than the difference between two steps, terminate optimization.
    jac : callable
        Callable gradient function.
    callback : callable, optional
        Called after each iteration, as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    decay : float
        Stepsize decay parameter of RMSProp.
    eps : float
        Small number to prevent division by zero.

    RETURNS
    -------
    OptimizeResult : OptimizeResult
        Scipy OptimizeResult object.
    '''

    bestx = x0
    besty = fun(x0)

    sqgrad = jac(x0)**2
    funcalls = 1  # tracks no. of function evals
    niter = 0
    improved = True
    stop = False

    testx = np.copy(bestx)
    testy = np.real(fun(testx, *args))
    while improved and not stop and niter < maxiter:
        improved = False
        niter += 1

        # compute gradient descent step, scaled with adaptive RMS learning rate
        sqgrad = decay*sqgrad + (1-decay)*jac(testx)**2
        testx = testx - stepsize*jac(testx)/(np.sqrt(sqgrad) + eps)
        testy = np.real(fun(testx, *args))

        if np.abs(besty-testy) < tol:
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

    return OptimizeResult(fun=besty, x=bestx, nit=niter,
                          nfev=funcalls, success=(niter > 1))


def newton_descent(fun, x0, args=(), maxfev=None, stepsize=0.01,
                   maxiter=100, tol=10**(-6), jac=None, hess=None, callback=None, **options):
    '''    
    Minimize a function `fun` using Newton's method, a second order method. 
    scipy.optimize.minimize compatible implementation of Newton's method for `method` == 'newton'.

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
    stepsize : float
        Step size of each gradient descent step.
    maxiter : int, optional
        Maximum number of iterations.
    tol : float
        Tolerance before the optimizer terminates; if `tol` is larger than the difference between two steps, terminate optimization.
    jac : callable
        Callable gradient function.
    hess : callable
        Callable hessian function.
    callback : callable, optional
        Called after each iteration, as ``callback(xk)``, where ``xk`` is the
        current parameter vector.

    RETURNS
    -------
    OptimizeResult : OptimizeResult
        Scipy OptimizeResult object.
    '''
    
    assert jac is not None, "This method needs jacobian specified"
    assert hess is not None, "This method needs hessian specified"

    bestx = x0
    besty = fun(x0)
    funcalls = 1
    niter = 0
    improved = True
    stop = False

    testx = np.copy(bestx)
    testy = np.real(fun(testx, *args))
    while improved and not stop and niter < maxiter:
        improved = False
        niter += 1

        # Scale gradient with inverse of Hessian
        scaled_gradient = np.linalg.solve(hess(testx), jac(testx))

        # compute Newton descent step
        testx = testx - stepsize*scaled_gradient
        testy = np.real(fun(testx, *args))

        if np.abs(besty-testy) < tol:
            improved = False
        else:
            # if testy < besty:
            besty = testy
            bestx = testx
            improved = True

        if callback is not None:
            callback(bestx)
        if maxfev is not None and funcalls >= maxfev:
            stop = True
            break

    return OptimizeResult(fun=besty, x=bestx, nit=niter,
                          nfev=funcalls, success=(niter > 1))


def natural_grad_descent(fun, x0, args=(), maxfev=None, stepsize=0.01,
                         maxiter=100, tol=10**(-6), lambd=0.001, jac=None, callback=None, **options):
    '''    
    Minimize a function `fun` using the natural gradient descent method.
    scipy.optimize.minimize compatible implementation of natural gradient descent for `method` == 'natural_grad_descent'.

    PARAMETERS
    ----------
    fun : QAOACost
        Function to minimize. Must be a `QAOACost` object, so that the method `qfim` can be called.
    x0 : ndarray
        Initial guess.
    args : sequence, optional
        Arguments to pass to `func`.
    maxfev : int, optional
        Maximum number of function evaluations.
    stepsize : float
        Step size of each gradient descent step.
    maxiter : int, optional
        Maximum number of iterations.
    tol : float
        Tolerance before the optimizer terminates; if `tol` is larger than the difference between two steps, terminate optimization.
    lambd : float
        Magnitude of the Identity regularization term to avoid singularity of the QFIM.
    jac : callable
        Callable gradient function.
    callback : callable, optional
        Called after each iteration, as ``callback(xk)``, where ``xk`` is the
        current parameter vector.

    RETURNS
    -------
    OptimizeResult : OptimizeResult
        Scipy OptimizeResult object.
    '''

    bestx = x0
    besty = fun(x0)
    n_params = len(x0)
    funcalls = 1
    niter = 0
    improved = True
    stop = False

    testx = np.copy(bestx)
    testy = np.real(fun(testx, *args))
    while improved and not stop and niter < maxiter:
        improved = False
        niter += 1

        # scale gradient by left-multiplication with qfi matrix.
        qfim = options['qfim']
        scaled_gradient = np.linalg.solve(
            qfim(testx) + lambd*(np.identity(n_params)), jac(testx))

        # compute natural gradient descent step
        testx = testx - stepsize*scaled_gradient
        testy = np.real(fun(testx, *args))

        if np.abs(besty-testy) < tol:
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

    return OptimizeResult(fun=besty, x=bestx, nit=niter,
                          nfev=funcalls, success=(niter > 1))


#### Experimental ############################################################

def SPSA(fun, x0, args=(), maxfev=None, a0=0.01, c0=0.01, A=1, alpha=0.602, gamma=0.101,
         maxiter=100, tol=10**(-6), jac=None, callback=None, **options):
    '''    
    Minimize a function `fun` using the Simultaneous Perturbation Stochastic Approximation (SPSA) method.
    scipy.optimize.minimize compatible implementation of SPSA for `method` == 'spsa'. Refer to https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF for in depth explanation of parameters and optimal parameter choices.

    PARAMETERS
    ----------
    fun : QAOACost
        Function to minimize. Must be a `QAOACost` object, so that the method `qfim` can be called.
    x0 : ndarray
        Initial guess.
    args : sequence, optional
        Arguments to pass to `func`.
    maxfev : int, optional
        Maximum number of function evaluations.
    a0 : float
        Initial stepsize.
    c0 : float
        Initial finite difference.
    A : float
        Initial decay constant. 
    alpha : float
        Decay rate parameter for `a`.
    gamma : float
        Decay rate parameter for `c`.
    maxiter : int, optional
        Maximum number of iterations.
    tol : float
        Tolerance before the optimizer terminates; if `tol` is larger than the difference between two steps, terminate optimization.
    jac : callable
        Callable gradient function.
    callback : callable, optional
        Called after each iteration, as ``callback(xk)``, where ``xk`` is the
        current parameter vector.

    RETURNS
    -------
    OptimizeResult : OptimizeResult
        Scipy OptimizeResult object.
    '''

    def grad_SPSA(params, c):
        delta = (2*np.random.randint(0, 2, size=len(params))-1)
        return np.real((fun(params + c*delta) - fun(params - c*delta))*delta/(2*c))

    bestx = x0
    besty = fun(x0)
    funcalls = 1  # tracks no. of function evals.
    niter = 0
    improved = True
    stop = False

    testx = np.copy(bestx)
    testy = np.real(fun(testx, *args))

    a = a0
    c = c0
    while improved and not stop and niter < maxiter:
        improved = False
        niter += 1

        # gain sequences
        a = a/(A+niter+1)**alpha
        c = c/(niter+1)**gamma

        # compute gradient descent step
        testx = testx - a*grad_SPSA(testx, c)
        testy = np.real(fun(testx, *args))

        if np.abs(besty-testy) < tol:
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

    return OptimizeResult(fun=besty, x=bestx, nit=niter,
                          nfev=funcalls, success=(niter > 1))


'''
def stochastic_grad_descent(fun, x0, jac, args=(), stepsize=0.001, mass=0.9, startiter=0, maxiter=1000, callback=None, **options ):
    """
    Scipy "OptimizeResult" object for method == 'sgd'
    scipy.optimize.minimize compatible implementation of stochastic gradient descent with momentum.
    Adapted from ``autograd/misc/optimizers.py``.

    """
    x = x0
    velocity = np.zeros_like(x)
    print("SGD Test")

    for i in range(startiter, startiter + maxiter):
        g = jac(x)

        velocity = mass * velocity - (1.0 - mass) * g
        x = x + stepsize * velocity

        if callback is not None:
            callback(x)

    return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i+1, nfev=i+1, success=True)
'''



def CANS(fun, x0, args=(), maxfev=None, stepsize=0.00001, n_shots_min=10, n_shots_max=None, n_shots_budget = None, 
            mu=0.99, b=1e-06, coeffs=None, maxiter=100, tol=10**(-6), jac_w_variance=None, callback=None, **options): 

    # check that the stepsize is small enough
    lipschitz = np.sum(np.abs(coeffs)) 
    if not stepsize < 2/lipschitz:
        raise ValueError("Stepsizec is bigger than 2/Lipschitz: it should be smaller than {0:.3g}".format(2/lipschitz))

    # initialize variables for the algorithm loop
    chi = np.zeros(len(x0))
    xi = 0
    n_shots = n_shots_min

    bestx = x0
    besty = np.real(fun(bestx, *args))
    funcalls = 1  # tracks no. of function evals.
    niter = 0
    n_shots_used_total = 0 # tracks no. of shots taken
    improved = True
    stop = False

    testx = np.copy(bestx)
    testy = besty

    while improved and not stop and niter < maxiter:

        # compute gradient and variance
        gradient, variance, n_shots_used = jac_w_variance(testx, n_shots=n_shots)

        # add the number of shots to the total
        n_shots_used_total += n_shots_used

        # compute gradient descent step
        testx = testx - stepsize*gradient
        testy = np.real(fun(testx, *args))

        # compute n_shots for next step
        chi = mu*chi + (1-mu)*gradient
        xi = mu*xi + (1-mu)*np.sum(variance)        
        n_shots = int(np.ceil(2*lipschitz*stepsize*xi/((2-lipschitz*stepsize)*(np.linalg.norm(chi)**2+b*mu**niter))))

        # clip the number of shots
        n_shots = max(n_shots, n_shots_min)
        n_shots = min(n_shots, n_shots_max) if n_shots_max else n_shots
     

        if np.abs(besty-testy) < tol:
            improved = False

        else:
            besty = testy
            bestx = np.copy(testx)
            improved = True

        if callback is not None:
            callback(bestx)
        if maxfev is not None and funcalls >= maxfev:
            stop = True
            break

        # if there is a maximum number of shots and we have reached it, stop
        if n_shots_budget != None:            
            if not n_shots_used_total < n_shots_budget:
                stop = True
                break

        niter += 1

    return OptimizeResult(fun=besty, x=bestx, nit=niter,
                          nfev=funcalls, success=(niter > 1))


def iCANS(fun, x0, args=(), maxfev=None, stepsize=0.00001, n_shots_min=10, n_shots_max=10000, n_shots_budget=None, 
            mu=0.99, b=1e-06, coeffs=None, maxiter=100, tol=10**(-6), jac_w_variance=None, callback=None, **options): 

    # check that the stepsize is small enough
    lipschitz = np.sum(np.abs(coeffs)) 
    if not stepsize < 2/lipschitz:
        raise ValueError("Stepsizec is bigger than 2/Lipschitz: it should be smaller than {0:.3g}".format(2/lipschitz))

    # define the initial values
    chi_ = np.zeros(len(x0))
    xi_ = np.zeros(len(x0))
    n_shots = [n_shots_min for _ in range(len(x0))]

    bestx = x0
    besty = np.real(fun(bestx, *args))
    funcalls = 1  # tracks no. of function evals.
    niter = 0
    n_shots_used_total = 0 # track no. of shots
    improved = True
    stop = False

    testx = np.copy(bestx)
    testy = besty

    while improved and not stop and niter < maxiter:

        # compute gradient and variance
        gradient, variance, n_shots_used = jac_w_variance(testx, n_shots=list(n_shots))
        
        # add the number of shots to the total 
        n_shots_used_total += n_shots_used # TODO : add n_shots_used for the 'testy' evaluation 

        # compute gradient descent step
        testx = testx - stepsize*gradient
        testy = np.real(fun(testx, *args))

        # update xi_ and chi_
        xi_  = (mu*xi_  + (1-mu)*variance) 
        chi_ = (mu*chi_ + (1-mu)*gradient) 

        # compute n_shots for next step
        xi   = xi_  / (1-mu**(niter+1))
        chi  = chi_ / (1-mu**(niter+1))
        n_shots = np.int32(np.ceil(2*lipschitz*stepsize*xi/((2-lipschitz*stepsize)*(chi**2+b*mu**niter))))

        n_shots = np.fmax(n_shots, 1) #to compute gain n_shots should be at least 1
        gain = ((stepsize-lipschitz*gradient**2/2)*chi**2-lipschitz*stepsize**2*xi/(2*n_shots))/n_shots

        # clip the number of shots
        n_shots = np.fmax(n_shots, n_shots_min)
        n_shots = np.fmin(n_shots, n_shots[np.argmax(gain)]) # max of n_shots is the one with the max gain
        n_shots = np.fmin(n_shots, n_shots_max) if n_shots_max else n_shots

        if np.abs(besty-testy) < tol:
            improved = False

        else:
            besty = testy
            bestx = np.copy(testx)
            improved = True

        if callback is not None:
            callback(bestx)
        if maxfev is not None and funcalls >= maxfev:
            stop = True
            break

        # if there is a maximum number of shots and we have reached it, stop
        if n_shots_budget != None:
            if not n_shots_used_total < n_shots_budget:
                stop = True
                break

        niter += 1

    return OptimizeResult(fun=besty, x=bestx, nit=niter,
                          nfev=funcalls, success=(niter > 1))