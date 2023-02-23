import numpy as np
from scipy.optimize import OptimizeResult


def rmsprop(
    fun,
    x0,
    args=(),
    maxfev=None,
    stepsize=0.01,
    maxiter=100,
    tol=10 ** (-6),
    jac=None,
    callback=None,
    decay=0.9,
    eps=1e-07,
    **options
):
    """
    Minimize a function `fun` using RMSProp. scipy.optimize.minimize
    compatible implementation of RMSProp for `method` == 'rmsprop'.

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
        Tolerance before the optimizer terminates; if `tol` is larger
        than the difference between two steps, terminate optimization.
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
    """

    bestx = x0
    besty = fun(x0)

    sqgrad = jac(x0) ** 2
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
        sqgrad = decay * sqgrad + (1 - decay) * jac(testx) ** 2
        testx = testx - stepsize * jac(testx) / (np.sqrt(sqgrad) + eps)
        testy = np.real(fun(testx, *args))

        if np.abs(besty - testy) < tol:
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

    return OptimizeResult(
        fun=besty, x=bestx, nit=niter, nfev=funcalls, success=(niter > 1)
    )
