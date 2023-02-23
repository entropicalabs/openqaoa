import numpy as np
from scipy.optimize import OptimizeResult


def grad_descent(
    fun,
    x0,
    args=(),
    maxfev=None,
    stepsize=0.01,
    maxiter=100,
    tol=10 ** (-6),
    jac=None,
    callback=None,
    **options
):
    """
    Minimize a function `fun` using gradient descent. scipy.optimize.minimize
    compatible implementation of gradient descent for `method` == 'vgd'.

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
        Tolerance before the optimizer terminates; if `tol` is larger than the
        difference between two steps, terminate optimization.
    jac : callable
        Callable gradient function.
    callback : callable, optional
        Called after each iteration, as ``callback(xk)``, where ``xk`` is the
        current parameter vector.

    RETURNS
    -------
    OptimizeResult : OptimizeResult
        Scipy OptimizeResult object.
    """

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
        testx = testx - stepsize * jac(testx)
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
