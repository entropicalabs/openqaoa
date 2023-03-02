import numpy as np
from scipy.optimize import OptimizeResult


def newton_descent(
    fun,
    x0,
    args=(),
    maxfev=None,
    stepsize=0.01,
    maxiter=100,
    tol=10 ** (-6),
    jac=None,
    hess=None,
    callback=None,
    **options
):
    """
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
        Tolerance before the optimizer terminates; if `tol` is larger
        than the difference between two steps, terminate optimization.
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
    """

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
        testx = testx - stepsize * scaled_gradient
        testy = np.real(fun(testx, *args))

        if np.abs(besty - testy) < tol:
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

    return OptimizeResult(
        fun=besty, x=bestx, nit=niter, nfev=funcalls, success=(niter > 1)
    )
