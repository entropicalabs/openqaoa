import numpy as np
from scipy.optimize import OptimizeResult


def natural_grad_descent(
    fun,
    x0,
    args=(),
    maxfev=None,
    stepsize=0.01,
    maxiter=100,
    tol=10 ** (-6),
    lambd=0.001,
    jac=None,
    callback=None,
    **options
):
    """
    Minimize a function `fun` using the natural gradient descent method.
    scipy.optimize.minimize compatible implementation of natural gradient
    descent for `method` == 'natural_grad_descent'.

    PARAMETERS
    ----------
    fun : QAOACost
        Function to minimize. Must be a `QAOACost` object, so that the
        method `qfim` can be called.
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
        Tolerance before the optimizer terminates; if `tol` is larger than
        the difference between two steps, terminate optimization.
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
    """

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
        qfim = options["qfim"]
        scaled_gradient = np.linalg.solve(
            qfim(testx) + lambd * (np.identity(n_params)), jac(testx)
        )

        # compute natural gradient descent step
        testx = testx - stepsize * scaled_gradient
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
