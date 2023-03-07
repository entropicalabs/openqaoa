import numpy as np
from scipy.optimize import OptimizeResult

#### Experimental ############################################################


def SPSA(
    fun,
    x0,
    args=(),
    maxfev=None,
    a0=0.01,
    c0=0.01,
    A=1,
    alpha=0.602,
    gamma=0.101,
    maxiter=100,
    tol=10 ** (-6),
    jac=None,
    callback=None,
    **options
):
    """
    Minimize a function `fun` using the Simultaneous Perturbation Stochastic Approximation (SPSA) method.
    scipy.optimize.minimize compatible implementation of SPSA for `method` == 'spsa'.
    Refer to https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF
    for in depth explanation of parameters and optimal parameter choices.

    PARAMETERS
    ----------
    fun :
        Function to minimize.
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
        Tolerance before the optimizer terminates;
        if `tol` is larger than the difference between
        two steps, terminate optimization.
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

    def grad_SPSA(params, c):
        delta = 2 * np.random.randint(0, 2, size=len(params)) - 1
        return np.real(
            (fun(params + c * delta) - fun(params - c * delta)) * delta / (2 * c)
        )

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
        a = a / (A + niter + 1) ** alpha
        c = c / (niter + 1) ** gamma

        # compute gradient descent step
        testx = testx - a * grad_SPSA(testx, c)
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
