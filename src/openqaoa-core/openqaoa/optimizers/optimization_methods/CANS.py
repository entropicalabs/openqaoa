import numpy as np
from scipy.optimize import OptimizeResult


def CANS(
    fun,
    x0,
    args=(),
    maxfev=None,
    stepsize=0.00001,
    n_shots_cost=None,
    n_shots_min=10,
    n_shots_max=None,
    n_shots_budget=None,
    mu=0.99,
    b=1e-06,
    coeffs=None,
    maxiter=100,
    tol=10 ** (-6),
    jac_w_variance=None,
    callback=None,
    **options
):
    # check that the stepsize is small enough
    lipschitz = np.sum(np.abs(coeffs))
    if not stepsize < 2 / lipschitz:
        raise ValueError(
            "Stepsizec is bigger than 2/Lipschitz: it should be smaller than {0:.3g}".format(
                2 / lipschitz
            )
        )

    # initialize variables for the algorithm loop
    chi = np.zeros(len(x0))
    xi = 0
    n_shots = n_shots_min

    bestx = x0
    besty = np.real(fun(bestx, *args))
    funcalls = 1  # tracks no. of function evals.
    niter = 0

    n_shots_used_total = n_shots_cost
    improved = True
    stop = False

    testx = np.copy(bestx)
    testy = besty

    while improved and not stop and niter < maxiter:
        # compute gradient and variance
        gradient, variance, n_shots_used = jac_w_variance(testx, n_shots=n_shots)

        # compute gradient descent step
        testx = testx - stepsize * gradient
        testy = np.real(fun(testx, *args))

        # add the number of shots used to the total (used in the
        # gradient computation and the cost function evaluation)
        n_shots_used_total += n_shots_used + n_shots_cost

        # compute n_shots for next step
        chi = mu * chi + (1 - mu) * gradient
        xi = mu * xi + (1 - mu) * np.sum(variance)
        n_shots = int(
            np.ceil(
                2
                * lipschitz
                * stepsize
                * xi
                / (
                    (2 - lipschitz * stepsize)
                    * (np.linalg.norm(chi) ** 2 + b * mu**niter)
                )
            )
        )

        # clip the number of shots
        n_shots = max(n_shots, n_shots_min)
        n_shots = min(n_shots, n_shots_max) if n_shots_max else n_shots

        if np.abs(besty - testy) < tol:
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
        if n_shots_budget != None and not n_shots_used_total < n_shots_budget:
            stop = True
            break

        niter += 1

    return OptimizeResult(
        fun=besty, x=bestx, nit=niter, nfev=funcalls, success=(niter > 1)
    )
