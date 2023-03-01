import numpy as np
from scipy.optimize import OptimizeResult


def iCANS(
    fun,
    x0,
    args=(),
    maxfev=None,
    stepsize=0.00001,
    n_shots_cost=None,
    n_shots_min=10,
    n_shots_max=10000,
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

    # define the initial values
    chi_ = np.zeros(len(x0))
    xi_ = np.zeros(len(x0))
    n_shots = [n_shots_min for _ in range(len(x0))]

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
        gradient, variance, n_shots_used = jac_w_variance(testx, n_shots=list(n_shots))

        # compute gradient descent step
        testx = testx - stepsize * gradient
        testy = np.real(fun(testx, *args))

        # add the number of shots used to the total (used in the gradient
        # computation and the cost function evaluation)
        n_shots_used_total += n_shots_used + n_shots_cost

        # update xi_ and chi_
        xi_ = mu * xi_ + (1 - mu) * variance
        chi_ = mu * chi_ + (1 - mu) * gradient

        # compute n_shots for next step
        xi = xi_ / (1 - mu ** (niter + 1))
        chi = chi_ / (1 - mu ** (niter + 1))
        n_shots = np.int32(
            np.ceil(
                2
                * lipschitz
                * stepsize
                * xi
                / ((2 - lipschitz * stepsize) * (chi**2 + b * mu**niter))
            )
        )

        n_shots = np.fmax(n_shots, 1)  # to compute gain n_shots should be at least 1
        gain = (
            (stepsize - lipschitz * gradient**2 / 2) * chi**2
            - lipschitz * stepsize**2 * xi / (2 * n_shots)
        ) / n_shots

        # clip the number of shots
        n_shots = np.fmax(n_shots, n_shots_min)
        n_shots = np.fmin(
            n_shots, n_shots[np.argmax(gain)]
        )  # max of n_shots is the one with the max gain
        n_shots = np.fmin(n_shots, n_shots_max) if n_shots_max else n_shots

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
