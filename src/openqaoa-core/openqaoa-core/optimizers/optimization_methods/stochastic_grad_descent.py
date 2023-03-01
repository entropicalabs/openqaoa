# def stochastic_grad_descent(
#     fun,
#     x0,
#     jac,
#     args=(),
#     stepsize=0.001,
#     mass=0.9,
#     startiter=0,
#     maxiter=1000,
#     callback=None,
#     **options
# ):
#     """
#     Scipy "OptimizeResult" object for method == 'sgd'
#     scipy.optimize.minimize compatible implementation of stochastic gradient descent with momentum.
#     Adapted from ``autograd/misc/optimizers.py``.

#     """
#     x = x0
#     velocity = np.zeros_like(x)
#     print("SGD Test")

#     for i in range(startiter, startiter + maxiter):
#         g = jac(x)

#         velocity = mass * velocity - (1.0 - mass) * g
#         x = x + stepsize * velocity

#         if callback is not None:
#             callback(x)

#     return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i + 1, nfev=i + 1, success=True)
