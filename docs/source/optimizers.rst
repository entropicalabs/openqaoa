Optimisers
==========
.. autoclass:: openqaoa.optimizers.training_vqa.OptimizeVQA
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:


SciPy Optimizers
----------------
.. autoclass:: openqaoa.optimizers.training_vqa.ScipyOptimizer
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: openqaoa.optimizers.training_vqa.CustomScipyGradientOptimizer
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: openqaoa.optimizers.training_vqa.PennyLaneOptimizer
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Optimization Methods
--------------------
.. automodule:: openqaoa.optimizers.optimization_methods
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. automodule:: openqaoa.optimizers.pennylane.optimization_methods_pennylane
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Derivate functions
------------------

Collection of functions to return derivative computation functions. Usually called from the `derivative_function` method of a `QAOABaseBackend` object.
New gradient/higher-order derivative computation methods can be added here. To add new computation methods:
#. Write function in the format : new_function(backend_obj, params_std, params_ext, gradient_options), or with less arguments.
#. Give this function a string identifier (eg: 'param_shift'), and add this to the list `derivative_methods` of the function `derivative`, and as a possible 'out'.


.. automodule:: openqaoa.derivative_functions
    :members:
    :show-inheritance:
    :inherited-members:

Qfim
----
.. automodule:: openqaoa.qfim
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Optimizers selectors
--------------------
.. automodule:: openqaoa.optimizers.qaoa_optimizer
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members: