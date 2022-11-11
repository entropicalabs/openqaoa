Workflows
=================================

Workflows are a simple reference API to build complex quantum optimisations problems. Currently, it supports creations of `QAOA` and `Recursive QAOA` workflows.

Workflows are designed to aid the user to focus on the optimisation problem, while delegating the construction and the execution of the specific algorithm to `OpenQAOA`


Workflows - ABC
---------------
.. autoclass:: openqaoa.workflows.optimizer.Optimizer
    :members:
    :undoc-members:
    :inherited-members:

QAOA
----
.. autoclass:: openqaoa.workflows.optimizer.QAOA
    :members:
    :undoc-members:
    :inherited-members:


Workflow QAOA Parameters
------------------------
.. autoclass:: openqaoa.workflows.parameters.qaoa_parameters.CircuitProperties
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa.workflows.parameters.qaoa_parameters.BackendProperties
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa.workflows.parameters.qaoa_parameters.ClassicalOptimizer
    :members:
    :undoc-members:
    :inherited-members:


RQAOA
-----
Recursive QAOA (RQAOA) is an iterative variant of QAOA, first introduced by Bravyi et. al in [1]. It runs QAOA recursively and uses the expectation values of the Hamiltonian terms to impose constraints and eliminate qubits for the problem at each step. Once the reduced problem reaches a preset cutoff size, it is solved exactly solved via classical methods. The final answer is then reconstructed by re-inserting the eliminated qubits in the classical solution in the appropriate order.

We currently offer two flavors of RQAOA, both of which can be found in the ``rqaoa`` module:

 * The ``Custom`` (default) strategy allows the user to define the number of eliminations to be performed at each step. This defined by the parameter ``steps``. If the parameter is set as an integer, the algorithm will use this number as the number of qubits to be eliminated at each step. Alternatively, it is possible to pass a list, which specifies the number of qubits to be eliminated at each step. For ``steps = 1``, the algorithm reduces to the original form of RQAOA presented in [1].

 * The ``Adaptive`` strategy adaptively selects how many qubits to eliminate at each step. The maximum number of allowed eliminations is given by the parameter ``n_max``. At each step, the algorithm selects the top ``n_max+1`` expectation values (ranked in magnitude), computes the mean among and uses the ones lying above this value for qubit elimination. This corresopnds to a maximum of ``n_max`` possible elimnations per step. For ``n_max= 1``, the algorithm reduces to the original form of RQAOA presented in [1].

The development of this method is associated with an internal research project at Entropica Labs and will be soon released publicly [2].

To choose the strategy, set the parameter ``rqaoa_type`` using the `set_rqaoa_parameters` method. To use the ``Adaptive`` strategy, pass ``rqaoa_type = 'adaptive'``. The default strategy is ``Custom``.

.. autoclass:: openqaoa.workflows.optimizer.RQAOA
    :members:
    :undoc-members:
    :inherited-members:
    

Workflow RQAOA Parameters
-------------------------
.. autoclass:: openqaoa.workflows.parameters.rqaoa_parameters.RqaoaParameters
    :members:
    :undoc-members:
    :inherited-members:

References
----------

[1] S. Bravyi, A. Kliesch, R. Koenig, and E. Tang, `Physical Review Letters 125, 260505 (2020) <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.260505>`_.
[2] E.I. Rodriguez Chiacchio, V. Sharma, E. Munro (Work in progress)


