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


RQAOA
-----
.. autoclass:: openqaoa.workflows.optimizer.RQAOA
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

.. autoclass:: openqaoa.workflows.parameters.qaoa_parameters.ExtraResults
    :members:
    :undoc-members:
    :inherited-members:

Workflow RQAOA Parameters
-------------------------
.. autoclass:: openqaoa.workflows.parameters.rqaoa_parameters.RqaoaParameters
    :members:
    :undoc-members:
    :inherited-members: