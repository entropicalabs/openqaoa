QAOA Parametrisations
=====================

We currently offer 7 different parametrisations for QAOA, which can be found in
the ``openqaoa.qaoa_components.variational_parameters`` module. They fall broadly into three categories:

* The ``Standard`` classes are parametrisations that have the :math:`\gamma` 's and :math:`\beta` 's as free parameters, as defined in the seminal paper by Farhi `et al` in `A Quantum Approximate Optimization Algorithm <https://arxiv.org/abs/1411.4028>`_.
* The ``Fourier`` classes have the discrete cosine and sine transforms of the :math:`\gamma` 's respective :math:`\beta`'s as free parameters, as proposed by Zhou et al. in `Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices <https://arxiv.org/abs/1812.01041>`_.
* The ``Annealing`` class is based on the idea of QAOA being a form of discretised, adiabatic annealing. Here the function values :math:`s(t_i)` at equally spaced times :math:`t_i` are the free parameters.

Except for the ``Annealing`` parameters, each class also comes in three levels of detail: 

* ``StandardParams`` and ``FourierParams`` offer the :math:`\gamma` 's and :math:`\beta`'s as proposed in above papers. 
* ``StandardWithBiasParams`` and ``FourierWithBiasParams`` allow for extra :math:`\gamma`'s for possible single-qubit bias terms, resp. their discrete sine transform. 
* ``ExtendedParams`` and ``FourierExtendedParams`` offer full control by having a seperate set of rotation angles for each term in the cost and mixer Hamiltonians, respective having a seperate set of Fourier coefficients for each term.

.. You can always convert parametrisations with fewer degrees of freedom to ones with more using the ``.from_other_parameters()`` classmethod. The full type
.. tree is shown below, where the arrows mark possible conversions:

.. code-block::

       ExtendedParams   <--------- FourierExtendedParams
              ^                         ^
              |                         |
    StandardWithBiasParams <------ FourierWithBiasParams
              ^                         ^
              |                         |
        StandardParams  <----------- FourierParams
              ^
              |
        AnnealingParams

Parameters
----------
.. autoclass:: openqaoa.qaoa_components.ansatz_constructor.baseparams.AnsatzDescriptor
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa.qaoa_components.ansatz_constructor.baseparams.QAOADescriptor
    :members:
    :undoc-members:
    :inherited-members:
    :noindex:


Standard Parameters
-------------------
.. autoclass:: openqaoa.qaoa_components.variational_parameters.standardparams.QAOAVariationalStandardParams
    :members:
    :inherited-members:

.. autoclass:: openqaoa.qaoa_components.variational_parameters.standardparams.QAOAVariationalStandardWithBiasParams
    :members:
    :inherited-members:


Extended Parameters
--------------------
.. autoclass:: openqaoa.qaoa_components.variational_parameters.extendedparams.QAOAVariationalExtendedParams
    :members:
    :inherited-members:


Fourier Parameters
-------------------
.. autoclass:: openqaoa.qaoa_components.variational_parameters.fourierparams.QAOAVariationalFourierParams
    :members:
    :inherited-members:

.. autoclass:: openqaoa.qaoa_components.variational_parameters.fourierparams.QAOAVariationalFourierWithBiasParams
    :members:
    :inherited-members:

.. autoclass:: openqaoa.qaoa_components.variational_parameters.fourierparams.QAOAVariationalFourierExtendedParams
    :members:
    :inherited-members:


Annealing Parameters
--------------------
.. autoclass:: openqaoa.qaoa_components.variational_parameters.annealingparams.QAOAVariationalAnnealingParams
    :members:
    :inherited-members: