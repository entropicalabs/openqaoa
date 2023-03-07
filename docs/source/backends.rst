Base Backends and Device
=================================

BaseDevice
----------

.. autoclass:: openqaoa.backends.devices_core.DeviceBase
    :members:
    :undoc-members:
    :inherited-members:

BaseBackends
------------

.. autoclass:: openqaoa.backends.basebackend.VQABaseBackend
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa.backends.basebackend.QAOABaseBackend
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa.backends.basebackend.QAOABaseBackendStatevector
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa.backends.basebackend.QAOABaseBackendShotBased
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa.backends.basebackend.QAOABaseBackendCloud
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa.backends.basebackend.QAOABaseBackendParametric
    :members:
    :undoc-members:
    :inherited-members:



Cloud Backends and Devices
=================================

Cloud Devices
-------------

.. autoclass:: openqaoa_qiskit.backends.devices.DeviceQiskit
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa_pyquil.backends.devices.DevicePyquil
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa_braket.backends.devices.DeviceAWS
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa_azure.backends.devices.DeviceAzure
    :members:
    :undoc-members:
    :inherited-members:

QCS - Rigetti
-------------
.. autoclass:: openqaoa_pyquil.backends.qaoa_pyquil_qpu.QAOAPyQuilQPUBackend
    :members:
    :undoc-members:
    :inherited-members:

IBM Quantum - IBM
-----------------
.. autoclass:: openqaoa_qiskit.backends.qaoa_qiskit_qpu.QAOAQiskitQPUBackend
    :members:
    :undoc-members:
    :inherited-members:

Amazon Braket - Amazon
----------------------
.. autoclass:: openqaoa_braket.backends.qaoa_braket_qpu.QAOAAWSQPUBackend
    :members:
    :undoc-members:
    :inherited-members:


Local Backend and Devices --- Simulators
========================================

Local Device
------------

.. autoclass:: openqaoa.backends.devices_core.DeviceLocal
    :members:
    :undoc-members:
    :inherited-members:

Vectorised
----------------------------------
.. autoclass:: openqaoa.backends.qaoa_vectorized.QAOAvectorizedBackendSimulator
    :members:
    :undoc-members:
    :inherited-members:
    
Analytical Simulator
----------------------------------
.. autoclass:: openqaoa.backends.qaoa_analytical_sim.QAOABackendAnalyticalSimulator
    :members:
    :undoc-members:
    :inherited-members:

Qiskit
----------------------------------
.. autoclass:: openqaoa_qiskit.backends.qaoa_qiskit_sim.QAOAQiskitBackendShotBasedSimulator
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa_qiskit.backends.qaoa_qiskit_sim.QAOAQiskitBackendStatevecSimulator
    :members:
    :undoc-members:
    :inherited-members:

PyQuil
----------------------------------
.. autoclass:: openqaoa_pyquil.backends.qaoa_pyquil_sim.QAOAPyQuilWavefunctionSimulatorBackend
    :members:
    :undoc-members:
    :inherited-members:

Backend selectors
=================================
.. automodule:: openqaoa.backends.qaoa_backend
    :members:
    :undoc-members:
    :inherited-members:
