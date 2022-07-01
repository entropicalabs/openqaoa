Abstract Backends
=================================
.. .. autoclass:: openqaoa.basebackend.QuantumCircuitBase
..     :members:
..     :undoc-members:
..     :inherited-members:

.. autoclass:: openqaoa.basebackend.VQABaseBackend
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa.basebackend.QAOABaseBackend
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa.basebackend.QAOABaseBackendStatevector
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa.basebackend.QAOABaseBackendShotBased
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa.basebackend.QAOABaseBackendCloud
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa.basebackend.QAOABaseBackendParametric
    :members:
    :undoc-members:
    :inherited-members:


Cloud Devices
=================================
QCS - Rigetti
----------------------------------
.. autoclass:: openqaoa.backends.qpus.qaoa_pyquil_qpu.QAOAPyQuilQPUBackend
    :members:
    :undoc-members:
    :inherited-members:

IBMQ - IBM
----------------------------------
.. autoclass:: openqaoa.backends.qpus.qaoa_qiskit_qpu.QAOAQiskitQPUBackend
    :members:
    :undoc-members:
    :inherited-members:

Devices
-------
.. autoclass:: openqaoa.devices.DeviceBase
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa.devices.DeviceLocal
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa.devices.DeviceQiskit
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa.devices.DevicePyquil
    :members:
    :undoc-members:
    :inherited-members:

Local devices --- Simulators
=================================
Vectorised
----------------------------------
.. autoclass:: openqaoa.backends.simulators.qaoa_vectorized.QAOAvectorizedBackendSimulator
    :members:
    :undoc-members:
    :inherited-members:

Qiskit
----------------------------------
.. autoclass:: openqaoa.backends.simulators.qaoa_qiskit_sim.QAOAQiskitBackendShotBasedSimulator
    :members:
    :undoc-members:
    :inherited-members:

.. autoclass:: openqaoa.backends.simulators.qaoa_qiskit_sim.QAOAQiskitBackendStatevecSimulator
    :members:
    :undoc-members:
    :inherited-members:


PyQuil
----------------------------------
.. autoclass:: openqaoa.backends.simulators.qaoa_pyquil_sim.QAOAPyQuilWavefunctionSimulatorBackend
    :members:
    :undoc-members:
    :inherited-members:

Backend selectors
=================================
.. automodule:: openqaoa.backends.qaoa_backend
    :members:
    :undoc-members:
    :inherited-members: