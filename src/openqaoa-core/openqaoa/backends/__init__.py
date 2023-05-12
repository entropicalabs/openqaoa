"""
Contains backend specific codes for implementing QAOA on supported simulators
and QPU backends

Currently Supported:
	Qiskit:
		QASM Simulator
		Statevector Simulator
		IBMQ available QPUs
	PyQuil:
		Rigetti QPUs
		Statevector Simulator
	Vectorized:
		Fast numpy native Statevector Simulator
"""
from .qaoa_vectorized import QAOAvectorizedBackendSimulator
from .qaoa_analytical_sim import QAOABackendAnalyticalSimulator
from .devices_core import DeviceLocal
from .qaoa_device import create_device
