from .backends import DeviceAzure

from openqaoa_qiskit.backends import QAOAQiskitQPUBackend

device_access = {DeviceAzure: QAOAQiskitQPUBackend}
