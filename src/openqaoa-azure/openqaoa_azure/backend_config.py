from .backends import DeviceAzure

from openqaoa_qiskit.backends import QAOAQiskitQPUBackend

device_access = {DeviceAzure: QAOAQiskitQPUBackend}
device_location = "azure"
device_plugin = DeviceAzure
device_args = {DeviceAzure: ["resource_id", "location"]}
