from .backends import QAOAQIRBackend
from openqaoa_azure.backends import DeviceAzure

device_access = {DeviceAzure: QAOAQIRBackend}
device_location = "azure"
device_plugin = DeviceAzure
device_args = {DeviceAzure: ["resource_id", "location"]}
backend_args = {QAOAQIRBackend: ["n_shots", "initial_qubit_mapping"]}
