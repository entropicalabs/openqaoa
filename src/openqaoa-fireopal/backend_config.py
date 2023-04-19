from .backends import DeviceFireOpal, QAOAFireOpalQPUBackend

device_access = {DeviceFireOpal: QAOAFireOpalQPUBackend}
device_location = "qctrl"
device_plugin = DeviceFireOpal
device_args = {DeviceFireOpal: ["device_name", "hub", "group", "project"]}
backend_args = {QAOAFireOpalQPUBackend: ["n_shots", "initial_qubit_mapping"]}
