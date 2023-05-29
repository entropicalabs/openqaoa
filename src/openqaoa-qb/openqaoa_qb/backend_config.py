from .backends import DeviceQristal, QAOAQristalQPUBackend

device_access = {DeviceQristal: QAOAQristalQPUBackend}
device_location = "qristal"
device_plugin = DeviceQristal
device_args = {DeviceQristal: []}
backend_args = {
    QAOAQristalQPUBackend: [
        "n_shots",
        "initial_qubit_mapping",
        "qiskit_optimization_level",
    ],
}