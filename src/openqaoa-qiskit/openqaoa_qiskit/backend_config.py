from .backends import (
    DeviceQiskit,
    QAOAQiskitQPUBackend,
    QAOAQiskitBackendShotBasedSimulator,
    QAOAQiskitBackendStatevecSimulator,
)

device_access = {DeviceQiskit: QAOAQiskitQPUBackend}
device_name_to_obj = {
    "qiskit.qasm_simulator": QAOAQiskitBackendShotBasedSimulator,
    "qiskit.shot_simulator": QAOAQiskitBackendShotBasedSimulator,
    "qiskit.statevector_simulator": QAOAQiskitBackendStatevecSimulator,
}
device_location = "ibmq"
device_plugin = DeviceQiskit
device_args = {DeviceQiskit: ["hub", "group", "project", "as_emulator"]}
backend_args = {
    QAOAQiskitBackendStatevecSimulator: [],
    QAOAQiskitBackendShotBasedSimulator: [
        "n_shots",
        "seed_simulator",
        "qiskit_simulation_method",
        "noise_model",
        "initial_qubit_mapping",
    ],
    QAOAQiskitQPUBackend: [
        "n_shots",
        "initial_qubit_mapping",
        "qiskit_optimization_level",
    ],
}
