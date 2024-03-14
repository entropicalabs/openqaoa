from .backends import (
    DeviceCirq,
    QAOACirqQPUBackend,
    QAOACirqBackendShotBasedSimulator,
    QAOACirqBackendStatevecSimulator,
)

device_access = {DeviceCirq: QAOACirqQPUBackend}
device_name_to_obj = {
    "cirq.qasm_simulator": QAOACirqBackendShotBasedSimulator,
    "cirq.shot_simulator": QAOACirqBackendShotBasedSimulator,
    "cirq.statevector_simulator": QAOACirqBackendStatevecSimulator,
}
device_location = "ibmq"
device_plugin = DeviceCirq
device_args = {DeviceCirq: ["hub", "group", "project", "as_emulator"]}
backend_args = {
    QAOACirqBackendStatevecSimulator: [],
    QAOACirqBackendShotBasedSimulator: [
        "n_shots",
        "seed_simulator",
        "cirq_simulation_method",
        "noise_model",
        "initial_qubit_mapping",
    ],
    QAOACirqQPUBackend: [
        "n_shots",
        "initial_qubit_mapping",
        "cirq_optimization_level",
    ],
}
