from .backends import (DeviceQiskit, QAOAQiskitQPUBackend, 
                       QAOAQiskitBackendShotBasedSimulator,
                       QAOAQiskitBackendStatevecSimulator)

device_access = {DeviceQiskit: QAOAQiskitQPUBackend}
device_name_to_obj = {"qiskit.qasm_simulator" : QAOAQiskitBackendShotBasedSimulator, 
                      "qiskit.shot_simulator" : QAOAQiskitBackendShotBasedSimulator, 
                      "qiskit.statevector_simulator" : QAOAQiskitBackendStatevecSimulator}