from .backends import (DevicePyquil, QAOAPyQuilQPUBackend, 
                       QAOAPyQuilWavefunctionSimulatorBackend)

device_access = {DevicePyquil: QAOAPyQuilQPUBackend}
device_name_to_obj = {"pyquil.statevector_simulator" : QAOAPyQuilWavefunctionSimulatorBackend}