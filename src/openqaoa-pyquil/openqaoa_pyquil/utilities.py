from .backends import (DevicePyquil, QAOAPyQuilQPUBackend, 
                       QAOAPyQuilWavefunctionSimulatorBackend)

device_access = {DevicePyquil: QAOAPyQuilQPUBackend}
device_name_to_obj = {"pyquil.statevector_simulator" : QAOAPyQuilWavefunctionSimulatorBackend}
device_location = 'qcs'
device_plugin = DevicePyquil
device_args = {DevicePyquil: ['as_qvm', 'noisy', 'compiler_timeout', 
                              'execution_timeout', 'client_configuration', 
                              'endpoint_id', 'engagement_manager']}
backend_args = {QAOAPyQuilWavefunctionSimulatorBackend: [], 
                QAOAPyQuilQPUBackend: ['n_shots', 'active_reset', 'rewiring', 
                                       'initial_qubit_mapping']}