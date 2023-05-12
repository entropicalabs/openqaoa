from .backends import DeviceAWS, QAOAAWSQPUBackend

device_access = {DeviceAWS: QAOAAWSQPUBackend}
device_location = "aws"
device_plugin = DeviceAWS
device_args = {DeviceAWS: ["s3_bucket_name", "aws_region", "folder_name"]}
backend_args = {
    QAOAAWSQPUBackend: ["n_shots", "disable_qubit_rewiring", "initial_qubit_mapping"]
}
