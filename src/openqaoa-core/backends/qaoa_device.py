from __future__ import annotations

from .plugin_finder import backend_boolean_list
from .devices_core import DeviceBase, DeviceLocal
# from openqaoa_braket.backends import DeviceAWS
# from openqaoa_qiskit.backends import DeviceQiskit
# from openqaoa_pyquil.backends import DevicePyquil
# from openqaoa_azure.backends import DeviceAzure

BACKEND_BOOLEAN_LIST = backend_boolean_list()

def device_class_arg_mapper(
    device_class: DeviceBase,
    hub: str = None,
    group: str = None,
    project: str = None,
    as_emulator: bool = None,
    as_qvm: bool = None,
    noisy: bool = None,
    compiler_timeout: float = None,
    execution_timeout: float = None,
    client_configuration: QCSClientConfiguration = None,
    endpoint_id: str = None,
    engagement_manager: EngagementManager = None,
    folder_name: str = None,
    s3_bucket_name: str = None,
    aws_region: str = None,
    resource_id: str = None,
    az_location: str = None,
) -> dict:
    
    DEVICE_ARGS_MAPPER = dict()
    
    if BACKEND_BOOLEAN_LIST[0]:
        from openqaoa_braket.utilities import backend_args
        DEVICE_ARGS_MAPPER[DeviceAWS] = {
            "s3_bucket_name": s3_bucket_name,
            "aws_region": aws_region,
            "folder_name": folder_name,
        }
        
    if BACKEND_BOOLEAN_LIST[1]:
        from openqaoa_azure.backends import DeviceAzure
        DEVICE_ARGS_MAPPER[DeviceAzure] = {
            "resource_id": resource_id, 
            "location": az_location
        }
    
    if BACKEND_BOOLEAN_LIST[2]:
        from openqaoa_qiskit.backends import DeviceQiskit
        DEVICE_ARGS_MAPPER[DeviceQiskit] = {
            "hub": hub,
            "group": group,
            "project": project,
            "as_emulator": as_emulator
        }
        
    if BACKEND_BOOLEAN_LIST[3]:
        from openqaoa_pyquil.backends import DevicePyquil
        DEVICE_ARGS_MAPPER[DevicePyquil] = {
            "as_qvm": as_qvm,
            "noisy": noisy,
            "compiler_timeout": compiler_timeout,
            "execution_timeout": execution_timeout,
            "client_configuration": client_configuration,
            "endpoint_id": endpoint_id,
            "engagement_manager": engagement_manager,
        }
    
    
    # DEVICE_ARGS_MAPPER = {
    #     DeviceQiskit: {
    #         "hub": hub,
    #         "group": group,
    #         "project": project,
    #         "as_emulator": as_emulator,
    #     },
    #     DevicePyquil: {
    #         "as_qvm": as_qvm,
    #         "noisy": noisy,
    #         "compiler_timeout": compiler_timeout,
    #         "execution_timeout": execution_timeout,
    #         "client_configuration": client_configuration,
    #         "endpoint_id": endpoint_id,
    #         "engagement_manager": engagement_manager,
    #     },
    #     DeviceAWS: {
    #         "s3_bucket_name": s3_bucket_name,
    #         "aws_region": aws_region,
    #         "folder_name": folder_name,
    #     },
    #     DeviceAzure: {"resource_id": resource_id, "location": az_location},
    # }

    final_device_kwargs = {
        key: value
        for key, value in DEVICE_ARGS_MAPPER[device_class].items()
        if value is not None
    }
    return final_device_kwargs


def create_device(location: str, name: str, **kwargs):
    """
    This function returns an instance of the appropriate device class.

    Parameters
    ----------
    device_name: str
        The name of the device to be accessed.
    device_location: str
        The location of the device to be accessed.
    kwargs: dict
        A dictionary of keyword arguments to be passed to the device class.
        These will be used to initialise the device.

    Returns
    -------
    device: DeviceBase
        An instance of the appropriate device class.
    """

    location = location.lower()
    if location == "ibmq" and BACKEND_BOOLEAN_LIST[2]:
        from openqaoa_qiskit.backends import DeviceQiskit
        device_class = DeviceQiskit
    elif location == "qcs" and BACKEND_BOOLEAN_LIST[3]:
        from openqaoa_pyquil.backends import DevicePyquil
        device_class = DevicePyquil
    elif location == "aws" and BACKEND_BOOLEAN_LIST[0]:
        from openqaoa_aws.backends import DeviceAWS
        device_class = DeviceAWS
    elif location == "local":
        device_class = DeviceLocal
    elif location == "azure" and BACKEND_BOOLEAN_LIST[1]:
        from openqaoa_azure.backends import DeviceAzure
        device_class = DeviceAzure
    else:
        raise ValueError(f"Invalid device location, Choose from: {location}")

    return device_class(device_name=name, **kwargs)
