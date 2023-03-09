from __future__ import annotations

from .plugin_finder import plugin_finder_dict
from .devices_core import DeviceBase, DeviceLocal

PLUGIN_DICT = plugin_finder_dict()

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
    
    local_vars = locals()
    
    for each_plugin_entrypoint in PLUGIN_DICT.values():
        if hasattr(each_plugin_entrypoint, 'device_args'):
            for each_key, each_value in each_plugin_entrypoint.device_args.items():
                # Convert list of accepted parameters into a dictionary with
                # the name of the variable as a key and the local value of the 
                # variable
                var_values = [local_vars[each_name] for each_name in each_value]
                input_dict = {each_key: dict(zip(each_value, var_values))}
                BACKEND_ARGS_MAPPER.update(input_dict)
    
#     if PLUGIN_DICT['braket']:
#         from openqaoa_braket.utilities import backend_args
#         DEVICE_ARGS_MAPPER[DeviceAWS] = {
#             "s3_bucket_name": s3_bucket_name,
#             "aws_region": aws_region,
#             "folder_name": folder_name,
#         }
        
#     if PLUGIN_DICT['azure']:
#         from openqaoa_azure.backends import DeviceAzure
#         DEVICE_ARGS_MAPPER[DeviceAzure] = {
#             "resource_id": resource_id, 
#             "location": az_location
#         }
    
#     if PLUGIN_DICT['qiskit']:
#         from openqaoa_qiskit.backends import DeviceQiskit
#         DEVICE_ARGS_MAPPER[DeviceQiskit] = {
#             "hub": hub,
#             "group": group,
#             "project": project,
#             "as_emulator": as_emulator
#         }
        
#     if PLUGIN_DICT['pyquil']:
#         from openqaoa_pyquil.backends import DevicePyquil
#         DEVICE_ARGS_MAPPER[DevicePyquil] = {
#             "as_qvm": as_qvm,
#             "noisy": noisy,
#             "compiler_timeout": compiler_timeout,
#             "execution_timeout": execution_timeout,
#             "client_configuration": client_configuration,
#             "endpoint_id": endpoint_id,
#             "engagement_manager": engagement_manager,
#         }

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
    
    location_device_mapper = dict()
    location_device_mapper.update({'local': DeviceLocal})
    location_device_mapper.update(zip([each_value.device_location for each_value in PLUGIN_DICT.values()], [each_value.device_plugin for each_value in PLUGIN_DICT.values()]))
    
    if location in location_device_mapper.keys():
        device_class = location_device_mapper[location]
    else:
        raise ValueError(f"Invalid device location, Choose from: {location}")

    return device_class(device_name=name, **kwargs)
