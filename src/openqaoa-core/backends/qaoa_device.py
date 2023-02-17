from __future__ import annotations

from .devices_core import DeviceBase, DeviceLocal
from openqaoa_braket.backends import DeviceAWS
from openqaoa_qiskit.backends import DeviceQiskit
from openqaoa_pyquil.backends import DevicePyquil
from openqaoa_azure.backends import DeviceAzure


def device_class_arg_mapper(
    device_class: DeviceBase,
    hub: str = None,
    group: str = None,
    project: str = None,
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
    DEVICE_ARGS_MAPPER = {
        DeviceQiskit: {"hub": hub, "group": group, "project": project},
        DevicePyquil: {
            "as_qvm": as_qvm,
            "noisy": noisy,
            "compiler_timeout": compiler_timeout,
            "execution_timeout": execution_timeout,
            "client_configuration": client_configuration,
            "endpoint_id": endpoint_id,
            "engagement_manager": engagement_manager,
        },
        DeviceAWS: {
            "s3_bucket_name": s3_bucket_name,
            "aws_region": aws_region,
            "folder_name": folder_name,
        },
        DeviceAzure: {"resource_id": resource_id, "location": az_location},
    }

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
    if location == "ibmq":
        device_class = DeviceQiskit
    elif location == "qcs":
        device_class = DevicePyquil
    elif location == "aws":
        device_class = DeviceAWS
    elif location == "local":
        device_class = DeviceLocal
    elif location == "azure":
        device_class = DeviceAzure
    else:
        raise ValueError(f"Invalid device location, Choose from: {location}")

    return device_class(device_name=name, **kwargs)
