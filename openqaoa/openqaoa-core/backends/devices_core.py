#   Copyright 2022 Entropica Labs
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import annotations

import abc
import logging

import numpy as np

logging.getLogger().setLevel(logging.ERROR)

SUPPORTED_LOCAL_SIMULATORS = [
    'qiskit.qasm_simulator', 'qiskit.shot_simulator',
    'qiskit.statevector_simulator','vectorized',
    'pyquil.statevector_simulator'
]


class DeviceBase(metaclass=abc.ABCMeta):
    """An object that contains the relevant information required to access 
    certain backends. Other Access Objects have to inherit from this object.
    """

    @abc.abstractmethod
    def check_connection(self) -> bool:
        """This method should allow a user to easily check if the credentials
        provided to access the remote QPU is valid.

        Returns
        -------
        bool 
            True if a connection can be established. If False, the error 
            should be logged and printable. (Not creating an exception 
            here is good for extendibility. i.e. able to try multiple
            providers without exiting the program.)
        """
        pass

class DeviceLocal(DeviceBase):
    """
    This class is a placeholder for all locally accessible devices.
    """
    def __init__(self, device_name: str):
        self.device_name = device_name
        self.device_location = 'local'

    def check_connection(self) -> bool:
        if self.device_name in SUPPORTED_LOCAL_SIMULATORS:
            return True
        else:
            return False

def device_class_arg_mapper(device_class:DeviceBase,
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
                            s3_bucket_name:str = None, 
                            aws_region: str = None, 
                            resource_id: str = None, 
                            az_location: str = None) -> dict:
    DEVICE_ARGS_MAPPER = {
        DeviceQiskit: {'hub': hub, 'group': group, 'project': project},

        DevicePyquil: {'as_qvm': as_qvm,
                        'noisy': noisy,
                        'compiler_timeout': compiler_timeout,
                        'execution_timeout': execution_timeout,
                        'client_configuration': client_configuration,
                        'endpoint_id': endpoint_id,
                        'engagement_manager': engagement_manager},
        
        DeviceAWS: {'s3_bucket_name': s3_bucket_name,
                    'aws_region': aws_region,
                    'folder_name': folder_name},
        
        DeviceAzure: {'resource_id': resource_id, 
                      'location': az_location}
    }

    final_device_kwargs = {key: value for key, value in DEVICE_ARGS_MAPPER[device_class].items()
                           if value is not None}
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
    if location == 'ibmq':
        device_class = DeviceQiskit
    elif location == 'qcs':
        device_class = DevicePyquil
    elif location == 'aws':
        device_class = DeviceAWS
    elif location == 'local':
        device_class = DeviceLocal
    elif location == 'azure':
        device_class = DeviceAzure
    else:
        raise ValueError(f'Invalid device location, Choose from: {location}')

    return device_class(device_name=name, **kwargs)