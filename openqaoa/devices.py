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

import abc
import numpy as np
from typing import Optional
from qiskit import IBMQ
from qiskit.providers.ibmq import IBMQAccountError
from qiskit.providers.ibmq.api.exceptions import RequestsApiError

from azure.quantum.qiskit import AzureQuantumProvider

SUPPORTED_LOCAL_SIMULATORS = [
    'qiskit.qasm_simulator', 'qiskit.shot_simulator',
    'qiskit.statevector_simulator','vectorized'
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

class DeviceAzure(DeviceBase):

    """
    Contains the required information and methods needed to access remote 
    Azure QPUs and Simulators.

    Parameters
    ----------
    available_qpus: `list`
        When connection to a provider is established, this attribute contains a list
        of backend names which can be used to access the selected backend by reinitialising
        the Access Object with the name of the available backend as input to the
        device_name parameter.
    """
    
    def __init__(self, device_name: str, resource_id: str, az_location: str):
        """
        Input parameters required for this can be found in the user's Azure 
        Quantum Workspace.
        
        Parameters
        ----------
        device_name: `str`
            The name of the Azure remote QPU/Simulator to be used
        resource_id: `str`
        az_location: `str`
        """
        
        self.resource_id = resource_id
        self.location = az_location
        self.device_name = device_name
        self.device_location = 'azure'
        
        self.provider_connected = None
        self.qpu_connected = None
        
    def check_connection(self):
        """
        """
        
        self.provider_connected = self._check_provider_connection()

        if self.provider_connected == False:
            return self.provider_connected

        self.available_qpus = [backend.name()
                               for backend in self.provider.backends()]

        if self.device_name == '':
            return self.provider_connected

        self.qpu_connected = self._check_backend_connection()

        if self.provider_connected and self.qpu_connected:
            return True
        else:
            return False
        
    def _check_backend_connection(self) -> bool:
        """Private method for checking connection with backend(s).
        """

        if self.device_name in self.available_qpus:
            self.backend_device = self.provider.get_backend(self.device_name)
            return True
        else:
            print(
                f"Please choose from {self.available_qpus} for this provider")
            return False

    def _check_provider_connection(self) -> bool:
        """
        Private method for checking connection with provider.
        """

        try:
            self.provider = AzureQuantumProvider(resource_id=self.resource_id, 
                                                 location=self.location)

            return True

        except ValueError as e:
            print('Either the resource id or location specified was invalid: {}'.format(e))
            return False

        except Exception as e:
            print('An Exception has occured when trying to connect with the \
            provider: {}'.format(e))
            return False


class DeviceQiskit(DeviceBase):
    """Contains the required information and methods needed to access remote
    qiskit QPUs.

    Parameters
    ----------
    available_qpus: `list`
        When connection to a provider is established, this attribute contains a list
        of backend names which can be used to access the selected backend by reinitialising
        the Access Object with the name of the available backend as input to the
        device_name parameter.
    """

    def __init__(self, device_name: str, api_token: str,
                 hub: str, group: str, project: str):
        """A majority of the input parameters required for this can be found in
        the user's IBMQ Experience account.

        Parameters
        ----------
        device_name: `str`
            The name of the IBMQ device to be used
        api_token: `str`
            Valid IBMQ Experience Token.
        hub: `str`
            Valid IBMQ hub name.
        group: `str`
            Valid IBMQ group name. 
        project: `str`
            The name of the project for which the experimental data will be 
            saved in on IBMQ's end.
        """

        self.api_token = api_token
        self.hub = hub
        self.group = group
        self.project = project
        self.device_name = device_name
        self.device_location = 'ibmq'

        self.provider_connected = None
        self.qpu_connected = None

    def check_connection(self) -> bool:
        """This method should allow a user to easily check if the credentials
        provided to access the remote QPU is valid.

        If no backend was specified in initialisation of object, just runs
        a test connection without a specific backend.
        If backend was specified, checks if connection to that backend
        can be established.

        Returns
        -------
        bool
            True if successfully connected to IBMQ or IBMQ and the QPU backend
            if it was specified. False if unable to connect to IBMQ or failure
            in the attempt to connect to the specified backend.
        """

        self.provider_connected = self._check_provider_connection()

        if self.provider_connected == False:
            return self.provider_connected

        self.available_qpus = [backend.name()
                               for backend in self.provider.backends()]

        if self.device_name == '':
            return self.provider_connected

        self.qpu_connected = self._check_backend_connection()

        if self.provider_connected and self.qpu_connected:
            return True
        else:
            return False

    def _check_backend_connection(self) -> bool:
        """Private method for checking connection with backend(s).
        """

        if self.device_name in self.available_qpus:
            self.backend_device = self.provider.get_backend(self.device_name)
            return True
        else:
            print(
                f"Please choose from {self.available_qpus} for this provider")
            return False

    def _check_provider_connection(self) -> bool:
        """
        Private method for checking connection with provider.
        """

        try:
            if IBMQ.active_account() is None:
                self.provider = IBMQ.enable_account(self.api_token, hub=self.hub,
                                                    group=self.group,
                                                    project=self.project)
            elif IBMQ.active_account()['token'] != self.api_token:
                IBMQ.disable_account()
                self.provider = IBMQ.enable_account(self.api_token, hub=self.hub,
                                                    group=self.group,
                                                    project=self.project)
            else:
                self.provider = IBMQ.get_provider(hub=self.hub, group=self.group,
                                                  project=self.project)

            return True

        except RequestsApiError as e:
            print('The api key used was invalid: {}'.format(e))
            return False

        except Exception as e:
            print('An Exception has occured when trying to connect with the \
            provider: {}'.format(e))
            return False


def device_class_arg_mapper(device_class: DeviceBase,
                            api_token: str = None,
                            hub: str = None,
                            group: str = None,
                            project: str = None,
                            resource_id: str = None, 
                            az_location: str = None) -> dict:
    DEVICE_ARGS_MAPPER = {
        DeviceQiskit: {'api_token': api_token,
                        'hub': hub,
                        'group': group,
                        'project': project},
        
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
    elif location == 'local':
        device_class = DeviceLocal
    elif location == 'azure':
        device_class = DeviceAzure
    else:
        raise ValueError(f'Invalid device location, Choose from: {location}')

    return device_class(device_name=name, **kwargs)