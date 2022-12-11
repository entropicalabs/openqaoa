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

from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._engagement_manager import EngagementManager
from pyquil import get_qc

from boto3.session import Session
from botocore.exceptions import NoRegionError
from braket.aws import AwsDevice
from braket.aws.aws_session import AwsSession

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

class DeviceQiskit(DeviceBase):
    """
    Contains the required information and methods needed to access remote
    qiskit QPUs.

    Attributes
    ----------
    available_qpus: `list`
      When connection to a provider is established, this attribute contains a list
      of backend names which can be used to access the selected backend by reinitialising
      the Access Object with the name of the available backend as input to the
      device_name parameter.
    n_qubits: `int`
        The maximum number of qubits available for the selected backend. Only
        available if check_connection method is executed and a connection to the
        qpu and provider is established.
    """

    def __init__(self, device_name: str, hub: str = None, group: str = None, 
                 project: str = None):
        """The user's IBMQ account has to be authenticated through qiskit in 
        order to use this backend. This can be done through `IBMQ.save_account`.
        
        See: https://quantum-computing.ibm.com/lab/docs/iql/manage/account/ibmq

        Parameters
        ----------
		device_name: `str`
			The name of the IBMQ device to be used
        hub: `str`
            Valid IBMQ hub name.
        group: `str`
            Valid IBMQ group name. 
        project: `str`
            The name of the project for which the experimental data will be 
            saved in on IBMQ's end.
        """

        self.device_name = device_name
        self.device_location = 'ibmq'
        self.hub = hub
        self.group = group
        self.project = project

        self.provider_connected = None
        self.qpu_connected = None

    def check_connection(self) -> bool:
        """
        This method should allow a user to easily check if the credentials
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
            self.n_qubits = self.backend_device.configuration().n_qubits
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
            self.provider = IBMQ.load_account()
            if any([self.hub, self.group, self.project]):
                self.provider = IBMQ.get_provider(hub=self.hub, group=self.group, project=self.project)
            return True
        except Exception as e:
            print('An Exception has occured when trying to connect with the provider. Please note that you are required to set up your IBMQ account locally first. See: https://quantum-computing.ibm.com/lab/docs/iql/manage/account/ibmq for how to save your IBMQ account locally: {}'.format(e))
            return False


class DevicePyquil(DeviceBase):
    """
    Contains the required information and methods needed to access remote
    Rigetti QPUs via Pyquil.
    
    Attributes
    ----------
    n_qubits: `int`
        The maximum number of qubits available for the selected backend. 
        Available upon proper initialisation of the class.
    """

    def __init__(self, device_name: str, as_qvm: bool = None, noisy: bool = None,
                 compiler_timeout: float = 20.0,
                 execution_timeout: float = 20.0,
                 client_configuration: QCSClientConfiguration = None,
                 endpoint_id: str = None,
                 engagement_manager: EngagementManager = None):
        """
        Parameters
        ----------
        device_name: str 
            The name of the desired quantum computer. This should correspond to 
            a name returned by :py:func:`list_quantum_computers`. Names ending 
            in "-qvm" will return a QVM. Names ending in "-pyqvm" will return a 
            :py:class:`PyQVM`. Names ending in "-noisy-qvm" will return a QVM 
            with a noise model. Otherwise, we will return a QPU with the given 
            name.
        as_qvm: bool 
            An optional flag to force construction of a QVM (instead of a QPU). 
            If specified and set to ``True``, a QVM-backed quantum computer will 
            be returned regardless of the name's suffix.
        noisy: bool
            An optional flag to force inclusion of a noise model. If specified 
            and set to ``True``, a quantum computer with a noise model will be 
            returned regardless of the name's suffix. The generic QVM noise 
            model is simple T1 and T2 noise plus readout error. See 
            :py:func:`~pyquil.noise.decoherence_noise_with_asymmetric_ro`. Note, 
            we currently do not support noise models based on QCS hardware; a 
            value of `True`` will result in an error if the requested QPU is a 
            QCS hardware QPU.
        compiler_timeout: float
            Time limit for compilation requests, in seconds.
        execution_timeout: float
            Time limit for execution requests, in seconds.
        client_configuration: QCSClientConfiguration
            Optional client configuration. If none is provided, a default one 
            will be loaded.
        endpoint_id: str
            Optional quantum processor endpoint ID, as used in the 
            `QCS API Docs`_.
        engagement_manager: EngagementManager
            Optional engagement manager. If none is provided, a default one will 
            be created.
        """

        self.device_name = device_name
        self.device_location = 'qcs'
        self.as_qvm = as_qvm
        self.noisy = noisy
        self.compiler_timeout = compiler_timeout
        self.execution_timeout = execution_timeout
        self.client_configuration = client_configuration
        self.endpoint_id = endpoint_id
        self.engagement_manager = engagement_manager

        self.quantum_computer = get_qc(name=self.device_name, as_qvm=self.as_qvm, noisy=self.noisy,
                                       compiler_timeout=self.compiler_timeout, execution_timeout=self.execution_timeout,
                                       client_configuration=self.client_configuration, endpoint_id=self.endpoint_id, engagement_manager=self.engagement_manager)
        self.n_qubits = len(self.quantum_computer.qubits())

    def check_connection(self) -> bool:
        """This method should allow a user to easily check if the credentials
        provided to access the remote QPU is valid.

        If no device was specified in initialisation of object, just runs
        a test connection without a specific device.
        If device was specified, checks if connection to that device
        can be established.

        TODO : 
        Accessing Rigetti's QCS is currently unsupported, so this part is empty until that is figured out.
        """

        return True
    
    
class DeviceAWS(DeviceBase):
    
    """
    Contains the required information and methods needed to access QPUs hosted
    on AWS Braket.
    
    Attributes
    ----------
	available_qpus: `list`
		When connection to AWS is established, this attribute contains a list
		of device names which can be used to access the selected device by
		reinitialising the Access Object with the name of the available device
		as input to the device_name parameter.
    n_qubits: `int`
        The maximum number of qubits available for the selected backend. Only
        available if check_connection method is executed and a connection to the
        qpu and provider is established.
    """
    
    def __init__(self, device_name: str, s3_bucket_name: str = None, 
                 aws_region: str = None, 
                 folder_name: str = 'openqaoa'):
        
        """Input the device arn and the name of the folder in which all the
        results for the QPU runs would be saved on the pre-defined s3 bucket. 
        Note that the user is required to authenticate through the AWS CLI 
        before being able to use this Device object.
        
        See: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html for further details.

        Parameters
        ----------
		device_name: `str`
			The ARN string of the braket QPU/simulator to be used
        s3_bucket_name: `str`
            The name of S3 Bucket where the Braket run results will be saved.
        aws_region: `str`
            The aws region in which the QPU/simulator is located. Defaults to the
            region set in aws cli config.
        folder_name: `str`
            The name of the folder in the s3 bucket that will contain the results
            from the tasks performed in this run.
        """
        
        self.device_name = device_name
        self.device_location = 'aws'
        self.s3_bucket_name = s3_bucket_name
        self.aws_region = aws_region
        self.folder_name = folder_name
        
        self.provider_connected = None
        self.qpu_connected = None
    
    def check_connection(self) -> bool:
        
        self.provider_connected = self._check_provider_connection()

        if self.provider_connected == False:
            return self.provider_connected
        
        # Only QPUs that are available for the specified aws region on Braket 
        # will be shown. We filter out QPUs that do not work with the circuit model
        sess_devices = self.aws_session.search_devices()
        
        device_filter = np.multiply(
            [each_dict['deviceStatus'] == 'ONLINE' for each_dict in sess_devices],
            [each_dict['providerName'] != 'D-Wave Systems' for each_dict in sess_devices]
        )
        active_devices = np.array(sess_devices)[device_filter].tolist()
        
        self.available_qpus = [backend_dict['deviceArn']
                               for backend_dict in active_devices]

        if self.device_name == '':
            return self.provider_connected

        self.qpu_connected = self._check_backend_connection()

        if self.provider_connected and self.qpu_connected:
            return True
        else:
            return False
        
    def _check_backend_connection(self) -> bool:
        
        if self.device_name in self.available_qpus:
            self.backend_device = AwsDevice(self.device_name, self.aws_session)
        else:
            print(
                '''
                These are the only available devices for this aws region: 
                {}. Try a different aws region if the device you are looking 
                for is not in the list.'
                '''.format(self.available_qpus))
            return False
        
        # Get the maximum number of qubits for that particular AWS Backend
        try:
            self.n_qubits = self.backend_device.properties.paradigm.qubitCount
        except AttributeError:
            print("OpenQAOA is unable to retrieve the number of qubits available in the selected QPU.")
            return False
        else:
            return True
    
    def _check_provider_connection(self) -> bool:
        
        try:
            sess = Session(region_name = self.aws_region)
            self.aws_session = AwsSession(sess, default_bucket=self.s3_bucket_name)
            self.aws_region = self.aws_session.region
            self.s3_bucket_name = self.aws_session.default_bucket()
            return True
        except NoRegionError:
            self.aws_session = None
            return True
        except Exception as e:
            print('An Exception has occured when trying to connect with the provider. You are required to authenticate through the AWS CLI in order to connect to the Braket QPUs. Please check if you have properly set it up. See: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html for further details. : {}'.format(e))
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
                            aws_region: str = None) -> dict:
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
                    'folder_name': folder_name}
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
    else:
        raise ValueError(f'Invalid device location, Choose from: {location}')

    return device_class(device_name=name, **kwargs)