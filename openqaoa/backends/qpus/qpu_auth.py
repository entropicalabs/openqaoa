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
from qiskit import IBMQ
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._engagement_manager import EngagementManager
from pyquil import get_qc


class AccessObjectBase(metaclass=abc.ABCMeta):
    """An object that contains the relevant information required to access 
    certain backends. Other Access Objects have to inherit from this object.
    """

    @abc.abstractmethod
    def check_connection(self) -> bool:
        """This method should allow a user to easily check if the credentials
        provided to access the remote QPU is valid.

        Returns:
            bool: True if a connection can be established. If False, the error 
                  should be logged and printable. (Not creating an exception 
                  here is good for extendibility. i.e. able to try multiple
                  providers without exiting the program.)
        """
        pass


class AccessObjectQiskit(AccessObjectBase):
    """Contains the required information and methods needed to access remote
    qiskit QPUs.

    Attributes:
        available_qpus (list): When connection to a provider is established, 
                               this attribute contains a list of backend names 
                               which can be used to access the selected backend 
                               by reinitialising the Access Object with the 
                               name of the available backend as input to the 
                               selected_backend parameter.
    """

    def __init__(self, api_token: str, hub: str, group: str, project: str,
                 selected_qpu: str = '') -> None:
        """A majority of the input parameters required for this can be found in
        the user's IBMQ Experience account.

        Parameters
        ----------
        api_token: str    
            Valid IBMQ Experience Token.
        hub: str
            Valid IBMQ hub name.
        group: str
            Valid IBMQ group name. 
        project: str
            The name of the project for which the experimental data will be 
            saved in on IBMQ's end.
        selected_qpu: str
            The name of the QPU in which the user would like to connect with.
        """

        self.api_token = api_token
        self.hub = hub
        self.group = group
        self.project = project
        self.selected_qpu = selected_qpu

        self.provider_connected = None
        self.qpu_connected = None

    def check_connection(self) -> bool:
        """This method should allow a user to easily check if the credentials
        provided to access the remote QPU is valid.

        If no backend was specified in initialisation of object, just runs
        a test connection without a specific backend.
        If backend was specified, checks if connection to that backend
        can be established.

        Returns:
            bool: True if successfully connected to IBMQ or IBMQ and the QPU 
                  backend if it was specified.
                  False if unable to connect to IBMQ or failure in the attempt 
                  to connect to the specified backend.
        """

        self.provider_connected = self._check_provider_connection()

        if self.provider_connected == False:
            return self.provider_connected

        self.available_qpus = [backend.name()
                               for backend in self.provider.backends()]

        if self.selected_qpu == '':
            return self.provider_connected

        self.qpu_connected = self._check_backend_connection()

        if self.provider_connected and self.qpu_connected:
            return True
        else:
            return False

    def _check_backend_connection(self) -> bool:
        """Private method for checking connection with backend(s).
        """

        if self.selected_qpu in self.available_qpus:
            self.backend_qpu = self.provider.get_backend(self.selected_qpu)
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
            if IBMQ.active_account() is None or IBMQ.active_account()['token'] != self.api_token:
                self.provider = IBMQ.enable_account(self.api_token, hub=self.hub,
                                                    group=self.group,
                                                    project=self.project)
            else:
                self.provider = IBMQ.get_provider(hub=self.hub, group=self.group,
                                                  project=self.project)

            return True

        except Exception as e:
            print('An Exception has occured when trying to connect with the \
            provider: {}'.format(e))
            return False


class AccessObjectPyQuil(AccessObjectBase):
    """
    Contains the required information and methods needed to access remote
    Rigetti QPUs.

    Attributes:
        available_qpus (list): When connection to AWS is established, this 
                               attribute contains a list of device names which 
                               can be used to access the selected device by 
                               reinitialising the Access Object with the name 
                               of the available device as input to the 
                               selected_device parameter.
    """

    def __init__(self, name: str, as_qvm: bool = None, noisy: bool = None,
                 compiler_timeout: float = 20.0,
                 execution_timeout: float = 20.0,
                 client_configuration: QCSClientConfiguration = None,
                 endpoint_id: str = None,
                 engagement_manager: EngagementManager = None) -> None:
        """
        Parameters
        ----------
        name: str 
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

        self.name = name
        self.as_qvm = as_qvm
        self.noisy = noisy
        self.compiler_timeout = compiler_timeout
        self.execution_timeout = execution_timeout
        self.client_configuration = client_configuration
        self.endpoint_id = endpoint_id
        self.engagement_manager = engagement_manager

        self.quantum_computer = get_qc(name=self.name, as_qvm=self.as_qvm, noisy=self.noisy,
                                       compiler_timeout=self.compiler_timeout, execution_timeout=self.execution_timeout,
                                       client_configuration=self.client_configuration, endpoint_id=self.endpoint_id, engagement_manager=self.engagement_manager)

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
