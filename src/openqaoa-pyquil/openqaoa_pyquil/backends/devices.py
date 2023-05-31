from typing import List
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._engagement_manager import EngagementManager
from pyquil import get_qc

from openqaoa.backends.devices_core import DeviceBase


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

    def __init__(
        self,
        device_name: str,
        as_qvm: bool = None,
        noisy: bool = None,
        compiler_timeout: float = 20.0,
        execution_timeout: float = 20.0,
        client_configuration: QCSClientConfiguration = None,
        endpoint_id: str = None,
        engagement_manager: EngagementManager = None,
    ):
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
        self.device_location = "qcs"
        self.as_qvm = as_qvm
        self.noisy = noisy
        self.compiler_timeout = compiler_timeout
        self.execution_timeout = execution_timeout
        self.client_configuration = client_configuration
        self.endpoint_id = endpoint_id
        self.engagement_manager = engagement_manager

        self.quantum_computer = get_qc(
            name=self.device_name,
            as_qvm=self.as_qvm,
            noisy=self.noisy,
            compiler_timeout=self.compiler_timeout,
            execution_timeout=self.execution_timeout,
            client_configuration=self.client_configuration,
            endpoint_id=self.endpoint_id,
            engagement_manager=self.engagement_manager,
        )
        self.n_qubits = len(self.quantum_computer.qubits())

    def check_connection(self) -> bool:
        """This method should allow a user to easily check if the credentials
        provided to access the remote QPU is valid.

        If no device was specified in initialisation of object, just runs
        a test connection without a specific device.
        If device was specified, checks if connection to that device
        can be established.

        TODO :
        Accessing Rigetti's QCS is currently unsupported, so this part
        is empty until that is figured out.
        """

        return True

    def connectivity(self) -> List[List[int]]:
        # returns a networkx graph of qubit topology
        G = self.quantum_computer.qubit_topology()
        connectivity_as_list = list(G.edges())
        return connectivity_as_list
