from qiskit_ibm_provider import IBMProvider
from qiskit_aer import AerSimulator
from typing import List
from openqaoa.backends.basedevice import DeviceBase


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

    def __init__(
        self,
        device_name: str,
        hub: str = None,
        group: str = None,
        project: str = None,
        as_emulator: bool = False,
    ):
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
        self.device_location = "ibmq"
        self.hub = hub
        self.group = group
        self.project = project
        self.as_emulator = as_emulator

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

        self.available_qpus = [backend.name for backend in self.provider.backends()]

        if self.device_name == "":
            return self.provider_connected

        self.qpu_connected = self._check_backend_connection()

        if self.provider_connected and self.qpu_connected:
            return True
        else:
            return False

    def _check_backend_connection(self) -> bool:
        """Private method for checking connection with backend(s)."""

        if self.device_name in self.available_qpus:
            self.backend_device = self.provider.get_backend(self.device_name)
            self.n_qubits = self.backend_device.configuration().n_qubits
            if self.as_emulator is True:
                self.backend_device = AerSimulator.from_backend(self.backend_device)
            return True
        else:
            print(f"Please choose from {self.available_qpus} for this provider")
            return False

    def _check_provider_connection(self) -> bool:
        """
        Private method for checking connection with provider.
        """

        try:
            # Use default
            self.provider = IBMProvider()
            # Unless exact instance is specified
            if all([self.hub, self.group, self.project]):
                instance_name = self.hub + "/" + self.group + "/" + self.project
                assert instance_name in self.provider.instances()
                self.provider = IBMProvider(instance=instance_name)
            elif any([self.hub, self.group, self.project]):
                # if only partially specified, print warning.
                raise Exception(
                    "You've only partially specified the instance name. Either"
                    "the hub, group or project is missing. hub: {}, group: {}, project: {}.\n"
                    "The default instance will be used instead. (This default can "
                    "be specified when doing `IBMProvider.save_account`)"
                )
            return True
        except Exception as e:
            print(
                "An Exception has occured when trying to connect with the provider."
                "Please note that you are required to set up your IBMQ account locally first."
                "See: https://quantum-computing.ibm.com/lab/docs/iql/manage/account/ibmq "
                "for how to save your IBMQ account locally. \n {}".format(e)
            )
            return False

    def connectivity(self) -> List[List[int]]:
        return self.backend_device.configuration().coupling_map
