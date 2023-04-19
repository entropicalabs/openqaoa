from qiskit_ibm_provider import IBMProvider
from openqaoa_qiskit.backends import DeviceQiskit
import fireopal


class DeviceFireOpal(DeviceQiskit):
    """
    Contains the required information and methods needed to access remote
    IBMQ QPUs via FireOpal.

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
    ):
        """The user's IBMQ account has to be authenticated through qiskit in
        order to use this backend. This can be done through `IBMQ.save_account`.

        See: https://quantum-computing.ibm.com/lab/docs/iql/manage/account/ibmq

        Parameters
        ----------
        device_name: `str`
            The name of the device to be used. The available devices can be
            found by executing the check_connection method.
        hub: `str`
            The IBMQ hub to be used. If not specified, the default hub is used.
        group: `str`
            The IBMQ group to be used. If not specified, the default group is used.
        project: `str`
            The IBMQ project to be used. If not specified, the default project is used.
        """
        super().__init__(device_name, hub, group, project, as_emulator=False)

    @property
    def credentials(self):
        """
        Returns the credentials for the IBMQ account.
        """
        token, hub, group, project = None, None, None, None
        if self.provider_connected is True:
            account = self.provider.active_account()
            token = account["token"]
            hub, group, project = account["instance"].split("/")
        creds = {
            "token": token,
            "hub": hub,
            "group": group,
            "project": project,
        }
        return creds

    def check_connection(self) -> bool:
        """
        Checks if the user's IBMQ account is authenticated and if the selected
        backend is available.

        Returns
        -------
        `bool`
            True if the user's IBMQ account is authenticated and the selected
            backend is available, False otherwise.
        """
        super().check_connection()

        # Set available backends accessible through IBMQ and FireOpal
        self.available_qpus = fireopal.show_supported_devices(
            credentials=self.credentials
        )["supported_devices"]
