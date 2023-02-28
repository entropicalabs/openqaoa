from typing import List

from azure.quantum.qiskit import AzureQuantumProvider
from openqaoa.backends.devices_core import DeviceBase


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
            The resource_id of the Workplace
        az_location: `str`
            The location of the Azure Workplace. e.g. "westus"
        """

        self.resource_id = resource_id
        self.location = az_location
        self.device_name = device_name
        self.device_location = "azure"

        self.provider_connected = None
        self.qpu_connected = None

    def check_connection(self):
        """ """

        self.provider_connected = self._check_provider_connection()

        if self.provider_connected == False:
            return self.provider_connected

        self.available_qpus = [backend.name() for backend in self.provider.backends()]

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
            return True
        else:
            print(f"Please choose from {self.available_qpus} for this provider")
            return False

    def _check_provider_connection(self) -> bool:
        """
        Private method for checking connection with provider.
        """

        try:
            self.provider = AzureQuantumProvider(
                resource_id=self.resource_id, location=self.location
            )

            return True

        except ValueError as e:
            print(
                "Either the resource id or location specified was invalid: {}".format(e)
            )
            return False

        except Exception as e:
            print(
                "An Exception has occured when trying to connect with the \
            provider: {}".format(
                    e
                )
            )
            return False

    def connectivity(self) -> List[List[int]]:
        return self.backend_device.configuration().coupling_map
