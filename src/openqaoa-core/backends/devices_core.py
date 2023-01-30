from __future__ import annotations
import abc
import logging

logging.getLogger().setLevel(logging.ERROR)

SUPPORTED_LOCAL_SIMULATORS = [
    'qiskit.qasm_simulator', 'qiskit.shot_simulator',
    'qiskit.statevector_simulator','vectorized',
    'pyquil.statevector_simulator', 'analytical_simulator'
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