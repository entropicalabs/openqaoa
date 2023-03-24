import abc
import logging

logging.getLogger().setLevel(logging.ERROR)

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

    @abc.abstractmethod
    def connectivity(self):
        """
        obtain the device connectivity as a list of qubit pairs

        Returns
        -------
            List[List[int]]
        """
        pass