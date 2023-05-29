import qb.core
from openqaoa.backends.devices_core import DeviceBase


class DeviceQristal(DeviceBase):
    def __init__(self, device_name):
        # Create a quantum computing session using the QB SDK
        self.sim = qb.core.session()
        # Set up meaningful defaults for session parameters
        self.sim.qb12()
        # Choose a simulator backend
        self.sim.acc = "qpp"
        self.name=device_name
        
        self.provider_connected = True
        self.qpu_connected = True

    def check_connection(self):
        return True

    def _check_backend_connection(self):
        return True

    def _check_provider_connection(self):
        return True
    
    def connectivity(self):
        return None
