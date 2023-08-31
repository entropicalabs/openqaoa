import unittest

from openqaoa.backends import DeviceLocal
from openqaoa.backends.devices_core import SUPPORTED_LOCAL_SIMULATORS


class TestingDeviceLocal(unittest.TestCase):

    """
    This tests check that the Device Object created for local devices have the
    appropriate behaviour.
    """

    def test_supported_device_names(self):
        for each_device_name in SUPPORTED_LOCAL_SIMULATORS:
            device_obj = DeviceLocal(each_device_name)

            self.assertEqual(device_obj.check_connection(), True)

    def test_unsupported_device_names(self):
        device_obj = DeviceLocal("unsupported_device")

        self.assertEqual(device_obj.check_connection(), False)


if __name__ == "__main__":
    unittest.main()
