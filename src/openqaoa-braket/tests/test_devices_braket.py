import unittest
import pytest

from openqaoa_braket.backends import DeviceAWS


class TestingDeviceAWS(unittest.TestCase):
    """These tests check the Object used to access AWS Braket and their
    available QPUs can be established.

    For any tests using provided credentials, the tests will only pass if those
    details provided are correct/valid with AWS Braket.
    """

    @pytest.mark.braket_api
    def test_changing_aws_region(self):
        device_obj = DeviceAWS(
            device_name="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            aws_region="us-east-1",
        )

        device_obj.check_connection()
        default_region = device_obj.aws_region

        self.assertEqual("us-east-1", default_region)

        device_obj = DeviceAWS(
            device_name="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            aws_region="us-west-1",
        )

        device_obj.check_connection()
        custom_region = device_obj.aws_region

        self.assertEqual("us-west-1", custom_region)

    @pytest.mark.braket_api
    def test_changing_s3_bucket_names(self):
        device_obj = DeviceAWS(
            device_name="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            s3_bucket_name="random_new_name",
        )

        device_obj.check_connection()
        custom_bucket = device_obj.s3_bucket_name

        self.assertEqual("random_new_name", custom_bucket)

    @pytest.mark.braket_api
    def test_check_connection_provider_no_backend_provided_credentials(self):
        """
        If no information about the device name, but the credentials
        used are correct, check_connection should return True.
        The provider_connected attribute should be updated to True.
        """

        device_obj = DeviceAWS(device_name="")

        self.assertEqual(device_obj.check_connection(), True)
        self.assertEqual(device_obj.provider_connected, True)
        self.assertEqual(device_obj.qpu_connected, None)

    @pytest.mark.braket_api
    def test_check_connection_provider_right_backend_provided_credentials(self):
        """
        If the correct device name is provided and the credentials
        used are correct, check_connection should return True.
        The provider_connected attribute should be updated to True.
        The qpu_connected attribute should be updated to True.
        """

        device_obj = DeviceAWS(device_name="")

        device_obj.check_connection()
        valid_qpu_name = device_obj.available_qpus[0]

        device_obj = DeviceAWS(device_name=valid_qpu_name)

        self.assertEqual(device_obj.check_connection(), True)
        self.assertEqual(device_obj.provider_connected, True)
        self.assertEqual(device_obj.qpu_connected, True)

    @pytest.mark.braket_api
    def test_check_connection_provider_wrong_backend_provided_credentials(self):
        """
        If device name provided is incorrect, and not empty, and the credentials
        used are correct, check_connection should return False.
        The provider_connected attribute should be updated to True.
        The qpu_connected attribute should be updated to False.
        """

        device_obj = DeviceAWS(device_name="random_invalid_backend")

        self.assertEqual(device_obj.check_connection(), False)
        self.assertEqual(device_obj.provider_connected, True)
        self.assertEqual(device_obj.qpu_connected, False)


if __name__ == "__main__":
    unittest.main()
