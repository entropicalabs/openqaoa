import unittest
import json
import itertools
import subprocess
import pytest

import openqaoa
from openqaoa_azure.backends import DeviceAzure


class TestingDeviceAzure(unittest.TestCase):

    """These tests check the Object used to access Azure and their
    available QPUs can be established.

    For any tests using provided credentials, the tests will only pass if those
    details provided are correct/valid with Azure.
    """

    @pytest.mark.api
    def setUp(self):

        bashCommand = "az resource list"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        if error is not None:
            print(error)
            raise Exception(
                "You must have the Azure CLI installed and must be logged in to use the Azure Quantum Backends"
            )
        else:
            output_json = json.loads(output)
            output_json_s = [
                each_json
                for each_json in output_json
                if each_json["name"] == "TestingOpenQAOA"
            ][0]
            self.RESOURCE_ID = output_json_s["id"]
            self.AZ_LOCATION = output_json_s["location"]

    @pytest.mark.api
    def test_check_connection_provider_no_resource_id_or_az_location(self):

        """
        If no information about about the workspace is provided, the resource id
        or az location, check_connection and provider_connected should return False.
        """

        for resource_id, az_location in itertools.product(
            ["", self.RESOURCE_ID], ["", self.AZ_LOCATION]
        ):

            if not (
                resource_id == self.RESOURCE_ID and az_location == self.AZ_LOCATION
            ):

                device_obj = DeviceAzure(
                    device_name="", resource_id=resource_id, az_location=az_location
                )

                self.assertEqual(device_obj.check_connection(), False)
                self.assertEqual(device_obj.provider_connected, False)
                self.assertEqual(device_obj.qpu_connected, None)
        
    @pytest.mark.api      
    def test_check_connection_provider_no_backend_provided_resource_id_and_az_location(self):
        
        """
        If no information about the device name, but the resource id and azure 
        location used are correct, check_connection should return True.
        The provider_connected attribute should be updated to True.
        """

        device_obj = DeviceAzure(
            device_name="", resource_id=self.RESOURCE_ID, az_location=self.AZ_LOCATION
        )

        self.assertEqual(device_obj.check_connection(), True)
        self.assertEqual(device_obj.provider_connected, True)
        self.assertEqual(device_obj.qpu_connected, None)

    @pytest.mark.api
    def test_check_connection_provider_right_backend_provided_resource_id_and_az_location(self):

        """
        If the correct device name is provided and the resource id and azure 
        location used are correct, check_connection should return True.
        The provider_connected attribute should be updated to True.
        The qpu_connected attribute should be updated to True.
        """

        device_obj = DeviceAzure(
            device_name="", resource_id=self.RESOURCE_ID, az_location=self.AZ_LOCATION
        )

        device_obj.check_connection()
        valid_qpu_name = device_obj.available_qpus[0]

        device_obj = DeviceAzure(
            device_name=valid_qpu_name,
            resource_id=self.RESOURCE_ID,
            az_location=self.AZ_LOCATION,
        )

        self.assertEqual(device_obj.check_connection(), True)
        self.assertEqual(device_obj.provider_connected, True)
        self.assertEqual(device_obj.qpu_connected, True)

    @pytest.mark.api
    def test_check_connection_provider_wrong_backend_provided_resource_id_and_az_location(self):
        
        """
        If device name provided is incorrect, and not empty, and the resource id 
        and azure location used are correct, check_connection should return False.
        The provider_connected attribute should be updated to True.
        The qpu_connected attribute should be updated to False.
        """

        device_obj = DeviceAzure(
            device_name="invalid_backend",
            resource_id=self.RESOURCE_ID,
            az_location=self.AZ_LOCATION,
        )

        self.assertEqual(device_obj.check_connection(), False)
        self.assertEqual(device_obj.provider_connected, True)
        self.assertEqual(device_obj.qpu_connected, False)


if __name__ == "__main__":
    unittest.main()
