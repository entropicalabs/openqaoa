import unittest
import json
import os
import itertools
import subprocess
import pytest

from openqaoa.backends import DeviceLocal
from openqaoa.backends.devices_core import SUPPORTED_LOCAL_SIMULATORS
from openqaoa_qiskit.backends import DeviceQiskit
from openqaoa_braket.backends import DeviceAWS
from openqaoa_azure.backends import DeviceAzure


class TestingDeviceQiskit(unittest.TestCase):

    """These tests check the Object used to access IBMQ and their available
    QPUs can be established.
    For any tests using provided credentials, the tests will only pass if those
    details provided are correct/valid with IBMQ.

    Please ensure that the provided api token, hub, group and project in the
    crendentials.json are correct.
    All of these can be found in your IBMQ Account Page.
    """

    @pytest.mark.api
    def setUp(self):

        self.HUB = "ibm-q"
        self.GROUP = "open"
        self.PROJECT = "main"
        self.INSTANCE = "ibm-q/open/main"

    @pytest.mark.api
    def test_changing_provider(self):

        """
        This test checks that the specified hub,group and project in the
        initialisation of DeviceQiskit changes the provider to the appropriate
        destination.
        """

        device_obj = DeviceQiskit(device_name="ibmq_manila")
        device_obj.check_connection()

        provider_instances = device_obj.provider.instances()

        if len(provider_instances) >= 2:

            for each_item in provider_instances[:2]:

                [hub, group, project] = each_item.split("/")
                device_obj2 = DeviceQiskit(
                    device_name="ibmq_manila", hub=hub, group=group, project=project
                )
                device_obj2.check_connection()

                self.assertEqual(device_obj2.provider._account.instance, each_item)

    @pytest.mark.api
    def test_check_connection_provider_no_backend_wrong_hub_group_project(self):

        """
        Hub, group and project must always be specified together.
        If either the hub, group or project is wrongly specified, check_connection should
        return False.
        If not all 3 are specified, check_connection should return False.
        The provider_connected attribute should be updated to False.
        Since the API Token is loaded from save_account, the api token will be
        checked by Qiskit.
        """

        for each_combi in itertools.product(
            ["invalid_hub", None], ["invalid_group", None], ["invalid_project", None]
        ):

            if each_combi != (None, None, None):

                device_obj = DeviceQiskit(
                    device_name="",
                    hub=each_combi[0],
                    group=each_combi[1],
                    project=each_combi[2],
                )

                self.assertEqual(device_obj.check_connection(), False)
                self.assertEqual(device_obj.provider_connected, False)
                self.assertEqual(device_obj.qpu_connected, None)

    @pytest.mark.api
    def test_check_connection_provider_no_backend_provided_credentials(self):

        """
        If no information about the device name, but the credentials
        used are correct, check_connection should return True.
        The provider_connected attribute should be updated to True.
        """

        device_obj = DeviceQiskit(
            device_name="", hub=self.HUB, group=self.GROUP, project=self.PROJECT
        )

        self.assertEqual(device_obj.check_connection(), True)
        self.assertEqual(device_obj.provider_connected, True)
        self.assertEqual(device_obj.qpu_connected, None)

    @pytest.mark.api
    def test_check_connection_provider_right_backend_provided_credentials(self):

        """
        If the correct device name is provided and the credentials
        used are correct, check_connection should return True.
        The provider_connected attribute should be updated to True.
        The qpu_connected attribute should be updated to True.
        """

        device_obj = DeviceQiskit(
            device_name="", hub=self.HUB, group=self.GROUP, project=self.PROJECT
        )

        device_obj.check_connection()
        valid_qpu_name = device_obj.available_qpus[0]

        device_obj = DeviceQiskit(
            device_name=valid_qpu_name,
            hub=self.HUB,
            group=self.GROUP,
            project=self.PROJECT,
        )

        self.assertEqual(device_obj.check_connection(), True)
        self.assertEqual(device_obj.provider_connected, True)
        self.assertEqual(device_obj.qpu_connected, True)

    @pytest.mark.api
    def test_check_connection_provider_wrong_backend_provided_credentials(self):

        """
        If device name provided is incorrect, and not empty, and the credentials
        used are correct, check_connection should return False.
        The provider_connected attribute should be updated to True.
        The qpu_connected attribute should be updated to False.
        """

        device_obj = DeviceQiskit(
            device_name="random_invalid_backend",
            hub=self.HUB,
            group=self.GROUP,
            project=self.PROJECT,
        )

        self.assertEqual(device_obj.check_connection(), False)
        self.assertEqual(device_obj.provider_connected, True)
        self.assertEqual(device_obj.qpu_connected, False)


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
    def test_check_connection_provider_no_backend_provided_resource_id_and_az_location(
        self,
    ):

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
    def test_check_connection_provider_right_backend_provided_resource_id_and_az_location(
        self,
    ):

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
    def test_check_connection_provider_wrong_backend_provided_resource_id_and_az_location(
        self,
    ):

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
