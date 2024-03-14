import unittest
import itertools
import pytest

from openqaoa_qiskit.backends import DeviceQiskit


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


if __name__ == "__main__":
    unittest.main()
