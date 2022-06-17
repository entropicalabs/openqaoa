#   Copyright 2022 Entropica Labs
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import unittest
import json

from openqaoa.devices import DeviceQiskit


"""
TODO: Logic for test_check_connection_aws_no_device_provided_credentials while 
correct. Returning true for the check_connection() function if no device was
provided might be a strange behaviour since check_connection() is also expected
to return true if the device provided was correct, and false if invalid.
"""


class TestingDeviceQiskit(unittest.TestCase):

    """This tests checks that Object used to access IBMQ and their available
    QPUs can be established.

    For any tests using provided credentials, the tests will only pass if those
    details provided are correct/valid with IBMQ.

    Please ensure that the provided api token in the crendentials.json is 
    correct.
    Note that the defaults for hub, group, project and valid backend can also be 
    changed in crendentials.json, in most cases, they can be left alone. 
    All of these can be found in your IBMQ Account Page.
    """

    def setUp(self):

        with open('./tests/credentials.json', 'r') as f:
            json_obj = json.load(f)['QISKIT']
            self.API_TOKEN = json_obj['API_TOKEN']
            self.HUB = json_obj['HUB']
            self.GROUP = json_obj['GROUP']
            self.PROJECT = json_obj['PROJECT']

        if self.API_TOKEN == "None":
            raise ValueError(
                "Please provide an appropriate API TOKEN in crendentials.json.")
        elif self.HUB == "None":
            raise ValueError(
                "Please provide an appropriate IBM HUB name in crendentials.json.")
        elif self.GROUP == "None":
            raise ValueError(
                "Please provide an appropriate IBMQ GROUP name in crendentials.json.")
        elif self.PROJECT == "None":
            raise ValueError(
                "Please provide an appropriate IBMQ Project name in crendentials.json.")

    def test_check_connection_provider_no_backend_wrong_credentials(self):

        device_obj = DeviceQiskit(device_name='', 
                                  api_token='', 
                                  hub='', group='',
                                  project='')

        self.assertEqual(device_obj.check_connection(), False)
        self.assertEqual(device_obj.provider_connected, False)
        self.assertEqual(device_obj.qpu_connected, None)

    def test_check_connection_provider_no_backend_provided_credentials(self):

        device_obj = DeviceQiskit(device_name='', 
                                  api_token=self.API_TOKEN,
                                  hub=self.HUB, group=self.GROUP,
                                  project=self.PROJECT)

        self.assertEqual(device_obj.check_connection(), True)
        self.assertEqual(device_obj.provider_connected, True)
        self.assertEqual(device_obj.qpu_connected, None)

    def test_check_connection_provider_right_backend_provided_credentials(self):

        device_obj = DeviceQiskit(device_name='', 
                                  api_token=self.API_TOKEN,
                                  hub=self.HUB, group=self.GROUP,
                                  project=self.PROJECT)

        device_obj.check_connection()
        valid_qpu_name = device_obj.available_qpus[0]

        device_obj = DeviceQiskit(device_name=valid_qpu_name, 
                                  api_token=self.API_TOKEN, 
                                  hub=self.HUB, group=self.GROUP,
                                  project=self.PROJECT)

        self.assertEqual(device_obj.check_connection(), True)
        self.assertEqual(device_obj.provider_connected, True)
        self.assertEqual(device_obj.qpu_connected, True)

    def test_check_connection_provider_wrong_backend_provided_credentials(self):

        device_obj = DeviceQiskit(device_name='random_invalid_backend', 
                                  api_token=self.API_TOKEN,
                                  hub=self.HUB, group=self.GROUP,
                                  project=self.PROJECT)

        self.assertEqual(device_obj.check_connection(), False)
        self.assertEqual(device_obj.provider_connected, True)
        self.assertEqual(device_obj.qpu_connected, False)


if __name__ == '__main__':
    unittest.main()
