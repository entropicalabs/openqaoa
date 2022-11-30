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
import os
import pytest
import itertools

from openqaoa.devices import DeviceQiskit, DeviceLocal, DeviceAWS
from openqaoa.backends.qaoa_backend import get_qaoa_backend, DEVICE_NAME_TO_OBJECT_MAPPER, DEVICE_ACCESS_OBJECT_MAPPER
from openqaoa.qaoa_parameters import Hamiltonian, create_qaoa_variational_params
from openqaoa.qaoa_parameters.baseparams import QAOACircuitParams
from openqaoa.utilities import X_mixer_hamiltonian
from openqaoa.devices import create_device
from openqaoa.basebackend import QAOABaseBackendShotBased

from qiskit import IBMQ


SUPPORTED_LOCAL_SIMULATORS = list(DEVICE_NAME_TO_OBJECT_MAPPER.keys())

class TestingDeviceLocal(unittest.TestCase):
    
    """
    These tests check that the Device Object created for local devices have the
    appropriate behaviour.
    """
    
    def test_supported_device_names(self):
        
        for each_device_name in SUPPORTED_LOCAL_SIMULATORS:
            device_obj = DeviceLocal(each_device_name)
            
            self.assertEqual(device_obj.check_connection(), True)
    
    def test_unsupported_device_names(self):
        
        device_obj = DeviceLocal('unsupported_device')
        
        self.assertEqual(device_obj.check_connection(), False)

    def test_get_counts_and_expectation_n_shots(self):
        """ 
        Check that the .get_counts admit n_shots as an argument, and works properly for the backend of all local devices.
        Also check that .expectation and .expecation_w_uncertainty methods admit n_shots as an argument for the QAOABaseBackendShotBased backends.
        """

        for device_name in SUPPORTED_LOCAL_SIMULATORS:

            device = create_device(location='local', name=device_name)

            cost_hamil = Hamiltonian.classical_hamiltonian([[0, 1]], [1], constant=0)
            mixer_hamil = X_mixer_hamiltonian(2)

            circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=1)
            variational_params_std = create_qaoa_variational_params(circuit_params, 'standard', 'ramp')
            backend = get_qaoa_backend(circuit_params=circuit_params, device=device, n_shots=1000)
                        
            assert sum(backend.get_counts(params=variational_params_std, n_shots=58).values())==58, "`n_shots` is not being respected for the local simulator `{}` when calling backend.get_counts(n_shots=58).".format(device_name)
            if isinstance(backend, QAOABaseBackendShotBased): 
                try: backend.expectation(params=variational_params_std, n_shots=58)
                except Exception: raise Exception("backend.expectation does not admit `n_shots` as an argument for the local simulator `{}`.".format(device_name))
                try: backend.expectation_w_uncertainty(params=variational_params_std, n_shots=58)
                except Exception: raise Exception("backend.expectation_w_uncertainty does not admit `n_shots` as an argument for the local simulator `{}`.".format(device_name))

class TestingDeviceQPUs(unittest.TestCase):
    """ These tests check methods of the backend of the Device Object for QPUs."""

    def setUp(self):
        try:
            opened_f = open('./tests/credentials.json', 'r')
        except FileNotFoundError:
            opened_f = open('credentials.json', 'r')
                
        with opened_f as f:
            json_obj = json.load(f)['QISKIT']
            
            try:
                api_token = os.environ['IBMQ_TOKEN']
                self.HUB = os.environ['IBMQ_HUB']
                self.GROUP = os.environ['IBMQ_GROUP']
                self.PROJECT = os.environ['IBMQ_PROJECT']
            except Exception:
                api_token = json_obj['API_TOKEN']
                self.HUB = json_obj['HUB']
                self.GROUP = json_obj['GROUP']
                self.PROJECT = json_obj['PROJECT']

        if api_token == "YOUR_API_TOKEN_HERE":
            raise ValueError(
                "Please provide an appropriate API TOKEN in crendentials.json.")
        elif self.HUB == "IBMQ_HUB":
            raise ValueError(
                "Please provide an appropriate IBM HUB name in crendentials.json.")
        elif self.GROUP == "IBMQ_GROUP":
            raise ValueError(
                "Please provide an appropriate IBMQ GROUP name in crendentials.json.")
        elif self.PROJECT == "IBMQ_PROJECT":
            raise ValueError(
                "Please provide an appropriate IBMQ Project name in crendentials.json.")
            
        IBMQ.save_account(token = api_token, hub=self.HUB, 
                          group=self.GROUP, project=self.PROJECT, 
                          overwrite=True)

    def test_get_counts_and_expectation_n_shots(self):
        """ Check that the .get_counts, .expectation and .expecation_w_uncertainty methods admit n_shots as an argument for the backends of all QPUs. """

        list_device_attributes = [ 
                                    {'device_name': 'ibmq_qasm_simulator', 'hub': self.HUB, 'group': self.GROUP, 'project': self.PROJECT}, 
                                    {'device_name': "2q-qvm", 'as_qvm': True, 'execution_timeout': 3, 'compiler_timeout': 3},       
                                    {'device_name': 'arn:aws:braket:::device/quantum-simulator/amazon/sv1'}
                                ]

        for (device, backend), device_attributes in zip(DEVICE_ACCESS_OBJECT_MAPPER.items(), list_device_attributes):
            cost_hamil = Hamiltonian.classical_hamiltonian([[0, 1]], [1], constant=0)
            mixer_hamil = X_mixer_hamiltonian(2)
            circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=1)
            device = device(**device_attributes)

            variational_params_std = create_qaoa_variational_params(circuit_params, 'standard', 'ramp')
            backend = backend(circuit_params = circuit_params, device = device, cvar_alpha = 1, n_shots=100, prepend_state = None, append_state = None, init_hadamard = True)

            ## TODO: get rid of the ``` if str(e) != "PyQuil does not support n_shots in get_counts" ``` 

            # Check that the .get_counts method admits n_shots as an argument
            try: assert sum(backend.get_counts(params=variational_params_std, n_shots=58).values()) == 58, "`n_shots` is not being respected for the QPU `{}` when calling backend.get_counts(n_shots=58).".format(device_attributes['device_name'])
            except Exception as e: 
                if str(e) != "PyQuil does not support n_shots in get_counts": raise e 

            try: backend.expectation(params=variational_params_std, n_shots=58)
            except Exception as e:
                if str(e) != "PyQuil does not support n_shots in get_counts":  
                    raise Exception("backend.expectation does not admit `n_shots` as an argument for the QPU `{}`.".format(device_attributes['device_name']))
            try: backend.expectation_w_uncertainty(params=variational_params_std, n_shots=58)
            except Exception: 
                if str(e) != "PyQuil does not support n_shots in get_counts": 
                    raise Exception("backend.expectation_w_uncertainty does not admit `n_shots` as an argument for the QPU `{}`.".format(device_attributes['device_name']))


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

        try:
            opened_f = open('./tests/credentials.json', 'r')
        except FileNotFoundError:
            opened_f = open('credentials.json', 'r')
                
        with opened_f as f:
            json_obj = json.load(f)['QISKIT']
            
            try:
                api_token = os.environ['IBMQ_TOKEN']
                self.HUB = os.environ['IBMQ_HUB']
                self.GROUP = os.environ['IBMQ_GROUP']
                self.PROJECT = os.environ['IBMQ_PROJECT']
            except Exception:
                api_token = json_obj['API_TOKEN']
                self.HUB = json_obj['HUB']
                self.GROUP = json_obj['GROUP']
                self.PROJECT = json_obj['PROJECT']

        if api_token == "YOUR_API_TOKEN_HERE":
            raise ValueError(
                "Please provide an appropriate API TOKEN in crendentials.json.")
        elif self.HUB == "IBMQ_HUB":
            raise ValueError(
                "Please provide an appropriate IBM HUB name in crendentials.json.")
        elif self.GROUP == "IBMQ_GROUP":
            raise ValueError(
                "Please provide an appropriate IBMQ GROUP name in crendentials.json.")
        elif self.PROJECT == "IBMQ_PROJECT":
            raise ValueError(
                "Please provide an appropriate IBMQ Project name in crendentials.json.")

        IBMQ.save_account(token = api_token, hub=self.HUB, 
                          group=self.GROUP, project=self.PROJECT, 
                          overwrite=True)
    
    @pytest.mark.api
    def test_changing_provider(self):
        
        """
        This test checks that the specified hub,group and project in the 
        initialisation of DeviceQiskit changes the provider to the appropriate
        destination.
        """
        
        device_obj = DeviceQiskit(device_name='ibmq_manila')
        device_obj.check_connection()
        
        self.assertEqual(device_obj.provider.credentials.hub, self.HUB)
        self.assertEqual(device_obj.provider.credentials.group, self.GROUP)
        self.assertEqual(device_obj.provider.credentials.project, self.PROJECT)
        
        device_obj2 = DeviceQiskit(device_name='ibmq_manila', 
                                   hub='ibm-q-startup')
        device_obj2.check_connection()
        self.assertEqual(device_obj2.provider.credentials.hub, 'ibm-q-startup')
    
    @pytest.mark.api
    def test_check_connection_provider_no_backend_wrong_hub_group_project(self):
        
        """
        If the wrong hub, group or project is specified, check_connection should 
        return False.
        The provider_connected attribute should be updated to False.
        Since the API Token is loaded from save_account, the api token will be
        checked by Qiskit.
        """
        
        for each_combi in itertools.product(['invalid_hub', None], 
                                            ['invalid_group', None], 
                                            ['invalid_project', None]):
            
            if each_combi != (None, None, None):
                
                device_obj = DeviceQiskit(device_name='',
                                          hub=each_combi[0], 
                                          group=each_combi[1],
                                          project=each_combi[2])
            
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

        device_obj = DeviceQiskit(device_name='',
                                  hub=self.HUB, group=self.GROUP,
                                  project=self.PROJECT)

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

        device_obj = DeviceQiskit(device_name='', 
                                  hub=self.HUB, group=self.GROUP,
                                  project=self.PROJECT)

        device_obj.check_connection()
        valid_qpu_name = device_obj.available_qpus[0]

        device_obj = DeviceQiskit(device_name=valid_qpu_name, 
                                  hub=self.HUB, group=self.GROUP,
                                  project=self.PROJECT)

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

        device_obj = DeviceQiskit(device_name='random_invalid_backend', 
                                  hub=self.HUB, group=self.GROUP,
                                  project=self.PROJECT)

        self.assertEqual(device_obj.check_connection(), False)
        self.assertEqual(device_obj.provider_connected, True)
        self.assertEqual(device_obj.qpu_connected, False)


class TestingDeviceAWS(unittest.TestCase):
    
    """These tests check the Object used to access AWS Braket and their 
    available QPUs can be established.

    For any tests using provided credentials, the tests will only pass if those
    details provided are correct/valid with AWS Braket.
    """
    
    @pytest.mark.api
    def test_changing_aws_region(self):
        
        device_obj = DeviceAWS(device_name='arn:aws:braket:::device/quantum-simulator/amazon/sv1')
        
        device_obj.check_connection()
        default_region = device_obj.aws_region
        
        self.assertEqual('us-east-1', default_region)
        
        device_obj = DeviceAWS(device_name='arn:aws:braket:::device/quantum-simulator/amazon/sv1', aws_region='us-west-1')
        
        device_obj.check_connection()
        custom_region = device_obj.aws_region
        
        self.assertEqual('us-west-1', custom_region)
        
    @pytest.mark.api
    def test_changing_s3_bucket_names(self):
        
        device_obj = DeviceAWS(device_name='arn:aws:braket:::device/quantum-simulator/amazon/sv1', s3_bucket_name='random_new_name')
        
        device_obj.check_connection()
        custom_bucket = device_obj.s3_bucket_name
        
        self.assertEqual('random_new_name', custom_bucket)
        
    @pytest.mark.api      
    def test_check_connection_provider_no_backend_provided_credentials(self):
        
        """
        If no information about the device name, but the credentials
        used are correct, check_connection should return True.
        The provider_connected attribute should be updated to True.
        """

        device_obj = DeviceAWS(device_name='')

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

        device_obj = DeviceAWS(device_name='')

        device_obj.check_connection()
        valid_qpu_name = device_obj.available_qpus[0]

        device_obj = DeviceAWS(device_name=valid_qpu_name)

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

        device_obj = DeviceAWS(device_name='random_invalid_backend')

        self.assertEqual(device_obj.check_connection(), False)
        self.assertEqual(device_obj.provider_connected, True)
        self.assertEqual(device_obj.qpu_connected, False)

if __name__ == '__main__':
    unittest.main()
