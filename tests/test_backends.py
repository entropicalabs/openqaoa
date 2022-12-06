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


def get_params():
    cost_hamil = Hamiltonian.classical_hamiltonian([[0, 1]], [1], constant=0)
    mixer_hamil = X_mixer_hamiltonian(2)

    circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=1)
    variational_params_std = create_qaoa_variational_params(circuit_params, 'standard', 'ramp')

    return circuit_params, variational_params_std

class TestingBackendLocal(unittest.TestCase):
    """
    These tests check that the methods of the local backends are working properly.
    """

    def test_get_counts_and_expectation_n_shots(self):
        """ 
        Check that the .get_counts admit n_shots as an argument, and works properly for the backend of all local devices.
        Also check that .expectation and .expecation_w_uncertainty methods admit n_shots as an argument for the QAOABaseBackendShotBased backends.
        """

        for device_name in DEVICE_NAME_TO_OBJECT_MAPPER.keys():

            circuit_params, variational_params_std= get_params()

            device = create_device(location='local', name=device_name)
            backend = get_qaoa_backend(circuit_params=circuit_params, device=device, n_shots=1000)
                        
            assert sum(backend.get_counts(params=variational_params_std, n_shots=58).values())==58, "`n_shots` is not being respected for the local simulator `{}` when calling backend.get_counts(n_shots=58).".format(device_name)
            if isinstance(backend, QAOABaseBackendShotBased): 
                try: backend.expectation(params=variational_params_std, n_shots=58)
                except Exception: raise Exception("backend.expectation does not admit `n_shots` as an argument for the local simulator `{}`.".format(device_name))
                try: backend.expectation_w_uncertainty(params=variational_params_std, n_shots=58)
                except Exception: raise Exception("backend.expectation_w_uncertainty does not admit `n_shots` as an argument for the local simulator `{}`.".format(device_name))

class TestingBackendQPUs(unittest.TestCase): 
    """ 
    These tests check methods of the QPU backends.

    For all of these tests, credentials.json MUST be filled with the appropriate
    credentials. 
    """

    @pytest.mark.qpu
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
            raise ValueError("Please provide an appropriate API TOKEN in crendentials.json.")
        elif self.HUB == "IBMQ_HUB":
            raise ValueError("Please provide an appropriate IBM HUB name in crendentials.json.")
        elif self.GROUP == "IBMQ_GROUP":
            raise ValueError("Please provide an appropriate IBMQ GROUP name in crendentials.json.")
        elif self.PROJECT == "IBMQ_PROJECT":
            raise ValueError("Please provide an appropriate IBMQ Project name in crendentials.json.")
            
        IBMQ.save_account(token = api_token, hub=self.HUB, 
                          group=self.GROUP, project=self.PROJECT, 
                          overwrite=True)

    @pytest.mark.qpu
    def test_get_counts_and_expectation_n_shots(self):
        """ Check that the .get_counts, .expectation and .expecation_w_uncertainty methods admit n_shots as an argument for the backends of all QPUs. """

        list_device_attributes = [ 
                                    {'QPU': 'Qiskit', 'device_name': 'ibmq_qasm_simulator', 'hub': self.HUB, 'group': self.GROUP, 'project': self.PROJECT}, 
                                    {'QPU': 'Pyquil', 'device_name': "2q-qvm", 'as_qvm': True, 'execution_timeout': 3, 'compiler_timeout': 3},       
                                    {'QPU': 'AWS', 'device_name': 'arn:aws:braket:::device/quantum-simulator/amazon/sv1'}
                                ]

        assert len(list_device_attributes) == len(DEVICE_ACCESS_OBJECT_MAPPER), "The number of QPUs in the list of tests is not the same as the number of QPUs in the DEVICE_ACCESS_OBJECT_MAPPER. The list should be updated."

        for (device, backend), device_attributes in zip(DEVICE_ACCESS_OBJECT_MAPPER.items(), list_device_attributes):

            circuit_params, variational_params_std = get_params()

            QPU_name = device_attributes.pop('QPU')
            print("Testing {} backend.".format(QPU_name))

            # TODO : remove the following if:
            if QPU_name == "Pyquil": 
                print("Skipping test for Pyquil backend.")
                continue # we need to skip this test for now, since the pyquil backend does not admit n_shots as an argument.

            try: 
                device = device(**device_attributes)
                backend = backend(circuit_params = circuit_params, device = device, cvar_alpha = 1, n_shots=100, prepend_state = None, append_state = None, init_hadamard = True)

                # Check that the .get_counts, .expectation and .expectation_w_variance methods admit n_shots as an argument
                assert sum(backend.get_counts(params=variational_params_std, n_shots=58).values()) == 58, "`n_shots` is not being respected when calling .get_counts(n_shots=58).".format(QPU_name)
                backend.expectation(params=variational_params_std, n_shots=58)
                backend.expectation_w_uncertainty(params=variational_params_std, n_shots=58)

            except Exception as e: 

                raise e from type(e)(f"Error raised for `{QPU_name}`: " + str(e))

            print("Test passed for {} backend.".format(QPU_name))




if __name__ == '__main__':
    unittest.main()
