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

from openqaoa.workflows.optimizer import QAOA
from openqaoa.backends.qaoa_backend import (DEVICE_NAME_TO_OBJECT_MAPPER,
                                            DEVICE_ACCESS_OBJECT_MAPPER)
import unittest
import networkx as nw
import numpy as np

from openqaoa.backends.qpus.qpu_auth import AccessObjectQiskit
from openqaoa.problems.problem import MinimumVertexCover

ALLOWED_LOCAL_SIMUALTORS = ['qiskit_shot_simulator', 'qiskit_statevec_simulator', 'qiskit_qasm_simulator', 'vectorized']
LOCAL_DEVICES = ALLOWED_LOCAL_SIMUALTORS + ['6q-qvm', 'Aspen-11']
"""
TODO: Logic for test_check_connection_aws_no_device_provided_credentials while 
correct. Returning true for the check_connection() function if no device was
provided might be a strange behaviour since check_connection() is also expected
to return true if the device provided was correct, and false if invalid.
"""


class TestingAccessObjectQiskit(unittest.TestCase):

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


class TestingVanillaQAOA(unittest.TestCase):

    """This 
    """

    def test_vanilla_qaoa_default_values(self):
        q = QAOA()
        assert q.circuit_properties.p == 1
        assert q.circuit_properties.param_type == 'standard'
        assert q.circuit_properties.init_type == 'ramp'
        assert q.device_properties.device_location == 'locale'
        assert q.device_properties.device == 'vectorized'

    def test_local_devices(self):
        """Check that the device is correctly initalised for all local devices
        """

        for device in LOCAL_DEVICES:
            q = QAOA()
            q.set_device_properties(
                device_location='locale', device_name=device)
            assert q.device_properties.device == device
            assert q.device_properties.device_name == device

    def test_end_to_end_vectorized(self):
        
        g = nw.circulant_graph(6, [1])
        vc = MinimumVertexCover(g, field =1.0, penalty=10).get_pubo_problem()

        q = QAOA()
        q.compile(vc)
        q.optimize()

        assert '010101' ==  str(list(q.results_information['probability progress list'][0].keys())[np.argmax(list((q.results_information['probability progress list'][0].values())))])



if __name__ == '__main__':
    unittest.main()
