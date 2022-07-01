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

from argparse import SUPPRESS
from openqaoa.workflows.optimizer import QAOA, RQAOA
from openqaoa.backends.qaoa_backend import (DEVICE_NAME_TO_OBJECT_MAPPER,
                                            DEVICE_ACCESS_OBJECT_MAPPER)
from openqaoa.devices import create_device,SUPPORTED_LOCAL_SIMULATORS
import unittest
import networkx as nw
import numpy as np

from openqaoa.problems.problem import MinimumVertexCover

ALLOWED_LOCAL_SIMUALTORS = SUPPORTED_LOCAL_SIMULATORS
LOCAL_DEVICES = ALLOWED_LOCAL_SIMUALTORS + ['6q-qvm', 'Aspen-11']


class TestingVanillaQAOA(unittest.TestCase):

    """This 
    """

    def test_vanilla_qaoa_default_values(self):
        q = QAOA()
        assert q.circuit_properties.p == 1
        assert q.circuit_properties.param_type == 'standard'
        assert q.circuit_properties.init_type == 'ramp'
        assert q.device.device_location == 'local'
        assert q.device.device_name == 'vectorized'

    def test_local_devices(self):
        """Check that the device is correctly initalised for all local devices
        """

        for device_name in LOCAL_DEVICES:
            q = QAOA()
            device = create_device('local', device_name)
            q.set_device(device)
            assert q.device.device_location == 'local'
            assert q.device.device_name == device_name

    def test_end_to_end_vectorized(self):
        
        g = nw.circulant_graph(6, [1])
        vc = MinimumVertexCover(g, field =1.0, penalty=10).get_qubo_problem()

        q = QAOA()
        q.set_classical_optimizer(optimization_progress = True)
        q.compile(vc)
        q.optimize()

        result = q.results.most_probable_states['solutions_bitstrings'][0]
        assert '010101' == result or '101010' == result


class TestingRQAOA(unittest.TestCase):

    """
    Unit test based testing of the RQAOA worlkflow class
    """

    def test_rqaoa_default_values(self):
        """
        Tests all default values are correct
        """
        r = RQAOA()

        assert isinstance(r.qaoa,QAOA)
        assert r.qaoa.circuit_properties.p == 1
        assert r.qaoa.circuit_properties.param_type == 'standard'
        assert r.qaoa.circuit_properties.init_type == 'ramp'
        assert r.qaoa.device.device_location == 'local'
        assert r.qaoa.device.device_name == 'vectorized'
        assert r.rqaoa_parameters.rqaoa_type == 'adaptive'
        assert r.rqaoa_parameters.n_cutoff == 5
        assert r.rqaoa_parameters.n_max == 1
        assert r.rqaoa_parameters.steps == 1

    def test_end_to_end_vectorized(self):
        """
        Test the full workflow with vectorized backend.
        """
        
        g = nw.circulant_graph(6, [1])
        vc = MinimumVertexCover(g, field =1.0, penalty=10).get_qubo_problem()

        r = RQAOA()
        r.compile(vc)
        r.optimize()

        # Computed solution
        sol_states = list(r.result['solution'].keys())

        # Correct solution
        exact_sol_states = ['101010','010101']

        # Check computed solutions are among the correct ones
        for sol in sol_states:
            assert sol in exact_sol_states

if __name__ == '__main__':
    unittest.main()
