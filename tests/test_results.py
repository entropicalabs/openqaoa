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
from openqaoa.devices import create_device,SUPPORTED_LOCAL_SIMULATORS
import unittest
import networkx as nw
import numpy as np
import itertools

from openqaoa.problems.problem import MinimumVertexCover

ALLOWED_LOCAL_SIMUALTORS = SUPPORTED_LOCAL_SIMULATORS


class TestingResultOutputs(unittest.TestCase):

    """
    Test the Results Output after an optimization loop
    """

    def test_flags_result_outputs_workflow(self):
        
        """
        Run an optimization problem for 5 iterations.
        Should expect certain fields of the results output to be filled based 
        on some of the users inputs. (Default settings)
        Can be checked for cobyla.
        
        Check for all available supported local backends.
        """
        
        g = nw.circulant_graph(3, [1])
        vc = MinimumVertexCover(g, field =1.0, penalty=10).get_qubo_problem()
        
        choice_combination = list(itertools.product([True, False], [True, False], [True, False]))
        recorded_evals = [0, 5]
        
        for device_name in ALLOWED_LOCAL_SIMUALTORS:
            
            for each_choice in choice_combination:
            
                q = QAOA()
                q.set_classical_optimizer(method = 'cobyla', 
                                          parameter_log = each_choice[0],
                                          cost_progress = each_choice[1],
                                          optimization_progress = each_choice[2], 
                                          maxiter = 5)
                device = create_device('local', device_name)
                q.set_device(device)
                q.compile(vc)
                q.optimize()
                
                self.assertEqual(recorded_evals[each_choice[0]], len(q.results.intermediate['angles log']))
                self.assertEqual(recorded_evals[each_choice[1]], len(q.results.intermediate['intermediate cost']))
                self.assertEqual(recorded_evals[each_choice[2]], len(q.results.intermediate['intermediate measurement outcomes']))
                

if __name__ == '__main__':
    unittest.main()
