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
from openqaoa.workflows.optimizer import QAOA
from openqaoa.problems.problem import QUBO
from openqaoa.devices import create_device
from openqaoa.devices import SUPPORTED_LOCAL_SIMULATORS


class TestSimulators(unittest.TestCase):
    
    def test_job_ids(self):
        """
        Test if correct job ids are generated and returned for all simulators
        """

        #define problem
        problem = QUBO.random_instance(3)

        #loop over all simulators
        for n in SUPPORTED_LOCAL_SIMULATORS:

            # initialize
            q = QAOA()

            # device
            device = create_device(location='local', name=n)
            q.set_device(device)

            # classical optimizer only 3 iterations
            q.set_classical_optimizer(maxiter=3)

            # compile
            q.compile(problem)

            # run
            q.optimize()

            # check if we have job ids
            opt_id = q.results.optimized['job_id']
            assert len(opt_id) == 36 and isinstance(opt_id, str), f'simulator {n}: job id is not a string of length 36, but {opt_id}'

            inter_id = q.results.intermediate['job_id']
            for id in inter_id:
                assert len(id) == 36 and isinstance(id, str), f'simulator {n}: on intermediate job id is not a string of length 36, but {id}'

            


if __name__ == "__main__":
    unittest.main()
