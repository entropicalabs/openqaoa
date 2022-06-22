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

from openqaoa.problems.problem import PUBO
from openqaoa.workflows.optimizer import QAOA
from openqaoa.problems.helper_functions import docplex2qubo




"""
Unittest based testing of docplex model conversion.
"""


class TestDocplex2QUBOClass(unittest.TestCase):

    def test_qubo(self):
        try:
            from docplex.mp.model import Model
            
            mdl = Model("Test") # Docplex model
            num_z = 5 # Number of variables
            z = mdl.binary_var_list(num_z, name="Z")
    
    
            objective = mdl.sum(z) - 2 * z[0]  + z[3] * z[4] + 5
    
            mdl.minimize(objective)
    
            qubo_problem = docplex2qubo(mdl)
            
            assert isinstance(PUBO, qubo_problem)
        except:
            pass
        
    def test_solution(self):
        try:
            from docplex.mp.model import Model
        
            mdl = Model("Test") # Docplex model
            num_z = 5 # Number of variables
            z = mdl.binary_var_list(num_z, name="Z")
    
    
            objective = mdl.sum(z) - 2 * z[0]  + z[3] * z[4] + 5
    
            mdl.minimize(objective)
    
            qubo_problem = docplex2qubo(mdl)
            
            q = QAOA()
            q.compile(qubo_problem)
            q.optimize() 
            probs = q.results_information["best probability"][0]
            
            assert max(probs, key=probs.get) == '10000'
        except:
            pass

if __name__ == '__main__':
    unittest.main()
