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

import json

from openqaoa.workflows.optimizer import Optimizer, RQAOA
from openqaoa.problems.problem import QUBO

def formatter(d):
    """
    This function is needed to make sure the method asdict() works as expected.
    It will removed from the final job routine
    """
    r = {}
    for k,v in d.items():
        if k.startswith("_"):
            r[k[1:]] = v
        else:
            r[k] = v
    return r

def create_aws_input_data(workflow : Optimizer, qubo: QUBO):
        """ 
        helper function to extract all the QAOA parameters from the QAOA object
        """
        rqaoa_parameters = None
        if workflow.algorithm == 'rqaoa': rqaoa_parameters = workflow.rqaoa_parameters.asdict()
        
        input_data = {
            'circuit_properties' : formatter(workflow.circuit_properties.asdict()),
            'backend_properties' :  formatter(workflow.backend_properties.asdict()),
            'classical_optimizer' : formatter(workflow.classical_optimizer.asdict()),
            'qubo' : formatter(qubo.asdict()),
            'rqaoa_parameters' : rqaoa_parameters
        }

        return input_data

def save_input_data(openqaoa_data: dict, openqaoa_data_path: str):
    '''
    Save the hyperparameter as a json file at a desired location
    '''

    with open(openqaoa_data_path, 'w') as f:
        json.dump(openqaoa_data, f)

