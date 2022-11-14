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

import copy
import json

from openqaoa.problems.helper_functions import convert2serialize, convert2serialize_complex
from openqaoa.workflows.parameters.qaoa_parameters import CircuitProperties, BackendProperties, ClassicalOptimizer
from openqaoa.workflows.parameters.rqaoa_parameters import RqaoaParameters

class RQAOAResults(dict):
    """
    A class to handle the results of RQAOA workflows
    It stores the results of the RQAOA optimization as a dictionary. With some custom methods.
    """

    def __init__(self):
        """
        Initializes the result class.
        """
        self.circuit_properties  = CircuitProperties()
        self.backend_properties  = BackendProperties()
        self.classical_optimizer = ClassicalOptimizer()
        self.rqaoa_parameters    = RqaoaParameters()
        self.device              = None

    def get_solution(self):
        """
        Returns the solution of the optimization.
        """
        return self['solution']

    def get_step(self, i):
        """
        Returns the QUBO problem and QAOA of the i-th intermidate step of the optimization.
        """
        return self['intermediate_steps'][i]

    def get_qaoa_step(self, i):
        """
        Returns the i-th qaoa step of the RQAOA.
        """
        return self['intermediate_steps'][i]['QAOA']

    def get_qaoa_step_optimized_angles(self, i):
        """
        Returns the optimized angles of the i-th qaoa step of the RQAOA.
        """
        return self.get_qaoa_step(i).results.optimized['optimized angles']

    def get_problem_step(self, i):
        """
        Returns the QUBO problem in the i-th step of the RQAOA.
        """
        return self['intermediate_steps'][i]['QUBO']

    def get_hamiltonian_step(self, i):
        """
        Returns the Hamiltonian of the i-th step of the RQAOA.
        """
        return self.get_problem_step(i).hamiltonian

    def dumps(self, human:bool=False, as_string:bool=False, indent:int=4):
        """
        Returns the result serialized 

        Parameters
        ----------
        human : bool
            If True, the result is serialized in a human readable format.
        as_string : bool
            If True, the result is returned as a string. Otherwise, it is returned as a dictionary.
        indent : int
            The number of spaces to indent the result in the json file. If None, the result is not indented.

        Returns
        -------
        str or dict
        """

        full_dict = copy.deepcopy(self)
        if as_string:
            full_dict['elimination_rules']   = [{str(key): value for key, value in dict.items()} for dict in full_dict['elimination_rules']] 
        if human:
            full_dict['intermediate_steps']  = [{'QUBO': step['QUBO'], 'QAOA': step['QAOA'].results.most_probable_states} for step in full_dict['intermediate_steps']]
        else:
            full_dict['intermediate_steps']  = [{'QUBO': step['QUBO'], 'QAOA': step['QAOA'].results} for step in full_dict['intermediate_steps']]
        full_dict['device']              = self.device
        full_dict['circuit_properties']  = self.circuit_properties
        full_dict['backend_properties']  = self.backend_properties
        full_dict['classical_optimizer'] = self.classical_optimizer
        full_dict['rqaoa_parameters']    = self.rqaoa_parameters

        if as_string:
            return json.dumps(convert2serialize_complex(full_dict), indent=indent)
        else:
            return convert2serialize(full_dict)

    def dump(self, file_path:str, human:bool=False, indent:int=4):
        """
        Saves the result as json file.

        Parameters
        ----------
        file_path : str
            The name of the file to save the result. If None, the result is saved as 'result.json'.
        human : bool
            If True, the result is serialized in a human readable format.
        indent : int
            The number of spaces to indent the result in the json file. If None, the result is not indented.
        """

        # adding .json extension if not present
        file_path = file_path + '.json' if '.json' != file_path[-5:] else file_path

        # saving the result in a json file
        with open(file_path, 'w') as f:
            f.write(self.dumps(human=human, as_string=True, indent=indent))

        print('Results saved as {}'.format(file_path))