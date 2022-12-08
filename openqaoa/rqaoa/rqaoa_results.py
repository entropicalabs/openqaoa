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

class RQAOAResults(dict):
    """
    A class to handle the results of RQAOA workflows
    It stores the results of the RQAOA optimization as a dictionary. With some custom methods.
    """

    def __init__(self):
        """
        Initializes the result class.
        """
        # initialize the attributes, so that they are always present
        self.circuit_properties  = None
        self.backend_properties  = None
        self.classical_optimizer = None
        self.rqaoa_parameters    = None
        self.device              = None

        self.__serializable_dict = {}

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
        return self['intermediate_steps'][i]['QAOA_results']

    def get_qaoa_step_optimized_angles(self, i):
        """
        Returns the optimized angles of the i-th qaoa step of the RQAOA.
        """
        return self.get_qaoa_step(i).optimized['optimized angles']

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

    def __serializable_dict_fun(self):
        """
        Returns all values and attributes of the result in a dictionary.
        """
        serializable_dict = self.copy()
        serializable_dict['parameters_used'] = {
                                                'circuit_properties': self.circuit_properties,
                                                'backend_properties': self.backend_properties,
                                                'classical_optimizer': self.classical_optimizer,
                                                'rqaoa_parameters': self.rqaoa_parameters,
                                                'device':  {'device_location': self.device.device_location, 'device_name': self.device.device_name},
                                            }
        serializable_dict['elimination_rules']   = [{str(key): value for key, value in dict.items()} for dict in serializable_dict['elimination_rules']] 

        for step in serializable_dict['intermediate_steps']:
            step['QAOA_results'] = {'optimized': step['QAOA_results'].optimized, 'most_probable_states': step['QAOA_results'].most_probable_states}
            step['QAOA_results']['optimized'].pop('optimized measurement outcomes')

        return serializable_dict

    def as_dict(self):
        """
        Returns all values and attributes of the result as a dictionary.
        """
        if self.__serializable_dict == {}: self.__serializable_dict = self.__serializable_dict_fun()

        return convert2serialize(self.__serializable_dict)

    def dumps(self, indent:int=2):
        """
        Returns a json string of the RQAOA result.

        Parameters
        ----------
        indent : int
            The number of spaces to indent the result in the json file. If None, the result is not indented.

        Returns
        -------
        str
        """
        if self.__serializable_dict == {}: self.__serializable_dict = self.__serializable_dict_fun()

        return json.dumps(convert2serialize(self.__serializable_dict), indent=indent)

    def dump(self, file_path:str, indent:int=2):
        """
        Saves the result as json file.

        Parameters
        ----------
        file_path : str
            The name of the file to save the result. If None, the result is saved as 'result.json'.
        indent : int
            The number of spaces to indent the result in the json file. If None, the result is not indented.
        """
        if self.__serializable_dict == {}: self.__serializable_dict = self.__serializable_dict_fun()

        # adding .json extension if not present
        file_path = file_path + '.json' if '.json' != file_path[-5:] else file_path

        # saving the result in a json file
        with open(file_path, 'w') as f:
            json.dump(convert2serialize(self.__serializable_dict), f, indent=indent)

        print('Results saved as {}'.format(file_path))
