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

class RQAOAResults(dict):
    """
    A class to handle the results of RQAOA workflows
    It stores the results of the RQAOA optimization as a dictionary. With some custom methods.

    TODO :  make it similar to the QAOA results class, some plotting functions could be added too
            QAQAOAResults class is in optimizers/result.py
    """

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