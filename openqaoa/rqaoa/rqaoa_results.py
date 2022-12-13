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

import matplotlib.pyplot as plt

class RQAOAResults(dict):
    """
    A class to handle the results of RQAOA workflows
    It stores the results of the RQAOA optimization as a dictionary. With some custom methods.
    """

    def get_solution(self):
        """
        Returns the solution of the optimization.
        """
        return self['solution']

    def get_qaoa_results(self, step):
        """
        Returns the i-th qaoa step of the RQAOA.
        """
        return self['intermediate_steps'][step]['qaoa_results']

    def get_qaoa_optimized_angles(self, step):
        """
        Returns the optimized angles of the i-th qaoa step of the RQAOA.
        """
        return self.get_qaoa_results(step).optimized['optimized angles']

    def get_problem(self, step):
        """
        Returns the QUBO problem in the i-th step of the RQAOA.
        """
        return self['intermediate_steps'][step]['problem']

    def get_hamiltonian(self, step):
        """
        Returns the Hamiltonian of the i-th step of the RQAOA.
        """
        return self.get_problem(step).hamiltonian

    def get_exp_vals_z(self, step):
        """
        Returns the expectation values of the Z operator of the i-th step of the RQAOA.
        """
        return self['intermediate_steps'][step]['exp_vals_z']

    def get_corr_matrix(self, step):
        """
        Returns the correlation matrix of the i-th step of the RQAOA. 
        """
        return self['intermediate_steps'][step]['corr_matrix']

    def plot_corr_matrix(self, step, cmap="cool"):
        """
        Plots the correlation matrix of the i-th step of the RQAOA.
        TODO : add more options
        """
        plt.imshow(self.get_corr_matrix(step=step), cmap=cmap)
        plt.colorbar()