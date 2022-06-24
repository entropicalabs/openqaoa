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

from logging.config import dictConfig
from re import I
import matplotlib.pyplot as plt
from typing import Type
import numpy as np

from .logger_vqa import Logger
from ..qaoa_parameters.operators import Hamiltonian
from ..utilities import qaoa_probabilities, bitstring_energy

def most_probable_bitstring(cost_hamiltonian, measurement_outcomes):

    if type(measurement_outcomes) == type(np.array([])):
        measurement_outcomes = qaoa_probabilities(measurement_outcomes)

    mea_out = list(measurement_outcomes.values())
    index_likliest_states = np.argwhere(mea_out == np.max(mea_out))
    # degeneracy = len(index_likliest_states)
    solutions_bitstrings = [list(measurement_outcomes.keys())[
        e[0]] for e in index_likliest_states]

    return {'solutions_bitstrings' : solutions_bitstrings, 
            'bitstring_energy' : bitstring_energy(cost_hamiltonian, solutions_bitstrings[0])}

class Result():
    '''
    A class to handle the results of QAOA workflows

    Parameters
    ----------
    log:

    method:
    '''

    def __init__(self,
                 log: Type[Logger],
                 method: Type[str],
                 cost_hamiltonian: Type[Hamiltonian]):

        self.log = log
        self.method = method

        self.most_probable_states = most_probable_bitstring(cost_hamiltonian, self.log.measurement_outcomes.best[0])

        self.evals = {
            'number of evals': self.log.func_evals.best[0],
            'jac evals': self.log.jac_func_evals.best[0],
            'qfim evals': self.log.qfim_func_evals.best[0]
        }

        self.intermediate = {
            'parameter log': np.array(self.log.param_log.history).tolist(),
            'intermediate cost': self.log.cost.history,
            'intermediate measurement outcomes': 
                self.log.measurement_outcomes.history
        }

        self.optimized = {
            'optimized param': np.array(self.log.param_log.best[0]).tolist(),
            'optimized cost': self.log.cost.best[0],
            'optimized measurement outcomes': 
                self.log.measurement_outcomes.best[0]
                if self.log.measurement_outcomes.best != [] else {}
        }

        self.hamiltonian = {'cost hamiltonian' : cost_hamiltonian}

    # def __repr__(self):
    #     """Return an overview over the parameters and hyperparameters
    #     Todo
    #     ----
    #     Split this into ``__repr__`` and ``__str__`` with a more verbose
    #     output in ``__repr__``.
    #     """
    #     string = "Optimization Results:\n"
    #     string += "\tThe solution is " + str(self.solution['degeneracy']) + " degenerate" "\n"
    #     string += "\tThe most probable bitstrings are: " + str(self.solution['bitstring']) + "\n"
    #     string += "\tThe associated cost is: " + str(self.optimized['optimized cost']) + "\n"

    #     return (string)


    def plot_cost(self, figsize=(10,8), label='Cost', linestyle='--', color='b'):
        """
        A simpler helper function to plot the cost associated to a QAOA workflow

        Parameters
        ----------
            figsize tuple
                The size of the figure to be plotted. Defaults to (10,8).
            label str
                The label of the cost line, defaults to 'Cost'.
            linestyle str
                The linestyle of the poloit. Defaults to '--'.
            color str
                The color of the line. Defaults to 'b'.
        """

        plt.figure(figsize=figsize)
        plt.plot(range(self.evals['number of evals']),
                self.intermediate['intermediate cost'], 
                label = label,
                linestyle=linestyle,
                color=color)

        plt.ylabel('Cost')
        plt.xlabel('Number of function evaluations')
        plt.legend()
        plt.title('Cost progress list')

        return
