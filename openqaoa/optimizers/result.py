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

from functools import update_wrapper
from logging.config import dictConfig
from re import I
import matplotlib.pyplot as plt
from typing import Type, List
import numpy as np
import json

from .logger_vqa import Logger
from ..qaoa_parameters.operators import Hamiltonian
from ..utilities import qaoa_probabilities, bitstring_energy, convert2serialize, delete_keys_from_dict
from ..basebackend import QAOABaseBackend, QAOABaseBackendStatevector


def most_probable_bitstring(cost_hamiltonian, measurement_outcomes):

    mea_out = list(measurement_outcomes.values())
    index_likliest_states = np.argwhere(mea_out == np.max(mea_out))
    # degeneracy = len(index_likliest_states)
    solutions_bitstrings = [
        list(measurement_outcomes.keys())[e[0]] for e in index_likliest_states
    ]

    return {
        "solutions_bitstrings": solutions_bitstrings,
        "bitstring_energy": bitstring_energy(cost_hamiltonian, solutions_bitstrings[0]),
    }


class Result:
    """
    A class to handle the results of QAOA workflows

    Parameters
    ----------
    log: `Logger`
        The raw logger generated from the training vqa part of the QAOA. 
    method: `str`
        Stores the name of the optimisation used by the classical optimiser
    """

    def __init__(
        self, log: Type[Logger], method: Type[str], cost_hamiltonian: Type[Hamiltonian], backend: Type[QAOABaseBackend]
    ):

        self.__backend = backend
        self.method = method
        self.cost_hamiltonian = cost_hamiltonian

        self.evals = {
            "number_of_evals": log.func_evals.best[0],
            "jac_evals": log.jac_func_evals.best[0],
            "qfim_evals": log.qfim_func_evals.best[0],
        }

        self.intermediate = {
            'angles': np.array(log.param_log.history).tolist(),
            'cost': log.cost.history,
            'measurement_outcomes':
                log.measurement_outcomes.history,
            'job_id': log.job_ids.history
        }

        self.optimized = {
            'angles':
                np.array(log.param_log.best[0]).tolist()
                if log.param_log.best != [] else [],
            'cost':
                log.cost.best[0]
                if log.cost.best != [] else None,
            'measurement_outcomes':
                log.measurement_outcomes.best[0]
                if log.measurement_outcomes.best != [] else {},
            'job_id': 
                log.job_ids.best[0] 
                if len(log.job_ids.best) != 0 else [],
            'eval_number': 
                log.eval_number.best[0] 
                if len(log.eval_number.best) != 0 else [],
        }

        self.most_probable_states = most_probable_bitstring(
            cost_hamiltonian, self.get_counts(log.measurement_outcomes.best[0])
        ) if log.measurement_outcomes.best != [] else []

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
    #     string += "\tThe associated cost is: " + str(self.optimized['cost']) + "\n"

    #     return (string)

    def asdict(self, keep_cost_hamiltonian:bool=True, complex_to_string:bool=False, intermediate_mesurements:bool=True, exclude_keys:List[str]=[]):
        """
        Returns a dictionary with the results of the optimization, where the dictionary is serializable. 
        If the backend is a statevector backend, the measurement outcomes will be the statevector, meaning that it is a list of complex numbers, which is not serializable. If that is the case, and complex_to_string is true the complex numbers are converted to strings.

        Parameters
        ----------
        keep_cost_hamiltonian: `bool`
            If True, the cost hamiltonian is kept in the dictionary. If False, it is removed.
        complex_to_string: `bool`
            If True, the complex numbers are converted to strings. If False, they are kept as complex numbers. This is useful for the JSON serialization. 
        intermediate_mesurements: bool, optional
            If True, intermediate measurements are included in the dump. If False, intermediate measurements are not included in the dump.
            Default is True.
        exclude_keys: `list[str]`
            A list of keys to exclude from the returned dictionary.

        Returns
        -------
        return_dict: `dict`
            A dictionary with the results of the optimization, where the dictionary is serializable.
        """

        return_dict = {}
        return_dict['method'] = self.method
        if keep_cost_hamiltonian: return_dict['cost_hamiltonian'] = convert2serialize(self.cost_hamiltonian)
        return_dict['evals'] = self.evals
        return_dict['most_probable_states'] = self.most_probable_states

        complx_to_str = lambda x: str(x) if isinstance(x, np.complex128) or isinstance(x, complex) else x
        
        # if the backend is a statevector backend, the measurement outcomes will be the statevector, meaning that it is a list of complex numbers, which is not serializable. If that is the case, and complex_to_string is true the complex numbers are converted to strings.
        if complex_to_string and issubclass(self.__backend, QAOABaseBackendStatevector):
            return_dict['intermediate'] = {}
            for key, value in self.intermediate.items():
                if intermediate_mesurements == False and 'measurement' in key: # if intermediate_mesurements is false, the intermediate measurements are not included in the dump
                    return_dict['intermediate'][key] = []
                elif 'measurement' in key and (isinstance(value, list) or isinstance(value, np.ndarray)):
                    return_dict['intermediate'][key] = [[complx_to_str(item) for item in list_] for list_ in value if (isinstance(list_, list) or isinstance(list_, np.ndarray))]
                else:
                    return_dict['intermediate'][key] = value 

            return_dict['optimized'] = {}
            for key, value in self.optimized.items():
                if 'measurement' in key and (isinstance(value, list) or isinstance(value, np.ndarray)):
                    return_dict['optimized'][key] = [complx_to_str(item) for item in value] 
                else:
                    return_dict['optimized'][key] = value
        else:
            return_dict['intermediate'] = self.intermediate
            return_dict['optimized'] = self.optimized

        return return_dict if exclude_keys == [] else delete_keys_from_dict(return_dict, exclude_keys)


    @staticmethod
    def get_counts(measurement_outcomes):
        """
        Converts probabilities to counts when the measurement outcomes are a numpy array, that is a state vector

        Parameters
        ----------
        measurement_outcomes: `Union[np.array, dict]`
            The measurement outcome as returned by the Logger. It can either be a statevector or a count dictionary
        Returns
        -------
        `dict`
            The count dictionary obtained either throught the statevector or the actual measurement counts
        """

        if type(measurement_outcomes) == type(np.array([])):
            measurement_outcomes = qaoa_probabilities(measurement_outcomes)

        return measurement_outcomes

    def plot_cost(
        self, figsize=(10, 8), label="Cost", linestyle="--", color="b", ax=None
    ):
        """
        A simpler helper function to plot the cost associated to a QAOA workflow

        Parameters
        ----------
        figsize: `tuple`
            The size of the figure to be plotted. Defaults to (10,8).
        label: `str`
            The label of the cost line, defaults to 'Cost'.
        linestyle: `str`
            The linestyle of the poloit. Defaults to '--'.
        color: `str`
            The color of the line. Defaults to 'b'.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            range(
                1,
                self.evals["number_of_evals"]
                - self.evals["jac_evals"]
                - self.evals["qfim_evals"] + 1
            ),
            self.intermediate["cost"],
            label=label,
            linestyle=linestyle,
            color=color,
        )

        ax.set_ylabel("Cost")
        ax.set_xlabel("Number of function evaluations")
        ax.legend()
        ax.set_title("Cost history")

        return

    def plot_probabilities(self, n_states_to_keep = None, figsize = (10,8),label='Probability distribution',color='tab:blue', ax=None):

        """
        Helper function to plot the probabilities corresponding to each basis states (with prob != 0) obtained from the optimized result

        Parameters
        ----------
        n_states_to_keep: 'int
            If the user passes a value, the plot will compile with the given value of states. 
            Else,  an upper bound will be calculated depending on the total size of the measurement outcomes.
        figsize: `tuple`
            The size of the figure to be plotted. Defaults to (10,8).
        label: `str`
            The label of the cost line, defaults to 'Probability distribution'.
        color: `str`
            The color of the line. Defaults to 'tab:blue'.
        ax: 'matplotlib.axes._subplots.AxesSubplot'
            Axis on which to plot the graph. Deafults to None
        """

        outcome = self.optimized['measurement_outcomes']

        # converting to counts dictionary if outcome is statevector
        if type(outcome) == type(np.array([])):
            outcome = self.get_counts(outcome)
            # setting norm to 1 since it might differ slightly for statevectors due to numerical preicision
            norm = np.float64(1)
        else: 
            # needed to be able to divide the tuple by 'norm'
            norm = np.float64(sum(outcome.values())) 

        # sorting dictionary. adding a callback function to sort by values instead of keys
        # setting reverse = True to be able to obtain the states with highest counts
        outcome_list = sorted(outcome.items(), key=lambda item: item[1], reverse=True)
        states, counts = zip(*outcome_list)

        # normalizing to obtain probabilities
        probs = counts/norm

        # total number of states / number of states with != 0 counts for shot simulators
        total = len(states)

        # number of states that fit without distortion in figure
        upper_bound = 40
        # default fontsize
        font = 'medium'

        if n_states_to_keep:
            if n_states_to_keep > total:
                raise ValueError(f"n_states_to_keep must be smaller or equal than the total number of states in measurement outcome: {total}")
            else:
                if n_states_to_keep>upper_bound:
                    print('number of states_to_keep exceeds the recommended value')
                    font = 'small'

        # if states_to_keep is not given
        else:
            if total > upper_bound:
                n_states_to_keep = upper_bound
            else:
                n_states_to_keep = total
        
        # formatting labels
        labels = [r'$\left|{}\right>$'.format(state) for state in states[:n_states_to_keep]]
        labels.append('rest')

        # represent the bar with the addition of all the remaining probabilites
        rest = sum(probs[n_states_to_keep:])

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        colors = [color for _ in range(n_states_to_keep)] + ['xkcd:magenta']

        ax.bar(labels,np.append(probs[:n_states_to_keep],rest), color=colors)
        ax.set_xlabel('Eigen-State')
        ax.set_ylabel('Probability')
        ax.set_title(label)
        ax.tick_params(axis='x', labelrotation = 75, labelsize=font)
        ax.grid(True, axis='y', linestyle='--')

        print('states kept:', n_states_to_keep)
        return


    def lowest_cost_bitstrings(self, n_bitstrings: int = 1) -> dict:
        """
        Find the minimium energy from cost_hamilonian given a set of measurement
        outcoms

        Parameters
        ----------

        n_bitstrings : int
            Number of the lowest energies bistrings to get

        Returns
        -------
        best_results : dict
            Returns a list of bitstring with the lowest values of the cost Hamiltonian.

        """

        if isinstance(self.optimized["measurement_outcomes"], dict):
            measurement_outcomes = self.optimized["measurement_outcomes"]
            solution_bitstring = list(measurement_outcomes.keys())
        elif isinstance(self.optimized["measurement_outcomes"], np.ndarray):
            measurement_outcomes = self.get_counts(
                self.optimized["measurement_outcomes"]
            )
            solution_bitstring = list(measurement_outcomes.keys())
        else:
            raise TypeError(
                f"The measurement outcome {type(self.optimized['measurement_outcomes'])} is not valid."
            )
        energies = [
            bitstring_energy(self.cost_hamiltonian, bitstring)
            for bitstring in solution_bitstring
        ]
        args_sorted = np.argsort(energies)
        if n_bitstrings > len(energies):
            n_bitstrings = len(energies)
        
        total_shots = sum(measurement_outcomes.values())
        best_results = {
            "solutions_bitstrings": [
                solution_bitstring[args_sorted[ii]] for ii in range(n_bitstrings)
            ],
            "bitstrings_energies": [
                energies[args_sorted[ii]] for ii in range(n_bitstrings)
            ],
            "probabilities": [
                measurement_outcomes[solution_bitstring[args_sorted[ii]]]/total_shots for ii in range(n_bitstrings)
            ]
        }
        return best_results
