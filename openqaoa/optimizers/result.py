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
        self, log: Type[Logger], method: Type[str], cost_hamiltonian: Type[Hamiltonian]
    ):

        self.method = method

        self.cost_hamilonian = cost_hamiltonian

        if log.param_log.best != []:
            p = len(log.param_log.best[0]) // 2
            angles_best = np.array(log.param_log.best[0])
        if log.param_log.history != []:
            angles_log = np.array(log.param_log.history)
            p = angles_log.shape[1] // 2

        

        self.evals = {
            "number of evals": log.func_evals.best[0],
            "jac evals": log.jac_func_evals.best[0],
            "qfim evals": log.qfim_func_evals.best[0],
        }

        self.intermediate = {
            "angles log": {"betas":angles_log[:,:p].tolist(),"gammas":angles_log[:,p:].tolist()}
                            if log.param_log.history != [] else {"betas":[], "gammas":[]},
            "intermediate cost": log.cost.history,
            "intermediate measurement outcomes": log.measurement_outcomes.history,
        }

        self.optimized = {
            "optimized angles": {"betas":angles_best[:p].tolist(),"gammas":angles_best[p:].tolist()}
                                if log.param_log.best != [] else {"betas":[], "gammas":[]},
            "optimized cost": log.cost.best[0] if log.cost.best != [] else np.nan,
            "optimized measurement outcomes": log.measurement_outcomes.best[0]
            if log.measurement_outcomes.best != []
            else {},
        }

        if log.measurement_outcomes.best != []:
            self.most_probable_states = most_probable_bitstring(
                cost_hamiltonian, self.get_counts(log.measurement_outcomes.best[0])
            )

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
                self.evals["number of evals"]
                - self.evals["jac evals"]
                - self.evals["qfim evals"]
            ),
            self.intermediate["intermediate cost"],
            label=label,
            linestyle=linestyle,
            color=color,
        )

        ax.set_ylabel("Cost")
        ax.set_xlabel("Number of function evaluations")
        ax.legend()
        ax.set_title("Cost history")

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
        measurement_outcomes = self.optimized["optimized measurement outcomes"]
        solution_bitstring = list(measurement_outcomes.keys())
        energies = [
            bitstring_energy(self.cost_hamiltonian, bitstring)
            for bitstring in solution_bitstring
        ]
        args_sorted = np.argsort(energies)
        if n_bitstrings < len(energies):
            best_results = {
                "solutions_bitstrings": [
                    solution_bitstring[args_sorted[ii]] for ii in range(n_bitstrings)
                ],
                "bitstring_energy": [
                    energies[args_sorted[ii]] for ii in range(n_bitstrings)
                ],
            }
        else:
            best_results = {
                "solutions_bitstrings": solution_bitstring,
                "bitstring_energy": energies,
            }
        return best_results
