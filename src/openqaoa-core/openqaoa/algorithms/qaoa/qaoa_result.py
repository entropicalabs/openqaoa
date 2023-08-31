from __future__ import annotations
from typing import Type, List, Union, TYPE_CHECKING

import copy
import numpy as np
import matplotlib.pyplot as plt

from ...qaoa_components import Hamiltonian

if TYPE_CHECKING:
    from ...optimizers.logger_vqa import Logger
from ...utilities import (
    qaoa_probabilities,
    bitstring_energy,
    convert2serialize,
    delete_keys_from_dict,
)
from ...backends.basebackend import QAOABaseBackend, QAOABaseBackendStatevector
from ...backends.qaoa_analytical_sim import QAOABackendAnalyticalSimulator


def most_probable_bitstring(cost_hamiltonian, measurement_outcomes):
    """
    Computing the most probable bitstring
    """
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


class QAOAResult:
    """
    A class to handle the results of QAOA workflows

    Parameters
    ----------
    log: `Logger`
        The raw logger generated from the training vqa part of the QAOA.
    method: `str`
        Stores the name of the optimisation used by the classical optimiser
    cost_hamiltonian: `Hamiltonian`
        The cost Hamiltonian for the problem statement
    type_backend: `QAOABaseBackend`
        The type of backend used for the experiment
    """

    def __init__(
        self,
        log: Logger,
        method: Type[str],
        cost_hamiltonian: Type[Hamiltonian],
        type_backend: Type[QAOABaseBackend],
    ):
        """
        init method
        """
        self.__type_backend = type_backend
        self.method = method
        self.cost_hamiltonian = cost_hamiltonian

        self.evals = {
            "number_of_evals": log.func_evals.best[0],
            "jac_evals": log.jac_func_evals.best[0],
            "qfim_evals": log.qfim_func_evals.best[0],
        }

        self.intermediate = {
            "angles": np.array(log.param_log.history).tolist(),
            "cost": log.cost.history,
            "measurement_outcomes": log.measurement_outcomes.history,
            "job_id": log.job_ids.history,
        }

        self.optimized = {
            "angles": np.array(log.param_log.best[0]).tolist()
            if log.param_log.best != []
            else [],
            "cost": log.cost.best[0] if log.cost.best != [] else None,
            "measurement_outcomes": log.measurement_outcomes.best[0]
            if log.measurement_outcomes.best != []
            else {},
            "job_id": log.job_ids.best[0] if len(log.job_ids.best) != 0 else [],
            "eval_number": log.eval_number.best[0]
            if len(log.eval_number.best) != 0
            else [],
        }

        self.most_probable_states = (
            most_probable_bitstring(
                cost_hamiltonian, self.get_counts(log.measurement_outcomes.best[0])
            )
            if type_backend != QAOABackendAnalyticalSimulator
            and log.measurement_outcomes.best != []
            else []
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
        #     string += "\tThe associated cost is: " + str(self.optimized['cost']) + "\n"

        #     return (string)

        # if we are using a shot adaptive optimizer, we need to add the number of shots to the result
        if log.n_shots.history != []:
            self.n_shots = log.n_shots.history

    def asdict(
        self,
        keep_cost_hamiltonian: bool = True,
        complex_to_string: bool = False,
        intermediate_measurements: bool = True,
        exclude_keys: List[str] = [],
    ):
        """
        Returns a dictionary with the results of the optimization, where the dictionary is serializable.
        If the backend is a statevector backend, the measurement outcomes will be the statevector,
        meaning that it is a list of complex numbers, which is not serializable.
        If that is the case, and complex_to_string is true the complex numbers are converted to strings.

        Parameters
        ----------
        keep_cost_hamiltonian: `bool`
            If True, the cost hamiltonian is kept in the dictionary. If False, it is removed.
        complex_to_string: `bool`
            If True, the complex numbers are converted to strings. If False, they are kept as complex numbers.
            This is useful for the JSON serialization.
        intermediate_measurements: bool, optional
            If True, intermediate measurements are included in the dump.
            If False, intermediate measurements are not included in the dump.
            Default is True.
        exclude_keys: `list[str]`
            A list of keys to exclude from the returned dictionary.

        Returns
        -------
        return_dict: `dict`
            A dictionary with the results of the optimization, where the dictionary is serializable.
        """

        return_dict = {}
        return_dict["method"] = self.method
        if keep_cost_hamiltonian:
            return_dict["cost_hamiltonian"] = convert2serialize(self.cost_hamiltonian)
        return_dict["evals"] = self.evals
        return_dict["most_probable_states"] = self.most_probable_states

        complex_to_str = (
            lambda x: str(x)
            if isinstance(x, np.complex128) or isinstance(x, complex)
            else x
        )

        # if the backend is a statevector backend, the measurement outcomes will be the statevector,
        # meaning that it is a list of complex numbers, which is not serializable.
        # If that is the case, and complex_to_string is true the complex numbers are converted to strings.
        if complex_to_string:
            return_dict["intermediate"] = {}
            for key, value in self.intermediate.items():
                # Measurements and Cost may require casting
                if "measurement" in key:
                    if len(value) > 0:
                        if intermediate_measurements is False:
                            # if intermediate_measurements is false, the intermediate measurements are not included
                            return_dict["intermediate"][key] = []
                        elif isinstance(
                            value[0], np.ndarray
                        ):  # Statevector -> convert complex to str
                            return_dict["intermediate"][key] = [
                                [complex_to_str(item) for item in list_]
                                for list_ in value
                                if (
                                    isinstance(list_, list)
                                    or isinstance(list_, np.ndarray)
                                )
                            ]
                        else:  # All other case -> cast numpy into
                            return_dict["intermediate"][key] = [
                                {k_: int(v_) for k_, v_ in v.items()} for v in value
                            ]
                    else:
                        pass
                elif "cost" == key and (
                    isinstance(value[0], np.float64) or isinstance(value[0], np.float32)
                ):
                    return_dict["intermediate"][key] = [float(item) for item in value]
                else:
                    return_dict["intermediate"][key] = value

            return_dict["optimized"] = {}
            for key, value in self.optimized.items():
                # If wavefunction do complex to str
                if "measurement" in key and (
                    isinstance(value, list) or isinstance(value, np.ndarray)
                ):
                    return_dict["optimized"][key] = [
                        complex_to_str(item) for item in value
                    ]
                # if dictionary, convert measurement values to integers
                elif "measurement" in key and (isinstance(value, dict)):
                    return_dict["optimized"][key] = {
                        k: int(v) for k, v in value.items()
                    }
                else:
                    return_dict["optimized"][key] = value

                if "cost" in key and (
                    isinstance(value, np.float64) or isinstance(value, np.float32)
                ):
                    return_dict["optimized"][key] = float(value)
        else:
            return_dict["intermediate"] = self.intermediate
            return_dict["optimized"] = self.optimized

        # if we are using a shot adaptive optimizer, we need to add the number of shots to the result,
        # so if attribute n_shots is not empty, it is added to the dictionary
        if getattr(self, "n_shots", None) is not None:
            return_dict["n_shots"] = self.n_shots

        return (
            return_dict
            if exclude_keys == []
            else delete_keys_from_dict(return_dict, exclude_keys)
        )

    @classmethod
    def from_dict(
        cls, dictionary: dict, cost_hamiltonian: Union[Hamiltonian, None] = None
    ):
        """
        Creates a Results object from a dictionary (which is the output of the asdict method)
        Parameters
        ----------
        dictionary: `dict`
            The dictionary to create the QAOA Result object from
        Returns
        -------
        `Result`
            The Result object created from the dictionary
        """

        # deepcopy the dictionary, so that the original dictionary is not changed
        dictionary = copy.deepcopy(dictionary)

        # create a new instance of the class
        result = cls.__new__(cls)

        # set the attributes of the new instance, using the dictionary
        for key, value in dictionary.items():
            setattr(result, key, value)

        # if there is an input cost hamiltonian, it is added to the result
        if cost_hamiltonian is not None:
            result.cost_hamiltonian = cost_hamiltonian

        # if the measurement_outcomes are strings, they are converted to complex numbers
        if not isinstance(
            result.optimized["measurement_outcomes"], dict
        ) and isinstance(result.optimized["measurement_outcomes"][0], str):
            for i in range(len(result.optimized["measurement_outcomes"])):
                result.optimized["measurement_outcomes"][i] = complex(
                    result.optimized["measurement_outcomes"][i]
                )

            for i in range(len(result.intermediate["measurement_outcomes"])):
                for j in range(len(result.intermediate["measurement_outcomes"][i])):
                    result.intermediate["measurement_outcomes"][i][j] = complex(
                        result.intermediate["measurement_outcomes"][i][j]
                    )

        # if the measurement_outcomes are complex numbers, the backend is set to QAOABaseBackendStatevector
        if not isinstance(
            result.optimized["measurement_outcomes"], dict
        ) and isinstance(result.optimized["measurement_outcomes"][0], complex):
            setattr(result, "_QAOAResult__type_backend", QAOABaseBackendStatevector)
        else:
            setattr(result, "_QAOAResult__type_backend", "")

        # return the object
        return result

    @staticmethod
    def get_counts(measurement_outcomes):
        """
        Converts probabilities to counts when the measurement outcomes are a numpy array,
        that is a state vector

        Parameters
        ----------
        measurement_outcomes: `Union[np.array, dict]`
            The measurement outcome as returned by the Logger.
            It can either be a statevector or a count dictionary

        Returns
        -------
        `dict`
            The count dictionary obtained either throught the statevector or
            the actual measurement counts.
        """

        if isinstance(measurement_outcomes, type(np.array([]))):
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
                - self.evals["qfim_evals"]
                + 1,
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

    def plot_probabilities(
        self,
        n_states_to_keep=None,
        figsize=(10, 8),
        label="Probability distribution",
        color="tab:blue",
        ax=None,
    ):
        """
        Helper function to plot the probabilities corresponding to each basis states
        (with prob != 0) obtained from the optimized result

        Parameters
        ----------
        n_states_to_keep: 'int
            If the user passes a value, the plot will compile with the given value of states.
            Else,  an upper bound will be calculated depending on the
            total size of the measurement outcomes.
        figsize: `tuple`
            The size of the figure to be plotted. Defaults to (10,8).
        label: `str`
            The label of the cost line, defaults to 'Probability distribution'.
        color: `str`
            The color of the line. Defaults to 'tab:blue'.
        ax: 'matplotlib.axes._subplots.AxesSubplot'
            Axis on which to plot the graph. Deafults to None
        """

        outcome = self.optimized["measurement_outcomes"]

        # converting to counts dictionary if outcome is statevector
        if isinstance(outcome, type(np.array([]))):
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
        probs = counts / norm

        # total number of states / number of states with != 0 counts for shot simulators
        total = len(states)

        # number of states that fit without distortion in figure
        upper_bound = 40
        # default fontsize
        font = "medium"

        if n_states_to_keep:
            if n_states_to_keep > total:
                raise ValueError(
                    "n_states_to_keep must be smaller or equal than the total number"
                    f"of states in measurement outcome: {total}"
                )
            else:
                if n_states_to_keep > upper_bound:
                    print("number of states_to_keep exceeds the recommended value")
                    font = "small"

        # if states_to_keep is not given
        else:
            if total > upper_bound:
                n_states_to_keep = upper_bound
            else:
                n_states_to_keep = total

        # formatting labels
        labels = [
            r"$\left|{}\right>$".format(state) for state in states[:n_states_to_keep]
        ]
        labels.append("rest")

        # represent the bar with the addition of all the remaining probabilities
        rest = sum(probs[n_states_to_keep:])

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        colors = [color for _ in range(n_states_to_keep)] + ["xkcd:magenta"]

        ax.bar(labels, np.append(probs[:n_states_to_keep], rest), color=colors)
        ax.set_xlabel("Eigen-State")
        ax.set_ylabel("Probability")
        ax.set_title(label)
        ax.tick_params(axis="x", labelrotation=75, labelsize=font)
        ax.grid(True, axis="y", linestyle="--")

        print("states kept:", n_states_to_keep)
        return

    def plot_n_shots(
        self,
        figsize=(10, 8),
        param_to_plot=None,
        label=None,
        linestyle="--",
        color=None,
        ax=None,
        xlabel="Iterations",
        ylabel="Number of shots",
        title="Evolution of number of shots for gradient estimation",
    ):
        """
        Helper function to plot the evolution of the number of shots used for each evaluation of
        the cost function when computing the gradient.
        It only works for shot adaptive optimizers: cans and icans.
        If cans was used, the number of shots will be the same for each parameter at each iteration.
        If icans was used, the number of shots could be different for each parameter at each iteration.

        Parameters
        ----------
        figsize: `tuple`
            The size of the figure to be plotted. Defaults to (10,8).
        param_to_plot: `list[int]` or `int`
            The parameteres to plot. If None, all parameters will be plotted. Defaults to None.
            If a int is given, only the parameter with that index will be plotted.
            If a list of ints is given, the parameters with those indexes will be plotted.
        label: `list[str]` or `str`
            The label for each parameter. Defaults to Parameter {i}.
            If only one parameter is plot the label can be a string, otherwise it must be a list of strings.
        linestyle: `list[str]` or `str`
            The linestyle for each parameter. Defaults to '--' for all parameters.
            If it is a string all parameters will use it, if it a list of strings the linestyle of
            each parameter will depend on one string of the list.
        color: `list[str]` or `str`
            The color for each parameter. Defaults to None for all parameters (matplotlib will choose the colors).
            If only one parameter is plot the color can be a string, otherwise it must be a list of strings.
        ax: 'matplotlib.axes._subplots.AxesSubplot'
            Axis on which to plot the graph. If none is given, a new figure will be created.

        """

        if ax is None:
            ax = plt.subplots(figsize=figsize)[1]

        # creating a list of parameters to plot
        # if param_to_plot is not given, plot all the parameters
        if param_to_plot is None:
            param_to_plot = list(range(len(self.n_shots[0])))
        # if param_to_plot is a single value, convert to list
        elif type(param_to_plot) == int:
            param_to_plot = [param_to_plot]
        # if param_to_plot is not a list, raise error
        if type(param_to_plot) != list:
            raise ValueError(
                "`param_to_plot` must be a list of integers or a single integer"
            )
        else:
            for param in param_to_plot:
                assert param < len(
                    self.n_shots[0]
                ), f"`param_to_plot` must be a list of integers between 0 and {len(self.n_shots[0]) - 1}"

        # if label is not given, create a list of labels for each parameter (only if there is more than 1 parameter)
        if len(self.n_shots[0]) > 1:
            label = (
                [f"Parameter {i}" for i in param_to_plot] if label is None else label
            )
        else:
            label = ["n. shots per parameter"]

        # if only one parameter is plotted, convert label and color to list if they are string
        if len(param_to_plot) == 1:
            if type(label) == str:
                label = [label]
            if type(color) == str:
                color = [color]

        # if param_top_plot is a list and label or color are not lists, raise error
        if (type(label) != list) or (type(color) != list and color is not None):
            raise TypeError("`label` and `color` must be list of str")
        # if label is a list, check that all the elements are strings
        for lab in label:
            assert type(lab) == str, "`label` must be a list of strings"
        # if color is a list, check that all the elements are strings
        if color is not None:
            for c in color:
                assert type(c) == str, "`color` must be a list of strings"

        # if label and color are lists, check if they have the same length as param_to_plot
        if len(label) != len(param_to_plot) or (
            color is not None and len(color) != len(param_to_plot)
        ):
            raise ValueError(
                f"`param_to_plot`, `label` and `color` must have the same length, \
                    `param_to_plot` is a list of {len(param_to_plot)} elements"
            )

        # linestyle must be a string or a list of strings, if it is a string, convert it to a list of strings
        if type(linestyle) != str and type(linestyle) != list:
            raise TypeError("`linestyle` must be str or list")
        elif type(linestyle) == str:
            linestyle = [linestyle for _ in range(len(param_to_plot))]
        elif len(linestyle) != len(param_to_plot):
            raise ValueError(
                f"`linestyle` must have the same length as param_to_plot \
                    (length of `param_to_plot` is {len(param_to_plot)}), or be a string"
            )
        else:
            for ls in linestyle:
                assert type(ls) == str, "`linestyle` must be a list of strings"

        # plot the evolution of the number of shots for each parameter that is in param_to_plot
        transposed_n_shots = np.array(self.n_shots).T
        for i, values in enumerate([transposed_n_shots[j] for j in param_to_plot]):
            if color is None:
                ax.plot(values, label=label[i], linestyle=linestyle[i])
            else:
                ax.plot(values, label=label[i], linestyle=linestyle[i], color=color[i])

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.legend()
        ax.set_title(title)

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
                measurement_outcomes[solution_bitstring[args_sorted[ii]]] / total_shots
                for ii in range(n_bitstrings)
            ],
        }
        return best_results

    def calculate_statistics(self, include_intermediate=False) -> dict:
        """
        A function to calculate statistics of measurement outcomes associated with a QAOA workflow

        Parameters
        ----------
        include_intermediate: `bool`
            Whether it is necessary to calculate statistics for intermediate results. Defaults to False.
        """
        if (
            len(self.intermediate["measurement_outcomes"]) == 0
            and include_intermediate == True
        ):
            raise ValueError(
                "The underlying QAOA object does not seem to have any intermediate measurement result. Please, consider saving "
                "intermediate measurements during optimization by setting `optimization_progress=True` in your workflow."
            )

        if isinstance(self.optimized["measurement_outcomes"], dict):
            optimized_measurement_outcomes = self.optimized["measurement_outcomes"]
        elif isinstance(self.optimized["measurement_outcomes"], np.ndarray):
            optimized_measurement_outcomes = self.get_counts(
                self.optimized["measurement_outcomes"]
            )
        else:
            raise TypeError(
                f"The measurement outcome {type(self.optimized['measurement_outcomes'])} is not valid."
            )

        if isinstance(self.intermediate["measurement_outcomes"], list):
            if len(self.intermediate["measurement_outcomes"]) > 0:
                if isinstance(self.intermediate["measurement_outcomes"][0], dict):
                    intermediate_measurement_outcomes = self.intermediate[
                        "measurement_outcomes"
                    ]
                elif isinstance(
                    self.intermediate["measurement_outcomes"][0], np.ndarray
                ):
                    intermediate_measurement_outcomes = [
                        self.get_counts(i)
                        for i in self.intermediate["measurement_outcomes"]
                    ]
                else:
                    raise TypeError(
                        f"The measurement outcome {type(self.intermediate['measurement_outcomes'][0])} is not valid."
                    )
        else:
            raise TypeError(
                f"The measurement outcome {type(self.intermediate['measurement_outcomes'])} is not valid."
            )

        def sorted_mean_std_deviation(counts: dict):
            values = list(counts.values())
            return {
                "sorted": dict(
                    sorted(counts.items(), key=lambda x: x[1], reverse=True)
                ),
                "mean": np.mean(values),
                "std_deviation": np.std(values),
            }

        return {
            "intermediate": [
                sorted_mean_std_deviation(i) for i in intermediate_measurement_outcomes
            ]
            if include_intermediate
            else [],
            "optimized": sorted_mean_std_deviation(optimized_measurement_outcomes),
        }
