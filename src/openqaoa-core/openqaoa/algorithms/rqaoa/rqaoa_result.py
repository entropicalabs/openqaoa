import copy
import numpy as np
from typing import List
import matplotlib.pyplot as plt

from ...utilities import delete_keys_from_dict
from .. import QAOAResult
from ...problems import QUBO


class RQAOAResult(dict):
    """
    A class to handle the results of RQAOA workflows
    It stores the results of the RQAOA optimization as a dictionary. With some custom methods.
    """

    def asdict(
        self,
        keep_cost_hamiltonian: bool = True,
        complex_to_string: bool = False,
        intermediate_measurements: bool = True,
        exclude_keys: List[str] = [],
    ):
        """
        Returns the results as a full dictionary, meaning that the objects of the
        intermediate steps are also converted to dictionaries.

        Parameters
        ----------
        keep_cost_hamiltonian : bool, optional
            If True, the cost Hamiltonian is kept in the dictionary, by default True.
        complex_to_string : bool, optional
            If True, the complex numbers are converted to strings, by default False.
            This is useful for JSON serialization.
        intermediate_measurements: bool, optional
            If True, intermediate measurements are included in the dump. If False,
            intermediate measurements are not included in the dump.
            Default is True.
        exclude_keys: `list[str]`, optional
            A list of keys to exclude from the returned dictionary.

        Returns
        -------
        dict
            The results as a dictionary.
        """

        results = {k: v for k, v in self.items()}
        results["intermediate_steps"] = []
        for step in self["intermediate_steps"]:
            results["intermediate_steps"].append(
                {
                    "counter": step["counter"],
                    "problem": step["problem"].asdict(),
                    "qaoa_results": step["qaoa_results"].asdict(
                        keep_cost_hamiltonian,
                        complex_to_string,
                        intermediate_measurements,
                    ),
                    "exp_vals_z": step["exp_vals_z"].tolist(),
                    "corr_matrix": step["corr_matrix"].tolist(),
                }
            )
        return (
            results
            if exclude_keys == []
            else delete_keys_from_dict(results, exclude_keys)
        )

    @classmethod
    def from_dict(cls, dictionary: dict):
        """
        Creates a RQAOAResult object from a dictionary (which is the output of the asdict method).
        Parameters
        ----------
        dictionary : dict
            The input dictionary.
        Returns
        -------
        RQAOAResult
            The RQAOAResult object.
        """

        # deepcopy the dictionary, so that the original dictionary is not changed
        dictionary = copy.deepcopy(dictionary)

        # create a new RQAOAResult object
        results = cls()

        # add the keys of the dictionary to the RQAOAResult object
        for key, value in dictionary.items():
            results[key] = value

        # convert the intermediate steps to objects
        for step in results["intermediate_steps"]:
            step["problem"] = QUBO.from_dict(step["problem"])
            step["qaoa_results"] = QAOAResult.from_dict(
                step["qaoa_results"], cost_hamiltonian=step["problem"].hamiltonian
            )
            step["exp_vals_z"] = np.array(step["exp_vals_z"])
            step["corr_matrix"] = np.array(step["corr_matrix"])

        return results

    def get_solution(self):
        """
        Returns the solution of the optimization.
        """
        return self["solution"]

    def get_qaoa_results(self, step):
        """
        Returns the i-th qaoa step of the RQAOA.
        """
        return self["intermediate_steps"][step]["qaoa_results"]

    def get_qaoa_optimized_angles(self, step):
        """
        Returns the optimized angles of the i-th qaoa step of the RQAOA.
        """
        return self.get_qaoa_results(step).optimized["angles"]

    def get_problem(self, step):
        """
        Returns the QUBO problem in the i-th step of the RQAOA.
        """
        return self["intermediate_steps"][step]["problem"]

    def get_hamiltonian(self, step):
        """
        Returns the Hamiltonian of the i-th step of the RQAOA.
        """
        return self.get_problem(step).hamiltonian

    def get_exp_vals_z(self, step):
        """
        Returns the expectation values of the Z operator of the i-th step of the RQAOA.
        """
        return self["intermediate_steps"][step]["exp_vals_z"]

    def get_corr_matrix(self, step):
        """
        Returns the correlation matrix of the i-th step of the RQAOA.
        """
        return self["intermediate_steps"][step]["corr_matrix"]

    def plot_corr_matrix(self, step, cmap="cool"):
        """
        Plots the correlation matrix of the i-th step of the RQAOA.
        TODO : add more options
        """
        plt.imshow(self.get_corr_matrix(step=step), cmap=cmap)
        plt.colorbar()
