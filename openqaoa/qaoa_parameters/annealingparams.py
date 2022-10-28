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

from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from ..qaoa_parameters.baseparams import (QAOACircuitParams,
                                          QAOAVariationalBaseParams,
                                          shapedArray,
                                          _is_iterable_empty)


class QAOAVariationalAnnealingParams(QAOAVariationalBaseParams):
    """
    QAOA parameters that implement a state preparation circuit of the form

    .. math::

        U = e^{-i (1-s(t_p)) H_M \Delta t} e^{-i s(t_p) H_C \Delta t} \cdots e^{-i(1-s(t_1)H_M \Delta t} e^{-i s(t_1) H_C \Delta t}

    where the :math:`s(t_i) =: s_i` are the variable parameters and
    :math:`\\Delta t= \\frac{T}{N}`.
    So the schedule is specified by specifying s(t) at evenly spaced timelayers.

    Parameters
    ----------
    hyperparameters:
        The hyperparameters containing the register, terms, weights, the number of layers
        and the total annealing time ``hyperparameters = (register, terms, weights,
        p, time)``
    parameters : Tuple
        Tuple containing ``(schedule values)`` of length ``p``

    Attributes
    ----------
    schedule: np.array
        An 1D array holding the values of the schedule function at each timestep.
    """

    def __init__(self,
                 qaoa_circuit_params: QAOACircuitParams,
                 total_annealing_time: float,
                 schedule: List[Union[float, int]]):

        # setup reg, qubits_singles and qubits_pairs
        super().__init__(qaoa_circuit_params)
        assert total_annealing_time is not None, f"Please specify total_annealing_time to use {type(self).__name__}"
        self.total_annealing_time = total_annealing_time
        self.schedule = np.array(schedule)
        self.schedule = np.array(schedule)

        self.dt = self.total_annealing_time / self.p
        self.mixer_time = (1 - self.schedule) * self.dt
        self.cost_time = self.schedule * self.dt

    def __repr__(self):
        string = "Annealing Parameterisation:\n"
        string += "\tp: " + str(self.p) + "\n"
        string += "Variational Parameters:\n"
        string += "\tschedule: " + str(self.schedule)
        return string

    def __len__(self):
        return self.p

    @shapedArray
    def schedule(self):
        return self.p

    @property
    def mixer_1q_angles(self):
        return 2 * np.outer(self.mixer_time, self.mixer_1q_coeffs)

    @property
    def mixer_2q_angles(self) -> np.ndarray:
        return 2 * np.outer(self.mixer_time, self.mixer_2q_coeffs)

    @property
    def cost_1q_angles(self):
        return 2 * np.outer(self.cost_time, self.cost_1q_coeffs)

    @property
    def cost_2q_angles(self):
        return 2 * np.outer(self.cost_time, self.cost_2q_coeffs)

    def update_from_raw(self, new_values):
        if len(new_values) != self.p:
            raise RuntimeWarning(
                "the new times should have length p")
        self.schedule = np.array(new_values)
        # udpate mixer_time and cost_time too
        self.mixer_time = (1 - self.schedule) * self.dt
        self.cost_time = self.schedule * self.dt

    def raw(self):
        return self.schedule

    @classmethod
    def linear_ramp_from_hamiltonian(cls,
                                     qaoa_circuit_params: QAOACircuitParams,
                                     total_annealing_time: float = None,
                                     time: float = None):
        """
        Returns
        -------
        AnnealingParams :
            An ``AnnealingParams`` object corresponding to
            a linear ramp schedule.
        """
        p = qaoa_circuit_params.p
        total_annealing_time = 0.7 * \
            p if total_annealing_time is None else total_annealing_time
        schedule = np.linspace(0.5 / p, 1 - 0.5 / p, p)

        # wrap it all nicely in a qaoa_parameters object
        #params = cls((register, terms, weights, p, time), (schedule))
        params = cls(qaoa_circuit_params, total_annealing_time, schedule)
        return params

    @classmethod
    def random(cls, qaoa_circuit_params: QAOACircuitParams,
               total_annealing_time: float, seed: int = None):
        """
        Returns
        -------
        AnnealingParams
            randomly initialised AnnealingParams object
        """
        if seed is not None:
            np.random.seed(seed)
        schedule = np.random.uniform(0, 1, qaoa_circuit_params.p)

        return cls(qaoa_circuit_params, total_annealing_time, schedule)

    @classmethod
    def empty(cls, qaoa_circuit_params: QAOACircuitParams, total_annealing_time: float):
        schedule = np.empty((qaoa_circuit_params.p))
        return cls(qaoa_circuit_params, total_annealing_time, schedule)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.schedule, marker="s", **kwargs)
        ax.set_xlabel("p", fontsize=14)
        ax.set_ylabel("s(t)", fontsize=14)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
