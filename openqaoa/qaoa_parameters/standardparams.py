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

from typing import List, Union
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .baseparams import QAOACircuitParams, QAOAVariationalBaseParams, shapedArray, _is_iterable_empty


class QAOAVariationalStandardParams(QAOAVariationalBaseParams):
    r"""
    QAOA parameters that implement a state preparation circuit with

    .. math::

        e^{-i \beta_p H_0}
        e^{-i \gamma_p H_c}
        \cdots
        e^{-i \beta_0 H_0}
        e^{-i \gamma_0 H_c}

    This corresponds to the parametrization used by Farhi in his
    original paper [https://arxiv.org/abs/1411.4028]

    Parameters
    ----------
    hyperparameters:
        The hyperparameters containing the register, terms, weights, and the number of layers
        ``hyperparameters = (register, terms, weights, p)``
    parameters:
        Tuple containing ``(betas, gammas)`` with dimensions
        ``(p, p)``

    Attributes
    ----------
    betas: np.array
        1D array with the betas from above
    gammas: np.array
        1D array with the gamma from above
    """

    def __init__(self,
                 qaoa_circuit_params: QAOACircuitParams,
                 betas: List[Union[float, int]],
                 gammas: List[Union[float, int]]):
        # setup reg, qubits_singles and qubits_pairs
        super().__init__(qaoa_circuit_params)
        self.betas = np.array(betas)
        self.gammas = np.array(gammas)

    def __repr__(self):
        string = "Standard Parameterisation:\n"
        string += "\tp: " + str(self.p) + "\n"
        string += "Variational Parameters:\n"
        string += "\tbetas: " + str(self.betas) + "\n"
        string += "\tgammas: " + str(self.gammas) + "\n"
        return(string)

    def __len__(self):
        return self.p * 2

    @shapedArray
    def betas(self):
        return self.p

    @shapedArray
    def gammas(self):
        return self.p

    @property
    def mixer_1q_angles(self):
        return 2*np.outer(self.betas, self.mixer_1q_coeffs)

    @property
    def mixer_2q_angles(self):
        return 2*np.outer(self.betas, self.mixer_2q_coeffs)

    @property
    def cost_1q_angles(self):
        return 2*np.outer(self.gammas, self.cost_1q_coeffs)

    @property
    def cost_2q_angles(self):
        return 2*np.outer(self.gammas, self.cost_2q_coeffs)

    def update_from_raw(self, new_values):
        # overwrite self.betas with new ones
        self.betas = np.array(new_values[0:self.p])
        new_values = new_values[self.p:]    # cut betas from new_values
        self.gammas = np.array(new_values[0:self.p])
        new_values = new_values[self.p:]

        if len(new_values) != 0:
            raise RuntimeWarning("Incorrect dimension specified for new_values"
                                 "to construct the new betas and new gammas")

    def raw(self):
        raw_data = np.concatenate((self.betas, self.gammas))
        return raw_data

    @classmethod
    def linear_ramp_from_hamiltonian(cls,
                                     qaoa_circuit_params: QAOACircuitParams,
                                     time: float = None):
        """
        Returns
        -------
        StandardParams
            A ``StandardParams`` object with parameters according
            to a linear ramp schedule for the Hamiltonian specified by register, terms, weights.
        """
        p = qaoa_circuit_params.p

        if time is None:
            time = float(0.7 * p)
        # create evenly spaced timelayers at the centers of p intervals
        dt = time / p
        # fill betas, gammas_singles and gammas_pairs
        betas = np.linspace((dt / time) * (time * (1 - 0.5 / p)),
                            (dt / time) * (time * 0.5 / p), p)
        gammas = betas[::-1]
        # wrap it all nicely in a qaoa_parameters object
        params = cls(qaoa_circuit_params, betas, gammas)

        return params

    @classmethod
    def random(cls, qaoa_circuit_params: QAOACircuitParams, seed: int = None):
        """
        Returns
        -------
        StandardParams
            Randomly initialised ``StandardParams`` object
        """
        if seed is not None:
            np.random.seed(seed)

        betas = np.random.uniform(0, np.pi, qaoa_circuit_params.p)
        gammas = np.random.uniform(0, np.pi, qaoa_circuit_params.p)

        params = cls(qaoa_circuit_params, betas, gammas)

        return params

    @classmethod
    def empty(cls, qaoa_circuit_params: QAOACircuitParams):
        """
        Initialise Standard Variational params with empty arrays
        """
        p = qaoa_circuit_params.p
        betas = np.empty(p)
        gammas = np.empty(p)

        return cls(qaoa_circuit_params, betas, gammas)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.betas, label="betas", marker="s", ls="", **kwargs)
        ax.plot(self.gammas, label="gammas", marker="^", ls="", **kwargs)
        ax.set_xlabel("p", fontsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        
    def convert_to_ext(self, args_std):
        """
        Method that converts a list of parameters in the standard parametrisation form (args_std) to an equivalent list of parameters in the extended parametrisation form.
        
        PARAMETERS
        ----------
        args_std : 
            Parameters (a list of float) in the standard parametrisation form. 

        RETURNS
        -------
        args_ext:
            Parameters (a list of float) in the extended parametrisation form. 
        
        """

        terms_lst = [len(self.mixer_1q_coeffs), len(self.mixer_2q_coeffs), len(self.cost_1q_coeffs), len(self.cost_2q_coeffs)]
        terms_lst_p = np.repeat(terms_lst, [self.p]*len(terms_lst))
        args_ext = []
        for i in range(4):  # 4 types of terms
            for j in range(self.p):
                for k in range(terms_lst_p[i*self.p + j]):
                    if i < 2:
                        args_ext.append(args_std[j])
                    else:
                        args_ext.append(
                            args_std[j + int(len(args_std)/2)])

        return args_ext


class QAOAVariationalStandardWithBiasParams(QAOAVariationalBaseParams):
    r"""
    QAOA parameters that implement a state preparation circuit with

    .. math::

        e^{-i \beta_p H_0}
        e^{-i \gamma_{\textrm{singles}, p} H_{c, \textrm{singles}}}
        e^{-i \gamma_{\textrm{pairs}, p} H_{c, \textrm{pairs}}}
        \cdots
        e^{-i \beta_0 H_0}
        e^{-i \gamma_{\textrm{singles}, 0} H_{c, \textrm{singles}}}
        e^{-i \gamma_{\textrm{pairs}, 0} H_{c, \textrm{pairs}}}

    where the cost hamiltonian is split into :math:`H_{c, \textrm{singles}}`
    the bias terms, that act on only one qubit, and
    :math:`H_{c, \textrm{pairs}}` the coupling terms, that act on two qubits.

    Parameters
    ----------
    hyperparameters:
        The hyperparameters containing the register, terms, weights, and the number of layers
        ``hyperparameters = (register, terms, weights, p)``
    parameters:
        Tuple containing ``(betas, gammas_singles, gammas_pairs)`` with
        dimensions ``(p, p, p)``

    Attributes
    ----------
    betas: np.array
        A 1D array containing the betas from above for each timestep
    gammas_pairs: np.array
        A 1D array containing the gammas_singles from above for each timestep
    gammas_singles: np.array
        A 1D array containing the gammas_pairs from above for each timestep
    """

    def __init__(self,
                 qaoa_circuit_params: QAOACircuitParams,
                 betas: List[Union[float, int]],
                 gammas_singles: List[Union[float, int]],
                 gammas_pairs: List[Union[float, int]]):

        super().__init__(qaoa_circuit_params)
        if not self.cost_1q_coeffs or not self.cost_2q_coeffs:
            raise RuntimeError(f"Please choose {type(self).__name__} parameterisation for "
                               "problems containing both Cost One-Qubit and Two-Qubit terms")

        self.betas = np.array(betas)
        self.gammas_singles = np.array(gammas_singles)
        self.gammas_pairs = np.array(gammas_pairs)

    def __repr__(self):
        string = "Standard with Bias Parameterisation:\n"
        string += "\tp: " + str(self.p) + "\n"
        string += "Variational Parameters:\n"
        string += "\tbetas: " + str(self.betas) + "\n"
        string += "\tgammas_singles: " + str(self.gammas_singles) + "\n"
        string += "\tgammas_pairs: " + str(self.gammas_pairs) + "\n"
        return(string)

    def __len__(self):
        return self.p * 3

    @shapedArray
    def betas(self):
        return self.p

    @shapedArray
    def gammas_singles(self):
        return self.p

    @shapedArray
    def gammas_pairs(self):
        return self.p

    @property
    def mixer_1q_angles(self):
        return 2*np.outer(self.betas, self.mixer_1q_coeffs)

    @property
    def mixer_2q_angles(self):
        return 2*np.outer(self.betas, self.mixer_2q_coeffs)

    @property
    def cost_1q_angles(self):
        return 2*np.outer(self.gammas_singles, self.cost_1q_coeffs)

    @property
    def cost_2q_angles(self):
        return 2*np.outer(self.gammas_pairs, self.cost_2q_coeffs)

    def update_from_raw(self, new_values):
        # overwrite self.betas with new ones
        self.betas = np.array(new_values[0:self.p])
        new_values = new_values[self.p:]    # cut betas from new_values
        self.gammas_singles = np.array(new_values[0:self.p])
        new_values = new_values[self.p:]
        self.gammas_pairs = np.array(new_values[0:self.p])
        new_values = new_values[self.p:]

        if len(new_values) != 0:
            raise RuntimeWarning("Incorrect dimension specified for new_values"
                                 "to construct the new betas and new gammas")

    def raw(self):
        raw_data = np.concatenate((self.betas,
                                   self.gammas_singles,
                                   self.gammas_pairs))
        return raw_data

    @classmethod
    def linear_ramp_from_hamiltonian(cls,
                                     qaoa_circuit_params: QAOACircuitParams,
                                     time: float = None):
        """
        Returns
        -------
        StandardParams
            A ``StandardParams`` object with parameters according
            to a linear ramp schedule for the Hamiltonian specified by register, terms, weights.
        """
        p = qaoa_circuit_params.p

        if time is None:
            time = float(0.7 * p)
        # create evenly spaced timelayers at the centers of p intervals
        dt = time / p
        # fill betas, gammas_singles and gammas_pairs
        betas = np.linspace((dt / time) * (time * (1 - 0.5 / p)),
                            (dt / time) * (time * 0.5 / p), p)
        gammas_singles = betas[::-1]
        gammas_pairs = betas[::-1]

        params = cls(qaoa_circuit_params, betas, gammas_singles, gammas_pairs)

        return params

    @classmethod
    def random(cls, qaoa_circuit_params: QAOACircuitParams, seed: int = None):
        """
        Returns
        -------
        StandardParams
            Randomly initialised ``StandardParams`` object
        """
        if seed is not None:
            np.random.seed(seed)

        betas = np.random.uniform(0, np.pi, qaoa_circuit_params.p)
        gammas_singles = np.random.uniform(0, np.pi, qaoa_circuit_params.p)
        gammas_pairs = np.random.uniform(0, np.pi, qaoa_circuit_params.p)

        params = cls(qaoa_circuit_params, betas, gammas_singles, gammas_pairs)
        return params

    @classmethod
    def empty(cls, qaoa_circuit_params: QAOACircuitParams):
        """
        Initialise Standard Variational params with empty arrays
        """
        p = qaoa_circuit_params.p
        betas = np.empty(p)
        gammas_singles = np.empty(p)
        gammas_pairs = np.empty(p)

        return cls(qaoa_circuit_params, betas, gammas_singles, gammas_pairs)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.betas, label="betas", marker="s", ls="", **kwargs)
        if not _is_iterable_empty(self.gammas_singles):
            ax.plot(self.gammas_singles,
                    label="gammas_singles", marker="^", ls="", **kwargs)
        if not _is_iterable_empty(self.gammas_pairs):
            ax.plot(self.gammas_pairs,
                    label="gammas_pairs", marker="v", ls="", **kwargs)
        ax.set_xlabel("p")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.grid(linestyle='--')
        ax.legend()
