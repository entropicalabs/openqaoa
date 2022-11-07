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

from typing import Tuple, List, Union
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .baseparams import QAOACircuitParams, QAOAVariationalBaseParams, shapedArray, _is_iterable_empty

class QAOAVariationalExtendedParams(QAOAVariationalBaseParams):
    """
    QAOA parameters in their most general form with different angles for each
    operator.

    This means, that at the i-th timestep the evolution hamiltonian is given by

    .. math::

        H(t_i) = \sum_{\textrm{qubits } j} \beta_{ij} X_j
               + \sum_{\textrm{qubits } j} \gamma_{\textrm{single } ij} Z_j
               + \sum_{\textrm{qubit pairs} (jk)} \gamma_{\textrm{pair } i(jk)} Z_j Z_k

    and the complete circuit is then

    .. math::

        U = e^{-i H(t_p)} \cdots e^{-iH(t_1)}.

    Attributes
    ----------
    qaoa_circuit_params: QAOACircuitParams
		Specify the circuit parameters to construct circuit angles to be 
		used for training
    betas_singles: list
        2D array with the gammas from above for each timestep and qubit.
        1st index goes over the timelayers, 2nd over the qubits.
    betas_pairs : list
    gammas_pairs: list
    gammas_singles: list
    """

    def __init__(self,
                 qaoa_circuit_params: QAOACircuitParams,
                 betas_singles:List[Union[float,int]],
                 betas_pairs:List[Union[float,int]],
                 gammas_singles:List[Union[float,int]],
                 gammas_pairs:List[Union[float,int]]):
        # setup reg, qubits_singles and qubits_pairs
        super().__init__(qaoa_circuit_params)
        

        self.betas_singles = betas_singles if self.mixer_1q_coeffs else []
        self.betas_pairs = betas_pairs if self.mixer_2q_coeffs else []
        self.gammas_singles = gammas_singles if self.cost_1q_coeffs else []
        self.gammas_pairs = gammas_pairs if self.cost_2q_coeffs else []

    def __repr__(self):
        string = "Extended Parameterisation:\n"
        string += "\tp: " + str(self.p) + "\n"
        string += "Parameters:\n"
        string += "\tbetas_singles: " + str(self.betas_singles).replace("\n", ",") + "\n"
        string += "\tbetas_pairs: " + str(self.betas_pairs).replace("\n", ",") + "\n"
        string += "\tgammas_singles: " + str(self.gammas_singles).replace("\n", ",") + "\n"
        string += "\tgammas_pairs: " + str(self.gammas_pairs).replace("\n", ",") + "\n"
        return string

    def __len__(self):
        return self.p * (len(self.mixer_1q_coeffs) + len(self.mixer_2q_coeffs) +
                         len(self.cost_1q_coeffs) + len(self.cost_2q_coeffs))

    @shapedArray
    def betas_singles(self):
        return (self.p, len(self.mixer_1q_coeffs))

    @shapedArray
    def betas_pairs(self):
        return (self.p, len(self.mixer_2q_coeffs))

    @shapedArray
    def gammas_singles(self):
        return (self.p, len(self.cost_1q_coeffs))

    @shapedArray
    def gammas_pairs(self):
        return (self.p, len(self.cost_2q_coeffs))

    @property
    def mixer_1q_angles(self):
        return 2 * (self.mixer_1q_coeffs * self.betas_singles)
    
    @property
    def mixer_2q_angles(self):
        return 2 * (self.mixer_2q_coeffs * self.betas_pairs)

    @property
    def cost_1q_angles(self):
        return 2 * (self.cost_1q_coeffs * self.gammas_singles)

    @property
    def cost_2q_angles(self):
        return 2 * (self.cost_2q_coeffs * self.gammas_pairs)

    def update_from_raw(self, new_values):

        self.betas_singles = np.array(new_values[:len(self.mixer_1q_coeffs) * self.p])
        self.betas_singles = self.betas_singles.reshape((self.p, len(self.mixer_1q_coeffs)))
        
        new_values = new_values[len(self.betas_singles.flatten()):]

        self.betas_pairs = np.array(new_values[:len(self.mixer_2q_coeffs) * self.p])
        self.betas_pairs = self.betas_pairs.reshape((self.p, len(self.mixer_2q_coeffs)))
        
        new_values = new_values[len(self.betas_pairs.flatten()):]

        self.gammas_singles = np.array(new_values[:len(self.cost_1q_coeffs)* self.p])
        self.gammas_singles = self.gammas_singles.reshape((self.p, len(self.cost_1q_coeffs)))

        new_values = new_values[len(self.gammas_singles.flatten()):]

        self.gammas_pairs = np.array(new_values[:len(self.cost_2q_coeffs) * self.p])
        self.gammas_pairs = self.gammas_pairs.reshape((self.p, len(self.cost_2q_coeffs)))

        new_values = new_values[len(self.gammas_pairs.flatten()):]
        # PEP8 complains, but new_values could be np.array and not list!
        if len(new_values) != 0:
            raise RuntimeWarning("Incorrect dimension specified for new_values"
                                 "to construct the new betas and new gammas")

    def raw(self):
        raw_data = np.concatenate((self.betas_singles.flatten(),
                                   self.betas_pairs.flatten(),
                                   self.gammas_singles.flatten(),
                                   self.gammas_pairs.flatten()))
        return raw_data

    @classmethod
    def linear_ramp_from_hamiltonian(cls,
                                     qaoa_circuit_params:QAOACircuitParams,
                                     time: float = None):
        """

        Returns
        -------
        ExtendedParams
            The initial parameters according to a linear ramp for the Hamiltonian specified by
            register, terms, weights.

        Todo
        ----
        Refactor this s.t. it supers from __init__
        """
        # create evenly spaced timelayers at the centers of p intervals
        p = qaoa_circuit_params.p
        if time is None:
            time = float(0.7 * p)

        dt = time / p

        n_gamma_singles = len(qaoa_circuit_params.cost_hamiltonian.single_qubit_coeffs)
        n_gamma_pairs = len(qaoa_circuit_params.cost_hamiltonian.pair_qubit_coeffs)
        n_beta_singles = len(qaoa_circuit_params.mixer_hamiltonian.single_qubit_coeffs)
        n_beta_pairs = len(qaoa_circuit_params.mixer_hamiltonian.pair_qubit_coeffs)

        betas = np.linspace((dt / time) * (time * (1 - 0.5 / p)),
                            (dt / time) * (time * 0.5 / p), p)
        gammas = betas[::-1]
        
        betas_singles = betas.repeat(n_beta_singles).reshape(p, n_beta_singles)
        betas_pairs = betas.repeat(n_beta_pairs).reshape(p, n_beta_pairs)
        gammas_singles = gammas.repeat(n_gamma_singles).reshape(p, n_gamma_singles)
        gammas_pairs = gammas.repeat(n_gamma_pairs).reshape(p, n_gamma_pairs)

        params = cls(qaoa_circuit_params,betas_singles, betas_pairs, gammas_singles, gammas_pairs)
        return params
    
    @classmethod
    def random(cls, qaoa_circuit_params:QAOACircuitParams, seed:int = None):
        """
        Returns
        -------
        ExtendedParams
            Randomly initialised ``ExtendedParams`` object
        """
        if seed is not None:
            np.random.seed(seed)

        p=qaoa_circuit_params.p
        n_gamma_singles = len(qaoa_circuit_params.cost_hamiltonian.single_qubit_coeffs)
        n_gamma_pairs = len(qaoa_circuit_params.cost_hamiltonian.pair_qubit_coeffs)
        n_beta_singles = len(qaoa_circuit_params.mixer_hamiltonian.single_qubit_coeffs)
        n_beta_pairs = len(qaoa_circuit_params.mixer_hamiltonian.pair_qubit_coeffs)

        betas_singles = np.random.uniform(0,np.pi,(p,n_beta_singles))
        betas_pairs = np.random.uniform(0,np.pi,(p,n_beta_pairs))
        gammas_singles = np.random.uniform(0,np.pi,(p,n_gamma_singles))
        gammas_pairs = np.random.uniform(0,np.pi,(p,n_gamma_pairs))

        params = cls(qaoa_circuit_params,betas_singles, betas_pairs, gammas_singles, gammas_pairs)
        return params
    
    @classmethod
    def empty(cls, qaoa_circuit_params: QAOACircuitParams):
        """
        Initialise Extended parameters with empty arrays
        """        

        p=qaoa_circuit_params.p
        n_gamma_singles = len(qaoa_circuit_params.cost_hamiltonian.single_qubit_coeffs)
        n_gamma_pairs = len(qaoa_circuit_params.cost_hamiltonian.pair_qubit_coeffs)
        n_beta_singles = len(qaoa_circuit_params.mixer_hamiltonian.single_qubit_coeffs)
        n_beta_pairs = len(qaoa_circuit_params.mixer_hamiltonian.pair_qubit_coeffs)

        betas_singles = np.empty((p,n_beta_singles))
        betas_pairs = np.empty((p,n_beta_pairs))
        gammas_singles = np.empty((p,n_gamma_singles))
        gammas_pairs = np.empty((p,n_gamma_pairs))

        params = cls(qaoa_circuit_params,betas_singles, betas_pairs, gammas_singles, gammas_pairs)
        return params
    
    def get_constraints(self):
        """Constraints on the parameters for constrained parameters.

        Returns
        -------
        List[Tuple]:
            A list of tuples (0, upper_boundary) of constraints on the
            parameters s.t. we are exploiting the periodicity of the cost
            function. Useful for constrained optimizers.

        """
        beta_constraints = [(0, math.pi)] * (len(self.betas_singles.flatten() + 
                                             len(self.betas_pairs.flatten())))

        beta_pair_constraints = [(0, math.pi / w) for w in self.mixer_2q_coeffs]
        beta_pair_constraints *= self.p

        beta_single_constraints = [(0, math.pi / w) for w in self.mixer_1q_coeffs]
        beta_single_constraints *= self.p

        gamma_pair_constraints = [(0, 2 * math.pi / w) for w in self.cost_2q_coeffs]
        gamma_pair_constraints *= self.p

        gamma_single_constraints = [(0, 2 * math.pi / w) for w in self.cost_1q_coeffs]
        gamma_single_constraints *= self.p

        all_constraints = beta_single_constraints + beta_pair_constraints + \
                          gamma_single_constraints + gamma_pair_constraints

        return all_constraints

    def plot(self, ax=None, **kwargs):

        list_names_ = ["betas singles", "betas pairs", "gammas singles", "gammas pairs"] 
        list_values_ = [self.betas_singles % (2*(np.pi)), self.betas_pairs % (2*(np.pi)), 
                        self.gammas_singles % (2*(np.pi)), self.gammas_pairs % (2*(np.pi))] 

        list_names, list_values = list_names_.copy(), list_values_.copy()

        n_pop = 0
        for i in range(len(list_values_)):
            if list_values_[i].size == 0: 
                list_values.pop(i-n_pop)
                list_names.pop(i-n_pop)
                n_pop += 1

        n = len(list_values)
        p = self.p

        if ax is None:
            fig , ax = plt.subplots((n+1)//2, 2, figsize =(9, 9 if n>2 else 5))

        fig.tight_layout(pad=4.0)

        for k, (name, values) in enumerate(zip(list_names, list_values)):
            i, j = k//2 , k%2
            axes = ax[i,j] if n>2 else ax[k]

            if values.size == p:
                axes.plot(values.T[0], marker="^", color="green", ls="", **kwargs)
                axes.set_xlabel("p", fontsize=12)
                axes.set_title(name)
                axes.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            else:
                n_terms = values.shape[1]
                plt1 = axes.pcolor(np.arange(p), np.arange(n_terms) , values.T, vmin=0, vmax=2*np.pi, cmap="seismic")
                axes.set_aspect(p/n_terms)
                axes.xaxis.set_major_locator(MaxNLocator(integer=True))
                axes.yaxis.set_major_locator(MaxNLocator(integer=True))
                axes.set_ylabel("terms")
                axes.set_xlabel("p")
                axes.set_title(name)

                plt.colorbar(plt1, **kwargs)

        if k == 0:
            ax[1].axis('off')
        elif k == 2:
            ax[1,1].axis('off')

