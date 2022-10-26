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
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.fftpack import dct, dst

from ..qaoa_parameters.baseparams import (QAOACircuitParams,
                                          QAOAVariationalBaseParams,
                                          shapedArray,
                                          _is_iterable_empty)


class QAOAVariationalFourierParams(QAOAVariationalBaseParams):
    """
    The QAOA parameters as the sine/cosine transform of the original gammas
    and x_rotation_angles. See "Quantum Approximate Optimization Algorithm:
    Performance, Mechanism, and Implementation on Near-Term Devices"
    [https://arxiv.org/abs/1812.01041] for a detailed description.

    Parameters
    ----------
    qaoa_circuit_params:
        The hyperparameters containing the register, terms, weights, the number of layers
        and the total annealing time
        ``hyperparameters = (register, terms, weights, p, q)``.
        ``q`` is the number of fourier coefficients. For ``q == p`` we
        have the full expressivity of ``StandardParams``.
        More are redundant.
    parameters:
        Tuple containing ``(v, u)`` with dimensions
        ``(q, q)``

    Attributes
    ----------
    q : int
        The number of coefficients for the discrete sine and cosine transforms
        below
    u : np.array
        The discrete sine transform of the ``gammas`` in
        ``StandardParams``
    v : np.array
        The discrete cosine transform of the ``betas`` in
        ``StandardParams``
    """

    def __init__(self,
                 qaoa_circuit_params: QAOACircuitParams,
                 q: int,
                 v: List[Union[int, float]],
                 u: List[Union[int, float]]):
        # setup reg, qubits_singles and qubits_pairs
        super().__init__(qaoa_circuit_params)
        assert q is not None, f"Depth q for {type(self).__name__} must be specified"
        self.q = q
        self.u = u
        self.v = v
        self.betas = dct(self.v, n=self.p)
        self.gammas = dst(self.u, n=self.p)

    def __repr__(self):
        string = "Fourier Parameterisation:\n"
        string += "\tp: " + str(self.p) + "\n"
        string += "\tq: " + str(self.q) + "\n"
        string += "Variational Parameters:\n"
        string += "\tv: " + str(self.v) + "\n"
        string += "\tu: " + str(self.u) + "\n"
        return(string)

    def __len__(self):
        return 2 * self.q

    @shapedArray
    def v(self):
        return self.q

    @shapedArray
    def u(self):
        return self.q

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
        # overwrite x_rotation_angles with new ones
        self.v = np.array(new_values[0:self.q])
        # cut x_rotation_angles from new_values
        new_values = new_values[self.q:]
        self.u = np.array(new_values[0:self.q])
        new_values = new_values[self.q:]

        if len(new_values) != 0:
            raise RuntimeWarning("Incorrect dimension specified for new_values"
                                 "to construct the new v's and new u's")

        # update the betas and gammas too
        self.betas = dct(self.v, n=self.p)
        self.gammas = dst(self.u, n=self.p)

    def raw(self):
        raw_data = np.concatenate((self.v, self.u))
        return raw_data

    @classmethod
    def linear_ramp_from_hamiltonian(cls,
                                     qaoa_circuit_params: QAOACircuitParams,
                                     q: int,
                                     time: float = None):
        """
        NOTE: rather than implement an exact linear schedule, 
        this instead implements the lowest frequency component, 
        i.e. a sine curve for gammas, and a cosine for betas.

        Parameters
        ----------
        qaoa_circuit_params:
                        QAOACircuitParams object containing information about terms,weights,register and p

        time:
            total time. Set to ``0.7*p`` if ``None`` is passed.

        Returns
        -------
        FourierParams:
            A ``FourierParams`` object with initial parameters
            corresponding to a the 0th order Fourier component 
            (a sine curve for gammas, cosine for betas)

        ToDo
        ----
        Make a more informed choice of the default value for ``q``. Probably
        depending on ``n_qubits``
        """
        assert q is not None, f"Depth q for {cls.__name__} must be specified"
        
        # Set default time
        if time is None:
            time = 0.7 * qaoa_circuit_params.p

        # fill x_rotation_angles, z_rotation_angles and zz_rotation_angles
        # Todo make this an easier expresssion
        v = np.zeros(q)
        v[0] = 0.5 * time / qaoa_circuit_params.p
        u = np.copy(v)

        # wrap it all nicely in a qaoa_parameters object
        params = cls(qaoa_circuit_params, q, v, u)
        return params

    @classmethod
    def random(cls, qaoa_circuit_params: QAOACircuitParams, q: int, seed: int = None):
        """
        Returns
        -------
        FourierParams
            randomly initialised ``FourierParams`` object
        """
        if seed is not None:
            np.random.seed(seed)

        v = np.random.uniform(0, np.pi, q)
        u = np.random.uniform(0, np.pi, q)

        params = cls(qaoa_circuit_params, q, v, u)
        return params

    @classmethod
    def empty(cls, qaoa_circuit_params: QAOACircuitParams, q: int):
        """
        Initialise the Fourier Params u,v with empty arrays
        """
        u = np.empty((q))
        v = np.empty((q))
        return cls(qaoa_circuit_params, q, v, u)

    def plot(self, ax=None, **kwargs):
        # warnings.warn("Plotting the gammas and x_rotation_angles through DCT "
        #              "and DST. If you are interested in v, u you can access "
        #              "them via params.v, params.u")

        if ax is None:        
            fig, ax = plt.subplots(2, figsize =(7, 9))

        fig.tight_layout(pad=4.0)

        ax[0].plot(self.v, label="v", marker="s", ls="", **kwargs)
        ax[0].plot(self.u, label="u", marker="^", ls="", **kwargs)
        ax[0].set_xlabel("q")
        ax[0].legend()        
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        ax[1].plot(dct(self.v, n=self.p),
                label="betas", marker="s", ls="", **kwargs)
        ax[1].plot(dst(self.u, n=self.p),
                label="gammas", marker="^", ls="", **kwargs)
        ax[1].set_xlabel("p")
        ax[1].legend()
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))


class QAOAVariationalFourierWithBiasParams(QAOAVariationalBaseParams):
    """
    The QAOA parameters as the sine/cosine transform of the original gammas
    and x_rotation_angles. See "Quantum Approximate Optimization Algorithm:
    Performance, Mechanism, and Implementation on Near-Term Devices"
    [https://arxiv.org/abs/1812.01041] for a detailed description.

    Parameters
    ----------
    qaoa_circuit_params:
        The hyperparameters containing the register, terms, weights, the number of layers
        and the total annealing time
        ``hyperparameters = (register, terms, weights, p, q)``.
        ``q`` is the number of fourier coefficients. For ``q == p`` we
        have the full expressivity of ``StandardWithBiasParams``.
        More are redundant.
    parameters:
        Tuple containing ``(v, u_singles, u_pairs)`` with dimensions
        ``(q, q, q)``

    Attributes
    ----------
    q : int
        The number of coefficients for the discrete sine and cosine transforms
        below
    u_pairs : np.array
        The discrete sine transform of the ``gammas_pairs`` in
        ``StandardWithBiasParams``
    u_singles : np.array
        The discrete sine transform of the ``gammas_singles`` in
        ``StandardWithBiasParams``
    v : np.array
        The discrete cosine transform of the betas in
        ``StandardWithBiasParams``
    """

    def __init__(self,
                 qaoa_circuit_params: QAOACircuitParams,
                 q: int,
                 v: List[Union[float, int]],
                 u_singles: List[Union[float, int]],
                 u_pairs: List[Union[float, int]]):

        # setup reg, qubits_singles and qubits_pairs
        super().__init__(qaoa_circuit_params)
        if not self.cost_1q_coeffs or not self.cost_2q_coeffs:
            raise RuntimeError(f"Please choose {type(self).__name__} parameterisation for problems "
                               "containing both Cost One-Qubit and Two-Qubit terms")
        assert q is not None, f"Depth q for {type(self).__name__} must be specified"

        self.q = q
        self.v = v
        self.u_singles = u_singles
        self.u_pairs = u_pairs

        self.betas = dct(self.v, n=self.p)
        self.gammas_singles = dst(self.u_singles, n=self.p)
        self.gammas_pairs = dst(self.u_pairs, n=self.p)

    def __repr__(self):
        string = "Fourier Parameterisation:\n"
        string += "\tp: " + str(self.p) + "\n"
        string += "\tq: " + str(self.q) + "\n"
        string += "Variational Parameters:\n"
        string += "\tv: " + str(self.v) + "\n"
        string += "\tu_singles: " + str(self.u_singles) + "\n"
        string += "\tu_pairs: " + str(self.u_pairs) + "\n"
        return(string)

    def __len__(self):
        return 3 * self.q

    @shapedArray
    def v(self):
        return self.q

    @shapedArray
    def u_singles(self):
        return self.q

    @shapedArray
    def u_pairs(self):
        return self.q

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

        # overwrite x_rotation_angles with new ones
        self.v = np.array(new_values[0:self.q])
        # cut x_rotation_angles from new_values
        new_values = new_values[self.q:]

        self.u_singles = np.array(new_values[0:self.q])
        new_values = new_values[self.q:]

        self.u_pairs = np.array(new_values[0:self.q])
        new_values = new_values[self.q:]

        if len(new_values) != 0:
            raise RuntimeWarning("Incorrect dimension specified for new_values"
                                 "to construct the new v's and new u's")

        # update the betas, gammas too
        self.betas = dct(self.v, n=self.p)
        self.gammas_singles = dst(self.u_singles, n=self.p)
        self.gammas_pairs = dst(self.u_pairs, n=self.p)

    def raw(self):
        raw_data = np.concatenate((self.v,
                                   self.u_singles,
                                   self.u_pairs))
        return raw_data

    @classmethod
    def linear_ramp_from_hamiltonian(cls,
                                     qaoa_circuit_params: QAOACircuitParams,
                                     q: int,
                                     time: float = None):
        """
        Parameters
        ----------
        qaoa_circuit_params:
            Hyper Parameters passed as an object of ``Type[QAOACircuitParams]``

        time: 
            Time for creating the linear ramp schedule.
            Defaults to ``0.7*p`` if None

        Returns
        -------
        FourierWithBiasParams:
            A ``FourierWithBiasParams`` object with initial parameters
            corresponding to a linear ramp annealing schedule

        ToDo
        ----
        Make a more informed choice of the default value for ``q``. Probably
        depending on ``n_qubits``
        """
        
        assert q is not None, f"Depth q for {cls.__name__} must be specified"

        if time is None:
            time = 0.7 * qaoa_circuit_params.p

        v = np.zeros(q)
        v[0] = 0.5 * time / qaoa_circuit_params.p
        u_singles = np.copy(v)
        u_pairs = np.copy(v)
        # wrap it all nicely in a qaoa_parameters object
        params = cls(qaoa_circuit_params, q, v, u_singles, u_pairs)
        return params

    @classmethod
    def random(cls, qaoa_circuit_params: QAOACircuitParams, q: int, seed: int = None):
        """
        Returns
        -------
        FourierWithBiasParams
            randomly initialised ``FourierWithBiasParams`` object
        """
        if seed is not None:
            np.random.seed(seed)

        v = np.random.uniform(0, np.pi, q)
        u_singles = np.random.uniform(0, np.pi, q)
        u_pairs = np.random.uniform(0, np.pi, q)

        params = cls(qaoa_circuit_params, q, v, u_singles, u_pairs)
        return params

    @classmethod
    def empty(cls, qaoa_circuit_params: QAOACircuitParams, q: int):
        """
        Initialise Fourier parameters with bias with an empty array
        """
        v = np.empty((q))
        u_singles = np.empty((q))
        u_pairs = np.empty((q))
        return cls(qaoa_circuit_params, q, v, u_singles, u_pairs)

    def plot(self, ax=None, **kwargs):
        # warnings.warn("Plotting the gammas and x_rotation_angles through DCT "
        #              "and DST. If you are interested in v, u_singles and "
        #              "u_pairs you can access them via params.v, "
        #              "params.u_singles, params.u_pairs")
        if ax is None:
            fig, ax = plt.subplots(2, figsize =(7, 9))

        fig.tight_layout(pad=4.0)

        ax[0].plot(self.v, label="v", marker="s", ls="", **kwargs)
        ax[0].plot(self.u_singles, label="u_singles", marker="^", ls="", **kwargs)
        ax[0].plot(self.u_pairs, label="u_pairs", marker="v", ls="", **kwargs)
        ax[0].set_xlabel("q")
        ax[0].legend()
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        ax[1].plot(dct(self.v, n=self.p),
                label="betas", marker="s", ls="", **kwargs)
        if not _is_iterable_empty(self.u_singles):
            ax[1].plot(dst(self.u_singles, n=self.p),
                    label="gammas_singles", marker="^", ls="", **kwargs)
        if not _is_iterable_empty(self.u_pairs):
            ax[1].plot(dst(self.u_pairs, n=self.p),
                    label="gammas_pairs", marker="v", ls="", **kwargs)
        ax[1].set_xlabel("p")
        ax[1].legend()
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))


class QAOAVariationalFourierExtendedParams(QAOAVariationalBaseParams):
    r"""
    The Fourier pendant to ExtendedParams.

    Parameters
    ----------
    qaoa_circuit_params: ``QAOACircuitParams``
        object containing information about terms,weights,register and p

    q: ``int``
        The parameter depth for u and v Fourier params

    parameters:
        Tuple containing ``(v, u_singles, u_pairs)`` with dimensions
        ``((q x nqubits), (q x nsingle_terms), (q x npair_terms))``

    Attributes
    ----------
    q : int
        The number of coefficients for the discrete sine and cosine transforms
        below
    v: np.array
        The discrete cosine transform of the ``betas`` in ``ExtendedParams``
    u_singles: np.array
        The discrete sine transform of the ``gammas_singles`` in
        ``ExtendedParams``
    u_pairs: np.array
        The discrete sine transform of the ``gammas_pairs`` in
        ``ExtendedParams``
    """

    def __init__(self,
                 qaoa_circuit_params: QAOACircuitParams,
                 q: int,
                 v_singles: List[Union[float, int]],
                 v_pairs: List[Union[float, int]],
                 u_singles: List[Union[float, int]],
                 u_pairs: List[Union[float, int]]):

        # setup reg, qubits_singles and qubits_pairs
        super().__init__(qaoa_circuit_params)
        assert q is not None, f"Depth q for {type(self).__name__} must be specified"
        self.q = q
        self.v_singles = v_singles
        self.v_pairs = v_pairs
        self.u_singles = u_singles
        self.u_pairs = u_pairs

        self.betas_singles = dct(self.v_singles, n=self.p, axis=0)
        self.betas_pairs = dct(self.v_pairs, n=self.p, axis=0)
        self.gammas_singles = dst(self.u_singles, n=self.p, axis=0)
        self.gammas_pairs = dst(self.u_pairs, n=self.p, axis=0)

    def __repr__(self):
        string = "Fourier Parameterisation:\n"
        string += "\tp: " + str(self.p) + "\n"
        string += "\tq: " + str(self.q) + "\n"
        string += "Variational Parameters:\n"
        string += "\tv_singles: " + \
            str(self.v_singles).replace("\n", ",") + "\n"
        string += "\tv_pairs: " + str(self.v_pairs).replace("\n", ",") + "\n"
        string += "\tu_singles: " + str(self.u_singles) + "\n"
        string += "\tu_pairs: " + str(self.u_pairs) + "\n"
        return(string)

    def __len__(self):
        return self.q * (len(self.mixer_1q_coeffs) + len(self.mixer_2q_coeffs)
                         + len(self.cost_1q_coeffs) + len(self.cost_2q_coeffs))

    @shapedArray
    def v_singles(self):
        return (self.q, len(self.mixer_1q_coeffs))

    @shapedArray
    def v_pairs(self):
        return (self.q, len(self.mixer_2q_coeffs))

    @shapedArray
    def u_singles(self):
        return (self.q, len(self.cost_1q_coeffs))

    @shapedArray
    def u_pairs(self):
        return (self.q, len(self.cost_2q_coeffs))

    @property
    def mixer_1q_angles(self):
        return 2 * (self.mixer_1q_coeffs * self.betas_singles)

    @property
    def mixer_2q_angles(self):
        return 2 * (self.mixer_2q_coeffs * self.betas_pairs)

    @property
    def cost_1q_angles(self):
        if self.u_singles.size > 0:
            return 2 * (self.cost_1q_coeffs * self.gammas_singles)
        else:
            return 2 * (self.cost_1q_coeffs * np.empty(shape=(self.p, 0)))

    @property
    def cost_2q_angles(self):
        if self.u_pairs.size > 0:
            return 2 * (self.cost_2q_coeffs * self.gammas_pairs)
        else:
            return 2 * (self.cost_2q_coeffs * np.empty(shape=(self.p, 0)))

    def update_from_raw(self, new_values):

        self.v_singles = np.array(
            new_values[:len(self.mixer_1q_coeffs) * self.q])
        self.v_singles = self.v_singles.reshape(
            (self.q, len(self.mixer_1q_coeffs)))
        new_values = new_values[self.q * len(self.mixer_1q_coeffs):]

        self.v_pairs = np.array(
            new_values[:len(self.mixer_2q_coeffs) * self.q])
        self.v_pairs = self.v_pairs.reshape(
            (self.q, len(self.mixer_2q_coeffs)))
        new_values = new_values[self.q * len(self.mixer_2q_coeffs):]

        self.u_singles = np.array(
            new_values[:len(self.cost_1q_coeffs) * self.q])
        self.u_singles = self.u_singles.reshape(
            (self.q, len(self.cost_1q_coeffs)))
        new_values = new_values[self.q * len(self.cost_1q_coeffs):]

        self.u_pairs = np.array(new_values[:len(self.cost_2q_coeffs) * self.q])
        self.u_pairs = self.u_pairs.reshape((self.q, len(self.cost_2q_coeffs)))
        new_values = new_values[self.q * len(self.cost_2q_coeffs):]

        # PEP8 complains, but new_values could be np.array and not list!
        if len(new_values) != 0:
            raise RuntimeWarning("Incorrect dimension specified for new_values"
                                 "to construct the new v's and new u's")

        # update the betas, gammas too
        self.betas_singles = dct(self.v_singles, n=self.p, axis=0)
        self.betas_pairs = dct(self.v_pairs, n=self.p, axis=0)
        self.gammas_singles = dst(self.u_singles, n=self.p, axis=0)
        self.gammas_pairs = dst(self.u_pairs, n=self.p, axis=0)

    def raw(self):
        raw_data = np.concatenate((self.v_singles.flatten(),
                                   self.v_pairs.flatten(),
                                   self.u_singles.flatten(),
                                   self.u_pairs.flatten()))
        return raw_data

    @classmethod
    def linear_ramp_from_hamiltonian(cls,
                                     qaoa_circuit_params: QAOACircuitParams,
                                     q: int,
                                     time: float = None):
        """
        Parameters
        ----------
        qaoa_circuit_params: ``QAOACircuitParams``
            an object containing information about the QAOA circuit

        q: ``int``
            The q-depth of the circuit parameters

        time: ``float`` 
            Time for creating the linear ramp schedule.
            Defaults to ``0.7*p`` if None

        Returns
        -------
        FourierExtendedParams
            The initial parameters according to a linear ramp for
            for the Hamiltonian specified by register, terms, weights.

        """
       
        assert q is not None, f"Depth q for {cls.__name__} must be specified"
        
        # create evenly spaced timelayers at the centers of p intervals
        p = qaoa_circuit_params.p

        if time is None:
            time = float(0.7 * p)

        n_u_singles = len(
            qaoa_circuit_params.cost_hamiltonian.single_qubit_coeffs)
        n_u_pairs = len(qaoa_circuit_params.cost_hamiltonian.pair_qubit_coeffs)
        n_v_singles = len(
            qaoa_circuit_params.mixer_hamiltonian.single_qubit_coeffs)
        n_v_pairs = len(
            qaoa_circuit_params.mixer_hamiltonian.pair_qubit_coeffs)

        v = np.zeros(q)
        v[0] = 0.5 * time / p
        u = np.copy(v)

        v_singles = v.repeat(n_v_singles).reshape(q, n_v_singles)
        v_pairs = v.repeat(n_v_pairs).reshape(q, n_v_pairs)
        u_singles = u.repeat(n_u_singles).reshape(q, n_u_singles)
        u_pairs = u.repeat(n_u_pairs).reshape(q, n_u_pairs)

        # wrap it all nicely in a qaoa_parameters object
        params = cls(qaoa_circuit_params, q,
                     v_singles, v_pairs, u_singles, u_pairs)
        return params

    @classmethod
    def random(cls, qaoa_circuit_params: QAOACircuitParams, q: int, seed: int = None):
        """
        Returns
        -------
        FourierExtendedParams
            randomly initialised ``FourierExtendedParams`` object
        """
        if seed is not None:
            np.random.seed(seed)

        p = qaoa_circuit_params.p
        n_u_singles = len(
            qaoa_circuit_params.cost_hamiltonian.single_qubit_coeffs)
        n_u_pairs = len(qaoa_circuit_params.cost_hamiltonian.pair_qubit_coeffs)
        n_v_singles = len(
            qaoa_circuit_params.mixer_hamiltonian.single_qubit_coeffs)
        n_v_pairs = len(
            qaoa_circuit_params.mixer_hamiltonian.pair_qubit_coeffs)

        v_singles = np.random.uniform(0, np.pi, (q, n_v_singles))
        v_pairs = np.random.uniform(0, np.pi, (q, n_v_pairs))
        u_singles = np.random.uniform(0, np.pi, (q, n_u_singles))
        u_pairs = np.random.uniform(0, np.pi, (q, n_u_pairs))

        params = cls(qaoa_circuit_params, q,
                     v_singles, v_pairs, u_singles, u_pairs)
        return params

    @classmethod
    def empty(cls, qaoa_circuit_params: QAOACircuitParams, q: int):
        """
        Initialise Fourier extended parameters with empty arrays
        """

        v_singles = np.empty(
            (q, len(qaoa_circuit_params.mixer_hamiltonian.single_qubit_coeffs)))
        v_pairs = np.empty(
            (q, len(qaoa_circuit_params.mixer_hamiltonian.pair_qubit_coeffs)))
        u_singles = np.empty(
            (q, len(qaoa_circuit_params.cost_hamiltonian.single_qubit_coeffs)))
        u_pairs = np.empty(
            (q, len(qaoa_circuit_params.cost_hamiltonian.pair_qubit_coeffs)))

        return cls(qaoa_circuit_params, q,
                   v_singles, v_pairs, u_singles, u_pairs)

    def plot(self, ax=None, **kwargs):
        # if ax is None:
        #     fig, ax = plt.subplots()

        # ax.plot(dct(self.v, n=self.p, axis=0),
        #         label="betas", marker="s", ls="", **kwargs)
        # if not _is_iterable_empty(self.u_singles):
        #     ax.plot(dst(self.u_singles, n=self.p),
        #             label="gammas_singles", marker="^", ls="", **kwargs)
        # if not _is_iterable_empty(self.u_pairs):
        #     ax.plot(dst(self.u_pairs, n=self.p),
        #             label="gammas_pairs", marker="v", ls="", **kwargs)
        # ax.set_xlabel("timestep")
        # ax.legend()

        betas_singles = dct(self.v_singles, n=self.p, axis=0)
        betas_pairs = dct(self.v_pairs, n=self.p, axis=0)
        gammas_singles = dst(self.u_singles, n=self.p, axis=0)
        gammas_pairs = dst(self.u_pairs, n=self.p, axis=0)

        p = self.p
        q = self.q
        list_pq_ = [q, q, q, q, p, p, p, p]
        list_pq_names_ = ["q", "q", "q", "q", "p", "p", "p", "p"]

        list_names_ = ["v singles", "v pairs", "u singles", "u pairs",
                        "betas singles", "betas pairs", "gammas singles", "gammas pairs"] 
        list_values_ = [self.v_singles % (2*(np.pi)) , self.v_pairs % (2*(np.pi)) , 
                        self.u_singles % (2*(np.pi)) , self.u_pairs % (2*(np.pi)) ,
                        betas_singles % (2*(np.pi)) , betas_pairs % (2*(np.pi)) , 
                        gammas_singles % (2*(np.pi)) , gammas_pairs % (2*(np.pi)) ]

        list_names, list_values = list_names_.copy(), list_values_.copy()
        list_pq, list_pq_names = list_pq_.copy(), list_pq_names_.copy()

        n_pop = 0
        for i in range(len(list_values_)):
            if list_values_[i].size == 0: 
                list_values.pop(i-n_pop)
                list_names.pop(i-n_pop)
                list_pq.pop(i-n_pop)
                list_pq_names.pop(i-n_pop)
                n_pop += 1

        n = len(list_values)

        if ax is None:
            fig , ax = plt.subplots((n+1)//2, 2, figsize =(9, 4*(n+1)//2))

        fig.tight_layout(pad=4.0)

        for k, (name, values) in enumerate(zip(list_names, list_values)):
            i, j = k//2 , k%2
            pq = list_pq[k]
            pq_name = list_pq_names[k]
            axes = ax[i,j] if n>2 else ax[k]

            if values.size == pq:
                axes.plot(values.T[0], marker="^", color="green", ls="", **kwargs)
                axes.set_xlabel(pq_name, fontsize=12)
                axes.set_title(name)
                axes.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            elif values.size > 0:
                n_terms = values.shape[1]
                plt1 = axes.pcolor(np.arange(pq), np.arange(n_terms) , values.T, vmin=0, vmax=2*np.pi, cmap="seismic")
                axes.set_aspect(pq/n_terms)
                axes.xaxis.set_major_locator(MaxNLocator(integer=True))
                axes.yaxis.set_major_locator(MaxNLocator(integer=True))
                axes.set_ylabel("terms")
                axes.set_xlabel(pq_name)
                axes.set_title(name)

                plt.colorbar(plt1, **kwargs)

        if j == 0:
            ax[i,j+1].axis('off')
