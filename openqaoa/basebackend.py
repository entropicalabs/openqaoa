# Copyright 2021 Entropica Labs
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

"""
#   Measure_Lib support will added in a future version. Particularly essential
#   when the Hamiltionian have non-commuting terms.

# from entropica_qaoa.vqe.measurelib import (append_measure_register,
#                                            commuting_decomposition,
#                                            sampling_expectation,
#                                            kron_eigs,
#                                            base_change_fun,
#                                            wavefunction_expectation)
"""
from abc import ABC, abstractmethod, abstractproperty
from typing import Union, List, Type, Dict, Optional, Any, Tuple
import numpy as np
from copy import deepcopy

from .devices import DeviceBase
from .qaoa_parameters.pauligate import PauliGate, TwoPauliGate
from .qaoa_parameters.baseparams import QAOACircuitParams, QAOAVariationalBaseParams
from .utilities import qaoa_probabilities
from .cost_function import cost_function

class QuantumCircuitBase:
    """
    Phantom class to indicate Quantum Circuits constructed using
    several acceptable services. For instance, IBMQ, PyQuil
    """
    pass


class VQABaseBackend(ABC):
    """
    This is the Abstract Base Class over which other classes will be built.
    Since, this is an Abstract Base class, in order to prevent its initialisation
    the class methods -- ``__init__`` and ``__cal__`` will be decorated as
    `abstractmethods`.

    The Child classes MUST implement and override these abstract methods in their
    implementation specific to their needs.

    NOTE:
        In addition one can also implement other methods which are not 
        necessitated by the ``VQABaseBackend`` Base Class


    Parameters
    ----------
    prepend_state: `Union[QuantumCircuitBase,List[complex],np.ndarray]`
        The initial state to start the quantum circuit in the backend.
    append_state: `Union[QuantumCircuitBase,np.ndarray]`
        The final state to append to the quantum circuit in the backend.
    """

    @abstractmethod
    def __init__(self,
                 prepend_state: Optional[Union[QuantumCircuitBase, List[complex], np.ndarray]],
                 append_state: Optional[Union[QuantumCircuitBase, np.ndarray]]):
        """The constructor. See class docstring"""
        self.prepend_state = prepend_state
        self.append_state = append_state

    @abstractmethod
    def expectation(self, params: Any) -> float:
        """
        Call the execute function on the circuit to compute the 
        expectation value of the Quantum Circuit w.r.t cost operator
        """
        pass

    @abstractmethod
    def expectation_w_uncertainty(self, params: Any) -> Tuple[float, float]:
        """
        Call the execute function on the circuit to compute the 
        expectation value of the Quantum Circuit w.r.t cost operator
        along with its uncertainty 
        """
        pass

    @abstractproperty
    def exact_solution(self):
        """
        Use linear algebra to compute the exact solution of the problem 
        Hamiltonian classically.
        """
        pass


class QAOABaseBackend(VQABaseBackend):
    """
    This class inherits from the VQABaseBackend and needs to be backend
    agnostic to QAOA implementations on different devices and their
    respective SDKs.

    Parameters
    ----------
    circuit_params: `QAOACircuitParams`
        This object handles the information to design the QAOA circuit ansatz
    prepend_state: `Union[QuantumCircuitBase,List[complex]]` 
        Warm Starting the QAOA problem with some initial state other than the regular
        $|+ \\rangle ^{otimes n}$ 
    append_state: `Union[QuantumCircuitBase,List[complex]]`
        Appending a user-defined circuit/state to the end of the QAOA routine    
    init_hadamard: ``bool``
        Initialises the QAOA circuit with a hadamard when ``True``
    """

    def __init__(self,
                 circuit_params: QAOACircuitParams,
                 prepend_state: Optional[Union[QuantumCircuitBase, List[complex], np.ndarray]],
                 append_state: Optional[Union[QuantumCircuitBase, np.ndarray]],
                 init_hadamard: bool,
                 cvar_alpha: float = 1):

        super().__init__(prepend_state, append_state)

        self.circuit_params = circuit_params
        self.cost_hamiltonian = circuit_params.cost_hamiltonian
        self.n_qubits = self.cost_hamiltonian.n_qubits
        self.init_hadamard = init_hadamard
        self.cvar_alpha = cvar_alpha

        self.pseudo_circuit = deepcopy(self.circuit_params.pseudo_circuit)

    def assign_angles(self, params: QAOAVariationalBaseParams) -> List[PauliGate]:
        """
        Assigns the angle values of the variational parameters to the circuit gates
        specified as a list of gates in the ``pseudo_circuit``.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The variational parameters(angles) to be assigned to the circuit gates
        """
        # if circuit is non-parameterised, then assign the angle values to the circuit
        pseudo_pauli_circuit = self.pseudo_circuit

        for each_pauli in pseudo_pauli_circuit:
            pauli_label_index = each_pauli.pauli_label[2:]
            if isinstance(each_pauli, TwoPauliGate):
                if each_pauli.pauli_label[1] == 'mixer':
                    angle = params.mixer_2q_angles[pauli_label_index[0],
                                                   pauli_label_index[1]]
                elif each_pauli.pauli_label[1] == 'cost':
                    angle = params.cost_2q_angles[pauli_label_index[0],
                                                  pauli_label_index[1]]
            elif isinstance(each_pauli, PauliGate):
                if each_pauli.pauli_label[1] == 'mixer':
                    angle = params.mixer_1q_angles[pauli_label_index[0],
                                                   pauli_label_index[1]]
                elif each_pauli.pauli_label[1] == 'cost':
                    angle = params.cost_1q_angles[pauli_label_index[0],
                                                  pauli_label_index[1]]
            each_pauli.rotation_angle = angle
        self.pseudo_circuit = pseudo_pauli_circuit

    def obtain_angles_for_pauli_list(self,
                                     input_pauli_list: List[PauliGate],
                                     params: QAOAVariationalBaseParams) -> List[float]:
        """
        This method uses the pauli gate list information to obtain the pauli angles
        from the VariationalBaseParams object. The floats in the list are in the order
        of the input PauliGates list.

        Parameters
        ----------
        input_pauli_list: `List[PauliGate]`
            The PauliGates list
        params: `QAOAVariationalBaseParams`
            The variational parameters(angles) to be assigned to the circuit gates

        Returns
        -------
        angles_list: `List[float]`
            The list of angles in the order of gates in the `PauliGate` list
        """
        angle_list = []

        for each_pauli in input_pauli_list:
            pauli_label_index = each_pauli.pauli_label[2:]
            if isinstance(each_pauli, TwoPauliGate):
                if each_pauli.pauli_label[1] == 'mixer':
                    angle_list.append(params.mixer_2q_angles[pauli_label_index[0],
                                                             pauli_label_index[1]])
                elif each_pauli.pauli_label[1] == 'cost':
                    angle_list.append(params.cost_2q_angles[pauli_label_index[0],
                                                            pauli_label_index[1]])
            elif isinstance(each_pauli, PauliGate):
                if each_pauli.pauli_label[1] == 'mixer':
                    angle_list.append(params.mixer_1q_angles[pauli_label_index[0],
                                                             pauli_label_index[1]])
                elif each_pauli.pauli_label[1] == 'cost':
                    angle_list.append(params.cost_1q_angles[pauli_label_index[0],
                                                            pauli_label_index[1]])

        return angle_list

    @abstractmethod
    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> QuantumCircuitBase:
        """
        Construct the QAOA circuit and append the parameter values to obtain the final
        circuit ready for execution on the device.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters as a 1D array (derived from an object of one of 
            the parameter classes, containing hyperparameters and variable parameters).

        Returns
        -------
        quantum_circuit: `QuantumCircuitBase`    
            A Quantum Circuit object of type created by the respective
            backend service
        """
        pass

    @abstractmethod
    def get_counts(self, params: QAOAVariationalBaseParams) -> dict:
        """
        This method will be implemented in the child classes according to the type
        of backend used.
        """
        pass

    def expectation(self, params: QAOAVariationalBaseParams) -> float:
        """
        Compute the expectation value w.r.t the Cost Hamiltonian

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters - an object of one of the parameter classes, containing 
            variable parameters.

        Returns
        -------
        `float`
            expectation value of cost operator wrt to quantum state produced by QAOA circuit
        """
        counts = self.get_counts(params)
        cost = cost_function(
            counts, self.circuit_params.cost_hamiltonian, self.cvar_alpha)
        return cost

    def expectation_w_uncertainty(self,
                                  params: QAOAVariationalBaseParams) -> Tuple[float, float]:
        """
        Compute the expectation value w.r.t the Cost Hamiltonian and its uncertainty

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters - an object of one of the parameter classes, containing 
            variable parameters.

        Returns
        -------
        `Tuple[float]`
            expectation value and its uncertainty of cost operator wrt 
            to quantum state produced by QAOA circuit.
        """
        counts = self.get_counts(params)
        cost = cost_function(
            counts, self.circuit_params.cost_hamiltonian, self.cvar_alpha)
        cost_sq = cost_function(counts,
                                self.circuit_params.cost_hamiltonian.hamiltonian_squared,
                                self.cvar_alpha)

        uncertainty = np.sqrt(cost_sq - cost**2)

        return (cost, uncertainty)

    @abstractmethod
    def reset_circuit(self):
        """
        Reset the circuit attribute 
        """
        pass

    @property
    def exact_solution(self):
        """
        Computes exactly the minimum energy of the cost function and its 
        corresponding configuration of variables using standard numpy module.

        Returns
        -------
        (energy, config): `Tuple[float, list]`
            - The minimum eigenvalue of the cost Hamiltonian,
            - The minimum energy eigenvector as a binary array
              configuration: qubit-0 as the first element in the sequence    
        """
        register = self.circuit_params.qureg
        terms = self.cost_hamiltonian.terms
        coeffs = self.cost_hamiltonian.coeffs
        constant_energy = self.cost_hamiltonian.constant

        diag = np.zeros((2**len(register)))
        for i, term in enumerate(terms):
            out = np.real(coeffs[i])
            for qubit in register:
                if qubit in term.qubit_indices:
                    out = np.kron([1, -1], out)
                else:
                    out = np.kron([1, 1], out)
            diag += out

        # add the constant energy contribution
        diag += constant_energy

        # index = np.argmin(diag)
        energy = np.min(diag)
        indices = []
        for idx in range(len(diag)):
            if diag[idx] == energy:
                indices.append(idx)

        config_strings = [np.binary_repr(index, len(register))[
            ::-1] for index in indices]
        configs = [np.array([int(x) for x in config_str])
                   for config_str in config_strings]

        return energy, configs

    def bitstring_energy(self, bitstring: Union[List[int], str]) -> float:
        """
        Computes the energy of a given bitstring with respect to the cost Hamiltonian.

        Parameters
        ----------
        bitstring : `Union[List[int],str]`
            A list of integers 0 and 1 of length `n_qubits` representing a configuration.

        Returns
        -------
        energy: `float`
            The energy of a given bitstring with respect to the cost Hamiltonian.
        """
        energy = 0
        string_rev = bitstring
        terms = self.cost_hamiltonian.terms
        coeffs = self.cost_hamiltonian.coeffs
        constant_energy = self.cost_hamiltonian.constant

        for i, term in enumerate(terms):
            variables_product = np.prod([(-1)**string_rev[k] for k in term])
            energy += coeffs[i]*variables_product
        energy += constant_energy

        return energy

    @abstractmethod
    def circuit_to_qasm(self):
        """
        Implement a method to construct a QASM string from the current
        state of the QuantumCircuit for the backends
        """
        pass


class QAOABaseBackendStatevector(QAOABaseBackend):
    """
    Base backend class for a statevector simulator backend
    """

    @abstractmethod
    def wavefunction(self,
                     params: QAOAVariationalBaseParams) -> List[complex]:
        """
        Get the wavefunction of the state produced by the QAOA circuit.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams` 
            The QAOA parameters - an object of one of the parameter classes, containing 
            the variational parameters (angles).

        Returns
        -------
        wf: `List[complex]`
            A list of the wavefunction amplitudes.
        """
        pass

    def sample_from_wavefunction(self,
                                 params: QAOAVariationalBaseParams,
                                 n_samples: int) -> np.ndarray:
        """
        Get the shot-based measurement results from the statevector. The return object is
        a list of shot-results.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters as a 1D array (derived from an object of one of the
            parameter classes, containing hyperparameters and variable parameters).
        n_samples: `int`
            The number of measurement samples required; specified as integer

        Returns
        -------
        meas_samples: `np.ndarray`
            A list of measurement outcomes sampled from a statevector
        """
        wf = self.wavefunction(params)
        prob_vec = np.real(np.conjugate(wf)*wf)
        samples = np.random.choice(len(prob_vec), p=prob_vec, size=n_samples)
        samples = [np.binary_repr(num, self.n_qubits)[::-1]
                   for num in samples]

        return samples

    def probability_dict(self, params: QAOAVariationalBaseParams):
        """
        Get the counts style probability dictionary with all basis states
        and their corresponding probabilities. Constructed using the complete
        statevector

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters as a 1D array (derived from an object of one of the
            parameter classes, containing hyperparameters and variable parameters). 

        Returns
        -------
        prob_dict: `Dict[str, float]`
            A dictionary of all basis states and their corresponding probabilities.
        """

        wf = self.wavefunction(params)
        return qaoa_probabilities(wf)

    def get_counts(self, params: QAOAVariationalBaseParams, n_shots: int) -> Dict:
        """
        Measurement outcome vs frequency information from a circuit execution
        represented as a python dictionary

        Parameters
        ----------
        params: `VariationalBaseParams`
            The QAOA parameters as a 1D array (derived from an object of one of the
            parameter classes, containing hyperparameters and variable parameters).
        n_shots: `int`
            The number of measurement shots required; specified as integer

        Returns
        -------
        counts: `Dict[str, float]`
            A dictionary of measurement outcomes vs frequency sampled from a statevector
        """
        samples = self.sample_from_wavefunction(params, n_shots)

        unique_nums, frequency = np.unique(samples, return_counts=True)
        # unique_shots = [np.binary_repr(num, self.n_qubits)[
        #     ::-1] for num in unique_nums]
        counts = dict(zip(unique_nums, frequency))

        return counts


class QAOABaseBackendShotBased(QAOABaseBackend):
    """
    Implementation of Backend object specific to shot-based simulators and QPUs
    """

    def __init__(self,
                 circuit_params: QAOACircuitParams,
                 n_shots: int,
                 prepend_state: Optional[QuantumCircuitBase],
                 append_state: Optional[QuantumCircuitBase],
                 init_hadamard: bool,
                 cvar_alpha: float = 1):

        super().__init__(circuit_params, prepend_state,
                         append_state, init_hadamard, cvar_alpha)

        # assert self.n_qubits >= len(prepend_state.qubits), \
        # "Cannot attach a bigger circuit to the QAOA routine"
        # assert self.n_qubits >= len(append_state.qubits), \
        # "Cannot attach a bigger circuit to the QAOA routine"
        self.n_shots = n_shots

    @abstractmethod
    def get_counts(self, params: QAOAVariationalBaseParams) -> dict:
        """
        Measurement outcome vs frequency information from a circuit execution
        represented as a python dictionary

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters as a 1D array (derived from an object of one of the
            parameter classes, containing hyperparameters and variable parameters).

        Returns
        -------
        counts: `Dict[str, float]`
            A dictionary of measurement outcomes vs frequency sampled from a statevector
        """
        pass


class QAOABaseBackendCloud:
    """
    QAOA backend that can be accessed over the cloud offered by the
    respective provider through an API based access
    """

    def __init__(self, device: DeviceBase):
        self.device = device
        self.device.check_connection()


class QAOABaseBackendParametric:
    """
    Base class to indicate Parametric Circuit Backend
    """
    @abstractmethod
    def parametric_qaoa_circuit(self):
        pass
