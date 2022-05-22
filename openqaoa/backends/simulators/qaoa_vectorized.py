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

"""
Wavefunction simulator with methods focused on fast QAOA implementations.
Can easily be extended to do the full suite of operations in an ordinary simulator.
"""
from typing import Union, List, Tuple, Type, Optional
import numpy as np
from scipy.sparse import csc_matrix, kron, diags
from scipy.linalg import expm

from ...utilities import qaoa_probabilities
from ...basebackend import QAOABaseBackendStatevector, QuantumCircuitBase
from ...qaoa_parameters.baseparams import QAOACircuitParams, QAOAVariationalBaseParams
from ...qaoa_parameters.operators import Hamiltonian


# Pauli gates
constI = csc_matrix(np.eye(2))
constX = csc_matrix(np.array([[0,1], [1,0]]))
constY = csc_matrix(np.array([[0, -1j], [1j, 0]]))
constZ = csc_matrix(np.array([[1,0], [0,-1]]))

# Single qubit gates
constH = csc_matrix((1/np.sqrt(2))*np.array([[1, 1], [1, -1]]))
constS = csc_matrix(np.array([[1, 0], [0, 1j]]))
constT = csc_matrix(np.array([[1, 0], [0, np.exp(1j*np.pi/4)]]))

# Two-qubit gates
constCNOT = csc_matrix(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))
constCZ = csc_matrix(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]))

# Projection (measurement) operators
P0 = csc_matrix(np.array([[1, 0],[0, 0]]))
P1 = csc_matrix(np.array([[0, 0],[0, 1]]))

# Parametrised rotations
def RX(theta: float) -> csc_matrix:
        return csc_matrix(expm(-1j*theta*constX/2))
            
def RY(theta: float) -> csc_matrix:
    return csc_matrix(expm(-1j*theta*constY/2))

def RZ(theta: float) -> csc_matrix:
    return csc_matrix(expm(-1j*theta*constZ/2))

def CR_Z(theta: float) -> csc_matrix:
    return kron(P0, constI, format='csc') + kron(P1, RZ(theta), format='csc') 

def ZZ(theta: float) -> csc_matrix:
    return diags([1,np.exp(-1j*theta),np.exp(-1j*theta),1],0,format='csc')

def CPHASE(theta: float) -> csc_matrix:
    return diags([1,1,1,np.exp(1j*theta)],0,format='csc')
    

def _get_perm(n_qubits: int, qubits: list) -> list:
    """
    Helper function for several methods below.

    Parameters
    ----------
    n_qubits:
        total number of qubits in the register

    qubits:
        the qubits being permuted to the beginning of the register

    Returns
    -------
    perm:
        the permutation to apply to the wavefunction to bring the
        active qubits to the start of the register

    perminv:
        the inverse permutation, taking them back to where they belong from the beginning
    """

    # Get modified indices of qubits to coincide with indexing from the right
    qubits = [np.arange(n_qubits)[::-1][i] for i in qubits]

    # Permute the active qubits to the beginning of the register
    other_qubits = list(set(np.arange(n_qubits)) - set(qubits))
    perm = qubits + other_qubits
    perminv = list(np.argsort(perm))

    return perm, perminv


def _permute_qubits(wavefn: np.ndarray, perm: list) -> np.ndarray:
    """
    Reorganises the wavefunction components according to the specified qubit permutation.

    Parameters
    ----------
    wavefn:
        The current wavefunction of the register

    perm:
        The permutation according to which the register is to be reorganised.

    Returns
    -------
    wavefn:
        The permuted wavefunction.
    """

    # Permute according to specified perm
    wavefn = np.transpose(wavefn, perm)

    return wavefn


def _build_cost_hamiltonian(n_qubits: int,
                            cost_hamiltonian: Type[Hamiltonian]) -> np.array:
    """
    Builds the cost Hamiltonian as a vector, since it is diagonal.
    Output is an ndarray of shape [2]*n_qubits, for use in the run_measure_exp_val value method.

    Parameters
    ----------
    n_qubits:
        number of qubits

    cost_hamiltonian:
        Hamiltonian object containing information about single/2-qubit terms and their weights.

    Returns
    -------
    ham_op:
        the Hamiltonian as a diagonal matrix reshaped to a [2]*n_qubits dimensional array

    """

    bias_qubits = cost_hamiltonian.qubits_singles
    biases = cost_hamiltonian.single_qubit_coeffs
    pairs = cost_hamiltonian.qubits_pairs
    coeffs = cost_hamiltonian.pair_qubit_coeffs

    terms = cost_hamiltonian.terms
    weights = cost_hamiltonian.coeffs

    ## ZZ operator on first two qubits, identity on all others to the right
    # Diagonal matrix (vector) for cost function
    iden_plus = np.ones(2 ** (n_qubits - 2), dtype=int)
    iden_minus = -1 * np.ones(2 ** (n_qubits - 1), dtype=int)
    ZZ_op = np.hstack((iden_plus, np.hstack((iden_minus, iden_plus))))
    ZZ_op.shape = [2] * n_qubits

    ## Z operator on first qubit, identity on all others to the right
    Z_op = np.hstack((np.ones(2 ** (n_qubits - 1), dtype=int), -1 * np.ones(2 ** (n_qubits - 1), dtype=int)))
    Z_op.shape = [2] * n_qubits

    ham_op = np.zeros([2] * n_qubits)
    for j in range(len(pairs)):
        perm, perminv = _get_perm(n_qubits, pairs[j].qubit_indices)
        ZZ_op = _permute_qubits(ZZ_op, perminv)

        ham_op += coeffs[j] * ZZ_op

        ZZ_op = _permute_qubits(ZZ_op, perm)

    for j in range(len(bias_qubits)):
        perm, perminv = _get_perm(n_qubits, [bias_qubits[j].qubit_indices])
        Z_op = _permute_qubits(Z_op, perminv)

        ham_op += biases[j] * Z_op

        Z_op = _permute_qubits(Z_op, perm)

    #add the constant term from the hamiltonian
    ham_op += cost_hamiltonian.constant

    return ham_op


class QAOAvectorizedBackendSimulator(QAOABaseBackendStatevector):
    """
    A simulator class for quantum circuits, oriented to QAOA in particular.

    Works by permuting the qubits to be acted upon by a gate to the beginning of the register,
    applying the gate, then undoing the permutation.

    Qubit labelling begins from the right, so that the right-most qubit has label 0,
    and the left-most has label n_qubits-1.

    Parameters
    ----------
    circuit_params: QAOACircuitParams
        circuit_params
    prepend_state: np.array
        prepend_state
    append_state: np.array
        append_state
    init_hadamard: bool
        init_hadamard
    cvar_alpha: float
    """
    def __init__(self,
                 circuit_params: QAOACircuitParams,
                 prepend_state: Optional[Union[np.ndarray, List[complex]]],
                 append_state: Optional[Union[np.ndarray, List[complex]]],
                 init_hadamard: bool,
                 cvar_alpha: float = 1):
        
        assert cvar_alpha == 1,  "Please use the shot-based simulator for simulations with cvar_alpha < 1"
        
        QAOABaseBackendStatevector.__init__(self, circuit_params,
                                         prepend_state,
                                         append_state,
                                         init_hadamard,
                                         cvar_alpha)

    # Apply gate methods
    def apply_rx(self, qubit_1: int, rotation_angle: float):
        """
        Applies the RX(`theta` = `rotation_angle`) gate on `qubit_1` in a vectorized way.

        Parameters
        ----------
        qubit_1:
            Qubit index to apply gate.

        rotation_angle:
            Angle to be rotatd.

        Returns
        -------
        None
        """

        C = np.cos(rotation_angle / 2)
        S = -1j * np.sin(rotation_angle / 2)

        self.wavefn = (C * self.wavefn) + (S * np.flip(self.wavefn, qubit_1))

    def apply_ry(self, qubit_1: int, rotation_angle: float):
        """
        Applies the RY(`theta` = `rotation_angle`) gate on `qubit_1` in a vectorized way.

        Parameters
        ----------
        qubit_1:
            Qubit index to apply gate.

        rotation_angle:
            Angle to be rotatd.

        Returns
        -------
        None
        """
        
        wfn = self.wavefn.copy()

        # flip
        wfn = np.flip(wfn, qubit_1)

        # multiply slices with i/-i
        slc_0 = tuple(0 if i == self.n_qubits - qubit_1 - 1
                       else slice(None) for i in range(self.n_qubits))
        slc_1 = tuple(1 if i == self.n_qubits - qubit_1 - 1
                       else slice(None) for i in range(self.n_qubits))
        wfn[slc_0] *= -1j
        wfn[slc_1] *= 1j

        C = np.cos(rotation_angle / 2)
        S = -1j * np.sin(rotation_angle / 2)

        self.wavefn = (C * self.wavefn) + (S * wfn)

    def apply_rz(self, qubit_1: int, rotation_angle: float):
        """
        Applies the RZ(`theta` = `rotation_angle`) gate on `qubit_1` in a vectorized way.

        Parameters
        ----------
        qubit_1:
            Qubit index to apply gate.

        rotation_angle:
            Angle to be rotatd.

        Returns
        -------
        None
        """

        slc_0 = tuple(0 if i == self.n_qubits - qubit_1 - 1
                       else slice(None) for i in range(self.n_qubits))
        slc_1 = tuple(1 if i == self.n_qubits - qubit_1 - 1
                       else slice(None) for i in range(self.n_qubits))

        self.wavefn[slc_0] *= np.exp(-1j * rotation_angle/2)
        self.wavefn[slc_1] *= np.exp(1j * rotation_angle/2)

    def apply_rxx(self, qubit_1: int, qubit_2: int, rotation_angle: float):
        """
        Applies the RXX(`theta` = `rotation_angle`) gate on `qubit_1` and `qubit_2` in a vectorized way.

        Parameters
        ----------
        qubit_1:
            First qubit index to apply gate.
            
        qubit_2:
            Second qubit index to apply gate.

        rotation_angle:
            Angle to be rotatd.

        Returns
        -------
        None
        """
        
        wfn = self.wavefn.copy()

        wfn = np.flip(np.flip(wfn, qubit_1), qubit_2)

        C = np.cos(rotation_angle / 2)
        S = -1j * np.sin(rotation_angle / 2)
        self.wavefn = (C * self.wavefn) + (S * wfn)

    def apply_ryy(self, qubit_1: int, qubit_2: int, rotation_angle: float):
        """
        Applies the RYY(`theta` = `rotation_angle`) gate on `qubit_1` and `qubit_2` in a vectorized way.

        Parameters
        ----------
        qubit_1:
            First qubit index to apply gate.
            
        qubit_2:
            Second qubit index to apply gate.

        rotation_angle:
            Angle to be rotatd.

        Returns
        -------
        None
        """
        
        wfn = self.wavefn.copy()

        wfn = np.flip(np.flip(wfn, qubit_1), qubit_2)

        slc_q1_0 = tuple(0 if i == self.n_qubits - qubit_1 - 1
                         else slice(None) for i in range(self.n_qubits))
        slc_q1_1 = tuple(1 if i == self.n_qubits - qubit_1 - 1
                         else slice(None) for i in range(self.n_qubits))
        slc_q2_0 = tuple(0 if i == self.n_qubits - qubit_2 - 1
                         else slice(None) for i in range(self.n_qubits))
        slc_q2_1 = tuple(1 if i == self.n_qubits - qubit_2 - 1
                         else slice(None) for i in range(self.n_qubits))

        wfn[slc_q1_0] *= np.exp(-1j * rotation_angle)
        wfn[slc_q1_1] *= np.exp(1j * rotation_angle)

        wfn[slc_q2_0] *= np.exp(-1j * rotation_angle)
        wfn[slc_q2_1] *= np.exp(1j * rotation_angle)

        C = np.cos(rotation_angle / 2)
        S = -1j * np.sin(rotation_angle / 2)
        self.wavefn = (C * self.wavefn) + (S * wfn)

    def apply_rzz(self, qubit_1: int, qubit_2: int, rotation_angle: float):
        """
        Applies the RZZ(`theta` = `rotation_angle`) gate on `qubit_1` and `qubit_2` in a vectorized way.

        Parameters
        ----------
        qubit_1:
            First qubit index to apply gate.
            
        qubit_2:
            Second qubit index to apply gate.

        rotation_angle:
            Angle to be rotatd.

        Returns
        -------
        None
        """
        
        slc_pair00 = tuple(0 if i in [self.n_qubits - qubit_1 - 1, self.n_qubits - qubit_2 - 1, ]
                           else slice(None) for i in range(self.n_qubits))
        slc_pair11 = tuple(1 if i in [self.n_qubits - qubit_1 - 1, self.n_qubits - qubit_2 - 1, ]
                           else slice(None) for i in range(self.n_qubits))

        self.wavefn[slc_pair00] *= np.exp(-1j * rotation_angle)
        self.wavefn[slc_pair11] *= np.exp(-1j * rotation_angle)


    def apply_rxy(self, qubit_1: int, qubit_2: int, rotation_angle: float):
        """
        Applies the RXY(`theta` = `rotation_angle`) gate on `qubit_1` and `qubit_2` in a vectorized way.

        Parameters
        ----------
        qubit_1:
            First qubit index to apply gate.
            
        qubit_2:
            Second qubit index to apply gate.

        rotation_angle:
            Angle to be rotatd.

        Returns
        -------
        None
        """
        
        wfn = self.wavefn.copy()

        # Action of X part
        wfn = np.flip(wfn, qubit_1)

        # Action of Y part
        wfn = np.flip(wfn, qubit_2)
        slc_q2_0 = tuple(0 if i == self.n_qubits - qubit_2 - 1
                         else slice(None) for i in range(self.n_qubits))
        slc_q2_1 = tuple(1 if i == self.n_qubits - qubit_2 - 1
                         else slice(None) for i in range(self.n_qubits))
        wfn[slc_q2_0] *= -1j
        wfn[slc_q2_1] *= 1j

        C = np.cos(rotation_angle / 2)
        S = -1j * np.sin(rotation_angle / 2)
        self.wavefn = (C * self.wavefn) + (S * wfn)

    def apply_rxz(self, qubit_1: int, qubit_2: int, rotation_angle: float):
        """
        Applies the RXZ(`theta` = `rotation_angle`) gate on `qubit_1` and `qubit_2` in a vectorized way.

        Parameters
        ----------
        qubit_1:
            First qubit index to apply gate.
            
        qubit_2:
            Second qubit index to apply gate.

        rotation_angle:
            Angle to be rotatd.

        Returns
        -------
        None
        """
        
        wfn = self.wavefn.copy()

        # Action of X part
        wfn = np.flip(wfn, qubit_1)

        # Action of Z part
        slc_q2_0 = tuple(0 if i == self.n_qubits - qubit_2 - 1
                      else slice(None) for i in range(self.n_qubits))
        slc_q2_1 = tuple(1 if i == self.n_qubits - qubit_2 - 1
                      else slice(None) for i in range(self.n_qubits))
        wfn[slc_q2_0] *= 1
        wfn[slc_q2_1] *= -1

        C = np.cos(rotation_angle / 2)
        S = -1j * np.sin(rotation_angle / 2)
        self.wavefn = (C * self.wavefn) + (S * wfn)

    def apply_ryz(self, qubit_1: int, qubit_2: int, rotation_angle: float):
        """
        Applies the RYZ(`theta` = `rotation_angle`) gate on `qubit_1` and `qubit_2` in a vectorized way.

        Parameters
        ----------
        qubit_1:
            First qubit index to apply gate.
            
        qubit_2:
            Second qubit index to apply gate.

        rotation_angle:
            Angle to be rotatd.

        Returns
        -------
        None
        """
        
        wfn = self.wavefn.copy()

        # Action of Y part
        wfn = np.flip(wfn, qubit_1)
        slc_q1_0 = tuple(0 if i == self.n_qubits - qubit_1 - 1
                         else slice(None) for i in range(self.n_qubits))
        slc_q1_1 = tuple(1 if i == self.n_qubits - qubit_1 - 1
                         else slice(None) for i in range(self.n_qubits))

        # Action of Z part
        slc_q2_0 = tuple(0 if i == self.n_qubits - qubit_2 - 1
                         else slice(None) for i in range(self.n_qubits))
        slc_q2_1 = tuple(1 if i == self.n_qubits - qubit_2 - 1
                         else slice(None) for i in range(self.n_qubits))
        wfn[slc_q2_0] *= 1
        wfn[slc_q2_1] *= -1

        C = np.cos(rotation_angle / 2)
        S = -1j * np.sin(rotation_angle / 2)
        self.wavefn = (C * self.wavefn) + (S * wfn)
        
    def apply_hadamard(self, qubit_1:int):
        """
        Applies the Hadamard gate on `qubit_1` in a vectorized way. Only used when init_hadamard is true.
        TODO : Combine init_hadamard and prepend_state into one.

        Parameters
        ----------
        qubit_1:
            First qubit index to apply gate.

        Returns
        -------
        None
        """
        
        # vectorized hadamard gate, for when init_hadamard = True
        wfn = self.wavefn.copy()

        slc_0 = tuple(0 if i == self.n_qubits - qubit_1 - 1
                       else slice(None) for i in range(self.n_qubits))
        slc_1 = tuple(1 if i == self.n_qubits - qubit_1 - 1
                       else slice(None) for i in range(self.n_qubits))
        wfn[slc_1] *= -1
        wfn[slc_0] += self.wavefn[slc_1]
        wfn[slc_1] += self.wavefn[slc_0]

        self.wavefn = wfn/np.sqrt(2)

    def qaoa_circuit(self,
                     params: Type[QAOAVariationalBaseParams]):
        """
        Executes the entire QAOA circuit, with angles specified within `params`.
        Steps:
        1) creates a (2,...,2) dimensional matrix that represents a 2**n dimensional wavefunction
        2) modify it according to the prepend_state option.
        3) Modify it according to init_hadamard option.
        4) Modify it according to list of gates in `params`.
        5) Modify it accoding to append_state option.

        Parameters
        ----------
        params:
            QAOAVariationalBaseParams object that contains rotation angles and gates to be applied.

        Returns
        -------
        None
        """
        
        if self.n_qubits > 0:
            self.wavefn = np.zeros((2**self.n_qubits,),dtype=complex)
            self.wavefn[0] = 1
            self.wavefn = self.wavefn.reshape([2] * self.n_qubits)
        else:
            self.wavefn = []
            
        # Handle prepend state
        if self.prepend_state is not None:

            if isinstance(self.prepend_state, np.ndarray) and (np.shape(self.prepend_state) == np.shape(self.wavefn) or np.shape(self.prepend_state) == (2**self.n_qubits,)):
                self.wavefn = self.prepend_state
            else:
                raise ValueError('Error : Unsupported prepend_state specified (Not an ndarray, not of shape (2**n,), or not of shape (2, 2, ..., 2)).')

        # Handle init_hadamard
        if self.init_hadamard:
            for i in range(self.n_qubits):
                self.apply_hadamard(i)
        
        # Assign angles and apply gates
        self.assign_angles(params)
        
        low_level_gate_list = []
        for each_gate in self.pseudo_circuit:
            low_level_gate_list.extend(each_gate.decomposition('trivial'))

        for each_tuple in low_level_gate_list:
            gate = each_tuple[0]()
            gate.apply_vector_gate(*each_tuple[1],self)

        # Handle append state
        if self.append_state is not None:
            
            if isinstance(self.append_state, np.ndarray) and np.shape(self.append_state) == (2**self.n_qubits, 2**self.n_qubits):
                
                # check unitarity of append_state matrix
                if np.allclose(np.eye(2**self.n_qubits), self.append_state.dot(self.append_state.conj().T)):
                    # Flatten (2,...,2) shaped wfn into a 2**n-dim column vector before multiplying with unitary matrix, ...
                    self.wavefn = np.matmul(self.append_state, self.wavefn.flatten())
                    # then re-shape it back to (2,...,2)
                    self.wavefn = self.wavefn.reshape([2] * self.n_qubits)
    
                else:
                    raise ValueError('append_state is not a unitary matrix')
                    
            else:
                raise ValueError('Unsupported append_state specified (Not an ndarray, or not of shape (2**n, 2**n).')
        

                
    def wavefunction(self,
                     params: Type[QAOAVariationalBaseParams] = None) -> list:

        """
        Get the wavefunction of the state produced by the parametric circuit.

        Parameters
        ----------
        params:
            The QAOA parameters - an object of one of the parameter classes, containing
            hyperparameters and variable parameters.

        Returns
        -------
        wf:
            A list of the wavefunction amplitudes.
        """

        self.qaoa_circuit(params)

        self.wavefn.shape = 2 ** self.n_qubits
        
        # Make format same as ProjectQ
        wf = [(component) for component in self.wavefn]

        return wf

    def expectation(self, params: Type[QAOAVariationalBaseParams]) -> float:
        """
        Call the execute function on the circuit to compute the
        expectation value of the Quantum Circuit w.r.t cost operator

        Returns
        -------
        exp_val:
            The expectation value of the cost function wrt the state generated by the circuit.
        """

        self.qaoa_circuit(params)

        # Reshape wavefunction
        wavefn_ = self.wavefn

        self.probability = qaoa_probabilities(self.wavefn.flatten())

        # Build the Hamiltonian operator as an array
        ham_op = _build_cost_hamiltonian(self.n_qubits, self.cost_hamiltonian)

        # Compute the expectation value and its standard deviation
        ham_wf = ham_op * wavefn_
        exp_val = np.real(np.vdot(wavefn_, ham_wf))

        out = exp_val

        self.reset_circuit()
        return out

    def expectation_w_uncertainty(self,
                                  params: Type[QAOAVariationalBaseParams]) -> Tuple[float, float]:
        """
        Call the execute function on the circuit to compute the
        expectation value of the Quantum Circuit w.r.t cost operator
        along with its uncertainty

        Returns
        -------
        exp_val:
            The expectation value of the cost function wrt the state generated by the circuit.
        std_dev:
            The standard deviation of the cost function wrt the state generated by the circuit.
        """

        self.qaoa_circuit(params)

        # Reshape wavefunction
        wavefn_ = self.wavefn

        self.probability = qaoa_probabilities(self.wavefn.flatten())

        # Build the Hamiltonian operator as an array
        ham_op = _build_cost_hamiltonian(self.n_qubits, self.cost_hamiltonian)

        # Compute the expectation value and its standard deviation
        ham_wf = ham_op * wavefn_
        exp_val = np.real(np.vdot(wavefn_, ham_wf))

        exp_val_sq = np.real(np.vdot(ham_wf, ham_wf))
        std_dev = (exp_val_sq - exp_val ** 2) ** 0.5
        out = exp_val, std_dev

        self.reset_circuit()
        return out

    def reset_circuit(self):
        """
        Reset the circuit by resetting the wavefunction. To be removed.
        """
        pass
            
    def circuit_to_qasm(self):
        """
        A method to convert the entire QAOA `QuantumCircuit` object into 
        a OpenQASM string
        """
        raise NotImplementedError()
